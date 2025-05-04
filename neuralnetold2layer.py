import numpy as np
import scipy.optimize as opt

class NeuralNetClassifier:
    def __init__(self, legalLabels, input_size, hidden_size, output_size, data_size, lmbda):
        self.input_size = input_size  # without bias node
        self.hidden_size = hidden_size  # without bias node
        self.output_size = output_size
        self.data_size = data_size
        self.legalLabels = legalLabels
        self.lmbda = lmbda

        self.input_activation = np.ones((self.input_size + 1, data_size))  # add bias node
        self.hidden_activation = np.ones((self.hidden_size + 1, data_size))  # add bias node
        self.output_activation = np.ones((self.output_size, data_size))

        self.bias = np.ones((1, data_size))

        self.input_change = np.zeros((self.hidden_size, self.input_size + 1))
        self.output_change = np.zeros((self.output_size, self.hidden_size + 1))

        self.hidden_epsilon = np.sqrt(6.0 / (self.input_size + self.hidden_size))
        self.output_epsilon = np.sqrt(6.0 / (self.input_size + self.output_size))

        self.input_weights = np.random.rand(self.hidden_size, self.input_size + 1) * 2 * self.hidden_epsilon - self.hidden_epsilon
        self.output_weights = np.random.rand(self.output_size, self.hidden_size + 1) * 2 * self.output_epsilon - self.output_epsilon

    def updatelmbda(self, lmbda):
        self.lmbda = lmbda

    def forwardprop(self, theta):
        #reshape theta into two weights matrices
        self.input_weights = theta[0:self.hidden_size * (self.input_size + 1)].reshape((self.hidden_size, self.input_size + 1))
        self.output_weights = theta[-self.output_size * (self.hidden_size + 1):].reshape((self.output_size, self.hidden_size + 1))

        #hidden_size activation
        hidden_z = self.input_weights.dot(self.input_activation)
        self.hidden_activation[:-1, :] = self.sigmoid(hidden_z)

        #output_size activation
        output_z = self.output_weights.dot(self.hidden_activation)
        self.output_activation = self.sigmoid(output_z)

        #calculate J
        costMatrix = self.output_truth * np.log(self.output_activation) + (1 - self.output_truth) * np.log(
            1 - self.output_activation)
        regulations = (np.sum(self.output_weights[:, :-1] ** 2) + np.sum(self.input_weights[:, :-1] ** 2)) * self.lmbda / 2
        return (-costMatrix.sum() + regulations) / self.data_size
    
    def backprop(self, thetaVec):
        #reshape thetaVec into two weights matrices
        self.input_weights = thetaVec[0:self.hidden_size * (self.input_size + 1)].reshape((self.hidden_size, self.input_size + 1))
        self.output_weights = thetaVec[-self.output_size * (self.hidden_size + 1):].reshape((self.output_size, self.hidden_size + 1))

        #calculate lower case delta
        outputError = self.output_activation - self.output_truth
        #calculate derivative
        hiddenError = self.output_weights[:, :-1].T.dot(outputError) * (self.hidden_activation[:-1:] * (1 - self.hidden_activation[:-1:]))

        #calculate upper case delta
        self.output_change = outputError.dot(self.hidden_activation.T) / self.data_size
        self.input_change = hiddenError.dot(self.input_activation.T) / self.data_size

        #add regulations
        self.output_change[:, :-1].__add__(self.lmbda * self.output_weights[:, :-1])
        self.input_change[:, :-1].__add__(self.lmbda * self.input_weights[:, :-1])

        return np.append(self.input_change.ravel(), self.output_change.ravel())


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.size_train = len(list(trainingData))
        features_train = []
        for datum in trainingData:
            feature = list(datum.values())
            features_train.append(feature)
        train_set = np.array(features_train, np.int32)

        iteration = 100
        self.input_activation[:-1, :] = train_set.transpose()
        self.output_truth = self.genTruthMatrix(trainingLabels)

        theta = np.append(self.input_weights.ravel(), self.output_weights.ravel())
        theta = opt.fmin_cg(self.forwardprop, theta, fprime=self.backprop, maxiter=iteration)
        self.input_weights = theta[0:self.hidden_size * (self.input_size + 1)].reshape((self.hidden_size, self.input_size + 1))
        self.output_weights = theta[-self.output_size * (self.hidden_size + 1):].reshape((self.output_size, self.hidden_size + 1))

    def genTruthMatrix(self, trainLabels):
            truth = np.zeros((self.output_size, self.data_size))
            for i in range(self.data_size):
                label = trainLabels[i]
                if self.output_size == 1:
                    truth[:,i] = label
                else:
                    truth[label, i] = 1
            return truth
    
    def classify(self, data):
        self.size_test = len(list(data))
        features_test = []
        for datum in data:
            feature = list(datum.values())
            features_test.append(feature)
        test_set = np.array(features_test, np.int32)
        feature_test_set = test_set.transpose()

        if feature_test_set.shape[1] != self.input_activation.shape[1]:
            self.input_activation = np.ones((self.input_size + 1, feature_test_set.shape[1]))
            self.hidden_activation = np.ones((self.hidden_size + 1, feature_test_set.shape[1]))
            self.output_activation = np.ones((self.output_size + 1, feature_test_set.shape[1]))
        self.input_activation[:-1, :] = feature_test_set

        hidden_z = self.input_weights.dot(self.input_activation)
        self.hidden_activation[:-1, :] = self.sigmoid(hidden_z)

        output_z = self.output_weights.dot(self.hidden_activation)
        self.output_activation = self.sigmoid(output_z)
        if self.output_size > 1:
            return np.argmax(self.output_activation, axis=0).tolist()
        else:
            return (self.output_activation>0.5).ravel()
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))