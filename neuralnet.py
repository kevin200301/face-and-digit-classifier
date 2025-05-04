import numpy as np
import scipy.optimize as opt

class NeuralNetClassifier:
    def __init__(self, legalLabels, input_size, hidden1_size, hidden2_size, output_size, data_size, lmbda):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.data_size = data_size
        self.legalLabels = legalLabels
        self.lmbda = lmbda

        self.input_activation = np.ones((self.input_size + 1, data_size))
        self.hidden1_activation = np.ones((self.hidden1_size + 1, data_size))
        self.hidden2_activation = np.ones((self.hidden2_size + 1, data_size))
        self.output_activation = np.ones((self.output_size, data_size))

        self.input_weights = self.init_weights(self.hidden1_size, self.input_size + 1)
        self.hidden1_weights = self.init_weights(self.hidden2_size, self.hidden1_size + 1)
        self.output_weights = self.init_weights(self.output_size, self.hidden2_size + 1)

    def init_weights(self, rows, cols):
        epsilon = np.sqrt(6.0 / (rows + cols))
        return np.random.rand(rows, cols) * 2 * epsilon - epsilon

    def updatelmbda(self, lmbda):
        self.lmbda = lmbda

    def forwardprop(self, theta):
        # Unroll theta
        idx1 = self.hidden1_size * (self.input_size + 1)
        idx2 = idx1 + self.hidden2_size * (self.hidden1_size + 1)
        self.input_weights = theta[:idx1].reshape(self.hidden1_size, self.input_size + 1)
        self.hidden1_weights = theta[idx1:idx2].reshape(self.hidden2_size, self.hidden1_size + 1)
        self.output_weights = theta[idx2:].reshape(self.output_size, self.hidden2_size + 1)

        # Forward pass
        self.hidden1_activation[:-1, :] = self.sigmoid(self.input_weights.dot(self.input_activation))
        self.hidden2_activation[:-1, :] = self.sigmoid(self.hidden1_weights.dot(self.hidden1_activation))
        self.output_activation = self.sigmoid(self.output_weights.dot(self.hidden2_activation))

        cost = self.output_truth * np.log(self.output_activation) + (1 - self.output_truth) * np.log(1 - self.output_activation)
        reg = (np.sum(self.input_weights[:, :-1]**2) + 
               np.sum(self.hidden1_weights[:, :-1]**2) + 
               np.sum(self.output_weights[:, :-1]**2)) * self.lmbda / 2
        return (-cost.sum() + reg) / self.data_size

    def backprop(self, theta):
        # Unroll theta
        idx1 = self.hidden1_size * (self.input_size + 1)
        idx2 = idx1 + self.hidden2_size * (self.hidden1_size + 1)
        self.input_weights = theta[:idx1].reshape(self.hidden1_size, self.input_size + 1)
        self.hidden1_weights = theta[idx1:idx2].reshape(self.hidden2_size, self.hidden1_size + 1)
        self.output_weights = theta[idx2:].reshape(self.output_size, self.hidden2_size + 1)

        # Errors
        output_delta = self.output_activation - self.output_truth
        hidden2_delta = self.output_weights[:, :-1].T.dot(output_delta) * self.hidden2_activation[:-1, :] * (1 - self.hidden2_activation[:-1, :])
        hidden1_delta = self.hidden1_weights[:, :-1].T.dot(hidden2_delta) * self.hidden1_activation[:-1, :] * (1 - self.hidden1_activation[:-1, :])

        # Gradients
        output_grad = output_delta.dot(self.hidden2_activation.T) / self.data_size
        hidden2_grad = hidden2_delta.dot(self.hidden1_activation.T) / self.data_size
        input_grad = hidden1_delta.dot(self.input_activation.T) / self.data_size

        # Regularization
        output_grad[:, :-1] += self.lmbda * self.output_weights[:, :-1]
        hidden2_grad[:, :-1] += self.lmbda * self.hidden1_weights[:, :-1]
        input_grad[:, :-1] += self.lmbda * self.input_weights[:, :-1]

        return np.concatenate([input_grad.ravel(), hidden2_grad.ravel(), output_grad.ravel()])

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        features_train = [list(datum.values()) for datum in trainingData]
        train_set = np.array(features_train, np.int32)
        self.input_activation[:-1, :] = train_set.transpose()
        self.output_truth = self.genTruthMatrix(trainingLabels)

        theta = np.concatenate([self.input_weights.ravel(), self.hidden1_weights.ravel(), self.output_weights.ravel()])
        theta = opt.fmin_cg(self.forwardprop, theta, fprime=self.backprop, maxiter=100)

        # Unroll final theta
        idx1 = self.hidden1_size * (self.input_size + 1)
        idx2 = idx1 + self.hidden2_size * (self.hidden1_size + 1)
        self.input_weights = theta[:idx1].reshape(self.hidden1_size, self.input_size + 1)
        self.hidden1_weights = theta[idx1:idx2].reshape(self.hidden2_size, self.hidden1_size + 1)
        self.output_weights = theta[idx2:].reshape(self.output_size, self.hidden2_size + 1)

    def genTruthMatrix(self, trainLabels):
        truth = np.zeros((self.output_size, self.data_size))
        for i in range(self.data_size):
            label = trainLabels[i]
            if self.output_size == 1:
                truth[:, i] = label
            else:
                truth[label, i] = 1
        return truth

    def classify(self, data):
        features_test = [list(datum.values()) for datum in data]
        test_set = np.array(features_test, np.int32).transpose()

        # Resize activations
        num_samples = test_set.shape[1]
        self.input_activation = np.ones((self.input_size + 1, num_samples))
        self.hidden1_activation = np.ones((self.hidden1_size + 1, num_samples))
        self.hidden2_activation = np.ones((self.hidden2_size + 1, num_samples))
        self.output_activation = np.ones((self.output_size, num_samples))

        self.input_activation[:-1, :] = test_set

        self.hidden1_activation[:-1, :] = self.sigmoid(self.input_weights.dot(self.input_activation))
        self.hidden2_activation[:-1, :] = self.sigmoid(self.hidden1_weights.dot(self.hidden1_activation))
        self.output_activation = self.sigmoid(self.output_weights.dot(self.hidden2_activation))

        if self.output_size > 1:
            return np.argmax(self.output_activation, axis=0).tolist()
        else:
            return (self.output_activation > 0.5).ravel()

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))