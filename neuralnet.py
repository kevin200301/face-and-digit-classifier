import forwardpass
import backpropagation
import weightupdate
import classificationMethod

def neuralnet(classificationMethod.ClassificationMethod):
    """
    Neural Network Classifier.
    
    This class implements a simple feedforward neural network with one hidden layer.
    It uses the sigmoid activation function and backpropagation for training.
    """
    def __init__(self, legalLabels):
        self.guess = None
        self.type = "neuralnet"

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        

    def classify(self, testData):
