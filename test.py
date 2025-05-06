import random
import samples
import util
import time
import perceptron
import neuralNetwork
import pytorchNeuralNetwork
import torch

numTraining = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
numTest = 100

def sampleTrainingData(rawData, labels, percent):
    assert len(rawData) == len(labels), "Data and labels must be the same length"
    assert 0 < percent <= 100, "Percent must be between 0 and 100"

    total = len(rawData)
    sample_size = int((percent / 100.0) * total)
    
    indices = list(range(total))
    random.shuffle(indices)
    selected_indices = indices[:sample_size]

    sampled_data = [rawData[i] for i in selected_indices]
    sampled_labels = [labels[i] for i in selected_indices]

    return sampled_data, sampled_labels

def basicFeatureExtractorDigit(datum):
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features



def main():
    data = 'faces'
    type = 'neuralnet'
    percent = 100

    if data == 'faces':
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)

        featureFunction = basicFeatureExtractorFace
        legalLabels = [0,1]
    elif data == 'digits':
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
        
        featureFunction = basicFeatureExtractorDigit
        legalLabels = [0,1,2,3,4,5,6,7,8,9]

    trainingData = list(map(featureFunction, rawTrainingData))
    validationData = list(map(featureFunction, rawValidationData))
    testData = list(map(featureFunction, rawTestData))

    sampledTrainingData, sampledTrainingLabels = sampleTrainingData(trainingData, trainingLabels, percent) # % of training data
    if type == 'perceptron':
        classifier = perceptron.PerceptronClassifier(legalLabels, 5) # 5 iterations
    elif type == 'neuralnet':
        if data == 'digits':
            classifier = neuralNetwork.NeuralNetClassifier(legalLabels, DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT, 128, 64, 10, percent, 0.001) # 100 ok for diff percentages?
        elif data == 'faces':
            classifier = neuralNetwork.NeuralNetClassifier(legalLabels, FACE_DATUM_WIDTH * FACE_DATUM_HEIGHT, 128, 64, 1, percent, 0.001) # 100 ok for diff percentages?
    elif type == 'pytorch':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if data == 'digits':
            classifier = pytorchNeuralNetwork.PytorchClassifier(legalLabels, 5, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT, device).to(device)
        elif data == 'faces':
            classifier = pytorchNeuralNetwork.PytorchClassifier(legalLabels, 5, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT, device).to(device)

    print ("Training...")
    start = time.time()
    classifier.train(sampledTrainingData, sampledTrainingLabels, validationData, validationLabels)
    end = time.time()
    print("Training Time:", end-start)

    print ("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    
    print ("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))

if __name__ == "__main__":
    main()