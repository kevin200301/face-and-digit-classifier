import torch
import torch.nn as nn
import torch.optim as optim
import samples
# import classificationMethod

class PyTorch(nn.Module):
    def __init__(self):
        super(PyTorch, self).__init__()

        self.linear1 = nn.Linear(28*28, 100)   # First hidden layer
        self.linear2 = nn.Linear(100, 50) # Second hidden layer
        self.final = nn.Linear(50, 10)  # Output layer
        self.relu = nn.ReLU()

        
    def forward(self, image):
        x = image.view(-1, 28*28)  # Flatten the image
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)

        return F.softmax(x)
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorch().to(device)

    numTraining = 100
    DIGIT_DATUM_WIDTH = 28
    DIGIT_DATUM_HEIGHT = 28
    numTest = 100
    # if(options.data=="faces"):   
    #     rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #     trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    #     rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #     validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    #     rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #     testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
    # else:    
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    
    trainingData = list(map(featureFunction, rawTrainingData)) # converted to list
    validationData = list(map(featureFunction, rawValidationData)) # converted to list
    testData = list(map(featureFunction, rawTestData)) # converted to list

    nn = PyTorch()

    cross_el = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn.parameters(), lr=0.001)
    epochs = 10

    for epoch in range(epochs):
        nn.train()

        for data in rawTrainingData:
            x, y = data
            optimizer.zero_grad()
            output = nn(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in rawTestData:
            x, y = data
            output = nn(x.view(-1, 28*28))
            for i, j in enumerate(output):
                if torch.argmax(j) == y[i]:
                    correct += 1
                total += 1
    print(f"Accuracy: {correct/total * 100}%")
