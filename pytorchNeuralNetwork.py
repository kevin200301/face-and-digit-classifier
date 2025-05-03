import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import classificationMethod

def dataToTensor(counter_list, width, height): # convert a list of counters to a tensors
    tensors = []
    for counter in counter_list:
        image_tensor = torch.zeros(height, width, dtype=torch.float32)
        for (x, y), val in counter.items():
            image_tensor[y][x] = float(val)
        tensors.append(image_tensor.flatten())
    return torch.stack(tensors)

def labelToTensor(labels): # convert a list of labels to a tensors
    return torch.tensor(labels, dtype=torch.long)

class PytorchClassifier(classificationMethod.ClassificationMethod, nn.Module):
    def __init__(self, legalLabels, maxIterations, width, height, device):
        classificationMethod.ClassificationMethod.__init__(self, legalLabels)
        nn.Module.__init__(self)
        
        self.linear1 = nn.Linear(width * height, 128)   # First hidden layer
        self.linear2 = nn.Linear(128, 64) # Second hidden layer
        self.final = nn.Linear(64, len(legalLabels))  # Output layer
        self.relu = nn.ReLU()

        self.legalLabels = legalLabels
        self.maxIterations = maxIterations
        self.width = width
        self.height = height
        self.device = device
        
    def forward(self, image):
        x = image.view(-1, self.width * self.height)  # Flatten the image
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        trainingData = dataToTensor(trainingData, self.width, self.height)
        trainingLabels = labelToTensor(trainingLabels)

        trainingData = TensorDataset(trainingData, trainingLabels)
        trainLoader = DataLoader(trainingData, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        nn.Module.train(self) # self.train()
        for epoch in range(self.maxIterations):
            for inputs, labels in trainLoader:
                inputs = inputs.to(self.device) # move data to correct device
                labels = labels.to(self.device) # move data to correct device

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def classify(self, data):
        nn.Module.train(self, mode=False) # self.eval()
        inputs = dataToTensor(data, self.width, self.height).to(self.device) # move data to correct device
        with torch.no_grad():
            outputs = self(inputs)
            _, predictions = torch.max(outputs, 1)
            return predictions.tolist()