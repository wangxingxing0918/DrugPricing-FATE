import torch
import torch.nn as nn
import torch.nn.functional as F
class FATEModel(nn.Module):
    def __init__(self):
        super(FATEModel, self).__init__()
        # Define the layers and their configurations for the FATE network
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(100, 100)
        # Add more layers as needed

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Apply more layers as needed
        return x

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        # Initialize the FATE network model
        self.fate_model = FATEModel()
        # Define additional layers for the ComplexModel
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(100, 100)
        # Add more layers as needed

    def forward(self, x):
        x = self.fate_model(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Apply more layers as needed
        return x
