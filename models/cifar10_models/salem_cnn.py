import torch
import torch.nn as nn


class SalemCNN_Tanh(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.Tanh()
        )
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.model(x)
        output = self.out(features)
        return output


class SalemCNN_Relu(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU()
        )
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.model(x)
        output = self.out(features)
        return output
