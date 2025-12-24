import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d()

    def forward(self, x):
        return x