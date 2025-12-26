import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(64 * 8 * 8, 256)
        self.lin2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x
    

# the RGB color channel is 3 matrices stacked on each other
# The convultion layer takes the 3 by 3 kernel and slides with a stride of 1 and outputs a feature map. Does 32 of these.
# The activation function is ReLU so the network doesn't collapse to a linear function
# Max pooling takes the max value in the kernel and puts it into new feature matrix. It downsizes feature map with most important parts.
# The linear layers produce the logits after flattening the feature maps. 
# For Regularization, we can do Dropout which randomly disables neurons during training to prevent overfitting

# output_size = (W − K + 2P) / S + 1

# Our image starts as (3, 32, 32) -> Conv1 (32, 32, 32) -> ReLU -> Pool (32, 16, 16) -> Conv2 (64, 16, 16) -> ReLU
# -> Pool (64, 8, 8) -> flatten (4096,) -> lin1 (256,) -> ReLU -> Dropout -> lin2 (10,)
