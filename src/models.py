from torch import nn as nn
from torch.nn import functional as F

# https://arxiv.org/pdf/1602.05629.pdf
# A CNN with two 5x5 convolution layers (the first with
# 32 channels, the second with 64, each followed with 2x2
# max pooling), a fully connected layer with 512 units and
# ReLu activation, and a final softmax output layer (1,663,370
# total parameters).

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
