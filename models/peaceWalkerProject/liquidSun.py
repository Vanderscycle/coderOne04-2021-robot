import torch
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

class AINetMK1(nn.Module):
    """
    Very simple fully connected neural network to confirm that the input and output matches.
    """
    def __init__(self):
        """
        Method were we defined the Input, hidden layers, and output of the neural network.
        """
        super().__init__()
        # input will be each square of the game
        self.fc1 = nn.Linear(10*12,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        # there are only 6 actions: ['','u','d','l','r','p']
        self.fc4 = nn.Linear(32,6)

    def forward(self, data):
        """
        Method that defines how the data will move through the neural network. We chose leaky RELUS as our non-linear activation function.
        """
        data = F.leaky_relu(self.fc1(data))
        data = F.leaky_relu(self.fc2(data))
        data = F.leaky_relu(self.fc3(data))
        data = self.fc4(data)
        return F.log_softmax(data, dim=1)

net = AINetMK1()
print(net)       
