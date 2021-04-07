import numpy as np

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
        # input will be each square of the game (10x12 grid) have to figureouyt the encoding.
        self.fc1 = nn.Linear(10*42,32)
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


# test data
# data to test a single pass in the NN
testLayout = np.array([['id', 'ob' ,None ,None ,'id', None, None, None, None, None, 'ob', 'id'],
                        ['sb', None, None, None, 'a', None, 'sb', None, None, None, None, 'sb'],
                        [None, 'id', 'ob', None, None, None, None, None, None, 'id', 'sb', None],
                        [None, 'sb', None, 'sb', 'sb', 'id', None, 'id', 'id', None, None, 'sb'],
                        [None, None, None, None, None, None, None, 'sb', 'id', 'sb', None, 'id'],
                        [None, None, 'sb', 'id', None, 'E' ,None ,'X', None, None, None, None],
                        [None, 'id', None, None, None, None, None, None, 'ob', None, None, None],
                        [None, 'id', 'sb', None, None, 'a' ,'sb' ,None ,'id', 'id', None, 'id'],
                        [None, 'ob', None, 'a' ,None ,None ,None ,None ,'id', None, None, 'sb'],
                        [None, None, None, None, 'a' ,'sb' ,None ,None ,None, 'sb', 'id', None]],dtype='object')

# test numpy array with nonetype entries
# print(testLayout)
for rowidx, row in enumerate(testLayout):
    for colidx, item in enumerate(row):
        if item is None:
            testLayout[rowidx][colidx] = 'floor'

environmentUniqueValue = np.unique(testLayout)

# test numpy array without nonetype entries
print(environmentUniqueValue)
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
encodedTestLayout = enc.fit_transform(testLayout)
print(encodedTestLayout)

print(testLayout.shape)
print(encodedTestLayout.shape)
testLayout = torch.from_numpy(encodedTestLayout)
# will have to adapt the input size of the NN due to the encoding
testLayout = testLayout.view(-1,10*42)
print(testLayout)
output = net(testLayout)
