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
        # one of the 6 action ['','u','d','l','r','p']
        return F.log_softmax(data, dim=1)


def arenaGridEncoder(grid):
    """
    Takes an object numpy array and convert it to a Tensor dtype float32.
    The encoding dictionary is specific to the coderone arena.
    """

    # Encoding the input data
    encoderDict = {None:0,'X':1,'E':2,'b':3,'a':4,'t':5,'ob':6,'sb':7,'id':8}
    for rowidx, row in enumerate(grid):
        for colidx, item in enumerate(row):
            grid[rowidx][colidx] = encoderDict[item]
    # the default is float32 
    # if I want to change later https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/8
    grid = grid.astype('float32')
    return torch.from_numpy(grid).view(-1,10*12)



if __name__ == '__main__':

    net = AINetMK1()

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

    testLayout = arenaGridEncoder(testLayout)
    output = net(testLayout)

    # need to map the output with an action.
    print(output)


    # saving the model (test)
    import os
    torch.save(net.state_dict(),os.path.join(os.path.dirname(__file__),'liquidSun_weights.pth'))

    # To load model weights, you need to create an instance of the same model first, and then load the parameters
    model = AINetMK1()
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'liquidSun_weights.pth')))
    print(model.eval)
