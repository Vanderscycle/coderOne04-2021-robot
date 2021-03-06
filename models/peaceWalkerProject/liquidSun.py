import numpy as np 
import random
import torch
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from collections import deque

REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training

class AINetMK1(nn.Module):
    """
    Very simple fully connected neural network to confirm that the input and output matches.

    Contains:
        - model layer definition
        - forward methods for the data to move through
        - optimizer for gradient descent
        - train method
    """
    def __init__(self):
        """
        Method were we defined the Input, hidden layers, and output of the neural network.
        """
        super().__init__()
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # tensorboard to track progress
        self.tensorboard = None
        # input will be each square of the game (10x12 grid)
        self.fc1 = nn.Linear(10*12,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        # there are only 6 actions: ['','u','d','l','r','p']
        self.fc4 = nn.Linear(32,6)


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


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


    def optimizer(self):
        """
        Optimizer for NN
        """
        return optim.Adam(net.parameters(), lr=0.001)


    def train(self):
        """
        Method that defines how the NN will be trained

        """
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
         # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)


if torch.cuda.is_available():  
    print('GPU available, using GPU')
    device = "cuda:0" 
else:  
    print('No GPU available, using CPU')
    device = "cpu"  


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
