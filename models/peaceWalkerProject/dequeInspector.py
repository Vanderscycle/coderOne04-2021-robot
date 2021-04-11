from collections import deque
import pickle
import os

REPLAY_MEMORY_SIZE = 50_000

dequeForInspection = deque(maxlen=REPLAY_MEMORY_SIZE)

#TODO deque init (for training)
# add the option to inspect later 
# maybe a config file generator too
# a simple meatl gear for testing trained ai (needs only arrayVision)


def emptyDeque(fileName):
    """
    init an empty deque for the nn
    """
    emptyDequeObj = deque(maxlen=REPLAY_MEMORY_SIZE)
    
    with open(os.path.join(os.path.dirname(__file__),f'{fileName}'),'wb') as dequeFile:
        pickle.dump(emptyDequeObj,dequeFile)


def dequeInspector(fileName):
    """
    Take the name of deque file adn return the deque.
    """
    with open(os.path.join(os.path.dirname(__file__),f'{fileName}'),'rb') as dequeFile:
        dequeForInspection = pickle.load(dequeFile)

        return dequeForInspection


if __name__ == '__main__':
    
    file = 'liquidSun_deque'
    #emptyDeque(file)
    print(dequeInspector(file))
