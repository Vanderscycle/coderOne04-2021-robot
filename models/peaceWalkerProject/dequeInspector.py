from collections import deque
import pickle
import os

REPLAY_MEMORY_SIZE = 50_000

file = 'liquidSun_deque'
dequeForInspection = deque(maxlen=REPLAY_MEMORY_SIZE)

#TODO deque init (for training)
# add the option to inspect later 
# maybe a config file generator too
# a simple meatl gear for testing trained ai (needs only arrayVision)
with open(os.path.join(os.path.dirname(__file__),f'{file}'),'rb') as dequeFile:
    dequeForInspection = pickle.load(dequeFile)

    print(dequeForInspection)
