import numpy as np
from tqdm import tqdm
import json
import os
# to call other python's scripts
from subprocess import (call,run)
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(BASE_DIR)
# Gym global values
EPISODES = 10

# Exploration settings
epsilon = 1  # not a constant, going to be decayed

with open(os.path.join(os.path.dirname(__file__),"../../modular_agent/metalGearZekeConfig.json")) as json_data_file:
    initConfig = json.load(json_data_file)

#loading the json data
initConfig['exploration_settings']['epsilon'] = epsilon
EPSILON_DECAY = initConfig['exploration_settings']['EPSILON_DECAY'] 
MIN_EPSILON = initConfig['exploration_settings']['MIN_EPSILON']

#resetting the training config file epsilon (randomness)
with open(os.path.join(os.path.dirname(__file__),"../../modular_agent/metalGearZekeConfig.json"),'w') as json_data_file:
    json.dump(initConfig,json_data_file)


# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 0 # we want the one given by game_state also starts at 0

    # Reset environment and get initial state
    #call([' python -m coderone.dungeon.main modular_agent/metalGearZEKE.py '],cwd=BASE_DIR, shell=True)

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        break #placeholder

    # update decay
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    initConfig['exploration_settings']['epsilon'] = epsilon
    with open(os.path.join(os.path.dirname(__file__),"../../modular_agent/metalGearZekeConfig.json"),"w") as json_data_file:
        json.dump(initConfig,json_data_file)
