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
EPISODES = 1

# Exploration settings
epsilon = 1  # not a constant, going to be decayed

with open(os.path.join(os.path.dirname(__file__),"../../modular_agent/metalGearZekeConfig.json")) as json_data_file:
    initConfig = json.load(json_data_file)

#loading the json data
initConfig['exploration_settings']['epsilon'] = epsilon
initConfig['episode_data']['reward'] = 0
EPSILON_DECAY = initConfig['exploration_settings']['EPSILON_DECAY'] 
MIN_EPSILON = initConfig['exploration_settings']['MIN_EPSILON']

#resetting the training config file epsilon (randomness)
with open(os.path.join(os.path.dirname(__file__),"../../modular_agent/metalGearZekeConfig.json"),'w') as json_data_file:
    json.dump(initConfig,json_data_file)


# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    if episode % 2000 == 0:
        # every 2000 episodes we want to see the progress of the NN.
        call([' python -m coderone.dungeon.main modular_agent/metalGearZEKE.py modular_agent/decoy.py --watch'],cwd=BASE_DIR, shell=True)
    else:
        # Reset environment and get initial state
        call([' python -m coderone.dungeon.main modular_agent/metalGearZEKE.py modular_agent/decoy.py --headless'],cwd=BASE_DIR, shell=True)

    
    # Every step we update replay memory and train main network
   # has to be done inside the program 
   # agent.update_replay_memory((current_state, action, reward, new_state, done))
   # agent.train(done, step)
    # update decay
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    #update and reset for the next run
    initConfig['exploration_settings']['epsilon'] = epsilon
    initConfig['episode_data']['reward'] = 0
    with open(os.path.join(os.path.dirname(__file__),"../../modular_agent/metalGearZekeConfig.json"),"w") as json_data_file:
        json.dump(initConfig,json_data_file)
