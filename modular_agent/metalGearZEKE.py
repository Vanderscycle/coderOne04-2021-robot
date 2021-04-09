    
'''
Metal Gear ZEKE was the first fully bipedal tank. It was designed by the Militaires Sans Frontières at the Mother Base by Huey Emmerich and Strangelove in 1974.
'''
import random
import numpy as np
# measuring efficiency
from time import time
from models.peaceWalkerProject.liquidSun import AINetMK1,arenaGridEncoder
import torch
import os
import json

with open(os.path.join(os.path.dirname(__file__),"metalGearZekeConfig.json")) as json_data_file:
    # note json text need a space between key: value
    trainingConfig = json.load(json_data_file)

# Exploration settings
epsilon = trainingConfig['exploration_settings']['epsilon']  # not a constant, going to be decayed
EPSILON_DECAY = trainingConfig['exploration_settings']['EPSILON_DECAY']
MIN_EPSILON = trainingConfig['exploration_settings']['MIN_EPSILON']

# General settings
info = trainingConfig['general_settings']['info']

class Agent:

    actions = {0:'',1:'u',2:'d',3:'l',4:'r',5:'p'}


    def __init__(self):
        """
        Example of a self, player_num, env)random agent
        """
        # want to know what was the action picked (whether it was random or NN)
        self.actionEpsilon = None
        # need to create method that can calculate when the agents picks a bomb.
        self.bombPicked = 0
        # init Zeke's NN
        self.ai = AINetMK1()
        # load the weight
        self.ai.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../models/peaceWalkerProject/liquidSun_weights.pth')))
        print('Weights loaded')
        print(self.ai.eval)


    def next_move(self, game_state, player_state):
        """ 
        Apparently the rule changed to elemination so there may be some changes (maybe treasure, ore blocks)
        
        Anyway, at the moment the agent selects an action at random.
        """
                
        # hacked the game files slightly to allow for one more round after game over to use the over flag to save the weights
        if game_state.is_over == True:
            print(f'Saving weights')
            torch.save(self.ai.state_dict(),os.path.join(os.path.dirname(__file__),'../models/peaceWalkerProject/liquidSun_weights.pth'))
            return None
        
        # measuring the speed of our program the goal being faster than 1e-3 seconds (100ms)
        begin = time()  
        # Non random action
        if np.random.random() > epsilon:
            self.actionEpsilon = 'NeuralNetwork'
            # Get action from Q table
            npVisionGrid = self.arrayVision(game_state, player_state)
            if info:
                print(npVisionGrid)

            decision = self.ai(arenaGridEncoder(npVisionGrid))
            action = Agent.actions[int(torch.argmax(decision))] 

        # random action (epsilon decays over training)
        else:
            self.actionEpsilon = 'random'
            # Get random action
            action = Agent.actions[random.randint(0,5)]

        end = time()

        #!!! update the config file with the reward. Once we get enough training samples we want to use next steps so that the nn can train (may take a while)
        if info:
            print(f'action choosen: {action}, action was decided by: {self.actionEpsilon}')

            #time metrics
            print(f"total runtime of the program is {round(end - begin,6)}")
            print(f"time left{round(.1 -(end - begin),6)}/100ms")

            # metrics for the gym to be used later
            print(f'GameOver? {game_state.is_over}, step:{game_state.tick_number}')
            # also want to add a reward everytime he picks a bomb
            print(f'agent hp:{player_state.hp} bomb Power:{player_state.power}')
        return action


    def arrayVision(self, game_state, player_state):
        """

        agent.py contains the gamestate config which is passed to the our agent:
        relevant method to access:    
            .all_blocks
                .indestructible_blocks (metal blocks)
                .ore_blocks (takes 3 hit to destroy) --not sure if they will be present
                .soft_blocks (take 1 hit to destroy)
            .ammo (1 extra bomb)
            .bombs ( Once the pin is pulled, Mr. Grenade is not our friend )

        returns: String array representing the board

            legend:
                X: our agent
                E: ennemy agent
                Xb: agent and bomb on same tile
                Eb: ennemy agent and bomb on same title
                id: indestructible block
                sb: wooden block
                ob: ore block
                b: bomb (armed)
                a: ammo (+1)
                t: treasure chest
                :function: TODO
        known issue: when E disappears (our robot or the ennemy agent has placed a bomb so must we ensure for that eventuality)
        Solved by changing the grid value assignement.

        quirk, once a bomb is placed you can move over time. When we place a bomb tho, our agent and the bomb can sit on the same tile tho.

        """

        visionArray = np.empty([10,12],dtype='object')

        playerStateAsset = {
            'playerInt':player_state.id,
            'playerLocation':player_state.location
            }
        # print(playerStateAsset['playerInt'])
        # print(playerStateAsset['playerLocation'])

        gameStateAsset = {
            'id': game_state.indestructible_blocks,
            'sb': game_state.soft_blocks,
            'ob': game_state.ore_blocks,
            'a': game_state.ammo,
            'b': game_state.bombs,
            't': game_state.treasure
            }
        
        # populating the np array with object representation of the game world
        for assetName, assetTupple in gameStateAsset.items():
            for singleAsset in assetTupple:
                #(x,y) - (x = columns, y = rows)
                x = singleAsset[0]
                y = 9 - singleAsset[1]
                # print(singleAsset)
                # print(f'x:{x},y:{y}')
                # print(assetName)
                # 10-y 12 -x 
                visionArray[y][x] = assetName

        # place our agent in an x,y location of our np array
        x,y = zip(playerStateAsset['playerLocation'])
        x = x[0]
        y = 9 - y[0]
        visionArray[y][x] = 'X'        

        # obtaining the opposite agent location
        opponentLocation = game_state.opponents(playerStateAsset['playerInt'])[0]
        # place the opposite agent in an x,y location of our np array
        x = opponentLocation[0]
        y = 9 - opponentLocation[1]
        # since its possible that we may be either agent 1 or 0 we want the target to always be constant (E)
        visionArray[y][x] = 'E'

        # also because we only have 100ms I am curious to see the time it takes to execute the action check (https://github.com/pyutils/line_profiler)
        return visionArray
