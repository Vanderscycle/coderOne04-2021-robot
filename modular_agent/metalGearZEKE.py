    
'''
Metal Gear ZEKE was the first fully bipedal tank. It was designed by the Militaires Sans Frontières at the Mother Base by Huey Emmerich and Strangelove in 1974.
'''
import random
import numpy as np
import pickle
# measuring efficiency
from time import time
from models.peaceWalkerProject.liquidSun import AINetMK1,arenaGridEncoder
import torch
import os
import json
from copy import deepcopy


with open(os.path.join(os.path.dirname(__file__),"metalGearZekeConfig.json"),'r') as json_data_file:
    # note json text need a space between key: value
    trainingConfig = json.load(json_data_file)

# Exploration settings
epsilon = trainingConfig['exploration_settings']['epsilon']  # not a constant, going to be decayed
EPSILON_DECAY = trainingConfig['exploration_settings']['EPSILON_DECAY']
MIN_EPSILON = trainingConfig['exploration_settings']['MIN_EPSILON']

# Episode specific 

# General settings
info = trainingConfig['general_settings']['info']

class Agent:

    actions = {0:'',1:'u',2:'d',3:'l',4:'r',5:'p'}
    encoderDict = {None:0,'X':1,'E':2,'b':3,'a':4,'t':5,'ob':6,'sb':7,'id':8}


    def __init__(self):
        """
        Example of a self, player_num, env)random agent
        """
        # tracked states for data in reinforcement learning
        self.lastAction = 0
        self.currentState = None
        self.currentStateNotEncoded = None
        self.lastReward = 0
        #session reward
        self.reward = 0
        self.episodeReward = 0
        #Agent hp
        self.currentHP = 3 #init value, we start with three
        # want to know what was the action picked (whether it was random or NN)
        self.actionEpsilon = None
        # need to create method that can calculate when the agents picks a bomb.
        self.bombCount = 3 # init value, we start with three
        # using GPU for training
        if torch.cuda.is_available():  
            print('GPU available, using GPU')
            self.device = "cuda:0" 
        else:  
            print('No GPU available, using CPU')
            self.device = "cpu"  
        # optimizing the NN
        self.policy_net = AINetMK1().to(self.device)
        self.target_net = AINetMK1().to(self.device)
        # init Zeke's NN
        self.ai = AINetMK1()
        # load the weight
        self.ai.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../models/peaceWalkerProject/liquidSun_weights.pth')))
        # init loading info
        print('Weights loaded')
        with open(os.path.join(os.path.dirname(__file__),'../models/peaceWalkerProject/liquidSun_deque'),'rb') as dequeFile:
            self.ai.replay_memory = pickle.load(dequeFile)
        print('Previous training data loaded')
        print(f'Showing the NN structure {self.ai.eval}')


    def next_move(self, game_state, player_state):
        """ 
        Apparently the rule changed to elemination so there may be some changes (maybe treasure, ore blocks)
        
        Anyway, at the moment the agent selects an action at random.
        """
                
        # measuring the speed of our program the goal being faster than 1e-3 seconds (100ms)
        begin = time()  

        # Get action from Q table
        encodedGrid = arenaGridEncoder(self.arrayVision(game_state, player_state))

        # at the start of the game there is no previous state or action 
        if game_state.tick_number == 0:
            print('Start of the game. setting past value to current for this once only')
            self.currentState = encodedGrid
            self.currentStateNotEncoded = self.arrayVision(game_state,player_state)

        self.reward = self.rewardSystem(player_state,game_state)
        self.episodeReward += self.reward
        
        # hacked the game files slightly to allow for one more round after game over to use the over flag to save the weights
        if game_state.is_over == True:
            # last action may have to be a number and not a string. double check rewards.
            self.ai.update_replay_memory((self.currentState, self.lastAction, self.reward, encodedGrid, game_state.is_over))
            self.gameOverSaving()
            return None

        else:

            # Non random action
            if np.random.random() > epsilon:
                self.actionEpsilon = 'NeuralNetwork'
                if info:
                    print(encodedGrid)

                decision = self.ai(encodedGrid)
                action = Agent.actions[int(torch.argmax(decision))] 

            # random action (epsilon decays over training)
            else:
                self.actionEpsilon = 'random'
                # Get random action
                action = Agent.actions[random.randint(0,5)]


        self.ai.update_replay_memory((self.currentState, self.lastAction, self.reward, encodedGrid, game_state.is_over))


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
            print(f'agent hp:{player_state.hp}, ammo: {player_state.ammo} bomb Power:{player_state.power}, current reward state: {self.reward} total reward {self.episodeReward}')
            print(f'size of the data {len(self.ai.replay_memory)}')
         
        self.lastAction = getKey(action,Agent.actions)
        self.currentState = encodedGrid
        self.currentStateNotEncoded = self.arrayVision(game_state,player_state)
        self.lastReward = self.reward
        return action


    def gameOverSaving(self):
        """
        Method that saves the game data. 

        Known issue: if our agent dies it doesn't allow to save
        """
        print(f'Saving weights')
        torch.save(self.ai.state_dict(),os.path.join(os.path.dirname(__file__),'../models/peaceWalkerProject/liquidSun_weights.pth'))

        #saving the trainingData
        with open(os.path.join(os.path.dirname(__file__),'../models/peaceWalkerProject/liquidSun_deque'),'wb') as dequeFile:
            print('Pickling the trainingData')
            pickle.dump(self.ai.replay_memory,dequeFile)
        # saving the relevant episode data in the config file
        trainingConfig['episode_data']['reward'] = self.episodeReward
        with open(os.path.join(os.path.dirname(__file__),"metalGearZekeConfig.json"),'w') as json_data_file:
            print('Saving the config file')
            json.dump(trainingConfig,json_data_file)
        

    def rewardSystem(self, player_state,game_state):
        """
        Reward system based on ammo picked and damage received.
        
        Need to find a way to get the opponent hp.
        """
        reward = 0
        # hp reward calculation
        if self.currentHP != player_state.hp:
            print('HP lost')
            self.currentHP = player_state.hp
            if self.currentHP == 2:
               reward -= 100
            elif self.currentHP == 1:
                reward -= 200
            elif self.currentHP == 0:
                reward -= 400
                print("Do not picture the Game Over screen, instead visualize how happy you'll be once the mission is complete")

        # ammo reward calculation
        #picked a bomb
        if self.bombCount < player_state.ammo:
            print('picked a bomb OSP')
            reward += 25
        # planted a bomb
        elif self.bombCount > player_state.ammo:
            print('bomb has been planted')
        # updating the bomb count
        self.bombCount = player_state.ammo

        if self.last2FramesSubstraction(deepcopy(self.currentStateNotEncoded),self.arrayVision(game_state, player_state),info=True):
            reward += 150
        return reward

    
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


    def last2FramesSubstraction(self, previousFrame, currentFrame, info=False, debug=False):
        """
        function that maps the blast radius of bombs and return if the ennemy agent was hit
        """
        explosionList = list()
        if debug:
            print('previousFrame')
            print(previousFrame)
            print('currentFrame')
            print(currentFrame)
        for rowidx, row in enumerate(currentFrame):
            for colidx, col in enumerate(row):

                if (currentFrame[rowidx][colidx] is None) and (previousFrame[rowidx][colidx] == 'b'):
                    # need to add x + y of what was touched (+2)
                    # also since there will be power ups we need to track who places what bombs
                    currentFrame[rowidx][colidx] = 'BOOM'
                    print(f'explosion at (y:{9 - rowidx},x:{colidx})')
                    explosionList.append((9 - rowidx,colidx))
                    #print(explosionList)
                # possible that the ob will be removed next patch
                elif (currentFrame[rowidx][colidx] in ['E','X','id','sb','ob']):
                    pass

                elif currentFrame[rowidx][colidx] is None and previousFrame[rowidx][colidx]!='b':
                    # could be sb,X,E taht moved
                    currentFrame[rowidx][colidx] = ''

                elif previousFrame[rowidx][colidx] == currentFrame[rowidx][colidx]:
                    #print(currentFrame[rowidx][colidx])
                    currentFrame[rowidx][colidx] = currentFrame[rowidx][colidx].replace(previousFrame[rowidx][colidx],'')

        for explosion in explosionList:

            # y boundaries
            minRow = 0
            maxRow = 9
            # x boundaries
            minCol = 0
            maxCol = 11
            # recap: done this way because of how numpy arrays are laid out
            # also we need to inverse row indexing so that's why we use maxrow - row
            # frame[row or y][col or x]

            # row edge cases (y coord)
            if explosion[0] - 3 < 0:
                minRow = 0
            else:
                minRow = explosion[0]-3
            
            if explosion[0] + 3 > 9:
                # 10 instead of 9 (the actual limit) because range ignores the last value
                maxRow = 10
            else:
                maxRow = explosion[0] + 3
            
            # columns edge cases (x coord)
            if explosion[1] - 2 < 0:
                minCol = 0
            else:
                minCol = explosion[1]-2
            
            if explosion[1] + 3 > 11:
                # 12 instead of 11 (the actual limit) because range ignores the last value
                maxCol = 12
            else:
                maxCol = explosion[1] + 3

            #print(minCol,explosion[1],maxCol)
            #print(minRow,explosion[0],maxRow)
            
            # mapping the rows blast
            for x in range(explosion[0], maxRow):

                # there's a wall at the above the explosion
                if (currentFrame[9-x][explosion[1]] in ['id','sb','ob']): 
                    break

                # empty space so the explosion propagate
                elif currentFrame[9-x][explosion[1]] == '': 
                    currentFrame[9-x][explosion[1]] = 'bl'
                # there's an player that was touched by the explosion
                else:
                    if currentFrame[9-x][explosion[1]][0]=='[' and currentFrame[9-x][explosion[1]][-1] ==']':
                        pass
                    else:
                        currentFrame[9-x][explosion[1]] = '['+currentFrame[9-x][explosion[1]]+']'

            for x in range(explosion[0], minRow,-1):

                # there's a wall at bellow explosion
                if (currentFrame[9-x][explosion[1]] in ['id','sb','ob']): 
                    break

                # empty space so the explosion propagate
                elif currentFrame[9-x][explosion[1]] == '': 
                    currentFrame[9-x][explosion[1]] = 'bl'
                # there's an player that was touched by the explosion
                else:
                    if currentFrame[9-x][explosion[1]][0]=='[' and currentFrame[9-x][explosion[1]][-1] ==']':
                        pass
                    else:
                        currentFrame[9-x][explosion[1]] = '['+currentFrame[9-x][explosion[1]]+']'

            # mapping the columns blast
            for y in range(explosion[1], maxCol):
                # there's a wall at the right of the explosion
                if currentFrame[9-explosion[0]][y] in ['id','sb','ob']:
                    break
                # empty space so the explosion propagate
                elif currentFrame[9-explosion[0]][y] == '': 
                    currentFrame[9-explosion[0]][y] = 'bl'
                # there's an player that was touched by the explosion
                else:
                    if currentFrame[9-explosion[0]][y][0] == '[' and currentFrame[9-explosion[0]][y][-1] == ']':
                        pass
                    else:
                        currentFrame[9-explosion[0]][y] = '['+currentFrame[9-explosion[0]][y]+']'

            for y in range(explosion[1], minCol -1,-1):
                if currentFrame[9-explosion[0]][y]  in ['id','sb','ob']:
                    break
                # empty space so the explosion propagate
                elif currentFrame[9-explosion[0]][y] == '': 
                    currentFrame[9-explosion[0]][y] = 'bl'
                # there's an player that was touched by the explosion
                else:
                    if currentFrame[9-explosion[0]][y][0] == '[' and currentFrame[9-explosion[0]][y][-1] == ']':
                        pass
                    else:
                        currentFrame[9-explosion[0]][y] = '['+currentFrame[9-explosion[0]][y]+']'
        ennemyAgentHit = np.where(currentFrame == '[E]')
        if info:
            print(f'explosion location {explosionList}')
            print(f'enemyhit? :{ennemyAgentHit}')
            print(currentFrame)
        if np.size(ennemyAgentHit):
            return True
        return False



# helper functions
def getKey(val, my_dict):
    """
    from https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    """
    for key, value in my_dict.items():
         if val == value:
             return key
