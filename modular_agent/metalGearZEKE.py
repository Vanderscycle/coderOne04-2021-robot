    
'''
Metal Gear ZEKE was the first fully bipedal tank. It was designed by the Militaires Sans Frontières at the Mother Base by Huey Emmerich and Strangelove in 1974.
'''
import random
import numpy as np


class Agent:

    actions = ['','u','d','l','r','p']
    def __init__(self):
        """
        Example of a self, player_num, env)random agent
        """
        pass

    def next_move(self, game_state, player_state):
        """ 
        Apparently the rule changed to elemination so there may be some changes (maybe treasure, ore blocks)
        
        Anyway, at the moment the agent selects an action at random.
        """
                
        action = random.choice(Agent.actions)
        print(self.arrayVision(game_state, player_state))#print(game_state.all_blocks)
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
                0: Player 1
                1: Player 2
                ib: indestructible block
                sb: wooden block
                ob: ore block
                b: bomb (armed)
                a: ammo (+1)
                t: treasure chest
                :function: TODO

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
        
        # place our agent in an x,y location of our np array
        x,y = zip(playerStateAsset['playerLocation'])
        x = x[0]
        y = 9 - y[0]
        visionArray[y][x] = playerStateAsset['playerInt']
        
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

        # obtaining the opposite agent location
        opponentLocation = game_state.opponents(playerStateAsset['playerInt'])[0]
        # place the opposite agent in an x,y location of our np array
        x = opponentLocation[0]
        y = 9 - opponentLocation[1]
        # since we can obtain our own int with the player state but we can't know at the satrt.
        if playerStateAsset['playerInt'] == 0:
            visionArray[y][x] = 1
        else:
            visionArray[y][x] = 0

       # print(opponentLocation)
       # print(f"Player at x:{opponentLocation[0]} y:{9 - opponentLocation[1]}")

        #for index, cell in np.ndenumerate(visionArray):
        #    #print(f'type:{type(cell)} -> {cell}')
        #    if cell is None:
        #        possiblePlayerLoc = game_state.entity_at(index)
        #        print(f'location: {index}, item:{possiblePlayerLoc}')

        # for values = None use game_state.entity_at(location) to find the other player (must pass a tupple)
        # also because we only have 100ms I am curious to see the time it takes to execute the action check (https://github.com/pyutils/line_profiler)
        return visionArray
