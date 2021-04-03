    
'''
Metal Gear ZEKE was the first fully bipedal tank. It was designed by the Militaires Sans Frontières at the Mother Base by Huey Emmerich and Strangelove in 1974.
'''
import random

class Agent:

    actions = ['','u','d','l','r','p']
    def __init__(self):
        """
        Example of a self, player_num, env)random agent
        """
        pass

    def next_move(self, game_state, player_state):
        """ 
        This method is called each time the agent is required to choose an action
        """

        action = random.choice(Agent.actions)
        print(action)
        return action
