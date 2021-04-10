'''
A life-sized, man-shaped decoy made of wood. It can be used to distract the enemy's attention, but the enemy will see through it immediately when they get close enough. Don't expect to rely on it too much - it is, after all, just a piece of wood.
'''
import time
import random
class agent:
    ACTION_PALLET = ['']

    def __init__(self):
            pass

    def next_move(self, game_state, player_state):
        dialogue = ["Kept you waiting, huh?","You're pretty good"]
        print(random.choice(dialogue))
        # Lets pretend that agent is doing some thinking
        time.sleep(1)

        return self.ACTION_PALLET[0]
