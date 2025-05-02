
import numpy as np



class Baseline_picking:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        # Assuming action_space is a Box space from gym
        return (self.action_space.high + self.action_space.low) / 2
    

class Baseline_planning:
    def __init__(self, action_space):
        self.action_space = action_space
        self.indexes = np.arange(self.action_space.n)

    def act(self):
        # Randomly pick an index over the size of the array
        return np.random.choice(self.indexes, size=1)[0]