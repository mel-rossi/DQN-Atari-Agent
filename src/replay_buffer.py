# Storage for game experiences - 
# Circular memory bank that stores the last 1 million game transitions
# (state, action, reward, next_state). 
# The agent randomly samples from this to learn, 
# which prevents it from only learning the recent experiences (important for stability).

import numpy as np
import collections 

class ReplayBuffer: 
    """ Circular replay buffer for storing transitions """

    def __init__(self, capacity=1000000):
        """
        Initialize replay buffer. 

        Args: 
            capacity: Maximum number of transitions to store
        """

        self.capacity = capacity
        self.data = []
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminal): 
        """
        Add a transition to the replay buffer.

        Args: 
            state: Current state (4, 84, 84)
            action: Action taken 
            reward: Reward received
            next_state: Next state
            terminal: Whether episode terminated
        """
        transition = (state, action, reward, next_state, terminal)

        if self.size < self.capacity: 
            self.data.append(transition)
            self.size += 1
        else: 
            self.data[self.index] = transition

        self.index = (self.index + 1) % self.capacity
