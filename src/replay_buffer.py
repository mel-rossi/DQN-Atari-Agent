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
