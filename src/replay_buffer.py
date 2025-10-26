import numpy as np
import collections 

class ReplayBuffer: 
    """ Circular replay buffer for storing transitions """

    # Setting up object's attributes (construct and initialization)
    def __init__(self, capacity=1000000):
        """
        Initialize replay buffer. 

        Args: 
            capacity: Maximum number of transitions to store
        """
        
        self.capacity = capacity
        self.data = [] # empty list
        self.index = 0 # initial state
        self.size = 0 # initial state

    # Store gameplay transition
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

        # Package info into truple
        transition = (state, action, reward, next_state, terminal)

        # Store package into self.data
        if self.size < self.capacity: 
            self.data.append(transition)
            self.size += 1
        else: 
            # Manage circular buffer logic (overwrites old memories when full)
            self.data[self.index] = transition

        # Updates index and size counters
        self.index = (self.index + 1) % self.capacity

    # Retrieve random batch of transitions (from memory) for training
    def sample(self, batch_size): 
        """
        Sample a batch of transitions. 

        Args: 
            batch_size: Number of transitions to sample

        Returns: 
            Tuple of batched (states, actions, rewards, next_states, terminals)
        """
        
        # Pick random indices - Selects n (batch_size) random positions from buffer
        indices = np.random.choice(self.size, batch_size, replace=False)

        states, actions, rewards, next_sattes, terminals = [], [], [], [], []

        # Extract transitions 
        for idx in indices: 
            # Specific transitions from self.data
            state, action, reward, next_state, terminal = self.data[idx]
            # Separate into arrays - unpacks each transition and groups them
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(terminal)

        # Convert to numpy arrays 
        # Returns one numpy array for each component to be fed to TensorFlow
        return(
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(terminals, dtype=np.float32)
        )

    # Returns size of buffer
    def __len__(self):
        return self.size

    # Confirm buffer is ready for training
    def is_ready(self, batch_size):
        """ Check if buffer has enough samples for training. """

        return self.size >= batch_size


