# Defines neural network architecture -
# 3 convolutional layers to process images, 
# 2 fully connected layers to decide which action is best. 
# Takes in a stack of 4 game frames (84x84 pixels) and 
# outputs a score for each possible action.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

class DQNNetwork(keras.Model):
    """
    Deep Q-Network (DQN) in TensorFlow/Keras - 
    Processes image states and outputs Q-values for each action.
    """

    # Setting up object's attributes (construct and initialization)
    def __init__(self, num_actions, name='dqn_network'):
        """
        Initialize DQN network. 

        Args:
            num_actions: Number of valid actions 
            name: Name of the network
        """

        # Assign name for model instance
        super(DQNNetwork, self).__init__(name=name)

        # Number of discrete actions the agent can take (size of output layer)
        self.num_action = num_actions

        # Convolutional layers 

        # Aggressively downsamples input while extracting coarse spatial features
        self.conv1 = layers.Conv2D( 
            32, 
            kernel_size = 8,
            strides = 4,
            activation = 'relu',
            name = 'conv1'
        )

        # Further downsamples and expands feature depth 
        # (captures mid-level patterns)
        self.conv2 = layers.Conv2D(
            64,
            kernal_size = 4,
            strides = 2,
            activation = 'relu'
            name = 'conv2'
        )

        # Preserves spatial resolution while enriching feature representation 
        # (fine-grained features)
        self.conv3 = layers.Conv2D(
            64,
            kernel_size = 3, 
            strides = 1
            activation = 'relu'
            name = 'conv3'
        )

        # Flatten layer - converts 3D feature map from conv layers 
        # into 1D vector for fully connected processing
        self.flatten = layers.Flatten()

        # Fully connected layers 

        # High-capacity latent representation that integrates features across whole frame
        self.dense1 = layers.Dense(512, activation = 'relu', name = 'fc1')

        # Outputs one Q-value per action 
        # (raw values are used by DQN for policy and TD targets)
        self.dense2 = layers.Dense(num_actions, name = 'fc2')

    # Forward pass and input handling 
    def call(self, state):
        """
        Forward pass through the network. 

        Args: 
            state: Input state tensor (batch_size, 4, 84, 84) or (batch_size, 84, 84, 4)

        Returns: 
            Q-values for each action 
        """

        # Note: TensorFlow expects (batch, height, width, channels)
        # If input is (batch, channels, height, width), we need to transpose 
        if state.shape[1] == 4: # channels first 
            state = tf.transpose(state, [0, 2, 3, 1])

        # Layer application sequence
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        q_values = self.dense2(x)

        return q_values
