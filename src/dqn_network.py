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

