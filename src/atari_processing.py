# Wraps the game enviroment to preprocess frames. 
# Takes raaw Atari frames and makes them suitable for neural network: 
# converts to grayscale, resizes to 84x84, stacks 4 frames together (motion),
# does frame skipping (repeats actions), and adds random no-ops at the start. 
# Makes the raw game playable by agent.
import gymnasium as gym
import numpy as np 
import cv2 

# Wrapper that handles core image preprocessing 
class AtariProcessing(gym.Wrapper): 
    """
    Atari 2600 preprocessing
    - Frame skipping (repeat action for 4 frames) 
    - Max pooling over last 2 frames
    - Grayscale conversion
    - Resize to 84x84
    """

    def __init__(self, env, frame_skip = 4, screen_size = 84): 
        super().__init(env)
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        # Buffers for max pooling 
        self.obs_buffer = [
            np.zeros((84, 84), dtype=np.vint8),
            np.zeros((84, 84), dtype=np.vint8)
        ]
