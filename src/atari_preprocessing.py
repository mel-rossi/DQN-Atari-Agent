from . import atari_setup
import gymnasium as gym
import numpy as np 
import cv2 

# Wrapper that handles core image processing
class AtariPreprocessing(gym.Wrapper): 
    """
    Atari 2600 preprocessing
    - Frame skipping (repeat action for 4 frames) 
    - Max pooling over last 2 frames
    - Grayscale conversion
    - Resize to 84x84
    """

    # Setting up object's attributes (construct and initialization)
    def __init__(self, env, frame_skip = 4, screen_size = 84): 
        """
        Initialize atari processing wrapper. 

        Args: 
            env: Base Gymnasium environment
            frame_skip: Number of frames to repeat each action
            screen_size: Target size for resized frames (default 84)
        """

        super().__init__(env) # Initialize enviroment
        self.frame_skip = frame_skip # initial state
        self.screen_size = screen_size # initial state

        # Buffers for max pooling (2 frames)
        self.obs_buffer = [
            np.zeros((84, 84), dtype=np.int8),
            np.zeros((84, 84), dtype=np.int8)
        ]

    # Reset environment  
    def reset(self, **kwargs): 
        """
        Reset the enviroment and return the first processed observation. 

        Returns: 
            obs: Preprocessed initial frame (84x84 grayscale)
            info: Additional enviroment info
        """

        obs, info = self.env.reset(**kwargs)

        return self._process_observation(obs), info

    # Preprocessing
    def step(self, action): 
        """
        Take a step in the environment with preprocessing 

        Args: 
            action: Action to apply

        Returns: 
            max_frame: Max_pooled frame (84x84 grayscale) 
            total_reward: Accumulated reward over skipped frames
            terminated: Whether the episode ended naturally 
            truncated: Whether the episode was truncted 
            info: Additional environment info
        """

        total_reward = 0.0 # initialize

        # Repeat the action for n (frame_skip) frames
        for i in range(self.frame_skip): 
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward # collect rewards across frames

            # Store last 2 frames for max pooling 
            if i >= self.frame_skip - 2:
                self.obs_buffer[i - (self.frame_skip - 2)] = self._process_observation(obs)

            if terminated or truncated: 
                break

        # Max pool over the last 2 frames 
        max_frame = np.maximum(self.obs_buffer[0], self.obs_buffer[1])

        return max_frame, total_reward, terminated, truncated, info 

    # Resize and convert frames to grayscale
    def _process_observation(self, obs): 
        """ 
        Convert raw RGB frame to grayscale and resize to 84x84. 

        Args: 
            obs: Raw RGB observation 

        Returns: 
            resized: Grayscale 84x84 frame
        """

        # Convert to grayscale 
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84
        resized = cv2.resize(gray, (self.screen_size, self.screen_size), 
                            interpolation=cv2.INTER_AREA)

        return resized

# Wrapper that handles frame stacking 
class FrameStack(gym.Wrapper): 
    """ Stack the last k frames into a single observation. """

    # Setting up object's attributes (construct and initialization)
    def __init__(self, env, k = 4): 
        """
        Initialize frame stack wrapper. 

        Args: 
            env: Base Gymnasium environment
            k: Number of frames to stack 
        """

        super().__init__(env) # initialize environment
        self.k = k # number of frames to stack
        self.frames = None # bufer to hold stacked frames

        # Update observation space
        low = np.repeat(self.observation_space.low[..., np.newaxis], k, axis = -1)
        high = np.repeat(self.observation_space.high[..., np.newaxis], k, axis = -1)
        self.observation_space = gym.spaces.Box(
            low = low, high = high, dtype = self.observation_space.dtype
        )

    # Reset environment
    def reset(self, **kwargs): 
        """
        Reset environment and initialize stacked frames. 

        Returns: 
            frames: Initial stacked observation (H, W, k)
            info: Additional environment info
        """

        obs, info = self.env.reset(**kwargs) # reset base environment
        # Fills buffer with k copies of the first observation (frame)
        self.frames = np.stack([obs] * self.k, axis = -1) 
        
        return self.frames, info 

    # Rolls the buffer 
    def step(self, action): 
        """
        Step environment and update stacked frames. 

        Args: 
            action: Action to apply 

        Returns: 
            frames: Updated stacked observation (H, W, k)
            reward: Reward from environment
            terminated: Whether the episode ended naturally
            truncated: Whether the episode was truncated
            info: Additional environment info
        """

        obs, reward, terminated, truncated, info = self.env.step(action) # take action
        self.frames = np.roll(self.frames, shift = 1, axis = -1) # shift frames left
        # Inserts newest frame at the end 
        self.frames[..., -1] = obs 
        
        # Updates observation space to reflect (H, W, k) instead of (H, W)
        return self.frames, reward, terminated, truncated, info 

# Wrapper that randomizes initial state 
class NoopResetEnv(gym.Wrapper): 
    """
    Sample initial states by taking random number of no-ops on reset. 
    No-op is assumed to be action 0. 
    """

    # Setting up object's attributes (construct and initialization)
    def __init__(self, env, noop_max = 30): 
        """
        Initialize No-op reset wrapper. 

        Args: 
            env: Base Gymnasium environment 
            noop_max: Maximum number of no-op actions to apply at reset
        """

        super().__init__(env) # Initialize environment
        self.noop_max = noop_max # Maximum number of no-op actions
        self.noop_action = 0 # No-op assumed to be action 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP' # Sanity check


    # Reset environment 
    def reset(self, **kwargs):
        """
        Reset environment with random no-op actions. 

        Returns: 
            obs: Observation after no-ops
            info: Additional environment info 
        """

        obs, info = self.env.reset(**kwargs) # Reset base environment
        noops = np.random.randint(1, self.noop_max + 1) # Sample random number of no-ops

        for _ in range (noops):
            # Apply no-op
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            
            if terminated or truncated: 
                obs, info = self.env.reset(**kwargs)

        return obs, info 

# Factory function
def make_atari_env(env_name='ALE/Pong-v5'):
    """ 
    Create an Atari environment with standard preprocessing. 

    Args: 
        env_name: Name of the Atari environment 

    Returns: 
        env: Preprocessed Gymnasium environment 
    """
 
    env = gym.make(env_name) # Creates base Atari environment
    env = NoopResetEnv(env, noop_max = 30) # Randomize start
    # Grayscale, resize, skip, max-pool
    env = AtariPreprocessing(env, frame_skip = 4, screen_size = 84) 
    env = FrameStack(env, k = 4) # Temporal context (stack last 4 frames)

    return env
