import numpy as np
import time
from atari_preprocessing import make_atari_env
from dqn_agent import DQNAgent

# Sets up Atari environment for evaluation
def play_game(
    checkpoint_dir = 'checkpoints',
    iteration = 5000,
    env_name = 'ALE/Breakout-v5',
    num_episodes = 5,
    render = True
):
    """
    Play game with trained agent.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        iteration: Iteration number to load
        env_name: Environment name
        num_episodes: Number of episodes to play
        render: Whether to render the game
    """

    # Create environment
    if render: # Builds a human-rendable environment with preprocessing weights
        env = make_atari_env(env_name.replace('ALE/', 'ALE/').replace('-v5', '-v5'))

        # For rendering, we need to use render_mode
        import gymnasium as gym
        render_env = gym.make(env_name, render_mode='human') # game window visible

        from atari_preprocessing import NoopResetEnv, AtariPreprocessing, FrameStack
        render_env = NoopResetEnv(render_env, noop_max=30)
        render_env = AtariPreprocessing(render_env, frame_skip=4, screen_size=84)
        render_env = FrameStack(render_env, k=4)
        env = render_env

    else:
        env = make_atari_env(env_name)

    # Action space
    num_actions = env.action_space.n # retrives the number of discrete actions available in  env

    # Agent creation
    print("Creating agent...")
    agent = DQNAgent(
        num_actions = num_actions, # initialize agent with correct action space
        observation_shape = env.observation_space.shape
    )

    # Load trained model
    print(f"Loading model from iteration {iteration}...")
    agent.load(checkpoint_dir, iteration) # load trained weights from checkpoint
    agent.set_eval_mode(True)
    
    print(f"\nPlaying {env_name} for {num_episodes} episodes...\n")
    
    episode_rewards = []

    # Episode loop
    for episode in range(num_episodes):
        state, _ = env.reset() # reset environment
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes} starting...")
        
        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
           
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
           
            # Rendering
            if render:
                time.sleep(0.02) # Small delay for smoother rendering  

        # Logging
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished: Reward = {episode_reward:.1f}, Steps = {steps}")
   
    # Cleanup
    env.close() # close environment
    
    print(f"\n{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Dev: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.1f}")
    print(f"Max Reward: {np.max(episode_rewards):.1f}")
    print(f"{'='*50}")

if __name__ == '__main__':

    # Play with trained model
    play_game(
        checkpoint_dir='checkpoints',
        iteration=5000,  # Change this to match your checkpoint
        env_name='ALE/Breakout-v5',
        num_episodes=5,
        render=True
    )
