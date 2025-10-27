import os 
import numpy as np 
import matplotlib.pyplot as plt
from .atari_preprocessing import make_atari_env
from .dqn_agent import DQNAgent 
import time

# Detect latest checkpoint iteration
def get_latest_checkpoint(checkpoint_dir):
    import os, re
    files = os.listdir(checkpoint_dir)
    pattern = r'online_network_(\d+)\.weights\.h5'
    iterations = [int(re.search(pattern, f).group(1)) for f in files if re.search(pattern, f)]
    return max(iterations) if iterations else 0

start_iteration = get_latest_checkpoint('checkpoints')

# Full training loop: 
# environment iteraction, replay buffer filling, training, 
# evaluation, logging, and checkpointing.
def train_dqn(
    env_name = 'ALE/Pong-v5', 
    num_iterations = 10000, 
    max_steps_per_episode = 1000, # Cap episode length: 27000 -> 1000 
    evaluation_period = 500, # Reduce evaluation frequency: 100 -> 500 
    num_eval_episodes = 10, 
    checkpoint_period = 500, 
    log_period = 10
):
    """
    Train DQN agent on Atari game. 

    Args: 
        env_name: Name of Atari environment
        num_iterations: Number of training iterations (episodes) 
        max_steps_per_episode: Maximum steps per episode
        evaluation_period: Evaluate every N iterations
        num_eval_episodes Number of episodes for evaluation
        checkpoint_period: Save checkpoint every N iterations
        log_period: Log progress every N iterations
    """

    # Create directories (saving checkpoints & logs) 
    os.makedirs('checkpoints', exist_ok = True)
    os.makedirs('logs', exist_ok = True) 

    # Create environment
    print("Creating environment...")
    env = make_atari_env(env_name) # wrapped Atari env with preprocessing
    num_actions = env.action_space.n # number of discrete actions

    print(f"Environment: {env_name}")
    print(f"Number of actions: {num_actions}")
    print(f"Observation shape: {env.observation_space.shape}")

    # Create agent 
    print("\nCreating DQN agent...")
    agent = DQNAgent(
        num_actions = num_actions,
        observation_shape = env.observation_space.shape,
        gamma = 0.99,
        min_replay_history = 50000,
        update_period = 4,
        target_update_period = 10000,
        epsilon_train = 0.01,
        epsilon_eval = 0.001,
        epsilon_decay_period = 1000000,
        replay_capacity = 1000000,
        batch_size = 256, # Increased batch size : 32 -> 128 / 256
        learning_rate = 0.00025
    )

    # Load latest checkpoint if available 
    start_iteration = get_latest_checkpoint('checkpoints')
    if start_iteration > 0:
        print(f"\nResuming from checkpoint at iteration {start_iteration}...")
        agent.load('checkpoints', start_iteration)

        # Load replay buffer
        replay_path = (f'checkpoints/replay_buffer_{start_iteration}.pkl')
        if os.path.exists(replay_path):
            agent.load_replay_buffer(replay_path, 'checkpoints', iteration)
            print(f"Replay buffer loaded from {replay_path}")
        else:
            print("Replay buffer not found — training will resume with empty buffer.")

    else:
        print("\nStarting fresh training...")

    # Start training from latest iteration
    iteration = start_iteration
    total_steps = 0

    # Training metrics 
    episode_rewards = [] # total reward per episode
    episode_lengths = [] # steps per episode
    training_losses = [] # mean loss per episode
    evaluation_rewards = [] # evaluation results over time

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    while iteration < num_iterations:
        # Training episode
        state, _ = env.reset() # reset environment
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        agent.set_eval_mode(False) # training mode

        for step in range(max_steps_per_episode):
            # Select (greedy-epsilon action) 
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Clip reward to [-1, 1]
            clipped_reward = np.sign(reward)

            # Store transition
            agent.store_transition(state, action, clipped_reward, next_state, done)

            # Train agent 
            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            agent.training_steps += 1
            
            if done:
                break

        # Record metrics 
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_losses:
            training_losses.append(np.mean(episode_losses))
        
        iteration += 1

        # Logging 
        if iteration % log_period == 0:
            avg_reward = np.mean(episode_rewards[-log_period:])
            avg_length = np.mean(episode_lengths[-log_period:])
            avg_loss = np.mean(training_losses[-log_period:]) if training_losses else 0
            
            print(f"Iteration {iteration}/{num_iterations} | "
                  f"Steps: {total_steps} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Avg Length: {avg_length:.0f} | "
                  f"Loss: {avg_loss:.4f}")

        # Evaluation
        if iteration % evaluation_period == 0:
            print(f"\n{'='*70}")
            print(f"Running evaluation at iteration {iteration}...")
            eval_reward = evaluate_agent(agent, env_name, num_eval_episodes)
            evaluation_rewards.append((iteration, eval_reward))
            print(f"Evaluation reward: {eval_reward:.2f}")
            print(f"{'='*70}\n")

        # Checkpointing and plotting
        if iteration % checkpoint_period == 0:
            
            agent.save('checkpoints', iteration)
            agent.save_replay_buffer(f'checkpoints/replay_buffer_{iteration}.pkl', 'checkpoints', iteration)
            plot_training_progress(
                episode_rewards, 
                training_losses, 
                evaluation_rewards,
                iteration
            )

    # Final save
    agent.save('checkpoints', iteration)
    agent.save_replay_buffer(f'checkpoints/replay_buffer_{iteration}.pkl', 'checkpoints', iteration)
    plot_training_progress(
        episode_rewards, 
        training_losses, 
        evaluation_rewards,
        iteration
    )
    
    env.close()
    print("\nTraining completed!")
    
    return agent

# Agent evaluation: runs agent in eval mode for multiple episodes and computes average reward.
def evaluate_agent(agent, env_name, num_episodes):
    """
    Evaluate agent performance.
    
    Args:
        agent: DQN agent
        env_name: Environment name
        num_episodes: Number of evaluation episodes
        
    Returns:
        Average reward over evaluation episodes
    """

    eval_env = make_atari_env(env_name) # fresh eval environment
    agent.set_eval_mode(True) # use eval epsilon
    
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state) # greedy or low-epsilon
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    eval_env.close()
    agent.set_eval_mode(False)
    
    return np.mean(eval_rewards) # average reward across epsiodes

# Visualizes learning curves and saves them to disk
def plot_training_progress(rewards, losses, eval_rewards, iteration):
    """Plot training progress."""

    fig, axes = plt.subplots(2, 2, figsize = (15, 10))
    
    # Plot episode rewards
    axes[0, 0].plot(rewards, alpha=0.6)
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(rewards)), moving_avg, 'r-', linewidth=2, 
                       label='Moving Avg (100)')
        axes[0, 0].legend()

    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training loss
    if losses:
        axes[0, 1].plot(losses, alpha=0.6)
        if len(losses) >= 100:
            loss_avg = np.convolve(losses, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(losses)), loss_avg, 'r-', linewidth=2,
                           label='Moving Avg (100)')
            axes[0, 1].legend()

        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True, alpha=0.3)

    
    # Plot evaluation rewards
    if eval_rewards:
        eval_iterations, eval_scores = zip(*eval_rewards)
        axes[1, 0].plot(eval_iterations, eval_scores, 'o-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Evaluation Performance')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot episode lengths (histogram of recent episodes)
    if len(rewards) >= 100:
        recent_rewards = rewards[-1000:] if len(rewards) >= 1000 else rewards
        axes[1, 1].hist(recent_rewards, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Recent Reward Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'logs/training_progress_{iteration}.png', dpi=150) # save figure
    plt.close()

if __name__ == '__main__':
    # Train the agent
    train_dqn(
        env_name='ALE/Pong-v5',
        num_iterations=start_iteration + 5000, # Updated (dynamic)
        max_steps_per_episode=1000, # Updated
        evaluation_period=500, # Updated
        num_eval_episodes=10,
        checkpoint_period=500,
        log_period=10
    )




