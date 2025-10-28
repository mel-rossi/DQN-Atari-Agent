import os
import torch
import ale_py  
import argparse 
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

# -- 1. Argument Parsing ---

parser = argparse.ArgumentParser(description="Train / resume training DQN agent for Pong.")

# Resume training from a saved model checkpoint
parser.add_argument(     
    "--load_model",
    type = str,
    default = None,
    help = (
        "Path to the model .zip file to load and continue training "
        "(e.g., models/best_model.zip)."
    )
)

# Control how long the agent trains
parser.add_argument(
    "--total_timesteps",
    type = int,
    default = 2_000_000,
    help = "Total number of timesteps for the training run (cumulative)."
)

args = parser.parse_args()

# --- 2. Configuration & Hyperparameters ---

# Gym environment to train on
ENV_ID = "PongNoFrameskip-v4"

# Tensorboard logs directory
LOG_DIR = "./logs/"

# Model checkpoints directory 
# (EvalCallback saves as best_model.zip, .load adds .zip)
MODEL_DIR = "./models/" 
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model"

# Create model directory if missing 
os.makedirs(MODEL_DIR, exist_ok = True)

# Selects compute device: mps (Metal Performance Shaders - Apple GPU) or cpu 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" 

# Training hyperparameters
TOTAL_TIMESTEPS = args.total_timesteps # Number of training steps
LEARNING_RATE = 1e-4 # Step size for gradient descent
BUFFER_SIZE = 200_000 # Size of replay buffer
LEARNING_STARTS = 50_000 # Number of steps before training begins
BATCH_SIZE = 32 # Number of samples per training batch
GAMMA = 0.99 # Discount factor for future rewards
TRAIN_FREQ = 4 # Agent trains every n (TRAIN_FREQ) steps
GRADIENT_STEPS = 1 # Number of gradient updates per training step
TARGET_UPDATE_INTERVAL = 1000 # Frequency of target network updates

# Evaluation settings
EVAL_FREQ = 10_000 # Run evaluation every n (EVAL_FREQ) steps
N_EVAL_EPISODES = 10 # Number of episodes to run during each evaluation phase

# --- 3. Create Environments ---

print(f"--- Using device: {DEVICE} ---")
print("--- Initializing Environment ---")

# Training Environment 
vec_env = make_atari_env(ENV_ID, n_envs = 4, seed = 0) # Creates vectorized Atari environment
vec_env = VecFrameStack(vec_env, n_stack = 4) # Wraps environment (stacking 4 frames)

# Evaluation Environment
eval_env = make_atari_env(ENV_ID, n_envs = 1, seed = 42) 
eval_env = VecFrameStack(eval_env, n_stack = 4)

# --- 4. Setup Callback ---

# Creates a EvalCallBack to periodically evaluate agent during training
eval_callback = EvalCallback(eval_env,
                             best_model_save_path = MODEL_DIR,
                             log_path = LOG_DIR,
                             eval_freq = EVAL_FREQ,
                             n_eval_episodes = N_EVAL_EPISODES,
                             deterministic = True,
                             render = False)

# --- 5. Define or Load Model ---

# Check if load latest checkpoint if available
if args.load_model is None: 
    print("--- Defining New DQN Model ---")

    # Initilize a new DQN agent using SB3
    model = DQN(
        "CnnPolicy", 
        vec_env, 
        learning_rate = LEARNING_RATE, 
        buffer_size = BUFFER_SIZE,
        learning_starts = LEARNING_STARTS, 
        batch_size = BATCH_SIZE, 
        gamma = GAMMA,
        train_freq = TRAIN_FREQ, 
        gradient_steps = GRADIENT_STEPS,
        target_update_interval = TARGET_UPDATE_INTERVAL, 
        verbose = 1,
        tensorboard_log = LOG_DIR, 
        device = DEVICE,
        optimize_memory_usage = True
    )

else: 
    print(f"--- Loading Existing Model from: {args.load_model} ---")
    load_path = args.load_model.replace('.zip', '')

    # Loads saved model checkpoint
    model = DQN.load(
        load_path, # DQN.load automatically adds .zip
        env = vec_env, 
        device = DEVICE, 
        tensorboard_log = LOG_DIR
    )

    # Ensure env is set for the loaded model (needed for .learn() to work properly)
    model.set_env(vec_env)
    print(f"--- Resuming training from timestep: {model.num_timesteps} ---")

# --- 6. Train the Model ---

print("--- Starting Model Training ---")

# Core training call
model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    # reset_num_timesteps = False allows continuing from loaded model's steps
    reset_num_timesteps = (args.load_model is None),
    callback = eval_callback,
    progress_bar = True
)

print("--- Training Complete ---")

# --- 7. Evaluate the *Best* Trained Model ---

print("--- Evaluating Best Model ---")

# Load best-performing model 
# a.k.a. 'best_model.zip' saved by EvalCallback during the *latest* run
model = DQN.load(
    BEST_MODEL_PATH, 
    device = DEVICE
)

# Run evaluation episodes
mean_reward, std_reward = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes = N_EVAL_EPISODES
)

print(f"=== FINAL EVALUATION ===")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"==========================")
