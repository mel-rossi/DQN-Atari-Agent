import gymnasium as gym
import ale_py  # Registers the Atari environments
import torch
import os

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. Configuration & Hyperparameters ---
# Environment and Paths
ENV_ID = "PongNoFrameskip-v4"
LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./models/"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model" 

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Training Hyperparameters
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 1e-4
BUFFER_SIZE = 200_000
LEARNING_STARTS = 50_000
BATCH_SIZE = 32
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1000

# Evaluation
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10

# --- 2. Create Environments ---
print(f"--- Using device: {DEVICE} ---")
print("--- Initializing Environment ---")

# Create 4 parallel environments for training
vec_env = make_atari_env(ENV_ID, n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Create a separate, single environment for evaluation
eval_env = make_atari_env(ENV_ID, n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)


# --- 3. Setup Callback ---
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=MODEL_DIR,
                             log_path=LOG_DIR,
                             eval_freq=EVAL_FREQ,
                             n_eval_episodes=N_EVAL_EPISODES,
                             deterministic=True,
                             render=False)
                            

# --- 4. Define the Model ---
print("--- Defining DQN Model ---")
model = DQN(
    "CnnPolicy",
    vec_env,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    train_freq=TRAIN_FREQ,
    gradient_steps=GRADIENT_STEPS,
    target_update_interval=TARGET_UPDATE_INTERVAL,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device=DEVICE
)

# --- 5. Train the Model ---
print("--- Starting Model Training ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True
)
print("--- Training Complete ---")

# --- 6. Evaluate the *Best* Trained Model ---
print("--- Evaluating Best Model ---")
# The .load() function will automatically add ".zip" to BEST_MODEL_PATH
# It will now correctly look for "./models/best_model.zip"
model = DQN.load(BEST_MODEL_PATH, device=DEVICE)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)

print(f"=== FINAL EVALUATION ===")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"==========================")