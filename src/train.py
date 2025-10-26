import gymnasium as gym
import ale_py  # Registers the Atari environments
import torch
import os
import argparse # <-- Import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Train or resume training a DQN agent for Pong.")
parser.add_argument(
    "--load_model",
    type=str,
    default=None,
    help="Path to the model .zip file to load and continue training (e.g., models/best_model.zip)."
)
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=2_000_000,
    help="Total number of timesteps for the training run (cumulative)."
)
args = parser.parse_args()

# --- 2. Configuration & Hyperparameters ---
ENV_ID = "PongNoFrameskip-v4"
LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./models/"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model" # EvalCallback saves as best_model.zip, .load adds .zip

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TOTAL_TIMESTEPS = args.total_timesteps
LEARNING_RATE = 1e-4
BUFFER_SIZE = 200_000
LEARNING_STARTS = 50_000
BATCH_SIZE = 32
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1000

EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10

# --- 3. Create Environments ---
print(f"--- Using device: {DEVICE} ---")
print("--- Initializing Environment ---")

vec_env = make_atari_env(ENV_ID, n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

eval_env = make_atari_env(ENV_ID, n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)

# --- 4. Setup Callback ---
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=MODEL_DIR,
                             log_path=LOG_DIR,
                             eval_freq=EVAL_FREQ,
                             n_eval_episodes=N_EVAL_EPISODES,
                             deterministic=True,
                             render=False)

# --- 5. Define or Load Model ---
if args.load_model is None:
    print("--- Defining New DQN Model ---")
    model = DQN(
        "CnnPolicy", vec_env, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE, gamma=GAMMA,
        train_freq=TRAIN_FREQ, gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL, verbose=1,
        tensorboard_log=LOG_DIR, device=DEVICE
    )
else:
    print(f"--- Loading Existing Model from: {args.load_model} ---")
    # Check if the path includes .zip, remove it for consistency if it does
    load_path = args.load_model.replace('.zip', '')
    model = DQN.load(
        load_path, # DQN.load automatically adds .zip
        env=vec_env, device=DEVICE, tensorboard_log=LOG_DIR
    )
    # Ensure env is set for the loaded model (needed for learn method)
    model.set_env(vec_env)
    print(f"--- Resuming training from timestep: {model.num_timesteps} ---")

# --- 6. Train the Model ---
print("--- Starting Model Training ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    # reset_num_timesteps=False allows continuing from loaded model's steps
    reset_num_timesteps=(args.load_model is None),
    callback=eval_callback,
    progress_bar=True
)
print("--- Training Complete ---")

# --- 7. Evaluate the *Best* Trained Model ---
print("--- Evaluating Best Model ---")
# Always load the 'best_model.zip' saved by EvalCallback during the *latest* run
model = DQN.load(BEST_MODEL_PATH, device=DEVICE)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)

print(f"=== FINAL EVALUATION ===")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"==========================")