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
BEST_MODEL_PATH = f"{MODEL_DIR}/dqn_pong_best_model.zip"

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Training Hyperparameters
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 1e-4
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1000

# Evaluation
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10