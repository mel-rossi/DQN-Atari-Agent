# Stores different training configurations -
# A config file with three presets: 'test' (quick testing), 'default' (normal training), 
# and 'heavy' (serious long training). 
# Contains all hyperparameters like learning rate, epsilon decay, replay buffer size, etc.
# Makes it easy to switch between different training setups without editing code. 



# config.py
# File to store all project hyperparameters and settings
import torch

# Environment and Paths 
ENV_ID = "PongNoFrameskip-v4"
LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./models/"
BEST_MODEL_PATH = f"{MODEL_DIR}/dqn_pong_best_model.zip"

# 2. Device 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#  3. Training Hyperparameters
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 1e-4
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1000

# 4. Evaluation 
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10