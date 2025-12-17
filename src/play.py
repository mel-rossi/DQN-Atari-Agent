# Copyright 2025 Melani Rossi Rodrigues, Angel Cortes Gildo, Maria Alexandra Lois Peralejo, and Colby Snook
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import time 
import torch
import ale_py  
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

# --- Configuration ---

# Should match training (train.py)
ENV_ID = "FreewayNoFrameskip-v4" 
MODEL_PATH = "./models/best_model"
N_STACK = 4

# --- 1. Create Environment ---

print("--- Initializing Environment ---")

# Create a single environment with rendering enabled
# render_mode="human" automatically pops up a window
env = make_atari_env(
    ENV_ID,
    n_envs = 1,
    seed = 0, # Any seed (initializes the environment state)
    env_kwargs = {"render_mode": "human"} # real-time rendering
)

# Frame stacking (same as during training)
env = VecFrameStack(env, n_stack = N_STACK)

# --- 2. Load Trained Model ---

print(f"--- Loading Model: {MODEL_PATH} ---")
model = DQN.load(MODEL_PATH, device = "cpu") # Loads DQN model from disk

print("--- Model Loaded. Starting Game... ---")

# --- 3. Run the Game Loop ---

# Resets environment to get initial observation
obs = env.reset()
episodes_played = 0

# Loop runs until 3 full episodes (games) are completed
while episodes_played < 3:
    # Predict next action
    # Get the agent's action (deterministic = True means no random exploration)
    action, _states = model.predict(obs, deterministic = True)

    # Executes action in the environment
    obs, rewards, dones, info = env.step(action)
    
    # Delay for visibility (makes the game watchable) 
    time.sleep(0.01)

    # Episode tracking (Check if the game(s) ended)
    if dones[0]:
        print(f"Game {episodes_played + 1} finished."
              f"Score: {info[0].get('episode', {}).get('r', 'N/A')}")

        episodes_played += 1
        obs = env.reset() 

# Clean up
print("--- Finished playing. Closing environment. ---")
env.close() # Close environment window
