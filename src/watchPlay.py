import gymnasium as gym
import ale_py  # Registers the Atari environments
import torch
import time # Optional: To slow down rendering

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

# --- Configuration ---
ENV_ID = "PongNoFrameskip-v4"
MODEL_PATH = "./models/best_model.zip"
N_STACK = 4 # keep it same as training

# --- 1. Create Environment ---
print("--- Initializing Environment ---")
# Create a single environment with rendering enabled
# render_mode="human" automatically pops up a window
env = make_atari_env(
    ENV_ID,
    n_envs=1,
    seed=0, # Use any seed, it just initializes the environment state
    env_kwargs={"render_mode": "human"}
)
# same frame stacking used during training
env = VecFrameStack(env, n_stack=N_STACK)

# --- 2. Load Trained Model ---
print(f"--- Loading Model: {MODEL_PATH} ---")
model = DQN.load(MODEL_PATH, device="cpu")
print("--- Model Loaded. Starting Game... ---")

# --- 3. Run the Game Loop ---
obs = env.reset()
episodes_played = 0
while episodes_played < 4: # Play 10 games then stop
    # Get the agent's action (deterministic=True means no random exploration)
    action, _states = model.predict(obs, deterministic=True)

    # Take the action in the environment
    obs, rewards, dones, info = env.step(action)

    
    # Decide the delay in game to make it easier to watch
    time.sleep(0.01)

    # Check if the game(s) ended
    if dones[0]:
        print(f"Game {episodes_played + 1} finished. Score: {info[0].get('episode', {}).get('r', 'N/A')}")
        episodes_played += 1
        obs = env.reset() 

print("--- Finished playing. Closing environment. ---")
env.close() # Close the environment window