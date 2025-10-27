import gymnasium as gym
import ale_py  # ensures ALE is available

# This line is the key: it registers Atari envs with Gymnasium
import gymnasium.envs.registration
gymnasium.envs.registration.register_envs(ale_py)

print([env for env in gym.envs.registry.keys() if "Pong" in env])

