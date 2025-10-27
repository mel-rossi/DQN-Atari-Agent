import gymnasium as gym
import ale_py
import gymnasium.envs.registration

# Register ALE environments once
gymnasium.envs.registration.register_envs(ale_py)

