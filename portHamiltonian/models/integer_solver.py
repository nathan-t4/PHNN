"""
RL environment is forward pass of PHNODE
"""

import gymnasium as gym
from gymnasium import spaces

from portHamiltonian.models import PHNODE

class PHNODEEnv(gym.Env):
    def __init__(self, num_entries: int, low, high, model):
        super().__init__()
        self.model: PHNODE = model
        self.action_space = spaces.Discrete(num_entries)
        self.observation_space = spaces.Box(low, high)
    
    def _reward():
        pass
    
    def step(self, action):
        self.last_J = self.last_J + action

        observation = (self.last_J - self.R) @ dH + self.g @ u

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...

# TODO: add to training code for PHNODE (make new file for training code!)
from stable_baselines3 import A2C

env = PHNODEEnv()
model = A2C("MlpPolicy", env)
model.learn(total_timesteps=1000)
