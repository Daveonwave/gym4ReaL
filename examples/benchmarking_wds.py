import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from gymnasium.utils.env_checker import check_env
import gymnasium as gym
import time
import os

os.chdir("..")

from gym4real.envs.wds.utils import parameter_generator


params = parameter_generator(hydraulic_step=3600, duration=3600*24*7, seed=1234)
env = gym.make(id="gym4real/wds-v0", **{'settings':params})
rewards = {}
alg = 'random'
rewards[alg] = {}
n_episodes = 1
rewards = defaultdict(dict)
start = time.perf_counter()
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset(options={'is_evaluation': True})
    done = False
    rewards[alg][episode] = []

    while not done:
        action = env.action_space.sample()  # Randomly select an action
        obs, reward, terminated, truncated, info = env.step(action)  # Return observation and reward
        done = terminated or truncated
        rewards[alg][episode].append(list(info['pure_rewards'].values()))
end = time.perf_counter()
print(f"One episode of WDSEnv required {end - start} seconds")