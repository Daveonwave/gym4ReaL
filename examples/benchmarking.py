import os
import time


os.chdir("..")

from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from gym4real.envs.dam. utils import parameter_generator

eval_params = parameter_generator(
    world_options='./gym4real/envs/dam/world_test.yaml',
    lake_params='./gym4real/envs/dam/lake.yaml'
)

eval_env = gym.make('gym4real/dam-v0', settings=eval_params)

print(type(eval_env))
n_episodes = 1
rewards = {}
alg = 'random'
rewards[alg] = []
start = time.perf_counter()
for episode in tqdm(range(n_episodes)):
    obs, info = eval_env.reset(options={'rewind_profiles': episode == 0})
    done = False
    rewards_episode = []

    while not done:
        action = eval_env.action_space.sample()  # Randomly select an action
        obs, reward, terminated, truncated, info = eval_env.step(action)  # Return observation and reward
        done = terminated or truncated
        rewards_episode.append(info['weighted_reward'])

    rewards[alg].append(rewards_episode)
end = time.perf_counter()
print(f"One episode of DamEnv required {end - start} seconds")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import gymnasium as gym

from gym4real.envs.elevator.utils import parameter_generator

params = parameter_generator(world_options='gym4real/envs/elevator/world.yaml', seed=1234)
env = gym.make(id="gym4real/elevator-v0", **{'settings': params})

alg = 'random'
rewards[alg] = {}
rewards = defaultdict(dict)
n_episodes = 1
start = time.perf_counter()
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    cumulated_reward = 0
    rewards[alg][episode] = {}
    rewards[alg][episode]['cum'] = []
    rewards[alg][episode]['reward'] = []

    while not done:
        action = env.action_space.sample()  # Randomly select an action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulated_reward += reward
        rewards[alg][episode]['cum'].append(cumulated_reward)
        rewards[alg][episode]['reward'].append(reward)
end = time.perf_counter()
print(f"One episode of ElevatorEnv required {end - start} seconds")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from gymnasium.utils.env_checker import check_env
import gymnasium as gym

from gym4real.envs.microgrid.utils import parameter_generator
params = parameter_generator(world_options='gym4real/envs/microgrid/world_test.yaml')
env = gym.make(id="gym4real/microgrid-v0", **{'settings':params})
alg = 'random'
rewards[alg] = {}
test_profiles = [i for i in range(370, 371)] #Testing one profile
rewards = defaultdict(dict)
start = time.perf_counter()
for profile in tqdm(test_profiles):
    obs, info = env.reset(options={'eval_profile': str(profile)})
    done = False
    rewards[alg][profile] = {}
    rewards[alg][profile]['pure'] = []

    while not done:
        action = env.action_space.sample()  # Randomly select an action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards[alg][profile]['pure'].append(list(info['pure_rewards'].values()))
end = time.perf_counter()
print(f"One episode of MicrogridEnv required {end - start} seconds")



from gym4real.envs.trading.utils import parameter_generator
from stable_baselines3.common.env_util import make_vec_env

params_train = parameter_generator(world_options='gym4real/envs/trading/world_train.yaml')

env_bnh = gym.make("gym4real/TradingEnv-v0", **{'settings':params_train})

rewards_bnh = []
daily_return_long = []
n_episodes = 1
start = time.perf_counter()
for _ in range(n_episodes):
    done = False
    env_bnh.reset()
    cum_rew_ep = 0
    while not done:
        next_obs, reward, terminated, truncated, _ = env_bnh.step(2)
        cum_rew_ep += reward
        rewards_bnh.append(reward)
        done = terminated or truncated
    daily_return_long.append((cum_rew_ep/ env_bnh.unwrapped._capital)*100)

daily_return_long = np.asarray(daily_return_long)
end = time.perf_counter()
print(f"One episode of TradingEnv required {end - start} seconds")