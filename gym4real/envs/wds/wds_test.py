import sys
import os

sys.path.append(os.getcwd())

from gym4real.envs.wds.utils import parameter_generator
import gymnasium as gym
from tqdm.rich import tqdm
import cProfile, pstats, functools


if __name__ == '__main__':    
    
    # Profiler setup
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    params = parameter_generator(world_options='gym4real/envs/wds/world_anytown.yaml',
                                 hydraulic_step=600,
                                 duration=24 * 3600 * 7,
                                 seed=42,
                                 reward_coeff={'dsr_coeff': 1.0, 'overflow_coeff': 1.0},
                                 use_reward_normalization=True)
    #dqn(params)
        
    env = gym.make("gym4real/wds-v0", **{'settings':params})
    n_episodes = 1
    rewards = []
    cumulated_reward = 0

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset(options={'is_evaluation': True})
        done = False

        while not done:
            action = 3  # Randomly select an action
            obs, reward, terminated, truncated, info = env.step(action)  # Return observation and reward
            done = terminated or truncated
            cumulated_reward += reward
    
    # Profiler teardown
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.dump_stats('wds.prof')
    
    
    
    