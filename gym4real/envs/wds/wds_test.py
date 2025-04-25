from gym4real.envs.wds.simulator.water_network import WaterNetwork
from gym4real.envs.wds.env_cps import WaterDistributionSystemEnv
from gym4real.envs.wds.utils import parameter_generator
from gym4real.algorithms.wds.dqn import dqn
from rich.pretty import pprint
from tqdm.rich import tqdm
import cProfile, pstats, functools


if __name__ == '__main__':    
    
    # Profiler setup
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    params = parameter_generator()    
    dqn(params)
    exit(0)
    
        
    env = WaterDistributionSystemEnv(params)

    n_episodes = 1
    rewards = []
    cumulated_reward = 0

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset(options={'is_evaluation': True})
        done = False

        while not done:
            action = env.action_space.sample()  # Randomly select an action
            obs, reward, terminated, truncated, info = env.step(action)  # Return observation and reward
            done = terminated or truncated
            cumulated_reward += reward
    
    # Profiler teardown
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.dump_stats('wds.prof')
    
    
    
    