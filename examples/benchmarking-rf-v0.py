import gym4real.envs.robofeeder.rf_picking_v0 as env
import numpy as np
import time
import shutil
import os
os.chdir("..")

default_config_file = "./gym4real/envs/robofeeder/configuration.yaml"
config_file = "./examples/robofeeder/notebooks/configuration_editable.yaml"
shutil.copy(default_config_file, config_file)

n_episodes = 1
rf_env = env.robotEnv(config_file)
rewards = []
start = time.perf_counter()

for _ in range(n_episodes):
    done = False
    rf_env.reset()
    while not done:
        action = np.array(rf_env.action_space.sample())
        next_obs, reward, terminated, truncated, _ = rf_env.step(action)
        rewards.append(reward)
        done = terminated or truncated

end = time.perf_counter()
print(f"One episode of Robofeeder-Picking-v0 required {end - start} seconds")
rf_env.close()
del rf_env

