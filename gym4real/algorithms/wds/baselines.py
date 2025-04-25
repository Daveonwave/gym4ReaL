from gym4real.envs.wds.env_cps import WaterDistributionSystemEnv
from tqdm import tqdm
import numpy as np
import os
import json


def run_baseline(env_params, args, test_profile):
    
    env = WaterDistributionSystemEnv(settings=env_params)
    
    if args['algo'][0] == 'random':
        random_action_policy(env, args['exp_name'], test_profile)

    else:
        print("Chosen baseline is not implemented or not existent!")
        exit(1)


def random_action_policy(env, exp_name, test_profile):
    
    print("######## RANDOM policy is running... ########")
    
    exp_results = {
        'test': test_profile,
        'pure_reward': {'r_trad': [], 'r_deg': [], 'r_clip': []},
        'norm_reward': {'r_trad': [], 'r_deg': [], 'r_clip': []},
        'weighted_reward': {'r_trad': [], 'r_deg': [], 'r_clip': []},
        'total_reward': 0,
        'actions': []
    }
    
    logdir = "./logs/{}/results/random/".format(exp_name)
    os.makedirs(logdir, exist_ok=True)

    env.reset(options={'eval_profile': test_profile})

    done = False
    pbar = tqdm(total=len(env.generation))
    while not done:
        act = env.action_space.sample()  # Randomly select an action
        obs, reward, terminated, truncated, info = env.step(act)  # Return observation and reward
        done = terminated or truncated
        pbar.update(1)
    
    pbar.close()
    output_file = logdir + 'test_{}.json'.format(test_profile)
    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(exp_results, f, allow_nan=False) 


def deterministic_action_policy(env, action:float, algo_name: str, exp_name: str, test_profile):
    assert 0 <= action <= 1, "The deterministic action must be between 0 and 1."

    print("######## {} policy is running... ########".format(algo_name.upper()))
    
    exp_results = {
        'test': test_profile,
        'pure_reward': {'r_trad': [], 'r_deg': [], 'r_clip': []},
        'norm_reward': {'r_trad': [], 'r_deg': [], 'r_clip': []},
        'weighted_reward': {'r_trad': [], 'r_deg': [], 'r_clip': []},
        'total_reward': 0,
        'actions': []
    }
    
    logdir = "./logs/{}/results/{}/".format(exp_name, algo_name)
    os.makedirs(logdir, exist_ok=True)

    env.reset(options={'eval_profile': test_profile})

    done = False
    pbar = tqdm(total=len(env.generation))
    while not done:
        act = np.array([action])  # Randomly select an action
        obs, reward, terminated, truncated, info = env.step(act)  # Return observation and reward
        done = terminated or truncated
        pbar.update(1)

    pbar.close()
    output_file = logdir + 'test_{}.json'.format(test_profile)
    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(exp_results, f, allow_nan=False) 

