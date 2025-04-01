from gym4real.envs.microgrid.env import MicroGridEnv
from gym4real.envs.microgrid.utils import parameter_generator
from tqdm import tqdm
import numpy as np
import os
import json


def run_baseline(env_params, args, test_profile, model_file=''):
    
    env = MicroGridEnv(settings=env_params)
    
    if args['algo'][0] == 'random':
        random_action_policy(env, args['exp_name'], test_profile)
    
    elif args['algo'][0] == 'only_market':
        deterministic_action_policy(env, action=0., algo_name=args['algo'][0], exp_name=args['exp_name'], test_profile=test_profile)
    
    elif args['algo'][0] == 'battery_first':
        deterministic_action_policy(env, action=1., algo_name=args['algo'][0], exp_name=args['exp_name'], test_profile=test_profile)
    
    elif args['algo'][0] == '20-80':
        deterministic_action_policy(env, action=0.2, algo_name=args['algo'][0], exp_name=args['exp_name'], test_profile=test_profile)
    
    elif args['algo'][0] == '80-20':
        deterministic_action_policy(env, action=0.8, algo_name=args['algo'][0], exp_name=args['exp_name'], test_profile=test_profile)
    
    elif args['algo'][0] == '50-50':
        deterministic_action_policy(env, action=0.5, algo_name=args['algo'][0], exp_name=args['exp_name'], test_profile=test_profile)
    
    elif args['algo'][0] == 'all_baselines':
        random_action_policy(env, args['exp_name'], test_profile)
        deterministic_action_policy(env, action=0., algo_name="only_market", exp_name=args['exp_name'], test_profile=test_profile)
        deterministic_action_policy(env, action=1., algo_name="battery_first", exp_name=args['exp_name'], test_profile=test_profile)
        deterministic_action_policy(env, action=0.2, algo_name="20-80", exp_name=args['exp_name'], test_profile=test_profile)
        deterministic_action_policy(env, action=0.5, algo_name="50-50", exp_name=args['exp_name'], test_profile=test_profile)
        deterministic_action_policy(env, action=0.8, algo_name="80-20", exp_name=args['exp_name'], test_profile=test_profile)

    else:
        print("Chosen baseline is not implemented or not existent!")
        exit(1)


def random_action_policy(env, exp_name, test_profile):
    
    print("######## RANDOM policy is running... ########")
    
    comparison_dict = {
        'test': test_profile,
        'pure_reward': {},
        'norm_reward': {},
        'weighted_reward': {},
        'total_reward': 0
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

    comparison_dict['total_reward'] = info['total_reward']
    comparison_dict['pure_reward'] = info['pure_reward_list']
    comparison_dict['norm_reward'] = info['norm_reward_list']
    comparison_dict['weighted_reward'] = info['weighted_reward_list']
    comparison_dict['actions'] = info['actions']
    comparison_dict['states'] = info['states']
    comparison_dict['traded_energy'] = info['traded_energy']
    comparison_dict['soh'] = info['soh']

    output_file = logdir + 'test_{}.json'.format(test_profile)

    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(comparison_dict, f, allow_nan=False) 


def deterministic_action_policy(env, action:float, algo_name: str, exp_name: str, test_profile):
    assert 0 <= action <= 1, "The deterministic action must be between 0 and 1."

    print("######## {} policy is running... ########".format(algo_name.upper()))
    
    comparison_dict = {
        'test': test_profile,
        'pure_reward': {},
        'norm_reward': {},
        'weighted_reward': {},
        'total_reward': 0
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

    comparison_dict['total_reward'] = info['total_reward']
    comparison_dict['pure_reward'] = info['pure_reward_list']
    comparison_dict['norm_reward'] = info['norm_reward_list']
    comparison_dict['weighted_reward'] = info['weighted_reward_list']
    comparison_dict['actions'] = info['actions']
    comparison_dict['states'] = info['states']
    comparison_dict['traded_energy'] = info['traded_energy']
    comparison_dict['soh'] = info['soh']

    output_file = logdir + 'test_{}.json'.format(test_profile)

    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(comparison_dict, f, allow_nan=False) 
