import sys
import os

sys.path.append(os.getcwd())

import gymnasium as gym
from collections import defaultdict
from tqdm.rich import tqdm
import numpy as np
import random
import pickle
import json
from gym4real.envs.elevator.utils import parameter_generator


def default_q_values():
        return np.zeros(3)
    
def obs_to_key(obs):
        # Manually flatten the dict to a tuple of ints
        return (
            obs['current_position'],
            obs['n_passengers'],
            #obs['speed'],
            *obs['floor_queues'],
            *obs['arrivals']
        )

def train_sarsa(env, args, eval_env_params, model_file=None):
    # Hyperparameters
    alpha = args['alpha']
    gamma = args['gamma']
    epsilon = args['epsilon']
    epsilon_decay = args['epsilon_decay']
    epsilon_min = args['epsilon_min']
    n_episodes = args['n_episodes']
    exp_name = args['exp_name']
    tol = args['tol']
    max_no_improvement = args['max_no_improvement']
    
    logdir = "./logs/" + exp_name + "/models/"
    os.makedirs(logdir, exist_ok=True)
    
    best_eval_reward = -np.inf
    best_eval_episode = 0
    learning_curve = {}
    
    actions_size = env.action_space.n
    Q = defaultdict(default_q_values)
    #Q = defaultdict(lambda: np.zeros(actions_size))
        
    for episode in tqdm(range(n_episodes)):
        if episode % 1000 == 0 and episode > 0:
            # Evaluate the Q-table every 1000 episodes
            eval_reward, cum_rewards = evaluate_sarsa(Q, eval_env_params, eval_episodes=30)
            learning_curve[episode] = cum_rewards
            
            if eval_reward > best_eval_reward + tol:
                best_eval_reward = eval_reward
                best_eval_episode = episode
                no_improvement = 0
                print(f"New best evaluation reward: {best_eval_reward} at episode {best_eval_episode}")
                with open(f"./logs/{exp_name}/models/best_q_table.pkl", "wb") as output_file:
                    pickle.dump(Q, output_file)
            else:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    with open(f"./logs/{exp_name}/models/learning_curve.json", "w", encoding="utf8") as output_file:
                        json.dump(learning_curve, output_file)
                    print(f"No improvement for 10 evaluations, stopping training at episode {episode}")
                    exit(0)
        
        obs, _ = env.reset()
        obs_key = obs_to_key(obs)
        
        cumulated_reward = 0        
        
        rng = np.random.default_rng(seed=42)
        done = False
        while not done:
            if rng.uniform() < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(Q[obs_key])     # Exploit learned values
            
            if action != 0 and action != 1 and action != 2:
                print(f"Invalid action {action}")
            
            new_obs, reward, terminated, truncated, _ = env.step(action) 
            new_obs_key = obs_to_key(new_obs)
                        
            cumulated_reward += reward
            done = terminated or truncated
            
            # Q-update
            next_action = env.action_space.sample() if rng.uniform() < epsilon else np.argmax(Q[new_obs_key])
            td_target = reward + gamma * Q[new_obs_key][next_action]
            Q[obs_key][action] += alpha * (td_target - Q[obs_key][action])

            obs, obs_key = new_obs, new_obs_key
            
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    with open(f"./logs/{exp_name}/models/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)
    print("Training complete!")


def evaluate_sarsa(Q, eval_env_params, eval_episodes=10):
    env = gym.make("gym4real/elevator-v0", **{'settings':eval_env_params})
    avg_cumulated_reward = 0
    ep_cum_rewards = []
    
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        obs_key = obs_to_key(obs)
        
        done = False
        cumulated_reward = 0
        
        while not done:
            action = np.argmax(Q[obs_key])  # Exploit learned values
            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_obs_key = obs_to_key(new_obs)
            
            cumulated_reward += reward
            done = terminated or truncated
            obs_key = new_obs_key
    
        ep_cum_rewards.append(cumulated_reward)
         
    avg_cumulated_reward = np.mean(ep_cum_rewards)     
    print(f"Evaluation finished with avg. cum. reward: {avg_cumulated_reward}")
    return avg_cumulated_reward, ep_cum_rewards

            
if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'elevator_no_speed/sarsa',
        'alpha': 0.1,
        'gamma': 1,
        'epsilon': 1.0,
        'epsilon_decay': 0.99,
        'epsilon_min': 0.05,
        'n_episodes': 100000,
        'tol': 0.1,
        'max_no_improvement': 10,
    }
    
    eval_env_params = parameter_generator(world_options='gym4real/envs/elevator/world.yaml', seed=1234)
    params = parameter_generator(world_options="gym4real/envs/elevator/world.yaml")
    
    env = gym.make("gym4real/elevator-v0", **{'settings':params})    
    train_sarsa(env=env, args=args, eval_env_params=eval_env_params)