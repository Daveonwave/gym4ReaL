import sys
import os

sys.path.append(os.getcwd())

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
import pandas as pd
from gym4real.envs.trading.utils import parameter_generator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def train_dqn(envs, args, train_env_params, eval_env_params, eval_env, model_file=None):
    print("######## DQN is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    plot_folder = "./logs/{}/plots/".format(args['exp_name'])
    os.makedirs(plot_folder, exist_ok=True)
    
    #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    """
    EvalCallbackTotalReward(eval_env, train_eval_env,  best_model_save_path=f"./logs_ppo2/seed_{seed}",
                                                 log_path=f"./logs_ppo2/seed_{seed}", eval_freq= 1 * env.get_trading_day_num() * 118 / 2,
                                                 deterministic=True, render=False, n_eval_episodes=eval_env.env.get_trading_day_num(), n_eval_episodes_train=train_eval_env.get_trading_day_num() )
    """
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/", 
                                 eval_freq= (1 * envs.env_method("get_trading_day_num")[0] * 598) / 2,
                                 n_eval_episodes=eval_env.unwrapped.get_trading_day_num(),
                                 deterministic=True, 
                                 render=False)
    
    callbacks = [eval_callback]
    
    if model_file is not None:
        model = DQN.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = DQN("MlpPolicy",
                    env=envs, 
                    verbose=args['verbose'], 
                    gamma=args['gamma'],
                    policy_kwargs=args['policy_kwargs'],
                    tensorboard_log="./logs/tensorboard/elevator/dqn/{}".format(args['exp_name']),
                    stats_window_size=1,
                    learning_rate=args['learning_rate'],
                    batch_size = args['batch_size'],
                    buffer_size = args['buffer_size'],
                    learning_starts = args['learning_starts'],
                    exploration_fraction = args['exploration_fraction'],
                    exploration_final_eps = args['exploration_final_eps'],
                    tau = args['tau'],
                    train_freq = args['train_freq'],
                    )
        
    model.learn(total_timesteps= args['n_episodes'] * envs.env_method("get_trading_day_num")[0] * 598,
                progress_bar=True,
                log_interval=args['log_rate'],
                tb_log_name="dqn_{}".format(args['exp_name']),
                callback=callbacks,
                reset_num_timesteps=True,
                )
    
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
    print("######## TRAINING is Done ########")

    train_env_params['sequential'] = True
    model = DQN.load(os.path.join(model_folder,"eval", "best_model"))
    evaluate_agent_with_baselines(model, train_env_params, plot_folder, None, 'train')
    evaluate_agent_with_baselines(model, eval_env_params, plot_folder, envs.env_method("get_scaler")[0] , 'valid')


def evaluate_agent_with_baselines(model, params, plot_folder, scaler, prefix):
    # model = PPO("MlpPolicy", env, verbose=1, batch_size=115, policy_kwargs=dict(net_arch= dict(pi=[256, 256], vf=[256, 256])), gamma=0.99, n_steps=598*5, seed=seed, tensorboard_log=f"./ppo_trading_tensorboard/seed_{seed}")
    env_agent = gym.make("gym4real/TradingEnv-v0",
                         **{'settings': params, 'scaler': scaler})

    rewards_agent_seed = []
    action_episodes = []
    rewards_agent = []
    for _ in range(env_agent.unwrapped.get_trading_day_num()):
        done = False
        action_episode = []
        obs, _ = env_agent.reset()

        while not done:
            action, _ = model.predict(observation=np.array(obs, dtype=np.float32))
            action_episode.append(action)
            next_obs, reward, terminated, truncated, _ = env_agent.step(action)
            rewards_agent.append(reward)
            obs = next_obs
            done = terminated or truncated

        action_episodes.append(action_episode)

    rewards_agent = np.asarray(rewards_agent)

    env_bnh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_bnh = []
    for _ in range(env_bnh.unwrapped.get_trading_day_num()):
        done = False
        env_bnh.reset()
        print(env_bnh.unwrapped._day)
        while not done:
            next_obs, reward, terminated, truncated, _ = env_bnh.step(2)
            rewards_bnh.append(reward)
            done = terminated or truncated

    env_snh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_snh = []
    for _ in range(env_snh.unwrapped.get_trading_day_num()):
        done = False
        env_snh.reset()
        print(env_snh.unwrapped._day)
        while not done:
            next_obs, reward, terminated, truncated, _ = env_snh.step(0)
            rewards_snh.append(reward)
            done = terminated or truncated


    rewards_bnh = np.asarray(rewards_bnh)
    rewards_snh = np.asarray(rewards_snh)

    plt.figure()
    plt.plot((rewards_bnh.cumsum() / env_snh.unwrapped._capital) * 100, label="B&H")
    plt.plot((rewards_snh.cumsum() / env_snh.unwrapped._capital) * 100, label="S&H")
    plt.plot((rewards_agent.cumsum() / env_agent.unwrapped._capital) * 100, label="Agent")
    plt.title(f"Performance on {prefix} Set")
    plt.xlabel("Time")
    plt.ylabel("P&L (%)")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, prefix+"_pnl"))

    action_agents = pd.DataFrame(action_episodes).fillna(0).astype(int)
    plt.figure()
    cmap = sns.color_palette(['red', 'white', 'green'])
    plt.title(f"Action Heatmap")
    sns.heatmap(action_agents, cmap=cmap)
    plt.savefig(os.path.join(plot_folder, prefix+"_action_distribution"))



if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'trading/dqn',
        'n_episodes': 30,
        'n_envs': 6,
        'policy_kwargs': dict(
                        net_arch=[256, 256]
                ),
        'verbose': False,
        'gamma': 0.99,
        'learning_rate': 0.0001,
        'log_rate': 10,
        'batch_size': 598,
        'buffer_size': 5000,
        'learning_starts': 10000,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.0,
        'tau': 0.005,
        'train_freq': 4,
        'save_model_as': 'dqn_trading_10eps',
    }
    
    # Example evaluation environment parameters
    train_env_params = parameter_generator(world_options='../../envs/trading/world_train.yaml', seed=1234)
    eval_env_params = parameter_generator(world_options='../../envs/trading/world_test.yaml', seed=1234)
    

    train_env = make_vec_env("gym4real/TradingEnv-v0", n_envs=args['n_envs'], env_kwargs={'settings':train_env_params})
    eval_env = gym.make("gym4real/TradingEnv-v0", **{'settings':eval_env_params, 'scaler':train_env.env_method('get_scaler')[0]})
    eval_env = Monitor(eval_env)
    train_dqn(envs= train_env, train_env_params=train_env_params, eval_env_params=eval_env_params, args=args, eval_env = eval_env)