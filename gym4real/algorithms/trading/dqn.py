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
from gym4real.algorithms.trading.utils import evaluate_agent_with_baselines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def train_dqn(args, train_env_params, eval_env_params, test_env_params, train=False):
    if train is True:
        for seed in args['seeds']:
            print("######## DQN is running... ########")
            logdir = "./logs/" + args['exp_name']
            os.makedirs(logdir, exist_ok=True)
            train_env = make_vec_env("gym4real/TradingEnv-v0", n_envs=args['n_envs'],
                                     env_kwargs={'settings': train_env_params, 'seed':seed})
            eval_env = gym.make("gym4real/TradingEnv-v0",
                                **{'settings': eval_env_params, 'scaler': train_env.env_method('get_scaler')[0], 'seed': seed})
            eval_env = Monitor(eval_env)

            # callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
            """
            EvalCallbackTotalReward(eval_env, train_eval_env,  best_model_save_path=f"./logs_ppo2/seed_{seed}",
                                                         log_path=f"./logs_ppo2/seed_{seed}", eval_freq= 1 * env.get_trading_day_num() * 118 / 2,
                                                         deterministic=True, render=False, n_eval_episodes=eval_env.env.get_trading_day_num(), n_eval_episodes_train=train_eval_env.get_trading_day_num() )
            """
            eval_callback = EvalCallback(eval_env,
                                         best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']+f"_seed_{seed}"),
                                         log_path="./logs/",
                                         eval_freq=(1 * train_env.env_method("get_trading_day_num")[0] * 118) / 2,
                                         n_eval_episodes=eval_env.unwrapped.get_trading_day_num(),
                                         deterministic=True,
                                         render=False)

            callbacks = [eval_callback]
            model = DQN("MlpPolicy",
                        env=train_env,
                        verbose=args['verbose'],
                        gamma=args['gamma'],
                        policy_kwargs=args['policy_kwargs'],
                        tensorboard_log="./logs/tensorboard/trading/dqn/{}".format(args['exp_name']+f"_seed_{seed}"),
                        stats_window_size=100,
                        learning_rate=args['learning_rate'],
                        batch_size=args['batch_size'],
                        buffer_size=args['buffer_size'],
                        learning_starts=args['learning_starts'],
                        exploration_fraction=args['exploration_fraction'],
                        exploration_final_eps=args['exploration_final_eps'],
                        tau=args['tau'],
                        train_freq=args['train_freq'],
                        seed=seed
                        )

            model.learn(total_timesteps=args['n_envs'] * args['n_episodes'] * train_env.env_method("get_trading_day_num")[0] * 598,
                        progress_bar=True,
                        log_interval=args['log_rate'],
                        tb_log_name="dqn_{}".format(args['exp_name']),
                        callback=callbacks,
                        reset_num_timesteps=True, )

            model.save("./logs/{}/models/{}".format(args['exp_name']+f"_seed_{seed}", args['save_model_as']))
            print("######## TRAINING is Done ########")
    else:
        train_env = make_vec_env("gym4real/TradingEnv-v0", n_envs=args['n_envs'],
                                 env_kwargs={'settings': train_env_params})

    train_env_params['sequential'] = True
    print("PLOTTING")
    models = []
    for seed in args['seeds']:
        model_folder = "./logs/{}/models/".format(args['exp_name'] + f"_seed_{seed}")
        model = DQN.load(os.path.join(model_folder, "eval", "best_model"))
        models.append(model)
    plot_folder = "./logs/{}/plots/".format(args['exp_name'])
    os.makedirs(plot_folder, exist_ok=True)
    evaluate_agent_with_baselines(models, train_env_params, plot_folder, None, 'Train', args['seeds'], 'DQN')
    evaluate_agent_with_baselines(models, eval_env_params, plot_folder, train_env.env_method("get_scaler")[0], 'Validation', args['seeds'], 'DQN')
    evaluate_agent_with_baselines(models, test_env_params, plot_folder, train_env.env_method("get_scaler")[0], 'Test', args['seeds'], 'DQN')


if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'trading/dqn',
        'n_episodes': 10,
        'n_envs': 6,
        'policy_kwargs': dict(
            net_arch=[512, 512]
        ),
        'verbose': False,
        'gamma': 0.90,
        'learning_rate': 0.0001,
        'log_rate': 10,
        'batch_size': 64,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.05,
        'tau': 1.0,
        'train_freq': 4,
        'save_model_as': 'dqn_trading_10eps',
        'seeds': [32517, 84029, 10473, 67288, 91352, 47605]
    }

    # Example evaluation environment parameters
    train_env_params = parameter_generator(world_options='../../envs/trading/world_train.yaml')
    eval_env_params = parameter_generator(world_options='../../envs/trading/world_validation.yaml')
    test_env_params = parameter_generator(world_options='../../envs/trading/world_test.yaml')


    train_dqn(train_env_params=train_env_params, eval_env_params=eval_env_params,test_env_params=test_env_params, args=args)