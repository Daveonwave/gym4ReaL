import sys
import os

sys.path.append(os.getcwd())

from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
import pandas as pd
from gym4real.envs.trading.utils import parameter_generator
from gym4real.algorithms.trading.utils import evaluate_agent_with_baselines, EvalCallbackSharpRatio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def train_ppo(args, train_env_params, eval_env_params, test_env_params, train = False):
    base_directory = args['log_dir']
    if train is True:
        for seed in args['seeds']:
            print("######## PPO is running... ########")
            train_env = make_vec_env("gym4real/TradingEnv-v0", n_envs=args['n_envs'],
                                     env_kwargs={'settings': train_env_params, 'seed': seed})
            eval_env = gym.make("gym4real/TradingEnv-v0",
                                **{'settings': eval_env_params, 'scaler': train_env.env_method('get_scaler')[0]})
            eval_env = Monitor(eval_env)

            base_logdir = os.path.join(base_directory, args['exp_name'])
            logdir = os.path.join(base_directory, args['exp_name'] + f"_seed_{seed}")
            tensordir = os.path.join(base_directory, "tensorboard", args['exp_name'])
            os.makedirs(logdir, exist_ok=True)
            os.makedirs(tensordir, exist_ok=True)
            os.makedirs(base_logdir, exist_ok=True)

            eval_callback = EvalCallback(eval_env,
                                         best_model_save_path=os.path.join(logdir, "models/eval"),
                                         log_path=None,
                                         eval_freq=(1 * train_env.env_method("get_trading_day_num")[0] * 118) / 2,
                                         n_eval_episodes=eval_env.unwrapped.get_trading_day_num(),
                                         deterministic=True,
                                         render=False)

            callbacks = [eval_callback]

            model = PPO("MlpPolicy",
                        env=train_env,
                        verbose=args['verbose'],
                        gamma=args['gamma'],
                        policy_kwargs=args['policy_kwargs'],
                        n_steps=args['n_steps'],
                        tensorboard_log=tensordir,
                        learning_rate=args['learning_rate'],
                        batch_size=args['batch_size'],
                        seed=seed
                        )

            model.learn(total_timesteps=args['n_episodes'] * train_env.env_method("get_trading_day_num")[0] * 598,
                        progress_bar=True,
                        log_interval=args['log_rate'],
                        tb_log_name="ppo_{}".format(args['exp_name'] + f"_seed_{seed}"),
                        callback=callbacks,
                        reset_num_timesteps=True, )

            # model.save("./logs/{}/models/{}".format(args['exp_name']+f"_seed_{seed}", args['save_model_as']))
            model.save(os.path.join(logdir, "models", args['save_model_as']))
        print("######## TRAINING is Done ########")
    else:
        train_env = make_vec_env("gym4real/TradingEnv-v0", n_envs=args['n_envs'],
                                 env_kwargs={'settings': train_env_params})

    train_env_params['sequential'] = True

    models = []
    for seed in args['seeds']:
        logdir = os.path.join(base_directory, args['exp_name'] + f"_seed_{seed}")
        model = PPO.load(os.path.join(logdir, "models/eval", "best_model"))
        models.append(model)

    plot_folder = os.path.join(base_directory, "{}/plots/".format(args['exp_name']))
    os.makedirs(plot_folder, exist_ok=True)
    evaluate_agent_with_baselines(models, train_env_params, plot_folder, None, 'Training', args['seeds'], 'PPO')
    evaluate_agent_with_baselines(models, eval_env_params, plot_folder, train_env.env_method("get_scaler")[0], 'Validation', args['seeds'], 'PPO')
    evaluate_agent_with_baselines(models, test_env_params, plot_folder, train_env.env_method("get_scaler")[0], 'Test', args['seeds'], 'PPO')





if __name__ == '__main__':
    # Example parameters

    args = {
        'exp_name': 'trading/ppo',
        'log_dir': "gym4real/algorithms/trading/logs",
        'n_episodes': 30,
        'n_envs': 6,
        'policy_kwargs': dict(net_arch=[512, 512]),
        'verbose': False,
        'gamma': 0.90,
        'learning_rate': 0.0001,
        'log_rate': 10,
        'batch_size': 236,
        'n_steps':  118*6,
        'ent_coeff': 0.,
        'save_model_as': 'ppo_10_eps',
        'seeds': [32517, 84029, 10473, 67288, 91352, 47605]
    }
    train = True
    # Example evaluation environment parameters
    train_env_params = parameter_generator(world_options='gym4real/envs/trading/world_train.yaml')
    eval_env_params = parameter_generator(world_options='gym4real/envs/trading/world_validation.yaml')
    test_env_params = parameter_generator(world_options='gym4real/envs/trading/world_test.yaml')


    train_ppo(train_env_params=train_env_params, eval_env_params=eval_env_params, test_env_params = test_env_params, args=args, train=train)