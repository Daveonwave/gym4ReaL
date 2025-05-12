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
from gym4real.algorithms.trading.utils import evaluate_agent_with_baselines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def train_dqn(envs, args, train_env_params, eval_env_params, eval_env, train = False):
    if train is True:
        for seed in args['seeds']:
            print("######## PPO is running... ########")
            logdir = "./logs/" + args['exp_name']
            os.makedirs(logdir, exist_ok=True)

            #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
            """
            EvalCallbackTotalReward(eval_env, train_eval_env,  best_model_save_path=f"./logs_ppo2/seed_{seed}",
                                                         log_path=f"./logs_ppo2/seed_{seed}", eval_freq= 1 * env.get_trading_day_num() * 118 / 2,
                                                         deterministic=True, render=False, n_eval_episodes=eval_env.env.get_trading_day_num(), n_eval_episodes_train=train_eval_env.get_trading_day_num() )
            """
            eval_callback = EvalCallback(eval_env,
                                         best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']+f"_seed_{seed}"),
                                         log_path="./logs/",
                                         eval_freq= (1 * envs.env_method("get_trading_day_num")[0] * 118) / 2,
                                         n_eval_episodes=eval_env.unwrapped.get_trading_day_num(),
                                         deterministic=True,
                                         render=False)

            callbacks = [eval_callback]

            model = PPO("MlpPolicy",
                        env=envs,
                        verbose=args['verbose'],
                        gamma=args['gamma'],
                        policy_kwargs=args['policy_kwargs'],
                        n_steps=args['n_steps'],
                        tensorboard_log="./logs/tensorboard/trading/ppo/{}".format(args['exp_name']+f"_seed_{seed}"),
                        learning_rate=args['learning_rate'],
                        batch_size = args['batch_size'],
                        seed=seed
                        )

            model.learn(total_timesteps= args['n_episodes'] * envs.env_method("get_trading_day_num")[0] * 598,
                        progress_bar=True,
                        log_interval=args['log_rate'],
                        tb_log_name="ppo_{}".format(args['exp_name']+f"_seed_{seed}"),
                        callback=callbacks,
                        reset_num_timesteps=True,)

            model.save("./logs/{}/models/{}".format(args['exp_name']+f"_seed_{seed}", args['save_model_as']))
        print("######## TRAINING is Done ########")

    train_env_params['sequential'] = True
    print("PLOTTING")
    models = []
    for seed in args['seeds']:
        model_folder = "./logs/{}/models/".format(args['exp_name'] + f"_seed_{seed}")
        model = PPO.load(os.path.join(model_folder, "eval", "best_model"))
        models.append(model)
    plot_folder = "./logs/{}/plots/".format(args['exp_name'])
    os.makedirs(plot_folder, exist_ok=True)
    evaluate_agent_with_baselines(models, train_env_params, plot_folder, None, 'train', args['seeds'])
    evaluate_agent_with_baselines(models, eval_env_params, plot_folder, envs.env_method("get_scaler")[0], 'valid', args['seeds'])





if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'trading/ppo',
        'n_episodes': 10,
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
        'seeds': [1234, 5678, 91011]
    }
    
    # Example evaluation environment parameters
    train_env_params = parameter_generator(world_options='../../envs/trading/world_train.yaml')
    eval_env_params = parameter_generator(world_options='../../envs/trading/world_test.yaml')
    

    train_env = make_vec_env("gym4real/TradingEnv-v0", n_envs=args['n_envs'], env_kwargs={'settings':train_env_params})
    eval_env = gym.make("gym4real/TradingEnv-v0", **{'settings':eval_env_params, 'scaler':train_env.env_method('get_scaler')[0]})
    eval_env = Monitor(eval_env)
    train_dqn(envs= train_env, train_env_params=train_env_params, eval_env_params=eval_env_params, args=args, eval_env = eval_env)