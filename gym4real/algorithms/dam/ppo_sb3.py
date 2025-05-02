import sys
import os

sys.path.append(os.getcwd())

print(os.getcwd())

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gym4real.envs.dam.gym_env_lake import LakeEnv
from gym4real.envs.dam.utils import parameter_generator

from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction, TransformReward, TransformObservation

def wrap_env(env, normalize_obs=True, normalize_reward=True, rescale_action=True, rescale_reward=False, rescale_obs=False):
    if normalize_obs:
        env = NormalizeObservation(env)
    if rescale_obs:
        env = TransformObservation(env, lambda o : o / np.array([1., 1., 1., 3000.][:env.observation_space.shape[0]]), None)
    if normalize_reward:
        env = NormalizeReward(env)
    if rescale_reward:
        env = TransformReward(env, lambda r: r / 200000.)
    if rescale_action:
        env = RescaleAction(env, min_action=-1., max_action=1.)

    return env


def train_ppo(envs, args, eval_env_params, model_file=None):
    print("######## PPO is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    
    eval_env = LakeEnv(settings=eval_env_params)
    eval_env = Monitor(eval_env)
    eval_env = wrap_env(eval_env)#, normalize_obs=False, rescale_obs=True, normalize_reward=False, rescale_reward=True)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/",
                                 eval_freq=365*5,
                                 n_eval_episodes=5,
                                 deterministic=True,
                                 render=False)
    
    callbacks = [callback_max_episodes, eval_callback]
    
    if model_file is not None:
        model = PPO.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = PPO(MlpPolicy, 
                    env=envs, 
                    verbose=args['verbose'],
                    batch_size=args['batch_size'],
                    gamma=args['gamma'],
                    ent_coef=args['ent_coef'],
                    tensorboard_log="./logs/tensorboard/dam/ppo/".format(args['exp_name']),
                    stats_window_size=1,
                    learning_rate=args['learning_rate'],
                    seed=args['seed'],
                    policy_kwargs=args['policy_kwargs'],
                    device='cpu'
                    )
        
    model.learn(total_timesteps=365 * args['n_envs'] * args['n_episodes'],
                progress_bar=True,
                log_interval=args['log_rate'],
                tb_log_name="ppo_{}".format(args['exp_name']),
                callback=callbacks,
                reset_num_timesteps=True,
                )
    
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
    print("######## TRAINING is Done ########")


if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'dam_100_episodes',
        'n_episodes': 100,
        'n_envs': 5,
        'batch_size': 256,
        'verbose': 1,
        'gamma': 0.995,
        'ent_coef': 0.01,
        'learning_rate': 0.0001,
        'log_rate': 10,
        'seed': 123,
        'save_model_as': 'dqn_ppo_100_episodes',
        'policy_kwargs': {'net_arch': [64, 64, 32], 'log_std_init':-0.5}
    }
    
    params = parameter_generator(world_options='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/world_train.yaml',
                                 lake_params='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/lake.yaml')
    
    eval_params = parameter_generator(world_options='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/world_test.yaml',
                                      lake_params='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/lake.yaml')
    
    # envs = make_vec_env("gym4real/dam-v0", wrapper_class=wrap_env,
    #                     n_envs=args['n_envs'], env_kwargs={'settings':params}, wrapper_kwargs={'normalize_obs':False, 'rescale_obs':True, 'normalize_reward':False, 'rescale_reward':True})


    envs = make_vec_env("gym4real/dam-v0", wrapper_class=wrap_env,
                        n_envs=args['n_envs'], env_kwargs={'settings': params},
                        # wrapper_kwargs={'normalize_obs': False, 'rescale_obs': True, 'normalize_reward': False,
                        #                 'rescale_reward': True}
                        )

    train_ppo(envs=envs, args=args, eval_env_params=params)