import os
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, \
    StopTrainingOnMaxEpisodes, EvalCallback

from callback import CustomEvalCallback
from stable_baselines3.ppo import MlpPolicy

# from gym_env_lakecomo_with_uniform_policy import LakeComoEnv
from gym_env_lake import LakeEnv
from stable_baselines3 import PPO, A2C
from utils import parameter_generator

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


if __name__ == '__main__':
    # def linear_schedule(initial_value: float):
    #     """
    #     Linear learning rate schedule.
    #
    #     :param initial_value: Initial learning rate.
    #     :return: schedule that computes
    #       current learning rate depending on remaining progress
    #     """
    #
    #     def func(progress_remaining: float) -> float:
    #         """
    #         Progress will decrease from 1 (beginning) to 0.
    #
    #         :param progress_remaining:
    #         :return: current learning rate
    #         """
    #         return progress_remaining * initial_value
    #
    #     return func

    # filename = '../../data/lakeComo/modified_settings_lakeComoHistory.txt'
    seed = 123
    num_episodes = 500
    verbose = True
    gamma = 0.995
    ent_coef = 0.01
    stats_window_size = 10 #1
    learning_rate = 2e-3
    days_in_year = 365
    log_rate = 1000
    policy_kwargs = {'net_arch': [64, 64, 32], 'log_std_init':-0.5}

    learning_rate_sched = lambda p : p*learning_rate

    params = parameter_generator(world_options='world.yaml',
                                 lake_params='lake.yaml')

    lake_env = LakeEnv(settings=params)

    lake_env = wrap_env(lake_env, normalize_obs=False, rescale_obs=True, normalize_reward=False, rescale_reward=True)


    print('Lake Como Environment Initialized')

    # for i in range(100):
    #     print(lake_env.step(0.))

    # def train_ppo(envs, args, eval_env_params, model_file=None):

    print('######## PPO is running... ########')

    logdir = './logs/' + 'first_exp'
    os.makedirs(logdir, exist_ok=True)
    model_folder = './logs/first_exp/models/'

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='logs/first_exp/models/',
        name_prefix='ppo',
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    callback_max_episodes = StopTrainingOnMaxEpisodes(
        max_episodes=num_episodes, verbose=1)

    params = parameter_generator(world_options='world.yaml',
                                 lake_params='lake.yaml')

    eval_env = LakeEnv(settings=params)

    eval_env = wrap_env(eval_env, normalize_obs=False, rescale_obs=True, normalize_reward=False, rescale_reward=True)

    eval_callback = CustomEvalCallback(eval_env,
                                       best_model_save_path='logs/first_exp/models/eval/',
                                       log_path='logs/',
                                       eval_freq=10000,
                                       n_eval_episodes=1,
                                       deterministic=True,
                                       render=False)
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    callbacks = [
        callback_max_episodes, eval_callback] # stop_train_callback]

    model = PPO(policy=MlpPolicy,
                batch_size=256,
                env=lake_env,
                verbose=verbose,
                gamma=gamma,
                tensorboard_log='./logs/tensorboard/ppo/first_exp',
                ent_coef=ent_coef,
                stats_window_size=stats_window_size,
                learning_rate=learning_rate_sched,
                policy_kwargs=policy_kwargs,
                device='cpu',
                seed=seed
                )

    # Option for loading agent parameters
    # else:
    #     model = PPO.load(path=model_folder + model_file, env=envs)
    #     model.set_env(envs)
    #     print('Loaded model from: {}'.format(model_file))

    model.learn(total_timesteps=days_in_year * num_episodes,
                progress_bar=True,
                log_interval=log_rate,
                tb_log_name='ppo_first_exp',
                callback=callbacks,
                reset_num_timesteps=True
                )

    model.save('./logs/first_exp/models/ppo_lakeComo')

    print('######## TRAINING is Done ########')

    # del model


    # H = 1000
    # for i in range(H):
    #     action = lakeComoEnv.action_space.sample()[0]
    #     print(action)
    #     observation, reward, done, info = lakeComoEnv.step(action)
    #
    #     # add here the code for retraining
    #     PPO(MlpPolicy, lakeComoEnv, verbose=False, gamma=0.9)
    #
    #
    #     if done:
    #         break
    #     print(f'Step: {i}, Action: {action}, Observation: {observation}, Reward: {reward}, Info: {info}')
    #
    # print('Episode is terminated')