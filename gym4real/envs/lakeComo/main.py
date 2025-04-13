import os

from stable_baselines3.common.callbacks import CheckpointCallback, \
    StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.ppo import MlpPolicy

# from gym_env_lakecomo_with_uniform_policy import LakeComoEnv
from gym_env_lakecomo import LakeComoEnv
from stable_baselines3 import PPO
from utils import parameter_generator

if __name__ == '__main__':

    # filename = "../../data/lakeComo/modified_settings_lakeComoHistory.txt"
    num_episodes = 500
    verbose = True
    gamma = 0.99
    ent_coef = 0.01
    stats_window_size = 1
    learning_rate = 0.01
    days_in_year = 365
    log_rate = 1000

    params = parameter_generator(world_options='world.yaml',
                                 lake_params='lake.yaml')

    lakeComoEnv = LakeComoEnv(settings=params)

    print("Lake Como Environment Initialized")


    # def train_ppo(envs, args, eval_env_params, model_file=None):

    print("######## PPO is running... ########")

    logdir = "./logs/" + 'first_exp'
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/first_exp/models/"

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./logs/first_exp/models/",
        name_prefix="ppo",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    callback_max_episodes = StopTrainingOnMaxEpisodes(
        max_episodes=num_episodes, verbose=1)

    params = parameter_generator(world_options='world.yaml',
                                 lake_params='lake.yaml')

    eval_env = LakeComoEnv(settings=params)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./logs/first_exp/models/eval/",
                                 log_path="./logs/",
                                 eval_freq=10000,
                                 n_eval_episodes=3,
                                 deterministic=True,
                                 render=False)
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    callbacks = [
        callback_max_episodes]  # eval_callback] # stop_train_callback]

    model = PPO(policy=MlpPolicy,
                env=lakeComoEnv,
                verbose=verbose,
                gamma=gamma,
                tensorboard_log="./logs/tensorboard/ppo/first_exp",
                ent_coef=ent_coef,
                stats_window_size=stats_window_size,
                learning_rate=learning_rate
            )

    # Option for loading agent parameters
    # else:
    #     model = PPO.load(path=model_folder + model_file, env=envs)
    #     model.set_env(envs)
    #     print('Loaded model from: {}'.format(model_file))

    model.learn(total_timesteps=days_in_year * num_episodes,
                progress_bar=True,
                log_interval=log_rate,
                tb_log_name="ppo_first_exp",
                callback=callbacks,
                reset_num_timesteps=True
                )

    model.save("./logs/first_exp/models/ppo_lakeComo")

    print("######## TRAINING is Done ########")

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
    #     print(f"Step: {i}, Action: {action}, Observation: {observation}, Reward: {reward}, Info: {info}")
    #
    # print("Episode is terminated")