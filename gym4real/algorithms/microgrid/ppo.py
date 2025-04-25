import sys
import os

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from gym4real.envs.microgrid.env import MicroGridEnv
from gym4real.envs.microgrid.utils import parameter_generator


def train_ppo(envs, args, eval_env_params, model_file=None):
    print("######## PPO is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    
    eval_env = MicroGridEnv(settings=eval_env_params)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/", 
                                 eval_freq=8760*5,
                                 n_eval_episodes=3,
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
                    gamma=args['gamma'], 
                    tensorboard_log="./logs/tensorboard/microgrid/ppo/".format(args['exp_name']),
                    stats_window_size=1,
                    learning_rate=args['learning_rate']
                    )
        
    model.learn(total_timesteps=len(envs.get_attr("generation")[0]) * args['n_envs'] * args['n_episodes'],
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
        'exp_name': 'microgrid_100_episodes',
        'n_episodes': 100,
        'n_envs': 5,
        'verbose': 1,
        'gamma': 0.99,
        'learning_rate': 0.001,
        'log_rate': 10,
        'save_model_as': 'dqn_ppo_100_episodes',
    }
    
    params = parameter_generator(world_options="gym4real/envs/microgrid/world_train.yaml",
                                 seed=42,
                                 min_soh=0.6)
    
    eval_params = parameter_generator(world_options="gym4real/envs/microgrid/world_test.yaml",
                                      seed=42,
                                      min_soh=0.6)
    
    envs = make_vec_env("gym4real/microgrid-v0", n_envs=args['n_envs'], env_kwargs={'settings':params})    
    
    train_ppo(envs=envs, args=args, eval_env_params=params)