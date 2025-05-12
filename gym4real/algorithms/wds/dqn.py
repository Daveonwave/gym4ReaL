import sys
import os

sys.path.append(os.getcwd())

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from gym4real.envs.wds.env import WaterDistributionSystemEnv
from gym4real.envs.wds.utils import parameter_generator


def train_dqn(envs, args, eval_env_params, model_file=None):
    print("######## DQN is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    
    eval_env = WaterDistributionSystemEnv(settings=eval_env_params)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/", 
                                 eval_freq=24*7*3600/300 * args['n_envs'],
                                 n_eval_episodes=5,
                                 deterministic=True, 
                                 render=False)
    
    callbacks = [callback_max_episodes, eval_callback]
    
    if model_file is not None:
        model = DQN.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = DQN("MlpPolicy", 
                    env=envs, 
                    verbose=args['verbose'], 
                    gamma=args['gamma'], 
                    tensorboard_log="./logs/tensorboard/wds/dqn/{}".format(args['exp_name']),
                    stats_window_size=1,
                    learning_rate=args['learning_rate']
                    )
        
    model.learn(total_timesteps=1200 * args['n_envs'] * args['n_episodes'],
                progress_bar=True,
                log_interval=args['log_rate'],
                tb_log_name="dqn_{}".format(args['exp_name']),
                callback=callbacks,
                reset_num_timesteps=True,
                )
    
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
    print("######## TRAINING is Done ########")


if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'wds_total_basedemand',
        'n_episodes': 20,
        'n_envs': 5,
        'verbose': 1,
        'gamma': 0.99,
        'learning_rate': 0.001,
        'log_rate': 10,
        'save_model_as': 'dqn',
    }
    
    # Example evaluation environment parameters
    eval_env_params = {
        'hydraulic_step': 600,
        'duration': 24 * 3600 * 7,
    }
    
    params = parameter_generator(world_options='gym4real/envs/wds/world_anytown.yaml',
                                 hydraulic_step=600,
                                 duration=24 * 3600 * 7,
                                 seed=42,
                                 reward_coeff={'dsr_coeff': 1.0, 'overflow_coeff': 1.0})
    
    envs = make_vec_env("gym4real/wds-v0", n_envs=args['n_envs'], env_kwargs={'settings':params})    
    
    train_dqn(envs=envs, args=args, eval_env_params=params)