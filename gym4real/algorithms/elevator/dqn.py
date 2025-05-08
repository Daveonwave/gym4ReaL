import sys
import os

sys.path.append(os.getcwd())

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from gym4real.envs.elevator.env import ElevatorEnv
from gym4real.envs.elevator.utils import parameter_generator


def train_dqn(envs, args, eval_env_params, model_file=None):
    print("######## DQN is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    
    eval_env = ElevatorEnv(settings=eval_env_params)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/", 
                                 eval_freq=100000,
                                 n_eval_episodes=30,
                                 deterministic=True, 
                                 render=False)
    
    callbacks = [callback_max_episodes, eval_callback]
    
    if model_file is not None:
        model = DQN.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = DQN("MultiInputPolicy", 
                    env=envs, 
                    verbose=args['verbose'], 
                    gamma=args['gamma'], 
                    tensorboard_log="./logs/tensorboard/elevator/dqn/{}".format(args['exp_name']),
                    stats_window_size=1,
                    learning_rate=args['learning_rate']
                    )
        
    model.learn(total_timesteps=3600 * args['n_envs'] * args['n_episodes'],
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
        'exp_name': 'elevator/dqn',
        'n_episodes': 1000,
        'n_envs': 5,
        'verbose': False,
        'gamma': 0.99,
        'learning_rate': 0.0001,
        'log_rate': 10,
        'save_model_as': 'dqn_elevator_1000eps',
    }
    
    # Example evaluation environment parameters
    eval_env_params = parameter_generator(world_options='gym4real/envs/elevator/world.yaml', seed=1234)
    
    params = parameter_generator(world_options='gym4real/envs/elevator/world.yaml')
    
    envs = make_vec_env("gym4real/elevator-v0", n_envs=args['n_envs'], env_kwargs={'settings':params})    
    
    train_dqn(envs=envs, args=args, eval_env_params=params)