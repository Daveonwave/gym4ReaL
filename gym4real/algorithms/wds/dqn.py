import sys
import os

sys.path.append(os.getcwd())

import json
from tqdm import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from gym4real.envs.wds.env import WaterDistributionSystemEnv
from gym4real.envs.wds.utils import parameter_generator
from warnings import filterwarnings
filterwarnings(action='ignore')
    

def train_dqn(envs, args, eval_env_params, model_file=None):
    print("######## DQN is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    
    eval_env = WaterDistributionSystemEnv(settings=eval_env_params)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/{}/".format(args['exp_name']), 
                                 eval_freq=24 * 7 * 2,
                                 n_eval_episodes=20,
                                 deterministic=True, 
                                 render=False)
    
    callbacks = [callback_max_episodes]#, eval_callback]
    
    if model_file is not None:
        model = DQN.load(path=model_file, 
                         env=envs, 
                         gamma=args['gamma'], 
                         tensorboard_log="./logs/tensorboard/wds/dqn/{}".format(args['exp_name']),
                         stats_window_size=1,
                         learning_rate=args['learning_rate'])
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
    
    res = {}
    for i in range(args['n_episodes']):
        model.learn(total_timesteps=168 * args['n_envs'],
                    progress_bar=True,
                    log_interval=args['log_rate'],
                    tb_log_name="dqn_{}".format(args['exp_name']),
                    callback=callbacks,
                    reset_num_timesteps=True,
                    )
    
        eval_env = make_vec_env("gym4real/wds-v0", n_envs=1, env_kwargs={'settings':eval_env_params})
        res[(i+1) * 168 * args['n_envs']] = []
        
        # Evaluate the model
        for _ in tqdm(range(10)):
            eval_env.set_options({'is_evaluation': True})
            obs = eval_env.reset()
        
            cumulated_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs)
                obs, r, dones, _ = eval_env.step(action)
                done = dones[0]
                cumulated_reward += r[0]
        
            res[(i+1) * 168 * args['n_envs']].append(cumulated_reward)
    
    # Save results in JSON format
    with open(os.path.join(logdir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=4)
    
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
    print("######## TRAINING is Done ########")


if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'wds_1h_step_curves',
        'n_episodes': 100,
        'n_envs': 8,
        'verbose': 0,
        'gamma': 0.99,
        'learning_rate': 0.001,
        'log_rate': 100,
        'save_model_as': 'dqn',
    }
    
    # Example evaluation environment parameters
    eval_env_params = {
        'hydraulic_step': 3600,
        'duration': 24 * 3600 * 7,
        'seed': 1234,
    }
    
    params = parameter_generator(world_options='gym4real/envs/wds/world_anytown.yaml',
                                 hydraulic_step=3600,
                                 duration=24 * 3600 * 7,
                                 seed=42,
                                 reward_coeff={'dsr_coeff': 1.0, 'overflow_coeff': 1.0})
    
    envs = make_vec_env("gym4real/wds-v0", n_envs=args['n_envs'], env_kwargs={'settings':params})    
    
    #model_file = "logs/wds_total_basedemand/models/eval/best_model.zip"
    model_file = None
    train_dqn(envs=envs, args=args, eval_env_params=params, model_file=model_file)