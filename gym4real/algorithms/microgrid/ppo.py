import sys
import os

sys.path.append(os.getcwd())

import itertools
import numpy as np
import cProfile, pstats, functools
from gymnasium import Wrapper
from stable_baselines3 import PPO
#from sbx import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes, EvalCallback
from gym4real.envs.microgrid.env import MicroGridEnv
from gym4real.envs.microgrid.utils import parameter_generator
from warnings import filterwarnings
filterwarnings(action='ignore')
import math


def cosine_schedule(initial_lr, total_timesteps):
    """
    Returns a cosine annealing learning rate schedule function.
    The LR will start at `initial_lr` and decay to 0 using cosine annealing.
    """
    def schedule(progress_remaining):
        # SB3 passes progress_remaining from 1.0 (start) to 0.0 (end)
        current_step = (1.0 - progress_remaining) * total_timesteps
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / total_timesteps))
        return lr

    return schedule



class ProfileInjectionEvalEnv(Wrapper):
    def __init__(self, env, demand_profiles, mode="cycle"):
        super().__init__(env)
        self.demand_profiles = demand_profiles
        self.mode = mode
        if mode == "cycle":
            self.profile_iterator = itertools.cycle(demand_profiles)

    def reset(self, **kwargs):
        # Select a profile
        if self.mode == "cycle":
            profile = next(self.profile_iterator)
        elif self.mode == "random":
            profile = np.random.choice(self.demand_profiles)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Inject it via the options dict
        options = kwargs.pop("options", {})
        options["eval_profile"] = profile
        return self.env.reset(options=options, **kwargs)


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        info = self.locals["infos"][0]  # SB3 returns list of infos
        self.logger.record("custom/reward_trading", info["pure_rewards"]["r_trad"])
        self.logger.record("custom/reward_degradation", info["pure_rewards"]["r_deg"])
        self.logger.record("custom/reward_clipping", info["pure_rewards"]["r_clip"])
        return True


def train_ppo(envs, args, eval_env_params, model_file=None):
    print("######## PPO is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    callback_reward = RewardLoggerCallback()
    

    # Wrap the raw eval env in DummyVecEnv, then VecNormalize
    def make_eval_env():
        base_env = lambda: ProfileInjectionEvalEnv(
            env=Monitor(MicroGridEnv(settings=eval_env_params)),
            demand_profiles=[str(i) for i in range(370, 380)],
            mode="cycle"
        )
        return VecNormalize(DummyVecEnv([base_env]), training=False, norm_obs=True, norm_reward=True)
    
    eval_env = make_eval_env()
    eval_env.obs_rms = envs.obs_rms  # Sync normalization stats
    eval_env.ret_rms = envs.ret_rms  # (Optional) sync returns normalization
    eval_env.training = False        # Ensure no stats update during eval
    eval_env.norm_reward = False     # Often preferred during evaluation
    
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/{}/".format(args['exp_name']), 
                                 eval_freq=8760*4,
                                 n_eval_episodes=10,
                                 deterministic=True, 
                                 render=False)
    
    callbacks = [callback_max_episodes, eval_callback, callback_reward]
    
    if model_file is not None:
        model = PPO.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = PPO("MlpPolicy", 
                    env=envs, 
                    gamma=args['gamma'], 
                    policy_kwargs=args['policy_kwargs'],
                    batch_size=args['batch_size'],
                    n_steps=args['n_steps'],
                    n_epochs=args['n_epochs'],
                    gae_lambda=args['gae_lambda'],
                    clip_range=args['clip_range'],
                    ent_coef=args['ent_coef'],
                    vf_coef=args['vf_coef'],
                    max_grad_norm=args['max_grad_norm'],
                    tensorboard_log="./logs/tensorboard/microgrid/ppo/".format(args['exp_name']),
                    #stats_window_size=1,
                    learning_rate=args['learning_rate'],
                    verbose=args['verbose'], 
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
    
    # Profiler setup
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    # Example parameters
    args = {
        #'exp_name': '100eps_envs8_NN64-32_gamma099_bs256_nsteps4096_epochs10_gae095_clip02_vf05_maxgrad05_lr-cos5e-5',
        'exp_name': 'prova',
        'n_episodes': 100,
        'n_envs': 8,
        'policy_kwargs': dict(net_arch=[64, 32], log_std_init=-1),
        'gamma': 0.99,
        'batch_size': 256,
        'n_steps': 4096,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.00,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'learning_rate': cosine_schedule(5e-5, 100 * 8 * 8760 * 4),
        'log_rate': 1,
        'verbose': 0,
        'save_model_as': 'ppo_non-modified-env_no-norm_100eps_envs8_NN128-64-32_gamma099_bs256_nsteps4096_epochs10_gae095_clip02_vf05_maxgrad05_lr-cos5e-5',
    }
    
    params = parameter_generator(world_options="gym4real/envs/microgrid/world_train.yaml",
                                 seed=42,
                                 min_soh=0.6,
                                 use_reward_normalization=True)
    
    eval_params = parameter_generator(world_options="gym4real/envs/microgrid/world_test.yaml",
                                      seed=1234,
                                      min_soh=0.6,
                                      use_reward_normalization=False)
    
    envs = make_vec_env("gym4real/microgrid-v0", n_envs=args['n_envs'], env_kwargs={'settings':params})    
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    
    train_ppo(envs=envs, args=args, eval_env_params=eval_params)
    
    # Profiler teardown
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.dump_stats('microgrid.prof')