import os
import sys

sys.path.append(os.getcwd())

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from gymnasium.wrappers import RecordEpisodeStatistics

from gym4real.envs.dam.env import DamEnv
from gym4real.envs.dam.utils import parameter_generator

from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction, TransformReward, TransformObservation

def my_wrap_env(env, normalize_obs=False, normalize_reward=False, rescale_action=True, rescale_reward=False, rescale_obs=False):
    env = RecordEpisodeStatistics(env)
    if normalize_obs:
        env = NormalizeObservation(env)
    if rescale_obs:
        env = TransformObservation(env, lambda o : o / np.array([1., 1., 1., 3000.][:env.observation_space.shape[0]]), None)
    if normalize_reward:
        env = NormalizeReward(env)
    if rescale_reward:
        env = TransformReward(env, lambda r: r / 300.)
    if rescale_action:
        env = RescaleAction(env, min_action=-1., max_action=1.)

    return env


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None, net_arch=(64, 64), clip_actions=False,
                 clip_log_std=True, initial_log_std=0., min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        layers = []
        input_dim = self.num_observations

        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim, device=device))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.num_actions, device=device))

        self.net = nn.Sequential(*layers)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device) + initial_log_std)

    def compute(self, inputs, role=''):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None, net_arch=(64, 64), clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        layers = []
        input_dim = self.num_observations

        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim, device=device))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1, device=device))

        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def train_ppo(params, args, eval_env_params, device='cuda'):

    set_seed(args.get('seed', 123))

    env = gym.make('gym4real/dam-v0', settings=params)
    env = my_wrap_env(env, normalize_reward=False, rescale_reward=False)
    env = wrap_env(env)

    eval_env = DamEnv(eval_env_params)
    eval_env = my_wrap_env(eval_env, normalize_reward=False, rescale_reward=False)
    eval_env = wrap_env(eval_env)

    memory = RandomMemory(memory_size=args['rollouts'], num_envs=env.num_envs, device=device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, net_arch=args['net_arch'], initial_log_std=args['initial_log_std'])#, clip_actions=True)
    models["value"] = Value(env.observation_space, env.action_space, device, net_arch=args['net_arch'])

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = args['rollouts']
    cfg["learning_epochs"] = args.get('learning_epochs', 10)
    cfg["mini_batches"] = args['mini_batches']
    cfg["discount_factor"] = args['discount_factor']
    cfg["lambda"] = args.get('lambda', 0.95)
    cfg["learning_rate"] = args['learning_rate']
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = args.get('grad_norm_clip', 0.5)
    cfg["ratio_clip"] = args.get('ratio_clip', 0.2)
    cfg["value_clip"] = args.get('value_clip', 0.2)
    cfg["clip_predicted_values"] = args.get('clip_predicted_values', False)
    cfg["entropy_loss_scale"] = args['ent_coef']
    cfg["value_loss_scale"] = args.get('value_loss_scale', 0.5)
    cfg["kl_threshold"] = args.get('kl_threshold', 0)
    cfg["mixed_precision"] = True
    if args.get('normalize_observations', True):
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 500
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = args.get('save_dir', "gym4real/algorithms/dam/runs")

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


    cfg_trainer = {"timesteps": args['training_timesteps'], "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()

    obs, _ = eval_env.reset()
    rew_cumul = 0.

    sep_reward_cumul = {'overflow_reward': 0.,
                        'daily_deficit_reward': 0.,
                        'wasted_water_reward': 0.,
                        'clipping_reward': 0.,
                        'starving_reward': 0.}

    agent.set_mode("eval")

    with torch.no_grad():

        for i in range(365*13):
            output = agent.act(obs, i, 0)
            act = output[-1]["mean_actions"]
            obs, tot_reward, terminated, truncated, info = eval_env.step(act)

            rew_cumul += tot_reward
            for key in sep_reward_cumul.keys():
                sep_reward_cumul[key] += info['weighted_reward'][key]
            if terminated or truncated:
                obs, _ = eval_env.reset()

    print(rew_cumul)
    print(sep_reward_cumul)

if __name__ == '__main__':
    # Example parameters
    args = {
        'training_timesteps': 200000,
        'mini_batches': 32,
        'learning_epochs': 10,
        'rollouts': 2048,
        'discount_factor': 0.995,
        'ent_coef': 0.,
        'learning_rate': 8e-6,
        'net_arch': [16, 16],
        'initial_log_std': -0.5,
        'seed': 123,
        'normalize_observations': True
    }

    params = parameter_generator(
        world_options='gym4real/envs/dam/world_train.yaml',
        lake_params='gym4real/envs/dam/lake.yaml')

    eval_params = parameter_generator(
        world_options='gym4real/envs/dam/world_test.yaml',
        lake_params='gym4real/envs/dam/lake.yaml')

    train_ppo(params, args, eval_params)