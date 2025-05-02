import gymnasium as gym
import numpy as np

import os

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from gymnasium.wrappers import RecordEpisodeStatistics


from gym4real.envs.dam.gym_env_lake import LakeEnv
from gym4real.envs.dam.utils import parameter_generator

from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction, TransformReward, TransformObservation


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

def my_wrap_env(env, normalize_obs=True, normalize_reward=True, rescale_action=True, rescale_reward=False, rescale_obs=False):
    env = RecordEpisodeStatistics(env)
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


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, net_arch=(64, 64), clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Build the network dynamically based on the provided architecture
        layers = []
        input_dim = self.num_observations

        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())  # Use ReLU activation for each layer
            input_dim = hidden_dim  # Update input dimension for the next layer

        # Final output layer for action mean
        layers.append(nn.Linear(input_dim, self.num_actions))

        # Combine all layers into a sequential model
        self.net = nn.Sequential(*layers)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, net_arch=(64, 64), clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Build the network dynamically based on the provided architecture
        layers = []
        input_dim = self.num_observations

        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())  # Use ReLU activation for each layer
            input_dim = hidden_dim  # Update input dimension for the next layer

        # Final output layer for the value estimate
        layers.append(nn.Linear(input_dim, 1))

        # Combine all layers into a sequential model
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

def train_ppo(params, args, eval_env_params, device='gpu'):
    # logdir = "./logs/" + args['exp_name']
    # os.makedirs(logdir, exist_ok=True)
    # model_folder = "./logs/{}/models/".format(args['exp_name'])

    env = gym.make_vec('gym4real/dam-v0', num_envs=args['n_envs'], wrappers=[my_wrap_env], vectorization_mode="sync", settings=params)
    env = wrap_env(env)

    eval_env = my_wrap_env(LakeEnv(eval_env_params))
    eval_env = wrap_env(eval_env)

    memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, net_arch=args['net_arch'], clip_actions=True)
    models["value"] = Value(env.observation_space, env.action_space, device, net_arch=args['net_arch'])

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1024  # memory_size
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 32
    cfg["discount_factor"] = 0.995
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = 0.5
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = False
    cfg["entropy_loss_scale"] = args['ent_coef']
    cfg["value_loss_scale"] = 0.5
    cfg["kl_threshold"] = 0
    cfg["mixed_precision"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 500
    # cfg["experiment"]['experiment_name'] = args['exp_name']
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = "runs/torch/lake"

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    cfg_trainer = {"timesteps": 50000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()


if __name__ == '__main__':
    # Example parameters
    args = {
        # 'exp_name': 'dam_100_episodes',
        # 'n_episodes': 100,
        'n_envs': 5,
        'batch_size': 256,
        'verbose': 1,
        'gamma': 0.995,
        'ent_coef': 0.01,
        'learning_rate': 8e-5,
        'log_rate': 10,
        'seed': 123,
        'save_model_as': 'ppo_dam_episodes',
        'net_arch': [64, 64, 32]
    }

    params = parameter_generator(
        world_options='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/world_train.yaml',
        lake_params='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/lake.yaml')

    eval_params = parameter_generator(
        world_options='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/world_test.yaml',
        lake_params='/media/samuele/Disco/PycharmProjectsUbuntu/gym4ReaL/gym4real/envs/dam/lake.yaml')

    train_ppo(params, args, eval_params, 'cuda')