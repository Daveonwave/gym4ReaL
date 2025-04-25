import sys
sys.path.append('/usr/src/data/')

from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

config_path = "/usr/src/data/configuration.yaml"


from stable_baselines3 import PPO
from Env_1 import robotEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

config_path = "/usr/src/data/configuration.yaml"

def make_env(path):
    def _init():
        env = Monitor(robotEnv(config_file=path))
        return env
    return _init


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        ks = 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=ks, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=ks, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=ks, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=ks, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Dynamically calculate CNN output size
        with th.no_grad():
            dummy_input = th.zeros(1, *observation_space.shape)
            flat_output = self.cnn(dummy_input)
            cnn_output_dim = flat_output.shape[1]
            #print(f"Raw CNN output dim: {cnn_output_dim}")

        # Final projection to fixed feature dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.cnn(observations)
        return self.linear(features)

if __name__ == "__main__":
    import os
    import torch as th

    num_cpu = 1  # or more, if needed
    env = SubprocVecEnv([make_env(config_path) for i in range(num_cpu)])

    policy_kwargs = dict(
        normalize_images=False,
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        optimizer_kwargs=dict(weight_decay=1e-5),
    )

    n_steps=2

    model = PPO(
        "CnnPolicy",
        env,
        n_steps=n_steps,
        batch_size=n_steps*num_cpu,
        n_epochs=20,
        learning_rate=0.003, 
        clip_range=0.3,
        #gamma=0.95,
        ent_coef=0.01, 
        #vf_coef=0.5,
        #max_grad_norm=.5,
        verbose=0,
        seed=123,
        tensorboard_log= ".",
        policy_kwargs=policy_kwargs,
        
    )

    model.learn(total_timesteps=100000,reset_num_timesteps=False,progress_bar=True)