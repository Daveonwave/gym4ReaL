import sys
import os

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from gym4real.envs.robofeeder.rf_picking_v0 import robotEnv
from stable_baselines3.common.policies import BaseFeaturesExtractor
import torch 
import torch.nn as nn
from gymnasium import spaces
import argparse



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
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            flat_output = self.cnn(dummy_input)
            cnn_output_dim = flat_output.shape[1]
            #print(f"Raw CNN output dim: {cnn_output_dim}")

        # Final projection to fixed feature dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.cnn(observations)
        return self.linear(features)

#planning
pi = [256, 128, 64]
vf = [256, 128, 64]

#picking
# pi = [512, 128]
# vf = [512, 128]

features_dim = 256
optimizer_kwargs= dict(weight_decay=2e-5,)

## Example of policy_kwargs with custom features extractor
# policy_kwargs = dict(normalize_images=False,
#                      features_extractor_class=CustomCNN,
#                      features_extractor_kwargs=dict(features_dim=features_dim),
#                      net_arch=dict(pi=pi, vf=vf),
#                      optimizer_kwargs=optimizer_kwargs
#                      )


policy_kwargs = dict(normalize_images=False , net_arch=dict(pi=pi, vf=vf))



def train_ppo(envs, args, model_file=None):
    print("######## PPO is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    if model_file is not None:
        model = PPO.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = PPO("CnnPolicy", 
                    env=envs, 
                    verbose=args['verbose'], 
                    gamma=args['gamma'], 
                    tensorboard_log="./logs/tensorboard/robofeeder-env0/ppo/".format(args['exp_name']),
                    n_steps=args['n_steps'],
                    n_epochs=args['n_epochs'],
                    batch_size=args['n_batches'],
                    learning_rate=args['learning_rate'],
                    # ent_coef=0.01,
                    policy_kwargs=policy_kwargs,
                    seed = 123
                    )
        
    model.learn(total_timesteps=args['n_episodes'] * args['n_envs'],
                progress_bar=True,
                tb_log_name="ppo_{}".format(args['exp_name']),
                reset_num_timesteps=False,
                )
    
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
    print("######## TRAINING is Done ########")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train PPO on RoboFeeder environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--exp_name', type=str, default='robofeeder_planning_5k', help='Experiment name for logging and saving models')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of steps per environment per update')
    parser.add_argument('--n_epochs', type=int, default=8, help='Number of epochs per update')
    parser.add_argument('--n_batches', type=int, default=128, help='Batch size for training')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--log_rate', type=int, default=1, help='Logging rate')
    parser.add_argument('--save_model_as', type=str, default='ppo_5k', help='Filename to save the trained model')
    parser.add_argument('--config_params', type=str, default="gym4real/envs/robofeeder/configuration.yaml", help='Path to environment configuration file')
    parser.add_argument('--env_id', type=str, default="gym4real/robofeeder-planning", help='Environment ID')
    parser.add_argument('--model_file', type=str, default=None, help='Path to a pre-trained model to resume training')

    args = vars(parser.parse_args())

    envs = make_vec_env(args['env_id'], n_envs=args['n_envs'], env_kwargs={'config_file': args['config_params']})

    train_ppo(envs=envs, args=args, model_file=args['model_file'])

    envs.close()
    print("######## PPO is Done ########")