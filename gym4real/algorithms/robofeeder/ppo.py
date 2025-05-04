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


pi = [256, 128, 64]
vf = [256, 128, 64]

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
                    )
        
    model.learn(total_timesteps=5000,
                progress_bar=True,
                tb_log_name="ppo_{}".format(args['exp_name']),
                reset_num_timesteps=False,
                )
    
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
    print("######## TRAINING is Done ########")

if __name__ == '__main__':
    # Example parameters
    args = {
        'exp_name': 'robofeeder_planning_5k',
        'n_episodes': 100,
        'n_envs': 8,
        'n_steps': 10,
        'n_epochs': 8,
        'n_batches': 128,
        'verbose': 1,
        'gamma': 0.99,
        'learning_rate': 0.005,
        'log_rate': 1,
        'save_model_as': 'ppo_5k',
    }
    
    config_params = "gym4real/envs/robofeeder/configuration.yaml"

    #envs = make_vec_env("gym4real/robofeeder-picking-v0", n_envs=args['n_envs'], env_kwargs={'config_file':config_params})    
    envs = make_vec_env("gym4real/robofeeder-planning", n_envs=args['n_envs'], env_kwargs={'config_file':config_params}) 

    train_ppo(envs=envs, args=args)

    envs.close()
    print("######## PPO is Done ########")