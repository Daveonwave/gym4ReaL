import yaml
import os
import pandas as pd

WORLD = "gym4real/envs/wds/world_anytown.yaml"


def read_csv(csv_file: str) -> pd.DataFrame:
    """
    Read data from csv files
    """
    # Check file existence
    if not os.path.isfile(csv_file):
        raise FileNotFoundError("The specified file '{}' doesn't not exist.".format(csv_file))
    df = None
    try:
        df = pd.read_csv(csv_file)
    except Exception as err:
        print("Error during the loading of '{}':".format(csv_file), type(err).__name__, "-", err)
    return df


def read_yaml(yaml_file: str):
    """

    Args:
        yaml_file (str): _description_

    Returns:
        _type_: _description_
    """
    with open(yaml_file, 'r') as fin:
        params = yaml.safe_load(fin)
    return params


def parameter_generator(world_options: str = WORLD,
                        hydraulic_step: int = None,
                        duration: int = None,
                        seed: int = 42,
                        reward_coeff: dict[str, float] = None,
                        use_reward_normalization: bool = True,
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """
    world_settings = read_yaml(world_options)
    
    params = world_settings.copy()
    
    params['demand'] = {'data_config': read_yaml(world_settings['demand']['path']),
                        'pattern_step': world_settings['demand']['pattern_step'],
                        'event_probs': world_settings['demand']['event_probs']}
    
    # Observations
    params['demand_moving_average'] = True if 'demand_moving_average' in world_settings['observations'] else False
    params['demand_exp_moving_average'] = True if 'demand_exp_moving_average' in world_settings['observations'] else False
    params['seconds_of_day'] = True if 'seconds_of_day' in world_settings['observations'] else False
    params['under_attack'] = True if 'under_attack' in world_settings['observations'] else False
    
    if params['under_attack']:
        params['attackers'] = {'data_config': read_yaml(world_settings['attackers']['path'])}
    # Time settings
    params['duration'] = duration if duration is not None else world_settings['duration']
    params['hyd_step'] = hydraulic_step if hydraulic_step is not None else world_settings['hyd_step']
    params['seed'] = seed if seed is not None else world_settings['seed']

    # Reward settings
    params['reward'] = reward_coeff if reward_coeff is not None else world_settings['reward']
    params['use_reward_normalization'] = use_reward_normalization if use_reward_normalization is not None else world_settings['use_reward_normalization']

    return params
