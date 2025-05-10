import yaml
import os
import pandas as pd

BATTERY = "gym4real/envs/microgrid/simulator/energy_storage/configuration/battery_pack.yaml"
INPUT = 'power'
ECM = "gym4real/envs/microgrid/simulator/energy_storage/configuration/models/electrical.yaml"
THERMAL = "gym4real/envs/microgrid/simulator/energy_storage/configuration/models/thermal.yaml"
AGING = "gym4real/envs/microgrid/simulator/energy_storage/configuration/models/aging.yaml"
WORLD = "gym4real/envs/microgrid/world_train.yaml"


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


def parameter_generator(battery_options: str = BATTERY,
                        world_options: str = WORLD,
                        input_var: str = INPUT,
                        electrical_model: str = ECM,
                        thermal_model: str = THERMAL,
                        aging_model: str = AGING,
                        step: int = None,
                        random_battery_init: bool = None,
                        random_data_init: bool = None,
                        seed: int = None,
                        max_iterations: int = None,
                        min_soh: float = None,
                        reward_coeff: dict[str, float] = None,
                        use_reward_normalization: bool = True,
                        spread_factor: float = 1.0,
                        replacement_cost: float = 3000.0,
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """
    world_settings = read_yaml(world_options)

    # Battery parameters retrieved with ErNESTO APIs.
    battery_params = read_yaml(battery_options)

    # Battery submodel configuration retrieved with ErNESTO APIs.
    models_config = [read_yaml(electrical_model),
                     read_yaml(thermal_model),
                     read_yaml(aging_model)]

    params = {'battery': battery_params['battery'],
              'input_var': input_var,
              'models_config': models_config,
              'demand': {'data': read_csv(world_settings['demand']['path']),
                         'timestep': world_settings['demand']['timestep'],
                         'data_usage': world_settings['demand']['data_usage']}}
    
    if replacement_cost is not None:
        params['battery']['params']['nominal_cost'] = replacement_cost
    
    params['soh'] = True if 'soh' in world_settings['observations'] else False

    # Exogenous variables data
    if 'generation' in world_settings['observations']:
        params['generation'] = {'data': read_csv(world_settings['generation']['path']),
                                'timestep': world_settings['generation']['timestep'],
                                'data_usage': world_settings['generation']['data_usage']}

    if 'market' in world_settings:
        params['market'] = {'data': read_csv(world_settings['market']['path']),
                            'timestep': world_settings['market']['timestep'],
                            'data_usage': world_settings['market']['data_usage'],
                            'spread_factor': spread_factor}
    
    if 'temp_amb' in world_settings:
        params['temp_amb'] = {'data': read_csv(world_settings['temp_amb']['path']),
                            'timestep': world_settings['temp_amb']['timestep'],
                            'data_usage': world_settings['temp_amb']['data_usage']}

    # Dummy information about world behavior
    params['dummy'] = world_settings['dummy']
    params['dummy']['market']['spread_factor'] = spread_factor
    
    # Time info among observations
    params['day_of_year'] = True if 'day_of_year' in world_settings['observations'] else False
    params['seconds_of_day'] = True if 'seconds_of_day' in world_settings['observations'] else False

    params['energy_level'] = True if 'energy_level' in world_settings['observations'] else False

    params['step'] = step if step is not None else world_settings['step']
    params['seed'] = seed if seed is not None else world_settings['seed']
    params['random_battery_init'] = random_battery_init if random_battery_init is not None else world_settings['random_battery_init']
    params['random_data_init'] = random_data_init if random_data_init is not None else world_settings['random_data_init']

    # Reward settings
    params['reward'] = reward_coeff if reward_coeff is not None else world_settings['reward']
    params['use_reward_normalization'] = use_reward_normalization if use_reward_normalization is not None else world_settings['use_reward_normalization']

    # Termination settings
    params['termination'] = {'max_iterations': max_iterations if max_iterations is not None else world_settings['termination']['max_iterations'],
                             'min_soh': min_soh if min_soh is not None else world_settings['termination']['min_soh']}
    
    return params
