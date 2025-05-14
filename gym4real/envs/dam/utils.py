import numpy as np
import yaml
import os
import pandas as pd

def read_csv(csv_file: str, **kwargs) -> pd.DataFrame:
    """
    Read data from csv files
    """
    # Check file existence
    if not os.path.isfile(csv_file):
        raise FileNotFoundError("The specified file '{}' doesn't not exist.".format(csv_file))
    df = None
    try:
        df = pd.read_csv(csv_file, **kwargs)
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

def parameter_generator(world_options: str,
                        lake_params: str) -> dict:

    lake_params = read_yaml(lake_params)
    lake_params['min_env_flow'] = read_csv(lake_params['min_env_flow'])['MEF'].to_numpy()
    lake_params['evaporation_rates'] = read_csv(lake_params['evaporation_rates'])['MEF'].to_numpy() if 'evaporation_rates' in lake_params.keys() else None

    world_settings = read_yaml(world_options)

    params = {
        'period': world_settings['period'],
        'integration': world_settings['integration'],
        'sim_horizon': world_settings['sim_horizon'],
        'doy': world_settings['doy'],
        'flood_level': world_settings['flood_level'],
        'starving_level': world_settings['starving_level'],
        'observations': world_settings['observations'],
        'action': world_settings['action'],
        'lake_params': lake_params,
        'reward_coeff': world_settings['reward'],
        'seed': world_settings['seed'],
        'random_init': world_settings['random_init'],
        'smooth_daily_deficit_coeff': world_settings['smooth_daily_deficit_coeff']
    }

    demand = read_csv(world_settings['demand'], index_col='year').T.to_dict(orient='list')
    inflow = read_csv(world_settings['inflow'], index_col='year').T.to_dict(orient='list')

    assert set(demand.keys()) == set(inflow.keys())

    for key in demand.keys():
        demand[key] = [x for x in demand[key] if not np.isnan(x)]
        inflow[key] = [x for x in inflow[key] if not np.isnan(x)]
        assert len(inflow[key]) == len(demand[key])

    params['demand'] = demand
    params['inflow'] = inflow

    if 'exponential_average_coeff' in world_settings.keys():
        params['exponential_average_coeff'] = world_settings['exponential_average_coeff']

    return params