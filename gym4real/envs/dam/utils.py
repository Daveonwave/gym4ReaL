import numpy as np
import random
import yaml
import os
import pandas as pd

class Utils:

    @staticmethod
    def load_matrix(file_name, row, col):
        data = np.loadtxt(file_name)
        return data.reshape((row, col)).tolist()

    @staticmethod
    def load_vector(file_name, length):
        return np.loadtxt(file_name, max_rows=length).tolist()

    @staticmethod
    def load_array(file_name, length):
        return np.loadtxt(file_name, max_rows=length)

    @staticmethod
    def log_vector(vec, file_name):
        with open(file_name, 'w') as f:
            for val in vec:
                f.write(f"{val}\n")

    @staticmethod
    def log_vector_append(vec, file_name):
        with open(file_name, 'a') as f:
            for val in vec:
                f.write(f"{val}\n")

    @staticmethod
    def interp_lin(X, Y, x):
        return np.interp(x, X, Y, left=None, right=None)

    # Unit Conversions
    @staticmethod
    def gallon_to_cubic_feet(x):
        return x * 0.13368

    @staticmethod
    def inches_to_feet(x):
        return x * 0.08333

    @staticmethod
    def cubic_feet_to_cubic_meters(x):
        return x * 0.0283

    @staticmethod
    def feet_to_meters(x):
        return x * 0.3048

    @staticmethod
    def acre_to_squared_feet(x):
        return x * 43560

    @staticmethod
    def acre_feet_to_cubic_feet(x):
        return x * 43560

    @staticmethod
    def cubic_feet_to_acre_feet(x):
        return x / 43560

    # Vector Operations
    @staticmethod
    def compute_sum(g):
        return sum(g)

    @staticmethod
    def compute_max(g):
        return max(g)

    @staticmethod
    def compute_min(g):
        return min(g)

    @staticmethod
    def compute_mean(g):
        return np.mean(g)

    @staticmethod
    def compute_variance(g):
        return np.var(g)

    # Normalization
    @staticmethod
    def normalize_vector(X, m, M):
        return [(x - mi) / (Ma - mi) for x, mi, Ma in zip(X, m, M)]

    @staticmethod
    def denormalize_vector(X, m, M):
        return [x * (Ma - mi) + mi for x, mi, Ma in zip(X, m, M)]

    # Standardization
    @staticmethod
    def standardize_vector(X, m, s):
        return [(x - mi) / si for x, mi, si in zip(X, m, s)]

    @staticmethod
    def destandardize_vector(X, m, s):
        return [x * si + mi for x, mi, si in zip(X, m, s)]

    # Random number
    @staticmethod
    def generate_random_unif(lower_bound, upper_bound):
        return random.uniform(lower_bound, upper_bound)

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
        'warmup': world_settings['warmup'],
        'doy': world_settings['doy'],
        'flood_level': world_settings['flood_level'],
        'observations': world_settings['observations'],
        'action': world_settings['action'],
        'lake_params': lake_params,
        'reward_coeff': world_settings['reward'],
        'seed': world_settings['seed'],
        'random_init': world_settings['random_init'],
    }

    demand = read_csv(world_settings['demand']).to_dict(orient='list')
    inflow = read_csv(world_settings['inflow']).to_dict(orient='list')

    assert set(demand.keys()) == set(inflow.keys())

    for key in demand.keys():
        demand[key] = [x for x in demand[key] if not np.isnan(x)]
        inflow[key] = [x for x in inflow[key] if not np.isnan(x)]
        assert len(inflow[key]) == len(demand[key])

    params['demand'] = demand
    params['inflow'] = inflow

    q_forecast = read_csv(world_settings['q_forecast'])['q_forecast'].to_numpy()
    params['q_forecast'] = q_forecast

    return params