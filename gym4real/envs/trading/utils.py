import yaml
import numpy as np
from sklearn.base import TransformerMixin
from gym4real.envs.elevator.simulator.passenger import generate_arrival_distribution

WORLD = "gym4real/envs/trading/world_train.yaml"


def read_yaml(yaml_file: str):
    with open(yaml_file, 'r') as fin:
        params = yaml.safe_load(fin)
    return params


def parameter_generator(world_options: str = WORLD,
                        data_directory: str = None,
                        years: [int] = None,
                        anomalous_days: [str] = None,
                        persistence: int = None,
                        number_of_deltas: int = None,
                        offset_delta: int = None,
                        sequential: bool = None,
                        fillna_features: bool = None,
                        ffill_limit_data: int = None,
                        capital: int = None,
                        fees: int = None,
                        trading_open: int = None,
                        trading_close: int = None,
                        use_month: bool =  None,
                        use_day_of_week: bool =  None,
                        use_day: bool = None,
                        sinusoidal_transformation: bool = None,
                        seed: int = None
                        ) -> dict:
    """
    Generates the parameters dict for `ElevatorEnv`.
    """
    world_settings = read_yaml(world_options)

    params = {}
    params['data_directory'] = data_directory if data_directory is not None else world_settings['data_directory']
    params['years'] = years if years is not None else world_settings['years']
    params['anomalous_days'] = anomalous_days if anomalous_days is not None else world_settings['anomalous_days']
    params['persistence'] = persistence if persistence is not None else world_settings['persistence']
    params['number_of_deltas'] = number_of_deltas if number_of_deltas is not None else world_settings['number_of_deltas']
    params['offset_delta'] = offset_delta if offset_delta is not None else world_settings['offset_delta']
    params['sequential'] = sequential if sequential is not None else world_settings['sequential']
    params['fillna_features'] = fillna_features if fillna_features is not None else world_settings['fillna_features']
    params['ffill_limit_data'] = ffill_limit_data if ffill_limit_data is not None else world_settings['ffill_limit_data']


    params['capital'] = capital if capital is not None else world_settings['capital']
    params['fees'] = fees if fees is not None else world_settings['fees']

    params['trading_open'] = trading_open if trading_open is not None else world_settings['trading_open']
    params['trading_close'] = trading_close if trading_close is not None else world_settings['trading_close']

    params['use_month'] = use_month if use_month is not None else world_settings['use_month']
    params['use_day'] = use_day if use_day is not None else world_settings['use_day']

    params['use_day_of_week'] = use_day_of_week if use_day_of_week is not None else world_settings[
        'use_day_of_week']
    params['sinusoidal_transformation'] = sinusoidal_transformation if sinusoidal_transformation is not None else world_settings['sinusoidal_transformation']
    #params['seed'] = seed if seed is not None else world_settings['seed']

    return params
