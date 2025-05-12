import yaml
import numpy as np
from gym4real.envs.elevator.simulator.passenger import generate_arrival_distribution

WORLD = "gym4real/envs/elevator/world.yaml"


def read_yaml(yaml_file: str):
    with open(yaml_file, 'r') as fin:
        params = yaml.safe_load(fin)
    return params


def parameter_generator(world_options: str = WORLD,
                        min_floor: int = None,
                        max_floor: int = None,
                        max_capacity: int = None,
                        movement_speed: float = None,
                        floor_height: int = None,
                        max_arrivals: int = None,
                        max_queue_length: int = None,
                        duration: int = None,
                        timestep: int = None,
                        goal_floor: int = None,
                        init_elevator_pos: int = None,
                        random_init_state: bool = None,
                        reward_coeff: float = None,
                        lambda_min: float = None,
                        lambda_max: float = None,
                        seed: int = None,
                        ) -> dict:
    """
    Generates the parameters dict for `ElevatorEnv`.
    """
    world_settings = read_yaml(world_options)
    
    params = {}
    params['min_floor'] = min_floor if min_floor is not None else world_settings['elevator']['min_floor']
    params['max_floor'] = max_floor if max_floor is not None else world_settings['elevator']['max_floor']
    params['max_capacity'] = max_capacity if max_capacity is not None else world_settings['elevator']['max_capacity']
    params['movement_speed'] = movement_speed if movement_speed is not None else world_settings['elevator']['movement_speed']
    params['floor_height'] = floor_height if floor_height is not None else world_settings['elevator']['floor_height']
    
    params['max_arrivals'] = max_arrivals if max_arrivals is not None else world_settings['floors']['max_arrivals']
    params['max_queue_length'] = max_queue_length if max_queue_length is not None else world_settings['floors']['max_queue_length']
    
    params['duration'] = duration if duration is not None else world_settings['duration']
    params['timestep'] = timestep if timestep is not None else world_settings['timestep']
    
    params['goal_floor'] = goal_floor if goal_floor is not None else world_settings['goal_floor']
    params['init_elevator_pos'] = init_elevator_pos if init_elevator_pos is not None else world_settings['init_elevator_pos']
    params['random_init_state'] = random_init_state if random_init_state is not None else world_settings['random_init_state']
    params['reward_coeff'] = reward_coeff if reward_coeff is not None else world_settings['reward_coeff']
    
    lambda_min = lambda_min if lambda_min is not None else world_settings['floors']['lambda_min']
    lambda_max = lambda_max if lambda_max is not None else world_settings['floors']['lambda_max']
    seed = seed if seed is not None else world_settings['floors']['seed']
    
    params['arrival_distributions'] = {
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'seed': seed,
    }
    
    return params
