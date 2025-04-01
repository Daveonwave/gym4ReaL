"""
This module primarily implements the `parameter_generator()` function which
generates the parameters dict for `EnergyStorageEnv`.
"""
import yaml
from pint import UnitRegistry
from ernestogym.ernesto.utils import read_csv
from ernestogym.ernesto import read_yaml, validate_yaml_parameters

BATTERY_OPTIONS = "ernestogym/ernesto/data/battery/cell.yaml"
INPUT_VAR = 'power'     # 'power'/'current'/'voltage'

ECM = "ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml"
R2C_THERMAL = "ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml"
BOLUN_MODEL = "ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml"
WORLD = "ernestogym/envs/single_agent/world_fading.yaml"

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)


def parameter_generator(battery_options: str = BATTERY_OPTIONS,
                        world_options: str = WORLD,
                        input_var: str = INPUT_VAR,
                        electrical_model: str = ECM,
                        thermal_model: str = R2C_THERMAL,
                        aging_model: str = BOLUN_MODEL,
                        use_degradation: bool = None,
                        use_fading: bool = None,
                        step: int = None,
                        random_battery_init: bool = None,
                        random_data_init: bool = None,
                        seed: int = None,
                        max_iterations: int = None,
                        min_soh: float = None,
                        reward_coeff: dict[str, float] = None,
                        use_reward_normalization: bool = None,
                        bypass_yaml_schema: bool = False,
                        spread_factor: float = 1.0,
                        replacement_cost: float = 3000.0
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """
    with open(world_options, "r") as fin:
        world_settings = yaml.safe_load(fin)

    # Battery parameters retrieved with ErNESTO APIs.
    battery_params = read_yaml(battery_options, yaml_type='battery_options', bypass_check=bypass_yaml_schema)
    battery_params['battery']['params'] = validate_yaml_parameters(battery_params['battery']['params'])

    # Battery submodel configuration retrieved with ErNESTO APIs.
    models_config = [read_yaml(electrical_model, yaml_type='model', bypass_check=bypass_yaml_schema),
                     read_yaml(thermal_model, yaml_type='model', bypass_check=bypass_yaml_schema)]

    params = {'battery': battery_params['battery'],
              'input_var': input_var,
              'models_config': models_config,
              'demand': {'data': read_csv(world_settings['demand']['path']),
                         'timestep': world_settings['demand']['timestep'],
                         'test_profiles': world_settings['demand']['test_profiles'],
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

    # Aging settings
    params['aging_options'] = {'degradation': use_degradation if use_degradation is not None else world_settings['aging_options']['degradation'],
                               'fading': use_fading if use_fading is not None else world_settings['aging_options']['fading']}

    assert not (params['aging_options']['degradation'] and params['aging_options']['fading']), \
        ("Degradation model and fading model cannot be used together (at the moment) since they depend on different "
         "variables.")

    if params['aging_options']['fading']:
        assert models_config[0]['use_fading'], ("The selected electrical model ({}) doesn't support parameter fading."
                                                .format(models_config[0]['class_name']))
    if params['aging_options']['degradation']:
        assert not models_config[0]['use_fading'], ("The selected electrical model is not compatible with the aging "
                                                    "model since it implements fading mechanisms.")
        models_config.append(read_yaml(aging_model, yaml_type='model', bypass_check=bypass_yaml_schema))

    # Reward settings
    params['reward'] = reward_coeff if reward_coeff is not None else world_settings['reward']
    params['use_reward_normalization'] = use_reward_normalization if use_reward_normalization is not None else world_settings['use_reward_normalization']

    # Termination settings
    params['termination'] = {'max_iterations': max_iterations if max_iterations is not None else world_settings['termination']['max_iterations'],
                             'min_soh': min_soh if min_soh is not None else world_settings['termination']['min_soh']}

    return params
