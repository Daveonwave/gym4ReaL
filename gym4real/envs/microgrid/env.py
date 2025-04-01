from typing import Any
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from datetime import timedelta
from gymnasium import Env
from gymnasium.spaces import Box
from .rewards import operational_cost, linearized_degradation, soh_cost
from ernestogym.ernesto.energy_storage.bess import BatteryEnergyStorageSystem
from ernestogym.ernesto import PVGenerator, EnergyDemand, EnergyMarket, DummyGenerator, DummyMarket, AmbientTemperature, DummyAmbientTemperature


class MicroGridEnv(Env):
    """
    """
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365.25

    def __init__(self,
                 settings: dict[str, Any],
                 ):
        """

        """
        metadata = {"render_modes": None}
        
        # Build the battery object
        self._battery = BatteryEnergyStorageSystem(
            models_config=settings['models_config'],
            battery_options=settings['battery'],
            input_var=settings['input_var']
        )

        # Save the initialization bounds for environment parameters from which we will sample at reset time
        self._reset_params = settings['battery']['init']
        self._params_bounds = settings['battery']['bounds']
        self._aging_options = settings['aging_options']
        self._random_battery_init = settings['random_battery_init']
        self._random_data_init = settings['random_data_init']
        self._seed = settings['seed']
        np.random.seed(self._seed)

        # Collect exogenous variables profiles
        self.demand = EnergyDemand(**settings["demand"])
        self.generation = PVGenerator(**settings["generation"]) if 'generation' in settings \
            else DummyGenerator(gen_value=settings['dummy']['generation'])
        self.market = EnergyMarket(**settings["market"]) if 'market' in settings \
            else DummyMarket(**settings['dummy']["market"])
        self.temp_amb = AmbientTemperature(**settings["temp_amb"]) if 'temp_amb' in settings \
            else DummyAmbientTemperature(temp_value=settings['dummy']['temp_amb'])

        # Timing variables of the simulation
        self.timeframe = 0
        self.elapsed_time = 0
        self.iterations = 0
        self._env_step = settings['step']
        self.termination = settings['termination']
        self.termination['max_iterations'] = len(self.generation) - 1 if self.termination['max_iterations'] is None else self.termination['max_iterations']

        # Reward coefficients
        self._trading_coeff = settings['reward']['trading_coeff'] if 'trading_coeff' in settings['reward'] else 0
        self._op_cost_coeff = settings['reward']['operational_cost_coeff'] if 'operational_cost_coeff' in settings['reward'] else 0
        self._deg_coeff = settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0
        self._clip_action_coeff = settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0
        self._use_reward_normalization = settings['use_reward_normalization']
        self._trad_norm_term = None
        self._max_op_cost = None
        self.traded_energy = []
        
        # To distinguish between learning and testing
        self.eval_profile = None

        # MDP information
        self._state = None
        self.total_reward = 0
        self.state_list: list[np.ndarray] = []
        self.action_list: list[np.ndarray] = []
        # Reward without normalization and weights
        self.pure_reward_list = {'r_trad': [], 'r_op': [], 'r_deg':[], 'r_clip': []}
        # Normalized value of reward
        self.norm_reward_list: list = {'r_trad': [], 'r_op': [], 'r_deg':[], 'r_clip': []}
        # Weighted value of reward multiplied by their coefficients
        self.weighted_reward_list: list = {'r_trad': [], 'r_op': [], 'r_deg':[], 'r_clip': []}

        # Observation space support dictionary
        self.spaces = OrderedDict()

        self.spaces['temperature'] = {'low': 250., 'high': 400.}
        self.spaces['soc'] = {'low': 0., 'high': 1.}
        self.spaces['demand'] = {'low': 0., 'high': np.inf}
        self._obs_keys = ['temperature', 'soc', 'demand']

        # Add optional 'State of Health' in observation space
        if settings['soh']:
            # spaces['soh'] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self._obs_keys.append('soh')
            self.spaces['soh'] = {'low': 0., 'high': 1.}

            # Add optional 'generation' in observation space
        if self.generation is not None:
            # spaces['generation'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)s
            self._obs_keys.append('generation')
            self.spaces['generation'] = {'low': 0., 'high': np.inf}

        # Add optional 'bid' and 'ask' of energy market in observation space
        if self.market is not None:
            # spaces['ask'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            # spaces['bid'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            self._obs_keys.append('market')
            self.spaces['ask'] = {'low': 0., 'high': np.inf}
            self.spaces['bid'] = {'low': 0., 'high': np.inf}

        if settings['day_of_year']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_keys.append('day_of_year')
            self.spaces['sin_day_of_year'] = {'low': -1, 'high': 1}
            self.spaces['cos_day_of_year'] = {'low': -1, 'high': 1}

        if settings['seconds_of_day']:
            # spaces['seconds_of_day'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_keys.append('seconds_of_day')
            self.spaces['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            self.spaces['cos_seconds_of_day'] = {'low': -1, 'high': 1}

        if settings['energy_level']:
            self._obs_keys.append('energy_level')
            min_energy = self._battery.nominal_capacity * self._battery.soc_min * self._battery.v_min
            max_energy = self._battery.nominal_capacity * self._battery.soc_max * self._battery.v_max
            self.spaces['energy_level'] = {'low': min_energy, 'high': max_energy}

        lows = [self.spaces[key]['low'] for key in self.spaces.keys()]
        highs = [self.spaces[key]['high'] for key in self.spaces.keys()]

        # Observation Space
        self.observation_space = Box(low=np.array(lows), high=np.array(highs), dtype=np.float32)

        # Action Space: percentage of generated energy to store
        self.action_space = Box(low=0., high=1., dtype=np.float32, shape=(1,))

    def _get_obs(self) -> dict[str, Any]:
        """
        Collect the observation from the environment.
        Note that 'demand' and 'generation' are considered at the previous step or as a forecast of the actual values.
        """
        obs = {}

        for key in self._obs_keys:
            match key:
                case 'temperature':
                    obs['temperature'] = self._battery.get_temp()

                case 'soc':
                    obs['soc'] = self._battery.soc_series[-1]

                case 'demand':
                    idx = self.demand.get_idx_from_times(time=self.timeframe - self._env_step)
                    _, _, obs['demand'] = self.demand[idx]

                case 'soh':
                    obs['soh'] = self._battery.soh_series[-1]

                case 'generation':
                    idx = self.generation.get_idx_from_times(time=self.timeframe - self._env_step)
                    _, _, obs['generation'] = self.generation[idx]

                case 'market':
                    idx = self.market.get_idx_from_times(time=self.timeframe)
                    _, _, obs['ask'], obs['bid'] = self.market[idx]

                case 'day_of_year':
                    sin_year = np.sin(2 * np.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * self.timeframe)
                    cos_year = np.cos(2 * np.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * self.timeframe)
                    obs['sin_day_of_year'] = sin_year
                    obs['cos_day_of_year'] = cos_year

                case 'seconds_of_day':
                    sin_day = np.sin(2 * np.pi / self.SECONDS_PER_DAY * self.timeframe)
                    cos_day = np.cos(2 * np.pi / self.SECONDS_PER_DAY * self.timeframe)
                    obs['sin_seconds_of_day'] = sin_day
                    obs['cos_seconds_of_day'] = cos_day

                case 'energy_level':
                    obs['energy_level'] = self._battery.get_c_max() * self._battery.get_v() * self._battery.soc_series[-1]

                case _:
                    raise KeyError(f'Unknown observation variable: {key}')

        return obs

    def _get_info(self):
        """
        Collect the actual information regarding 'demand' and 'generation' to execute the step and compute the reward.
        """
        # TODO: partial status of the battery + info about env
        info = {}

        idx = self.demand.get_idx_from_times(time=self.timeframe)
        _, _, info['demand'] = self.demand[idx]

        if self.generation is not None:
            idx = self.generation.get_idx_from_times(time=self.timeframe)
            _, _, info['generation'] = self.generation[idx]

        return info

    def reset(self, seed=None, options=None):
        """

        """
        super().reset(seed=seed, options=options)

        print('Resetting the environment...')
        
        # Initialize the episode counter
        self.state_list = []
        self.action_list = []
        
        # Reset reward collections
        self.pure_reward_list = {'r_trad': [], 'r_op': [], 'r_deg':[], 'r_clip': []}
        self.norm_reward_list: list = {'r_trad': [], 'r_op': [], 'r_deg':[], 'r_clip': []}
        self.weighted_reward_list: list = {'r_trad': [], 'r_op': [], 'r_deg':[], 'r_clip': []}
        
        self.traded_energy = []

        self.total_reward = 0
        self._trad_norm_term = None
        self._max_op_cost = None

        self.elapsed_time = 0
        self.iterations = 0

        # Randomly sample a profile within the dataset
        if options is not None and 'eval_profile' in options:
            self.eval_profile = options['eval_profile']
            self.demand.profile = self.eval_profile
        else:
            self.demand.profile = np.random.choice(self.demand.labels)
        print("profile: ", self.demand.profile)

        # If seed is -1 we take datasets from the beginning
        if not self._random_data_init:
            gen_idx = 1
        # Otherwise we take an index between [1,len-1] so that we won't have out-of-index issues
        else:
            gen_idx = np.random.randint(low=1, high=len(self.generation) - 1)
        
        _, sampled_time, _ = self.generation[gen_idx]
        self.timeframe = sampled_time % (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR)
        
        # Initialize randomly the environment setting for a new run
        if self._random_battery_init:
            init_info = {key: np.random.uniform(low=value['low'], high=value['high']) for key, value in
                         self._params_bounds.items()}
        else:
            init_info = {key: value for key, value in self._reset_params.items()}
            idx = self.temp_amb.get_idx_from_times(time=self.timeframe)
            _, _, init_info['temperature'] = self.temp_amb[idx]
            _, _, init_info['temp_ambient'] = self.temp_amb[idx]

        # Initialize the battery object
        self._battery.reset()
        self._battery.init(init_info=init_info)
                
        self._state = np.array(list(self._get_obs().values()), dtype=np.float32)
        self.state_list.append(self._state)

        # info = self._get_info()
        info = {}
        
        return self._state, info

    def step(self, action: np.ndarray):
        """

        """
        #assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid" (TODO: this assertion generates error with np.ndarray)
        assert self._state is not None, "Call reset before using step method."

        self.action_list.append(action)

        # Retrieve the actual amount of demand, generation and market
        obs_pre_step, info_pre_step = self._get_obs(), self._get_info()

        self.timeframe += self._env_step

        # Compute the fraction of energy to store/use and the fraction to sell/buy
        margin = info_pre_step['generation'] - info_pre_step['demand']

        last_v = self._battery.get_v()
        i_max, i_min = self._battery.get_feasible_current(last_soc=self._battery.soc_series[-1], dt=self._env_step)

        # Clip the chosen action so that it won't exceed the SoC limits
        to_load = np.clip(a=margin * action[0], a_min=last_v * i_min, a_max=last_v * i_max)
        to_trade = margin - to_load
        
        self.traded_energy.append(to_trade)

        # Current ambient temperature
        idx = self.temp_amb.get_idx_from_times(time=self.timeframe)
        _, _, t_amb = self.temp_amb[idx]        
                
        # Step of the battery model and update of internal state
        self._battery.step(load=to_load, dt=self._env_step, k=self.iterations, t_amb=t_amb)
        self._battery.t_series.append(self.elapsed_time)
        self.elapsed_time += self._env_step
        self.iterations += 1
                                
        # Termination condition
        terminated = bool(self._battery.soh_series[-1] <= self.termination['min_soh'])

        # Truncation conditions (due to the end of data)
        truncated = bool(
            (self.termination['max_iterations'] is not None and
             self.iterations >= self.termination['max_iterations'])
            or self.demand.is_run_out_of_data()
            or self.generation.is_run_out_of_data()
            or self.market.is_run_out_of_data()
        )

        # Trading reward with market and cost of degradation
        r_trading = to_trade * obs_pre_step['ask'] if to_trade < 0 else to_trade * obs_pre_step['bid']

        # Clipping penalty from unfeasible actions
        r_clipping = -abs(margin * action[0] - to_load)

        # Operational cost penalty and degradation penalty
        r_operation, r_deg = self._optional_reward()
        
        pure_reward_terms = [r_trading, r_operation, r_deg, r_clipping]
        normalized_reward_terms = self._normalize_reward(deepcopy(pure_reward_terms))
        weighted_reward_terms = [self._trading_coeff * normalized_reward_terms[0], self._op_cost_coeff * normalized_reward_terms[1],
                                 self._deg_coeff * normalized_reward_terms[2], self._clip_action_coeff * normalized_reward_terms[3]]

        self._update_reward_collections(pure_reward_terms, normalized_reward_terms, weighted_reward_terms)

        # Combining reward terms
        reward = sum(weighted_reward_terms)
        self.total_reward += reward
        
        self._state = np.array(list(self._get_obs().values()), dtype=np.float32)
        self.state_list.append(self._state)
        info = self._get_info()
        
        if truncated or terminated:
            info['total_reward'] = self.total_reward
            info['pure_reward_list'] = self.pure_reward_list
            info['norm_reward_list'] = self.norm_reward_list
            info['weighted_reward_list'] = self.weighted_reward_list            
            info['actions'] = [action.tolist() for action in self.action_list]
            info['states'] = [state.tolist() for state in self.state_list]
            info['traded_energy'] = self.traded_energy
            info['soh'] = self._battery.soh_series
            
        return self._state, reward, terminated, truncated, info

    def _optional_reward(self):
        """
        Compute the reward related to optional aspect of the environment. In particular, the operational cost term, the
        degradation term and the action clipping penalty term.
        NOTE: the reward related to trading is the only mandatory term.
        """
        op_cost_term = 0
        deg_term = 0

        # Linearized degradation penalty
        deg_term = soh_cost(delta_soh=abs(self._battery.soh_series[-2] - self._battery.soh_series[-1]),
                            replacement_cost=self._battery.nominal_cost,
                            soh_limit=self.termination['min_soh'])

        # Operational cost penalty
        op_cost_term = (
            operational_cost(replacement_cost=self._battery.nominal_cost,
                             C_rated=self._battery.nominal_capacity * self._battery.nominal_voltage / 1000,
                             C=self._battery.get_c_max() * self._battery.nominal_voltage / 1000,
                             DoD_rated=self._battery.nominal_dod,
                             L_rated=self._battery.nominal_lifetime,
                             v_rated=self._battery.nominal_voltage,
                             K_rated=self._battery.get_polarization_resistance(),
                             p=self._battery.get_p(),
                             r=self._battery.get_internal_resistance(),
                             soc=self._battery.soc_series[-1],
                             is_discharging=bool(self._battery.get_p() <= 0))
        )
            
        if self._max_op_cost is None: 
            self._max_op_cost = operational_cost(replacement_cost=self._battery.nominal_cost,
                                                 C_rated=self._battery.nominal_capacity * self._battery.nominal_voltage / 1000,
                                                 C=self._battery.nominal_capacity * self._battery.nominal_voltage / 1000,
                                                 DoD_rated=self._battery.nominal_dod,
                                                 L_rated=self._battery.nominal_lifetime,
                                                 v_rated=self._battery.nominal_voltage,
                                                 K_rated=self._battery.get_polarization_resistance(nominal=True),
                                                 p=self._battery.get_feasible_current(last_soc=self._battery.soc_min, dt=self._env_step)[0] * self._battery.v_max / 1000,
                                                 r=self._battery.get_internal_resistance(nominal=True),
                                                 soc=self._battery.soc_min,
                                                 is_discharging=True)

        return -op_cost_term, -deg_term

    def _normalize_reward(self, rewards: list):
        """
        Min-max normalization of reward values.
        """
        if self._use_reward_normalization:
            if self._trad_norm_term is None:
                self._trad_norm_term = max(self.generation.max_gen * self.market.max_bid, self.demand.max_demand * self.market.max_ask)
        
            # OlD NORMALIZATION of r_trad
            #min_trading = -self.demand.max_demand * self.market.max_ask
            #max_trading = self._battery.v_max * (0.6 * self._battery.nominal_capacity) * self.market.max_bid
            #max_trading = self.generation.max_gen * self.market.max_bid
            #rewards[0] = -1 + 2 * (rewards[0] - min_trading) / (max_trading - min_trading)
            
            rewards[0] = rewards[0] / self._trad_norm_term
            #rewards[1] = rewards[1] / 40 / 21
            #print("before:", rewards[1])
            #print("MAX:", self._max_op_cost)
            
            #rewards[1] = rewards[1] / self._max_op_cost
            rewards[1] = rewards[1] / self._battery.nominal_cost
            #rewards[1] = rewards[1] / 410            
            
            #rewards[3] = rewards[3] / max(self.demand.max_demand, self.generation.max_gen)
            rewards[3] = rewards[3] / max(abs(self.demand.max_demand - self.generation.min_gen), 
                                          abs(self.generation.max_gen - self.demand.min_demand))            
            #rewards[3] = rewards[3] / self.demand.max_demand

        return rewards
    
    def _update_reward_collections(self, pure_terms: list, norm_terms: list, weighted_terms: list):
        labels = ['r_trad', 'r_op', 'r_deg', 'r_clip']
        
        for i, label in enumerate(labels):
            self.pure_reward_list[label].append(pure_terms[i])
            self.norm_reward_list[label].append(norm_terms[i])
            self.weighted_reward_list[label].append(weighted_terms[i])

    def render(self):
        raise NotImplementedError("Rendering not implemented yet.")

    def close(self):
        raise NotImplementedError("Rendering not implemented yet.")

