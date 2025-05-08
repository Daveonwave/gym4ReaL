from typing import Any
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from .rewards import soh_cost
from .simulator.energy_storage.bess import BatteryEnergyStorageSystem
from .simulator import PVGenerator, EnergyDemand, EnergyMarket, DummyGenerator, DummyMarket, AmbientTemperature, DummyAmbientTemperature


class MicroGridEnv(Env):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365.25

    def __init__(self,
                 settings: dict[str, Any],
                 render_mode=None
                 ):
        metadata = {"render_modes": None}
        """
        Initialize the MicroGrid environment.

        This method sets up the environment, including the battery system, exogenous variables, 
        observation and action spaces, and reward coefficients.

        Args:
            settings (dict[str, Any]): A dictionary containing configuration settings for the environment.
        """        
        # Build the battery object
        self._battery = BatteryEnergyStorageSystem(
            models_config=settings['models_config'],
            battery_options=settings['battery'],
            input_var=settings['input_var']
        )

        # Save the initialization bounds for environment parameters from which we will sample at reset time
        self._reset_params = settings['battery']['init']
        self._params_bounds = settings['battery']['bounds']
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
        self._deg_coeff = settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0
        self._clip_action_coeff = settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0
        self._use_reward_normalization = settings['use_reward_normalization']
        self._trad_norm_term = None

        
        # Reward without normalization and weights
        self.pure_rewards = {'r_trad':0, 'r_deg':0, 'r_clip': 0}
        # Normalized value of reward
        self.norm_rewards = {'r_trad':0, 'r_deg':0, 'r_clip':0}
        # Weighted value of reward multiplied by their coefficients
        self.weighted_rewards = {'r_trad':0, 'r_deg':0, 'r_clip':0}

        # Observation space support dictionary
        spaces = {}
        spaces['temperature'] = {'low': 250., 'high': 400.}
        spaces['soc'] = {'low': 0., 'high': 1.}
        spaces['demand'] = {'low': 0., 'high': np.inf}
        self._obs_keys = ['temperature', 'soc', 'demand']

        # Add optional 'State of Health' in observation space
        if settings['soh']:
            self._obs_keys.append('soh')
            spaces['soh'] = {'low': 0., 'high': 1.}

            # Add optional 'generation' in observation space
        if self.generation is not None:
            self._obs_keys.append('generation')
            spaces['generation'] = {'low': 0., 'high': np.inf}

        # Add optional 'bid' and 'ask' of energy market in observation space
        if self.market is not None:
            self._obs_keys.append('market')
            spaces['ask'] = {'low': 0., 'high': np.inf}
            spaces['bid'] = {'low': 0., 'high': np.inf}

        if settings['day_of_year']:
            self._obs_keys.append('day_of_year')
            spaces['sin_day_of_year'] = {'low': -1, 'high': 1}
            spaces['cos_day_of_year'] = {'low': -1, 'high': 1}

        if settings['seconds_of_day']:
            self._obs_keys.append('seconds_of_day')
            spaces['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            spaces['cos_seconds_of_day'] = {'low': -1, 'high': 1}

        lows = [spaces[key]['low'] for key in spaces.keys()]
        highs = [spaces[key]['high'] for key in spaces.keys()]

        # Gym spaces
        self.observation_space = Box(low=np.array(lows), high=np.array(highs), dtype=np.float32)
        self.action_space = Box(low=0., high=1., dtype=np.float32, shape=(1,))

    def _get_obs(self) -> dict[str, Any]:
        """
        Collect the observation from the environment.

        This method gathers the current state of the environment based on the observation keys defined during initialization.
        It includes variables such as temperature, state of charge (SOC), demand, generation, market data, and time-related features.

        Returns:
            dict[str, Any]: A dictionary containing the current observation values for each key.
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

                case _:
                    raise KeyError(f'Unknown observation variable: {key}')

        return obs

    def _get_actual_state(self) -> dict[str, Any]:
        """
        Collect the actual information regarding 'demand' and 'generation' to execute the step and compute the reward.

        This method retrieves the real-time values of demand and generation at the current timeframe to be used
        for environment dynamics and reward calculation.

        Returns:
            dict[str, Any]: A dictionary containing the actual 'demand' and 'generation' values.
        """
        info = {}

        idx = self.demand.get_idx_from_times(time=self.timeframe)
        _, _, info['demand'] = self.demand[idx]

        if self.generation is not None:
            idx = self.generation.get_idx_from_times(time=self.timeframe)
            _, _, info['generation'] = self.generation[idx]

        return info
    
    def _get_info(self, to_trade: float) -> dict[str, Any]:
        """
        Retrieve detailed information about the current environment state.

        This method provides a snapshot of the environment's status, including cumulative rewards, 
        the most recent action taken, individual reward components, and battery state.

        Args:
            to_trade (float): The amount of energy traded (positive for selling, negative for buying).

        Returns:
            dict[str, Any]: A dictionary containing the following keys:
                - "cumulated_reward": The total reward accumulated so far.
                - "action": The last action taken by the agent.
                - "pure_rewards": The raw reward components before normalization and weighting.
                - "norm_rewards": The normalized reward components.
                - "weighted_rewards": The weighted reward components after applying coefficients.
                - "traded_energy": The amount of energy traded in the current step.
                - "battery": A snapshot of the battery's current state.
        """
        info = {
            "profile": self.demand.profile,
            "pure_rewards": self.pure_rewards,
            "norm_rewards": self.norm_rewards,
            "weighted_rewards": self.weighted_rewards,
            "traded_energy": to_trade,
            "battery": self._battery.get_snapshot()
        }
        
        return info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        This method resets the environment, including the battery system, reward collections, and timing variables.
        It also initializes the environment with random or predefined settings based on the configuration.

        Args:
            seed (int, optional): A seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial state and an empty info dictionary.
        """
        super().reset(seed=seed, options=options)
        
        self.total_reward = 0
        self._trad_norm_term = None
        self.elapsed_time = 0
        self.iterations = 0

        # Randomly sample a profile within the dataset
        if options is not None and 'eval_profile' in options:
            self.demand.profile = options['eval_profile']
        else:
            self.demand.profile = np.random.choice(self.demand.labels)
        
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
                
        state = np.array(list(self._get_obs().values()), dtype=np.float32)
        return state, {}

    def step(self, action: np.ndarray):
        """
        Perform a single step in the environment.

        This method updates the environment state based on the action taken by the agent. It computes the reward, 
        checks termination and truncation conditions, and returns the new state, reward, and additional information.

        Args:
            action (np.ndarray): The action taken by the agent, representing the fraction of energy to store.

        Returns:
            tuple: A tuple containing the new state, reward, termination flag, truncation flag, and info dictionary.
        """
        # Retrieve the actual amount of demand, generation and market
        obs, actual_state = self._get_obs(), self._get_actual_state()

        self.timeframe += self._env_step

        # Compute the fraction of energy to store/use and the fraction to sell/buy
        margin = actual_state['generation'] - actual_state['demand']

        last_v = self._battery.get_v()
        i_max, i_min = self._battery.get_feasible_current(last_soc=self._battery.soc_series[-1], dt=self._env_step)

        # Clip the chosen action so that it won't exceed the SoC limits
        to_load = np.clip(a=margin * action[0], a_min=last_v * i_min, a_max=last_v * i_max)
        to_trade = margin - to_load

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
        r_trading = to_trade * obs['ask'] if to_trade < 0 else to_trade * obs['bid']

        # Operational cost penalty and degradation penalty
        r_deg = -soh_cost(delta_soh=abs(self._battery.soh_series[-2] - self._battery.soh_series[-1]),
                          replacement_cost=self._battery.nominal_cost,
                          soh_limit=self.termination['min_soh'])
        
        # Clipping penalty from unfeasible actions
        r_clipping = -abs(margin * action[0] - to_load)
        
        self.pure_rewards = {'r_trad': r_trading, 'r_deg': r_deg, 'r_clip': r_clipping}
        self._normalize_rewards(rewards=list(self.pure_rewards.values()))
        self.weighted_rewards = {'r_trad': self.norm_rewards['r_trad'] * self._trading_coeff,
                                 'r_deg': self.norm_rewards['r_deg'] * self._deg_coeff,
                                 'r_clip': self.norm_rewards['r_clip'] * self._clip_action_coeff}   

        # Combining reward terms
        reward = sum(self.weighted_rewards.values())
        
        state = np.array(list(self._get_obs().values()), dtype=np.float32)
        info = self._get_info(to_trade=to_trade)
            
        return state, reward, terminated, truncated, info
    
        
    def _normalize_rewards(self, rewards: list):
        """
        Normalize reward values using min-max normalization.

        This method normalizes the reward components based on predefined terms and coefficients. 
        If normalization is disabled, the raw rewards are used as-is.

        Args:
            rewards (list): A list of raw reward values to be normalized.
        """
        if self._use_reward_normalization:
            if self._trad_norm_term is None:
                self._trad_norm_term = max(self.generation.max_gen * self.market.max_bid, 
                                           self.demand.max_demand * self.market.max_ask)
            
            self.norm_rewards['r_trad'] = rewards[0] / self._trad_norm_term
            self.norm_rewards['r_deg'] = rewards[1] 
            self.norm_rewards['r_clip'] = rewards[2] / max(abs(self.demand.max_demand - self.generation.min_gen), 
                                          abs(self.generation.max_gen - self.demand.min_demand))          
        else:
            self.norm_rewards['r_trad'] = rewards[0]
            self.norm_rewards['r_deg'] = rewards[1]
            self.norm_rewards['r_clip'] = rewards[2]


