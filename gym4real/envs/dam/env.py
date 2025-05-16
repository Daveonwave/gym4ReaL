import math
import random

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from math import sin, cos, pi

from pygame.examples.midi import key_class

from gym4real.envs.dam.lake import Lake

from collections import OrderedDict

class DamEnv(Env):
    """
    An OpenAI Gym environment for simulating the management of a lake's water resources.

    The environment models the dynamics of a lake, including inflow, demand, and release, and provides 
    a framework for reinforcement learning agents to optimize water management strategies.

    Attributes:
        DAYS_IN_YEAR (int): Number of days in a year (365).
    """

    DAYS_IN_YEAR = 365

    def __init__(self, settings):
        """
        Initialize the Gym environment for lake management.

        Args:
            settings (dict): A dictionary containing the configuration parameters for the environment.
                Keys include:
                    - 'period' (int): Simulation period.
                    - 'integration' (float): Integration step size.
                    - 'sim_horizon' (int): Simulation horizon.
                    - 'doy' (int): Initial day of the year.
                    - 'reward_coeff' (float): Coefficients for reward calculation.
                    - 'exponential_average_coeff' (float, optional): Coefficient for exponential averaging of demand. Defaults to 0.8.
                    - 'smooth_daily_deficit_coeff' (bool): Whether to smooth the daily deficit.
                    - 'lake_params' (dict): Parameters for the lake model.
                    - 'flood_level' (float): Flood level threshold.
                    - 'starving_level' (float): Starving level threshold.
                    - 'demand' (dict): Demand data as an ordered dictionary.
                    - 'inflow' (dict): Inflow data as an ordered dictionary.
                    - 'action' (dict): Action space configuration with keys:
                        - 'low' (float): Minimum action value.
                        - 'high' (float): Maximum action value.
                    - 'observations' (list): List of observation keys to include in the observation space.
                    - 'seed' (int): Random seed for reproducibility.
                    - 'random_init' (bool): Whether to initialize the environment randomly.

        Attributes:
            T (int): Simulation period.
            integ_step (float): Integration step size.
            H (int): Simulation horizon.
            init_day (int): Initial day of the year.
            reward_coeff (float): Coefficients for reward calculation.
            exponential_average_coeff (float): Coefficient for exponential averaging of demand.
            smooth_daily_deficit_coeff (float): Whether to smooth the daily deficit.
            lake (Lake): Lake model instance.
            flood_level (float): Flood level threshold.
            starving_level float: Starving level threshold.
            demand_data (OrderedDict): Demand data.
            inflow_data (OrderedDict): Inflow data.
            action_space (Box): Action space for the environment.
            obs_keys (list): List of observation keys.
            observation_space (Box): Observation space for the environment.
            _rng (random.Random): Random number generator instance.
            random_init (bool): Whether to initialize the environment randomly.
            current_step (int): Tracks the current step in the simulation.
            exponential_average_demand (float): Tracks the exponential average of demand.
            level (list): Stores the water levels of the lake at each step.
            storage (list): Stores the storage values of the lake at each step.
            release (list): Stores the release values of the lake at each step.
            doy (list): Tracks the day of the year for each step.
            actions (list): Stores the actions taken at each step.
            curr_year_data (str): The current year's data being used for demand and inflow.
            demand (list): The demand data for the current year.
            inflow (list): The inflow data for the current year.

        """

        super().__init__()
        self.T = settings['period']
        self.integ_step = settings['integration']
        self.H = settings['sim_horizon']
        self.init_day = settings['doy']
        self.reward_coeff = settings['reward_coeff']
        self.exponential_average_coeff = settings.get('exponential_average_coeff', 0.8)
        self.smooth_daily_deficit_coeff = settings['smooth_daily_deficit_coeff']

        # Model components
        self.lake = Lake(settings['lake_params'])

        self.flood_level = settings['flood_level']
        self.starving_level = settings['starving_level']
        self.demand_data = OrderedDict(settings['demand'])
        self.inflow_data = OrderedDict(settings['inflow'])
        self.demand = None
        self.inflow = None

        # Action space = single release decision
        self.action_space = Box(
            low=settings['action']['low'],
            high=settings['action']['high'],
            dtype=np.float32
        )

        low = []
        high = []


        self.obs_keys = []

        if 'level' in settings['observations']:
            self.obs_keys.append('level')
            low.append(-np.inf)
            high.append(np.inf)
        if 'day_of_year' in settings['observations']:
            self.obs_keys.extend(['sin_day_of_year', 'cos_day_of_year'])
            low.extend([-1., -1.])
            high.extend([1., 1.])
        if 'exponential_average_demand' in settings['observations']:
            self.obs_keys.append('exponential_average_demand')
            low.append(0.)
            high.append(np.inf)

        low = np.array(low)
        high = np.array(high)

        self.observation_space = Box(
            low=low, high=high, dtype=np.float32
        )

        self._rng = random.Random(settings['seed'])
        self.random_init = settings['random_init']

        self.curr_year_data = None

        self._init_internal_state()

    def _init_internal_state(self, seed=None):
        """
        Initializes the internal state of the environment.

        Args:
            seed (int, optional): A seed for random number generation. Defaults to None.

        Attributes:
            current_step (int): Tracks the current step in the simulation.
            exponential_average_demand (float): Tracks the exponential average of demand.
            level (list): Stores the water levels of the lake at each step.
            storage (list): Stores the storage values of the lake at each step.
            release (list): Stores the release values of the lake at each step.
            doy (list): Tracks the day of the year for each step.
            actions (list): Stores the actions taken at each step.
            curr_year_data (str): The current year's data being used for demand and inflow.
            demand (list): The demand data for the current year.
            inflow (list): The inflow data for the current year.

        Notes:
            - If `random_init` is True, a random year is selected for the simulation data.
            - If `random_init` is False, the year cycles through the available data years.
        """

        self.current_step = 0

        self.exponential_average_demand = 0.

        self.level = []
        self.storage = []
        self.release = []
        self.doy = []
        self.actions = []

        self.level.append(self.lake.init_level)
        self.storage.append(self.lake.level_to_storage(self.level[0]))
        self.release.append(0.)

        if self.random_init:
            if seed is None:
                rng = self._rng
            else:
                rng = random.Random(seed)
            self.curr_year_data = rng.choice(list(self.demand_data.keys()))
        else:
            data_years = list(self.demand_data.keys())
            if self.curr_year_data is None:
                self.curr_year_data = data_years[0]
            else:
                self.curr_year_data = data_years[(data_years.index(self.curr_year_data)+1)%len(data_years)]

        self.demand = self.demand_data[self.curr_year_data]
        self.inflow = self.inflow_data[self.curr_year_data]

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (int, optional): A seed value for random number generation to ensure reproducibility. Defaults to None.
            options (dict, optional): A dictionary of options for resetting the environment. If the key 
                'rewind_profiles' is set to True, the current year data will be rewinded to the beginning, defaults to False.

        Returns:
            tuple: A tuple containing:
                - observation: The initial observation of the environment.
                - info (dict): A dictionary containing additional information, including the key 
                  'exponential_average_demand' initialized to 0.
        """

        if options is not None and options.get('rewind_profiles', False):
            self.curr_year_data = None
        self._init_internal_state(seed=seed)

        info = {'exponential_average_demand': self.exponential_average_demand,
                'storage': self.storage[-1],
                'level': self.level[-1]}

        return self._get_observation(), info

    def step(self, action):
        """
        Executes a single step in the environment using the provided action.

        Args:
            action (float): The action to be taken, which is clipped to the valid range
                            defined by the action space.

        Returns:
            tuple:
                - observation (object): The current observation of the environment after the step.
                - reward (float): The total reward obtained from taking the action.
                - done (bool): Always False, as this environment does not use the 'done' flag.
                - truncated (bool): Whether the episode has reached its maximum step limit.
                - info (dict): Additional information about the environment state, including:
                    - 'flood' (float): Indicator of whether the lake level exceeds the flood level.
                    - 'deficit' (float): The daily water deficit.
                    - 'reward' (float): The total reward for the step.
                    - 'pure_reward' (float): The unweighted reward dictionary for the step.
                    - 'weighted_reward' (float): The weighted reward dictionary for the step.
                    - 'storage' (float): The current storage level of the lake.
                    - 'release' (float): The water release for the step.
                    - 'action' (float): The action taken for the step.
                    - 'level' (float): The current water level of the lake.
                    - 'demand' (float): The water demand for the current day.
                    - 'exponential_average_demand' (float): The current exponentially averaged water demand.
        """

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high).item()
        self.actions.append(clipped_action)

        t = self.current_step

        # Day of year
        self.doy.append((self.init_day + t - 1) % self.T + 1)

        demand = self.demand[int(self.doy[t]) - 1]
        self.exponential_average_demand = self.exponential_average_coeff * self.exponential_average_demand + (1 - self.exponential_average_coeff) * demand

        inflow = self.get_inflow(t)

        new_storage, new_release = self.lake.integration(
            self.integ_step, self.storage[t], clipped_action, inflow, self.doy[t]
        )

        self.storage.append(new_storage)
        self.release.append(new_release)

        self.level.append(self.lake.storage_to_level(new_storage))

        # Compute reward
        tot_reward, pure_reward, weighted_reward = self._calculate_reward(t)

        # Update step
        self.current_step += 1

        truncated = self.current_step >= self.H

        info = {
            'flood': float(self.level[-1] > self.flood_level),
            'deficit': self._daily_deficit(t),
            'reward': tot_reward,
            'pure_reward': pure_reward,
            'weighted_reward': weighted_reward,
            'storage': new_storage,
            'release': new_release,
            'action': action,
            'level': self.level[-1],
            'demand': demand,
            'exponential_average_demand': self.exponential_average_demand
        }

        return self._get_observation(), tot_reward, False, truncated, info

    def _get_observation(self):
        """
        Generate the current observation for the environment.

        This method computes the observation based on the current step and 
        the specified observation keys.

        Returns:
            np.ndarray: A NumPy array containing the observation values 
            as specified by `self.obs_keys`. The array is of type `np.float32`.

        Possible observation Keys:
            - 'level': The water level corresponding to the current day of the year.
            - 'sin_day_of_year': The sine of the day of the year.
            - 'cos_day_of_year': The cosine of the day of the year.
            - 'exponential_average_demand': The exponential moving average of the demand.
        """

        t = self.current_step
        doy = (self.init_day + t - 1) % self.T #+ 1
        obs = []

        for key in self.obs_keys:
            match key:
                case 'level':
                    obs.append(self.level[doy])
                case 'sin_day_of_year':
                    obs.append(sin(2 * pi * doy / self.T))
                case 'cos_day_of_year':
                    obs.append(cos(2 * pi * doy / self.T))
                case 'exponential_average_demand':
                    obs.append(self.exponential_average_demand)

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, t):
        """
        Calculate the reward for a given time step based on various penalties and coefficients.

        Args:
            t (int): The current time step index.

        Returns:
            tuple:
                - tot_reward (float): The total weighted reward calculated as the sum of all weighted penalties.
                - pure_reward (dict): A dictionary containing the unweighted individual penalties:
                    - 'overflow_reward' (float): Penalty for water level exceeding the flood level.
                    - 'starving_reward' (float): Penalty for water level dropping below the starving level.
                    - 'daily_deficit_reward' (float): Penalty for daily water deficit.
                    - 'wasted_water_reward' (float): Penalty for water released when not needed.
                    - 'clipping_reward' (float): Penalty for the difference between action and actual release.
                - weighted_reward (dict): A dictionary containing the weighted individual penalties:
                    - 'overflow_reward' (float): Weighted penalty for water level exceeding the flood level.
                    - 'starving_reward' (float): Weighted penalty for water level dropping below the starving level.
                    - 'daily_deficit_reward' (float): Weighted penalty for daily water deficit.
                    - 'wasted_water_reward' (float): Weighted penalty for water released when not needed.
                    - 'clipping_reward' (float): Weighted penalty for the difference between action and actual release.

        Notes:
            - The penalties are calculated based on the current state of the environment, including water levels,
              release actions, and predefined coefficients for each penalty type.
            - The reward coefficients are accessed from `self.reward_coeff` and are used to weight the penalties.
        """

        action = self.actions[t]
        release = self.release[t+1]

        # Overflow penalty
        overflow_reward = - int(self.level[t+1] > self.flood_level)

        # Starving penalty
        starving_reward = - int(self.level[t+1] < self.starving_level)

        # Deficit penalty
        daily_deficit_reward = self._daily_deficit(t)
        daily_deficit_reward = - daily_deficit_reward

        assert release == self.release[t+1]

        # Wasted water penalty
        wasted_water_reward = - self._wasted_water(t)

        # Clipping penalty
        clipping_reward = - (action - release) ** 2

        pure_reward = {'overflow_reward': overflow_reward,
                       'starving_reward': starving_reward,
                       'daily_deficit_reward': daily_deficit_reward,
                       'wasted_water_reward': wasted_water_reward,
                       'clipping_reward': clipping_reward}

        weighted_reward = {'overflow_reward': overflow_reward * self.reward_coeff['overflow_coeff'],
                           'starving_reward': starving_reward * self.reward_coeff['starving_coeff'],
                           'daily_deficit_reward': daily_deficit_reward * self.reward_coeff['daily_deficit_coeff'],
                           'wasted_water_reward': wasted_water_reward * self.reward_coeff['wasted_water_coeff'],
                           'clipping_reward': clipping_reward * self.reward_coeff['clip_action_coeff']}

        tot_reward = 0.
        for r in weighted_reward.values():
            tot_reward += r

        return tot_reward, pure_reward, weighted_reward

    def _daily_deficit(self, t):
        """
        Calculate the daily water deficit based on release, demand, and day of the year.

        Args:
            t (int): The current time step index.

        Returns:
            float: The calculated daily water deficit.

        Notes:
            - The deficit is calculated as the difference between the demand for the day
              of the year and the excess release above the minimum flow (MEF).
            - If the `smooth_daily_deficit_coeff` is enabled, the deficit is scaled using
              a cosine-based smoothing factor depending on the day of the year.
            - Otherwise, during the period between days 121 and 243 (inclusive), the deficit
              is doubled to account for seasonal variations.
        """

        doy = int(self.doy[t]) - 1

        qdiv = self.release[t+1] - self.lake.get_mef(doy)
        qdiv = max(qdiv, 0.0)
        d = max(self.demand[doy] - qdiv, 0.0)

        # scale depending on day of the year
        if self.smooth_daily_deficit_coeff:
            d *= 0.5 * (3 - math.cos(doy * 2*math.pi/self.DAYS_IN_YEAR))
        else:
            if 120 < self.doy[t] <= 243:
                d *= 2

        return d

    def _wasted_water(self, t):
        """
        Calculate the amount of wasted water for a given time step.

        Wasted water is defined as the excess water released beyond the demand
        for a specific day of the year.

        Args:
            t (int): The current time step index.

        Returns:
            float: The amount of wasted water. Returns 0 if the release does not
            exceed the demand.
        """

        doy = int(self.doy[t]) - 1
        demand = self.demand[doy]
        release = self.release[t+1]

        wasted = max(release - demand, 0.)
        return wasted

    def render(self, mode='human'):
        """
        Renders the current state of the environment.
        Args:
            mode (str): The mode in which to render the environment. Defaults to 'human'.
        Prints:
            A message displaying the current day and the corresponding level value
            formatted to two decimal places.
        """

        print(f'Day {self.current_step}, Level: {self.level[self.current_step]:.2f}')

    def close(self):
        """
        Clean up resources associated with the lake environment.

        This method is called to release any resources or perform any cleanup
        operations when the environment is no longer needed. It sets the `lake`
        attribute to `None` to indicate that the environment has been closed.
        """

        self.lake = None

    def get_inflow(self, pt):
        return self.inflow[pt]

