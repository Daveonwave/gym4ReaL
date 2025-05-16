import pandas as pd
import numpy as np
import random
from pathlib import Path
from gymnasium import Env, spaces
from gym4real.envs.wds.simulator.wn import WaterNetwork
from gym4real.envs.wds.simulator.demand import WaterDemandPattern
from gym4real.envs.wds.rewards import *


class WaterDistributionSystemEnv(Env):
    """
    Custom Environment that follows gym interface for the Water Distribution System.
    This environment is a simplified version of the environment, which does not include the PLCs and the attacks.
    """
    metadata = {"render_modes": []}
    SECONDS_PER_DAY = 3600 * 24
    DAYS_PER_WEEK = 7

    def __init__(self,
                 settings: dict,
                 ):
        super().__init__()
        
        # Simulation variables
        self.elapsed_time = None
        self.timestep = None
        self._duration = settings['duration']
        self._seed = settings['seed']
        np.random.seed(self._seed)

        self.demand_moving_average = None
        self._use_attacks = False

        # Physical process of the environment
        self._wn = WaterNetwork(settings['inp_file'])
        self._wn.set_time_params(duration=settings['duration'], 
                                 hydraulic_step=settings['hyd_step'], 
                                 pattern_step=settings['demand']['pattern_step'])
        
        # Demand patterns
        self._demand = WaterDemandPattern(**settings['demand'])
        
        # Reward components
        self._dsr = 0
        self._overflow = 0
        
        # Reward weights
        self.dsr_coeff = settings['reward']['dsr_coeff']
        self.overflow_coeff = settings['reward']['overflow_coeff']
        
        self._obs_keys = []
        obs_space = {}
        for key in settings['observations']:
            if key.startswith('T'):
                obs_space[key] = {'low': 0., 'high': self._wn.tanks[key].maxlevel}
                self._obs_keys.append(key)
            if key.startswith('J'):
                obs_space[key] = {'low': 0., 'high': np.inf}
                self._obs_keys.append(key)
                    
        # Add optional 'Demand Moving Average' in observation space
        if settings['demand_moving_average']:
            self._obs_keys.append('demand_SMA')
            obs_space['demand_SMA'] = {'low': 0., 'high': np.inf}
            
        # Add optional 'Demand Exponential Weighted Moving Average' in observation space
        if settings['demand_exp_moving_average']:
            self._obs_keys.append('demand_EWMA')
            obs_space['demand_EWMA'] = {'low': 0., 'high': np.inf}

        if settings['seconds_of_day']:
            self._obs_keys.append('seconds_of_day')
            obs_space['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            obs_space['cos_seconds_of_day'] = {'low': -1, 'high': 1}
        
        # Add optional 'Under attack' in observation space
        if settings['under_attack']:
            self._use_attacks = True
            self._obs_keys.append('under_attack')
            obs_space['under_attack'] = {'low': 0., 'high': 1.}

        lows = [obs_space[key]['low'] for key in obs_space.keys()]
        highs = [obs_space[key]['high'] for key in obs_space.keys()]

        # Observation space
        self.observation_space = spaces.Box(low=np.array(lows), high=np.array(highs), shape=(len(lows),), dtype=np.float32)
        # Two possible values for each pump: 2 ^ n_pumps
        #   -> 3 ACTIONS because pumps are overlapped
        self.action_space = spaces.Discrete(2 ** len(self._wn.pumps))

    def _get_obs(self):
        """
        Returns the current observation
        :return:
        """
        """
        Build current state list, which can be used as input of the nn saved_models
        :param readings:
        :return:
        """
        obs = {}

        for key in self._obs_keys:
            match key:
                case key if key.startswith('T'):
                    obs[key] = self._wn.nodes[key].pressure.iloc[-1] if self.elapsed_time > 0 else 0
                    
                case key if key.startswith('J'):
                    obs[key] = self._wn.nodes[key].pressure.iloc[-1] if self.elapsed_time > 0 else 0

                case 'demand_SMA':
                    obs['demand_SMA'] = self._demand.moving_average[(self.elapsed_time // self._demand._pattern_step) % len(self._demand.pattern)]
                
                case 'demand_EWMA':
                    obs['demand_EWMA'] = self._demand.exp_moving_average[(self.elapsed_time // self._demand._pattern_step) % len(self._demand.pattern)]

                case 'seconds_of_day':
                    sin_day = np.sin(2 * np.pi / self.SECONDS_PER_DAY * self.elapsed_time)
                    cos_day = np.cos(2 * np.pi / self.SECONDS_PER_DAY * self.elapsed_time)
                    obs['sin_seconds_of_day'] = sin_day
                    obs['cos_seconds_of_day'] = cos_day

                case _:
                    raise KeyError(f'Unknown observation variable: {key}')
                
        #print(f"Current observation: {obs}")
        return obs
    
    def _get_info(self):
        """
        Returns the current observation
        :return:
        """
        return {'profile': self._demand.pattern,
                'elapsed_time': self.elapsed_time,
                'pure_rewards': {'dsr': self._dsr, 'overflow': self._overflow},
                'weighted_rewards': {'dsr': self._dsr * self.dsr_coeff, 'overflow': self._overflow * self.overflow_coeff}}
    
    def reset(self, seed=None, options=None):
        """
        Called at the beginning of each episode
        :param state:
        :return:
        """
        print("Resetting the environment...")
        self._wn.reset()
        self._wn.solved = False
        
        self._dsr = 0
        self._overflow = 0

        self.elapsed_time = 0
        self.timestep = 1
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset if testing
        if options is not None and 'is_evaluation' in options and options['is_evaluation']:
            self._demand.draw_pattern(is_evaluation=True)            
            self._wn.set_demand_pattern('junc_demand', self._demand.pattern, self._wn.junctions)
        # Reset if training
        else:
            self._demand.draw_pattern()
            self._wn.set_demand_pattern('junc_demand', self._demand.pattern, self._wn.junctions)

        if 'demand_SMA' in self._obs_keys:
            self._demand.set_moving_average(window_size=6, total_basedemand=sum([junc.basedemand for junc in self._wn.junctions.values()]))
        if 'demand_EWMA' in self._obs_keys:
            self._demand.set_exp_moving_average(window_size=6, total_basedemand=sum([junc.basedemand for junc in self._wn.junctions.values()]))
        
        self._wn.init_simulation()
        
        state = np.array(list(self._get_obs().values()), dtype=np.float32)
        
        info = self._get_info()
        return state, info

    def step(self, action):
        """
        Execute one time step within the environment
        """
        pump_actuation = {pump_id: 0 for pump_id in self._wn.pumps.keys()}
        bin_action = '{0:0{width}b}'.format(action, width=int(np.log2(self.action_space.n)))
                
        for i, key in enumerate(pump_actuation.keys()):
            pump_actuation[key] = int(bin_action[i])
        self._wn.update_pumps(new_status=pump_actuation)

        # Simulate the next hydraulic step
        self.timestep = self._wn.simulate_step(self.elapsed_time)
                
        # Retrieve current state and reward from the chosen action
        reward = self._compute_reward()

        terminated = False
        truncated = self.timestep == 0
        self._wn.solved = self.timestep == 0
        self.elapsed_time += self.timestep
        
        state = np.array(list(self._get_obs().values()), dtype=np.float32)
        info = self._get_info()

        return state, reward, terminated, truncated, info  
    
    def _compute_reward(self):
        """
        Compute the reward for the current step. It depends on the step_DSR

        :param step_pump_updates:
        :return:
        """
        state = self._wn.get_snapshot((self.elapsed_time // self._demand._pattern_step) % len(self._demand.pattern))
        
        reward = 0
        self._dsr = dsr(state)
        self._overflow = overflow(state)
        reward += self._dsr * self.dsr_coeff
        reward -= self._overflow * self.overflow_coeff
        
        return reward

