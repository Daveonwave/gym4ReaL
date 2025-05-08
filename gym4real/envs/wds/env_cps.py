import pandas as pd
import numpy as np
import random
import yaml
from pathlib import Path
from gymnasium import Env, spaces
from gym4real.envs.wds.simulator.plc import SensorPLC, ActuatorPLC
from gym4real.envs.wds.simulator.wn import WaterNetwork
from gym4real.envs.wds.simulator.demand import WaterDemandPattern
from gym4real.envs.wds.simulator.attacker import AttackScheduler
from gym4real.envs.wds.rewards import *


class WaterDistributionSystemEnv(Env):
    """
    Custom Environment that follows gym interface for the Water Distribution System.
    This environment is the Cyber-Physical System (CPS) version of the Water Distribution System.
    This version of the environment is designed to be used with PLCs.
    It includes the simulation of the water distribution system, demand patterns,
    and the ability to handle attacks on sensors and actuators.
    The environment provides a discrete action space for controlling pumps and
    a continuous observation space for monitoring tank levels and junction pressures.
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
        self._seed = settings['seed']
        np.random.seed(self._seed)

        self._use_attacks = False

        # Physical process of the environment
        self._wn = WaterNetwork(settings['inp_file'])
        self._wn.set_time_params(duration=settings['duration'], hydraulic_step=settings['hyd_step'], pattern_step=settings['demand']['pattern_step'])
        
        # Demand patterns
        self._demand = WaterDemandPattern(**settings['demand'])
        # Attackers
        if settings['under_attack']:
            self._attack_scheduler = AttackScheduler(**settings['attackers'])
        # PLCs creation
        self._sensor_plcs = [SensorPLC(name=sensor['name'],
                                       wn=self._wn,
                                       plc_variables=sensor['vars'])
                            for sensor in settings['plcs'] if sensor['type'] == 'sensor']
        self._actuator_plcs = [ActuatorPLC(name=sensor['name'],
                                           wn=self._wn,
                                           plc_variables=sensor['vars'],
                                           n_actions=2 ** len(settings['actions']))
                              for sensor in settings['plcs'] if sensor['type'] == 'actuator']
        
        # Reward weights
        self.dsr_coeff = settings['reward']['dsr_coeff']
        self.overflow_coeff = settings['reward']['overflow_coeff']
        
        self._rewards = {'dsr': 0, 'overflow': 0}

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
        self.action_space = spaces.Discrete(2 ** len(self._wn.pumps))

    def _get_obs(self, readings):
        """
        Returns the current observation
        :return:
        """
        """
        Build current state list, which can be used as input of the nn saved_models
        :param readings:
        :param reset:
        :return:
        """
        obs = {}

        for key in self._obs_keys:
            match key:
                case key if key.startswith('T'):
                    obs[key] = readings[key]['pressure'] if key in readings else 0
                
                case key if key.startswith('J'):
                    obs[key] = readings[key]['pressure'] if key in readings else 0

                case 'demand_SMA':
                    obs['demand_SMA'] = self._demand.moving_average[self.elapsed_time // self._demand._pattern_step]
                
                case 'demand_EWMA':
                    obs['demand_EWMA'] = self._demand.exp_moving_average[self.elapsed_time // self._demand._pattern_step]

                case 'seconds_of_day':
                    sin_day = np.sin(2 * np.pi / self.SECONDS_PER_DAY * self.elapsed_time)
                    cos_day = np.cos(2 * np.pi / self.SECONDS_PER_DAY * self.elapsed_time)
                    obs['sin_seconds_of_day'] = sin_day
                    obs['cos_seconds_of_day'] = cos_day
                    
                case 'under_attack':
                    obs['under_attack'] = readings['under_attack'] if key in readings else 0

                case _:
                    raise KeyError(f'Unknown observation variable: {key}')
        return obs
    
    def _get_info(self):
        """
        Returns the current observation
        :return:
        """
        info = {
            'elapsed_time': self.elapsed_time,
            'timestep': self.timestep,
            'demand_profile': self._demand.pattern,
            'demand_pattern_type': self._demand.pattern_type,
            'reward_components': self._rewards,
       }
        return info
    
    def reset(self, seed=None, options=None):
        """
        Called at the beginning of each episode
        :param state:
        :return:
        """
        self._wn.reset()
        self._wn.solved = False

        self.elapsed_time = 0
        self.timestep = 1
        self.readings = {}

        self.attackers = []
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset if testing
        if options is not None and 'is_evaluation' in options and options['is_evaluation']:
            self._demand.draw_pattern(is_evaluation=True)            
            self._wn.set_demand_pattern('junc_demand', self._demand.pattern, self._wn.junctions)
            if self._use_attacks:
                self._attack_scheduler.draw_attacks(is_evaluation=True)
        # Reset if training
        else:
            self._demand.draw_pattern()
            self._wn.set_demand_pattern('junc_demand', self._demand.pattern, self._wn.junctions)
            if self._use_attacks:
                self._attack_scheduler.draw_attacks()
                
        if 'demand_SMA' in self._obs_keys:
            self._demand.set_moving_average(window_size=6)
        if 'demand_EWMA' in self._obs_keys:
            self._demand.set_exp_moving_average(window_size=6)

        # Set attacks relative to PLCs
        if self._use_attacks:
            for sensor in self._sensor_plcs:
                sensor.set_attackers([attacker for attacker in self._attack_scheduler.attacks 
                                    if attacker.target == sensor.name])
            for actuator in self._actuator_plcs:
                actuator.set_attackers([attacker for attacker in self._attack_scheduler.attacks 
                                        if attacker.target == actuator.name])

        self._wn.init_simulation()
        
        readings = {tank: {'pressure':sensor.init_readings(tank, prop='pressure')} for sensor in self._sensor_plcs 
                    for tank in sensor.variables if tank.startswith('T')}
        state = np.array(list(self._get_obs(readings=readings).values()), dtype=np.float32)
        
        info = self._get_info()
        return state, info

    def step(self, action):
        """
        Execute one time step within the environment
        """
        pump_updates = {}
        readings = {}
        non_altered_readings = {}

        # Actuator PLCs apply the action into the physical process
        for plc in self._actuator_plcs:
            # Return a dict with 1 if the actuator has been updated, 0 otherwise
            pump_updates.update(plc.apply(action, self.elapsed_time))
            #self.total_updates += sum(pump_updates.values())

        # Simulate the next hydraulic step
        self.timestep = self._wn.simulate_step(self.elapsed_time)

        # Pass information to sensor PLCs #TODO: do not update under_attack if 1
        for sensor in self._sensor_plcs:
            applied = sensor.apply()
            readings = readings | applied[0]
            non_altered_readings = non_altered_readings | applied[1]
                
        # Retrieve current state and reward from the chosen action
        reward = self._compute_reward(readings=readings)

        terminated = False
        truncated = self.timestep == 0
        self._wn.solved = self.timestep == 0
        
        self.elapsed_time += self.timestep
        
        state = np.array(list(self._get_obs(readings=readings).values()), dtype=np.float32)
        info = self._get_info()

        return state, reward, terminated, truncated, info  
    
    def _compute_reward(self, readings):
        """
        Compute the reward for the current step. It depends on the step_DSR

        :param step_pump_updates:
        :return:
        """
        reward = 0
        self._rewards['dsr'] = dsr(readings) * self.dsr_coeff
        self._rewards['overflow'] = -overflow(readings) * self.overflow_coeff
        
        reward = self._rewards['dsr'] + self._rewards['overflow']
        return reward

