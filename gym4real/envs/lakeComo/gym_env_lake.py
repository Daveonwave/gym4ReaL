import math

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from math import sin, cos, pi
from lakecomo import LakeComo

class LakeEnv(Env):

    DAYS_IN_YEAR = 365

    def __init__(self, settings):
        super().__init__()
        self.T = settings['period']
        self.integ_step = settings['integration']
        self.H = settings['sim_horizon']
        self.warmup = settings['warmup']
        self.init_day = settings['doy']
        self.inflow = settings['inflow']
        self.obs_keys = settings['observations']
        self.reward_coeff = settings['reward_coeff']

        # Model components
        self.lake = LakeComo(settings['lake_params'])

        self.flood_level = settings['flood_level']
        self.demand = settings['demand']
        self.qForecast = settings['q_forecast']

        # Action space = single release decision
        self.action_space = Box(
            low=settings['action']['low'],
            high=settings['action']['high'],
            dtype=np.float32
        )

        low = []
        high = []

        if 'level' in self.obs_keys:
            low.append(0.)
            high.append(np.inf)
        if 'day_of_year' in self.obs_keys:
            low.extend([0., 0.])
            high.extend([1., 1.])
        if 'q_forecast' in self.obs_keys:
            low.append(-np.inf)
            high.append(np.inf)

        low = np.array(low)
        high = np.array(high)

        self.observation_space = Box(
            low=low, high=high, dtype=np.float32
        )

        self._init_internal_state()

    def _init_internal_state(self):
        self.current_step = 0

        self.level = []
        self.storage = []
        self.release = []
        self.doy = []
        self.actions = []

        self.level.append(self.lake.init_level)
        self.storage.append(self.lake.level_to_storage(self.level[0]))
        self.release.append(0.)

    def reset(self, seed=None, options=None):
        self._init_internal_state()
        return self._get_observation(), {}

    def step(self, action):

        action = action.item()
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high).item()
        self.actions.append(clipped_action)

        t = self.current_step

        # Day of year
        self.doy.append((self.init_day + t - 1) % self.T + 1)

        inflow = self.get_inflow(t)

        new_storage, new_release = self.lake.integration(
            self.integ_step, t, self.storage[t], clipped_action, inflow, self.doy[t]
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
            'demand': self.demand[int(self.doy[t]) - 1]
        }

        return self._get_observation(), tot_reward, False, truncated, info

    def _get_observation(self):
        t = self.current_step
        doy = (self.init_day + t - 1) % self.T #+ 1
        obs = []

        if 'level' in self.obs_keys:
            obs.append(self.level[t])
        if 'day_of_year' in self.obs_keys:
            obs.append(sin(2 * pi * doy / self.T))
            obs.append(cos(2 * pi * doy / self.T))
        if 'q_forecast' in self.obs_keys:
            obs.append(self.qForecast[doy])

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, t):
        # FIXME redefine the reward function?
        """Negative cost: combination of flood + deficit + wasted water + clip"""

        action = self.actions[t]
        release = self.release[t+1]

        # Overflow penalty
        overflow_reward = - int(self.level[t+1] > self.flood_level)

        # Deficit penalty
        daily_deficit_reward = self._daily_deficit(t)
        daily_deficit_reward = math.sqrt(daily_deficit_reward)
        daily_deficit_reward = - daily_deficit_reward

        assert release == self.release[t+1]

        # Wasted water penalty
        wasted_water_reward = - self._wasted_water(t)

        # Clipping penalty
        clipping_reward = - (action - release) ** 2

        pure_reward = {'overflow_reward': overflow_reward,
                       'daily_deficit_reward': daily_deficit_reward,
                       'wasted_water_reward': wasted_water_reward,
                       'clipping_reward': clipping_reward}

        weighted_reward = {'overflow_reward': overflow_reward * self.reward_coeff['overflow_coeff'],
                           'daily_deficit_reward': daily_deficit_reward * self.reward_coeff['daily_deficit_coeff'],
                           'wasted_water_reward': wasted_water_reward * self.reward_coeff['wasted_water_coeff'],
                           'clipping_reward': clipping_reward * self.reward_coeff['clip_action_coeff']}

        tot_reward = 0.
        for r in weighted_reward.values():
            tot_reward += r

        return tot_reward, pure_reward, weighted_reward

    def _daily_deficit(self, t):
        doy = int(self.doy[t]) - 1
        # FIXME
        # why is there here a t+1?
        qdiv = self.release[t+1] - self.lake.get_mef(doy)
        qdiv = max(qdiv, 0.0)
        d = max(self.demand[doy] - qdiv, 0.0)
        d_pure = d
        if 120 < self.doy[t] <= 243:
            d *= 2
        # d *= 0.5 * (3 - math.cos(doy * math.pi/self.DAYS_IN_YEAR))

        return d**2

    def _wasted_water(self, t):
        doy = int(self.doy[t]) - 1
        demand = self.demand[doy]
        release = self.release[t+1]

        wasted = max(release - demand, 0.)
        return wasted

    def render(self, mode='human'):
        print(f'Day {self.current_step}, Level: {self.level[self.current_step]:.2f}')

    def close(self):
        self.lake = None

    def get_inflow(self, pt):
        """
        Retrieve inflow for simulation point ps and time step pt.
        """
        return self.inflow[pt]


def from_str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None

