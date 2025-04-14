import os

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from math import sin, cos, pi
from utils import Utils
# from catchment import Catchment
from lakecomo import LakeComo

class LakeComoEnv(Env):
    def __init__(self, settings):
        super().__init__()

        self.Nsim = settings['num_sim']
        self.NN = settings['dim_ensemble']
        self.T = settings['period']
        self.integ_step = settings['integration']
        self.H = settings['sim_horizon']
        self.Nobj = settings['num_objs']
        self.Nvar = settings['num_vars']
        self.warmup = settings['warmup']
        self.init_day = settings['doy']
        self.inflow = settings['inflow']
        self.obs_keys = settings['observations']

        # Model components
        self.lake_como = LakeComo(settings['lake_params'])

        # min_input = self.p_param["mIn"]
        # max_input = self.p_param["MIn"]
        # min_output = self.p_param["mOut"]
        # max_output = self.p_param["MOut"]

        self.flood_level = settings['flood_level']
        self.demand = settings['demand']
        self.qForecast = settings['q_forecast']

        # Action space = single release decision
        # FIXME these bounds should be modified
        self.action_space = Box(
            low=0.,
            high=2.,
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

        self.level.append(self.lake_como.get_init_cond())
        self.storage.append(self.lake_como.level_to_storage(self.level[0]))
        self.release.append(0.)

    def reset(self, seed=None, options=None):
        self._init_internal_state()
        return self._get_observation(), {}

    def step(self, action):
        t = self.current_step
        ps = 0

        # if self.done:
        #     raise RuntimeError("Episode is done. Call reset().")

        # Day of year
        self.doy.append((self.init_day + t - 1) % self.T + 1)

        # Inflow + integration
        # qIn = self.ComoCatchment.get_inflow(t, ps)
        inflow = self.get_inflow(t, ps)

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

        self.actions.append(clipped_action)
        # self.u[t] = np.clip(action[0], *self.action_space.bounds)

        new_storage, new_release = self.lake_como.integration(
            self.integ_step, t, self.storage[t], clipped_action, inflow, self.doy[t], ps
        )

        self.storage.append(new_storage)
        self.release.append(new_release)

        self.level.append(self.lake_como.storage_to_level(new_storage))

        # Compute reward
        reward = self._calculate_reward(t)

        # Update step
        self.current_step += 1
        truncated = self.current_step >= self.H
        info = {
            "flood": float(self.level[-1] > self.flood_level),
            "deficit": self._daily_deficit(t)
        }

        return self._get_observation(), reward, False, truncated, info

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
        """Negative cost: combination of flood + deficit"""
        penalty = 0.0

        if self.level[t + 1] > self.flood_level:
            penalty += 1.0

        # Deficit penalty
        penalty += self._daily_deficit(t)

        return -penalty

    def _daily_deficit(self, t):
        doy = int(self.doy[t]) - 1
        # FIXME
        # why is there here a t+1?
        qdiv = self.release[t + 1] - self.lake_como.get_mef(doy)
        qdiv = max(qdiv, 0.0)
        d = max(self.demand[doy] - qdiv, 0.0)
        if 120 < self.doy[t] <= 243:
            d *= 2
        return d**2

    def render(self, mode='human'):
        print(f"Day {self.current_step}, Level: {self.level[self.current_step]:.2f}")

    def close(self):
        # self.ComoCatchment = None
        self.lake_como = None
        # self.mPolicy = None

    def get_inflow(self, pt, ps):
        """
        Retrieve inflow for simulation point ps and time step pt.
        """
        return self.inflow[pt]
        # return self.inflow[ps][pt]    #fixme why ps?


def from_str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None

