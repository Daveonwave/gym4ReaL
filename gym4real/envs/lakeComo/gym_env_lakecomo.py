import os

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from math import sin, cos, pi
from utils import Utils
from catchment import Catchment
from lakecomo import LakeComo

class LakeComoEnv(Env):
    def __init__(self, filename):
        super().__init__()
        self.read_file_settings(filename)

        # Model components
        self.ComoCatchment = Catchment(self.Como_catch_param)
        self.LakeComo = LakeComo()
        self.LakeComo.set_evap(0)
        self.LakeComo.set_mef({
            "filename": "../data/MEF_como.txt",
            "row": self.T
        })
        self.LakeComo.set_surface(145900000)
        self.LakeComo.set_init_cond(self.Como_param["initCond"])

        # min_input = self.p_param["mIn"]
        # max_input = self.p_param["MIn"]
        # min_output = self.p_param["mOut"]
        # max_output = self.p_param["MOut"]

        self.h_flo = 1.24
        self.demand = Utils.load_vector("../data/comoDemand.txt", 365)
        self.qForecast = Utils.load_vector("../data/qSimAnomL51.txt", self.H)

        # Action space = single release decision
        # FIXME these bounds should be modified
        self.action_space = Box(
            low=0.,
            high=2.,
            # low=np.array(0),
            # high=np.array(2),
            dtype=np.float32
        )

        # Observation = sin/cos(doy), lake level, forecast
        # TODO see whether we need to modify this
        obs_dim = 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._init_internal_state()

    def _init_internal_state(self):
        self.current_step = 0
        self.done = False

        self.h = np.full(self.H + 1, -999.0)
        self.s = np.full(self.H + 1, -999.0)
        self.r = np.full(self.H + 1, -999.0)
        self.doy = np.full(self.H, -999, dtype=int)
        self.u = np.full(self.H, -999.0)

        self.h[0] = self.LakeComo.get_init_cond()
        self.s[0] = self.LakeComo.level_to_storage(self.h[0])

    def reset(self, seed=None, options=None):
        self._init_internal_state()
        return self._get_observation()

    def step(self, action):
        t = self.current_step
        ps = 0

        # if self.done:
        #     raise RuntimeError("Episode is done. Call reset().")

        # Day of year
        self.doy[t] = (self.initDay + t - 1) % self.T + 1

        # Inflow + integration
        qIn = self.ComoCatchment.get_inflow(t, ps)

        self.u[t] = np.clip(action, self.action_space.low, self.action_space.high)
        # self.u[t] = np.clip(action[0], *self.action_space.bounds)

        self.s[t + 1], self.r[t + 1] = self.LakeComo.integration(
            self.integStep, t, self.s[t], self.u[t], qIn, self.doy[t], ps
        )
        self.h[t + 1] = self.LakeComo.storage_to_level(self.s[t + 1])

        # Compute reward
        reward = self._calculate_reward(t)

        # Update step
        self.current_step += 1
        self.done = self.current_step >= self.H

        return self._get_observation(), reward, self.done, {
            "flood": float(self.h[t + 1] > self.h_flo),
            "deficit": self._daily_deficit(t)
        }

    def _get_observation(self):
        t = self.current_step
        doy = (self.initDay + t - 1) % self.T + 1
        h_t = self.h[t]
        obs = [
            sin(2 * pi * doy / self.T),
            cos(2 * pi * doy / self.T),
            h_t
        ]
        # Commented for the moment since it goes beyond the boundary when t > 365
        # if self.p_param["policyInput"] > 3:
        # obs.append(self.qForecast[t])
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, t):
        # FIXME redefine the reward function?
        """Negative cost: combination of flood + deficit"""
        penalty = 0.0

        if self.h[t + 1] > self.h_flo:
            penalty += 1.0

        # Deficit penalty
        penalty += self._daily_deficit(t)

        return -penalty

    def _daily_deficit(self, t):
        doy = int(self.doy[t]) - 1
        # FIXME
        # why is there here a t+1?
        qdiv = self.r[t + 1] - self.LakeComo.get_mef(doy)
        qdiv = max(qdiv, 0.0)
        d = max(self.demand[doy] - qdiv, 0.0)
        if 120 < self.doy[t] <= 243:
            d *= 2
        return d * d

    def render(self, mode='human'):
        print(f"Day {self.current_step}, Level: {self.h[self.current_step]:.2f}")

    def close(self):
        self.ComoCatchment = None
        self.LakeComo = None
        self.mPolicy = None


    def read_file_settings(self, filename):
        self.Como_catch_param = {"inflow_file": {}}
        self.Como_param = {"minEnvFlow": {}}

        print(os.getcwd())
        filename = os.path.join(os.getcwd(), filename)
        with open(filename, 'r') as f:
            lines = f.readlines()

        def find_and_read(key, return_line=False):
            for i, line in enumerate(lines):
                print(line.strip())
                if line.strip().startswith(key):
                    if return_line:
                        return i
                    print(line.split()[1])
                    if from_str_to_num(line.split()[1]) != None:
                        value = from_str_to_num(line.split()[1])
                    else:
                        raise ValueError(f"Invalid value for {key}: {line.split()[1]}")
                    return value

        self.Nsim = int(find_and_read("<NUM_SIM>"))
        self.NN = int(find_and_read("<DIM_ENSEMBLE>"))
        self.T = int(find_and_read("<PERIOD>"))
        self.integStep = int(find_and_read("<INTEGRATION>"))
        self.H = int(find_and_read("<SIM_HORIZON>"))
        self.Nobj = int(find_and_read("<NUM_OBJ>"))
        self.Nvar = int(find_and_read("<NUM_VAR>"))
        self.warmup = int(find_and_read("<WARMUP>"))

        initDay_or_file = find_and_read("<DOY>", False)

        # This part has been simplified with respect to the cpp implementation
        self.initDay = int(initDay_or_file)

        self.Como_catch_param["CM"] = int(find_and_read("<CATCHMENT>"))
        catchment_index = find_and_read("<CATCHMENT>", return_line=True)
        self.Como_catch_param["inflow_file"]["filename"] = lines[catchment_index + 1].strip()
        self.Como_catch_param["inflow_file"]["row"] = self.NN
        self.Como_catch_param["inflow_file"]["col"] = self.H

        self.Como_param["initCond"] = float(find_and_read("<INIT_CONDITION>"))

        minEnvFlow_index = find_and_read("<MIN_ENV_FLOW_FILE>", return_line=True)
        self.Como_param_minEnvFlow_filename = lines[
            minEnvFlow_index + 1].strip()
        print(self.Como_param_minEnvFlow_filename)
        self.Como_param["minEnvFlow"] = Utils.load_matrix(
            self.Como_param_minEnvFlow_filename,
            self.NN,
            self.H
        )


def from_str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None

