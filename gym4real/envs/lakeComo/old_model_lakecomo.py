import numpy as np
from math import sin, cos, pi
from utils import Utils
from catchment import Catchment
from lakecomo import LakeComo


class ModelLakeComo:
    def __init__(self, filename):
        self.read_file_settings(filename)

        # create catchment
        self.ComoCatchment = Catchment(self.Como_catch_param)

        # create lake
        self.LakeComo = LakeComo()
        self.LakeComo.set_evap(0)
        self.Como_param["minEnvFlow"]["filename"] = "../data/MEF_como.txt"
        self.Como_param["minEnvFlow"]["row"] = self.T
        self.LakeComo.set_mef(self.Como_param["minEnvFlow"])
        self.LakeComo.set_surface(145900000)
        self.LakeComo.set_init_cond(self.Como_param["initCond"])

        # policy
        if self.p_param["tPolicy"] == 4:
            self.mPolicy = NcRBF(
                self.p_param["policyInput"],
                self.p_param["policyOutput"],
                self.p_param["policyStr"]
            )
        else:
            raise ValueError("Policy architecture not defined")

        self.mPolicy.set_max_input(self.p_param["MIn"])
        self.mPolicy.set_min_input(self.p_param["mIn"])
        self.mPolicy.set_max_output(self.p_param["MOut"])
        self.mPolicy.set_min_output(self.p_param["mOut"])

        # Objectives
        self.h_flo = 1.24
        self.demand = Utils.load_vector("../data/comoDemand.txt", 365)

# useless
    def clear(self):
        del self.LakeComo
        del self.mPolicy
        del self.ComoCatchment

    def get_nobj(self):
        return self.Nobj

    def get_nvar(self):
        return self.Nvar

    def evaluate(self, var, obj):
        self.mPolicy.set_parameters(var)

        if self.Nsim < 2:
            J = self.simulate(0)
            for i in range(self.Nobj):
                obj[i] = J[i]
        else:
            raise NotImplementedError("Monte Carlo simulation not implemented")

        self.mPolicy.clear_parameters()

    def simulate(self, ps):
        s = np.full(self.H + 1, -999.0)
        h = np.full(self.H + 1, -999.0)
        u = np.full(self.H, -999.0)
        r = np.full(self.H + 1, -999.0)
        doy = np.full(self.H, -999.0)

        qForecast = Utils.load_vector("../data/qSimAnomL51.txt", self.H)

        h[0] = self.LakeComo.get_init_cond()
        s[0] = self.LakeComo.level_to_storage(h[0])

        for t in range(self.H):
            doy[t] = (self.initDay + t - 1) % self.T + 1
            qIn = self.ComoCatchment.get_inflow(t, ps)

            input_vec = [
                sin(2 * pi * doy[t] / self.T),
                cos(2 * pi * doy[t] / self.T),
                h[t]
            ]
            if self.p_param["policyInput"] > 3:
                input_vec.append(qForecast[t])

            uu = self.mPolicy.get_norm_output(input_vec)
            u[t] = uu[0]

            sh_rh = self.LakeComo.integration(
                self.integStep, t, s[t], u[t], qIn, doy[t], ps
            )

            s[t + 1], r[t + 1] = sh_rh[0], sh_rh[1]
            h[t + 1] = self.LakeComo.storage_to_level(s[t + 1])

        # remove warmup
        h = h[1 + self.warmup:]
        r = r[1 + self.warmup:]
        doy = doy[self.warmup:]

        return [
            self.flood_days(h, self.h_flo) / (self.H // self.T),
            self.avg_deficit_beta(r, self.demand, doy)
        ]

    def flood_days(self, h, h_flo):
        return sum(level > h_flo for level in h)

    def avg_deficit_beta(self, q, w, doy):
        gt = 0.0
        for i in range(len(q)):
            qdiv = q[i] - self.LakeComo.get_mef(int(doy[i]) - 1)
            qdiv = max(qdiv, 0.0)
            d = max(w[int(doy[i]) - 1] - qdiv, 0.0)
            if 120 < doy[i] <= 243:
                d *= 2
            gt += d * d
        return gt / len(q)

    def read_file_settings(self, filename):
        self.p_param = {
            "mIn": [], "MIn": [],
            "mOut": [], "MOut": [],
        }
        self.Como_catch_param = {"inflow_file": {}}
        self.Como_param = {"minEnvFlow": {}}

        with open(filename, 'r') as f:
            lines = f.readlines()

        def find_and_read(key, is_float=True, multi=1):
            for i, line in enumerate(lines):
                if line.strip().startswith(key):
                    values = list(map(float if is_float else str, lines[i + 1].split()))
                    return values[0] if multi == 1 else values

        self.Nsim = int(find_and_read("<NUM_SIM>", True))
        self.NN = int(find_and_read("<DIM_ENSEMBLE>"))
        self.T = int(find_and_read("<PERIOD>"))
        self.integStep = float(find_and_read("<INTEGRATION>"))
        self.H = int(find_and_read("<SIM_HORIZON>"))
        self.Nobj = int(find_and_read("<NUM_OBJ>"))
        self.Nvar = int(find_and_read("<NUM_VAR>"))
        self.warmup = int(find_and_read("<WARMUP>"))

        initDay_or_file = find_and_read("<DOY>", False)
        if initDay_or_file.isdigit():
            self.initDay = int(initDay_or_file)
        else:
            self.doy_file = Utils.load_vector(initDay_or_file, self.H)

        self.Como_catch_param["CM"] = int(find_and_read("<CATCHMENT>"))
        self.Como_catch_param["inflow_file"]["filename"] = lines[lines.index("<CATCHMENT>\n") + 2].strip()
        self.Como_catch_param["inflow_file"]["row"] = self.NN
        self.Como_catch_param["inflow_file"]["col"] = self.H

        self.Como_param["initCond"] = float(find_and_read("<INIT_CONDITION>"))
        self.p_param["tPolicy"] = int(find_and_read("<POLICY_CLASS>"))
        self.p_param["policyInput"] = int(find_and_read("<NUM_INPUT>"))
        i_line = lines.index("<NUM_INPUT>\n") + 2
        for _ in range(self.p_param["policyInput"]):
            i1, i2 = map(float, lines[i_line].split())
            self.p_param["mIn"].append(i1)
            self.p_param["MIn"].append(i2)
            i_line += 1

        self.p_param["policyOutput"] = int(find_and_read("<NUM_OUTPUT>"))
        o_line = lines.index("<NUM_OUTPUT>\n") + 2
        for _ in range(self.p_param["policyOutput"]):
            o1, o2 = map(float, lines[o_line].split())
            self.p_param["mOut"].append(o1)
            self.p_param["MOut"].append(o2)
            o_line += 1

        self.p_param["policyStr"] = int(find_and_read("<POLICY_STRUCTURE>"))
