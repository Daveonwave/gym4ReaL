import numpy as np

class Lake:

    SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, params):
        self.init_level = params['init_level']
        self.evaporation = params['evaporation']
        self.evap_rates = params['evaporation_rates']
        self.rating_curve = []
        self.lsv_rel = []
        self.surface = params['surface']
        self.tailwater = []
        self.min_env_flow = params['min_env_flow']

        self.min_level = params['min_level']
        self.max_level = params['max_level']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.C_r = params['C_r']

        self.linear_slope = params['linear_slope']
        self.linear_intercept = params['linear_intercept']
        self.linear_limit = params['linear_limit']

    def integration(self, step, tt, init_storage, to_release, inflow, cday):
        """
        Simulates lake behavior over a discretized period.
        Returns a tuple (final_storage, mean_release)
        """
        sim_step = self.SECONDS_PER_DAY / step  # seconds per step
        release = []

        # Initial condition

        curr_storage = init_storage

        for i in range(step):
            # Compute actual release
            release.append(self.actual_release(to_release, curr_storage, cday))

            # Compute evaporation
            if self.evaporation:
                surf = self.level_to_surface(self.storage_to_level(curr_storage))
                evaporated = self.evap_rates[cday-1] / 1000 * surf / self.SECONDS_PER_DAY  # m³/s
            else:
                evaporated = 0.

            # System transition
            curr_storage = curr_storage + sim_step * (inflow - release[-1] - evaporated)

        mean_release = np.mean(release)

        return curr_storage, mean_release

    def actual_release(self, to_release, storage, cday):
        """
        Returns the actual release amount based on decision and storage constraints.
        """
        release_min = self.min_release(storage, cday)
        release_max = self.max_release(storage, cday)

        return min(release_max, max(release_min, to_release))

    def min_release(self, s, cday):
        """
        Computes the minimum release (q) for a given storage and day of year.
        Based on elevation-dependent piecewise rules.
        """
        DMV = self.min_env_flow[cday - 1]
        h = self.storage_to_level(s)

        if h <= self.min_level:
            q = 0.
        elif h <= self.max_level:
            q = DMV
        else:
            q = self.C_r * ((h - self.alpha) ** self.beta)

        return q

    def max_release(self, s, cday):
        """
        Computes the maximum release (q) for a given storage and day of year.
        Based on elevation-dependent piecewise rules.
        """
        h = self.storage_to_level(s)

        if h <= self.min_level:
            q = 0.
        elif h <= self.linear_limit:
            q = self.linear_slope * h + self.linear_intercept
        else:
            q = self.C_r * ((h - self.alpha) ** self.beta)

        return q

    def rel_to_tailwater(self, r):
        """
        Converts release to tailwater level using rating curve interpolation.
        """
        if self.tailwater:
            return np.interp(r, self.tailwater[0], self.tailwater[1], left=None, right=None)
        return 0.

    def get_mef(self, pDoy):
        return self.min_env_flow[pDoy]

    def storage_to_level(self, storage):
        """
        Converts storage (m³) to water level (m).
        Linear relationship: h = s/A + h0
        """
        return storage / self.surface + self.min_level

    def level_to_storage(self, h):
        """
        Converts water level (m) to storage (m³).
        Inverse of storage_to_level.
        """
        return self.surface * (h - self.min_level)

    def level_to_surface(self, h):
        """
        Returns surface area (m²) at level h.
        Constant surface area is assumed for Lake Como.
        """
        return self.surface