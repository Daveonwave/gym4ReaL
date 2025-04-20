from abc import ABC, abstractmethod
import numpy as np

class Lake(ABC):
    def __init__(self, params):
        self.init_level = params['init_level']
        self.evaporation = params['evaporation']
        self.evap_rates = []
        self.rating_curve = []
        self.lsv_rel = []
        self.surface = params['surface']
        self.tailwater = []
        self.minEnvFlow = params['min_env_flow']

    def integration(self, step, tt, init_storage, to_release, inflow, cday):
        """
        Simulates lake behavior over a discretized period.
        Returns a tuple (final_storage, mean_release)
        """
        sim_step = 3600 * 24 / step  # seconds per step
        release = []

        # Initial condition

        curr_storage = init_storage

        for i in range(step):
            # Compute actual release
            release.append(self.actual_release(to_release, curr_storage, cday))

            # Compute evaporation
            if self.evaporation == 1:
                surf = self.level_to_surface(self.storage_to_level(curr_storage))
                evaporated = self.evap_rates[cday - 1] / 1000 * surf / 86400  # m³/s
            elif self.evaporation > 1:
                # E = compute_evaporation()  # Not implemented
                evaporated = 0.
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

    def rel_to_tailwater(self, r):
        """
        Converts release to tailwater level using rating curve interpolation.
        """
        if self.tailwater:
            return np.interp(r, self.tailwater[0], self.tailwater[1], left=None, right=None)
        return 0.0

    def get_mef(self, pDoy):
        return self.minEnvFlow[pDoy]

    @abstractmethod
    def min_release(self, s, cday):
        # Define logic or interpolation using rating_curve / lsv_rel
        pass

    @abstractmethod
    def max_release(self, s, cday):
        # Define logic or interpolation using rating_curve / lsv_rel
        pass

    @abstractmethod
    def storage_to_level(self, storage):
        # Define transformation from storage to level
        pass

    @abstractmethod
    def level_to_surface(self, level):
        # Define transformation from level to surface area
        return self.surface  # Simple constant surface fallback
