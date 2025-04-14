from utils import Utils
import numpy as np

class Lake:
    def __init__(self, params):
        self.init_condition = params['init_condition']
        self.evaporation = params['evaporation']
        self.evap_rates = []
        self.rating_curve = []
        self.lsv_rel = []
        self.surface = params['surface']
        self.tailwater = []
        self.minEnvFlow = params['min_env_flow']

    def integration(self, step, tt, init_storage, to_release, inflow, cday, ps):
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
            return Utils.interp_lin(self.tailwater[0], self.tailwater[1], r)
        return 0.0

    def set_init_cond(self, ci):
        self.init_condition = ci

    def get_init_cond(self):
        return self.init_condition

    def set_evap(self, pEV):
        self.evaporation = pEV

    def set_evap_rates(self, pEvap):
        self.evap_rates = Utils.load_vector(pEvap["filename"], pEvap["row"])

    def set_rat_curve(self, pRatCurve):
        self.rating_curve = Utils.load_matrix(pRatCurve["filename"], pRatCurve["row"], pRatCurve["col"])

    def set_lsv_rel(self, pLSV_Rel):
        self.lsv_rel = Utils.load_matrix(pLSV_Rel["filename"], pLSV_Rel["row"], pLSV_Rel["col"])

    def set_surface(self, pA):
        self.surface = pA

    def set_tailwater(self, pTailWater):
        self.tailwater = Utils.load_matrix(pTailWater["filename"], pTailWater["row"], pTailWater["col"])

    def set_mef(self, pMEF):
        self.minEnvFlow = Utils.load_vector(pMEF["filename"], pMEF["row"])

    def get_mef(self, pDoy):
        return self.minEnvFlow[pDoy]

    # Placeholder methods to be defined as needed
    def min_release(self, s, cday):
        # Define logic or interpolation using rating_curve / lsv_rel
        return 0.0

    def max_release(self, s, cday):
        # Define logic or interpolation using rating_curve / lsv_rel
        return 1000.0

    def storage_to_level(self, storage):
        # Define transformation from storage to level
        return 0.0

    def level_to_surface(self, level):
        # Define transformation from level to surface area
        return self.surface  # Simple constant surface fallback
