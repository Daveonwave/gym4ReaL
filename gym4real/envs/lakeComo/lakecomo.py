from lake import Lake
import math

class LakeComo(Lake):
    def __init__(self):
        super().__init__()

    def storage_to_level(self, s):
        """
        Converts storage (m³) to water level (m).
        Linear relationship: h = s/A + h0
        """
        h0 = -0.5
        return s / self.A + h0

    def level_to_storage(self, h):
        """
        Converts water level (m) to storage (m³).
        Inverse of storage_to_level.
        """
        h0 = -0.5
        return self.A * (h - h0)

    def level_to_surface(self, h):
        """
        Returns surface area (m²) at level h.
        Constant surface area is assumed for Lake Como.
        """
        return self.A

    def min_release(self, s, cday):
        """
        Computes the minimum release (q) for a given storage and day of year.
        Based on elevation-dependent piecewise rules.
        """
        DMV = self.minEnvFlow[cday - 1]
        h = self.storage_to_level(s)

        if h <= -0.50:
            q = 0.0
        elif h <= 1.25:
            q = DMV
        else:
            q = 33.37 * ((h + 2.5) ** 2.015)

        return q

    def max_release(self, s, cday):
        """
        Computes the maximum release (q) for a given storage and day of year.
        Based on elevation-dependent piecewise rules.
        """
        h = self.storage_to_level(s)

        if h <= -0.5:
            q = 0.0
        elif h <= -0.40:
            q = 1488.1 * h + 744.05
        else:
            q = 33.37 * ((h + 2.5) ** 2.015)

        return q
