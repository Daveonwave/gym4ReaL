import numpy as np

class Lake:
    """
    Lake class simulates the behavior of a lake, including its storage, release, and evaporation dynamics.
    """

    SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, params):
        """
        Initializes the Lake class with the given parameters.
        Args:
            params (dict): A dictionary containing the following keys:
                - 'init_level' (float): The initial water level of the lake.
                - 'evaporation' (bool): Whether to take water evaporation into account.
                - 'evaporation_rates' (list): A list of evaporation rates for different days.
                - 'surface' (float): The surface area of the lake.
                - 'min_env_flow' (float): The minimum environmental flow required.
                - 'min_level' (float): The minimum allowable water level of the lake.
                - 'max_level' (float): The maximum allowable water level of the lake.
                - 'alpha' (float): Parameter for the rating curce of water level dynamic.
                - 'beta' (float): Parameter for the rating curce of water level dynamic.
                - 'C_r' (float): Parameter for the rating curce of water level dynamic.
                - 'linear_slope' (float): The slope of the linear relationship for level dynamic.
                - 'linear_intercept' (float): The intercept of the linear relationship for level dynamic.
                - 'linear_limit' (float): The limit for the linear relationship in the level dynamic.
        Attributes:
            init_level (float): The initial water level of the lake.
            evaporation (float): The evaporation rate of the lake.
            evap_rates (list): A list of evaporation rates for different conditions.
            rating_curve (list): A list to store the rating curve data.
            lsv_rel (list): A list to store level-storage-volume relationships.
            surface (float): The surface area of the lake.
            tailwater (list): A list to store tailwater levels.
            min_env_flow (float): The minimum environmental flow required.
            min_level (float): The minimum allowable water level of the lake.
            max_level (float): The maximum allowable water level of the lake.
            alpha (float): Parameter for the rating curce of water level dynamic.
            beta (float): Parameter for the rating curce of water level dynamic.
            C_r (float): Parameter for the rating curce of water level dynamic.
            linear_slope (float): The slope of the linear relationship for level dynamic.
            linear_intercept (float): The intercept of the linear relationship for level dynamic.
            linear_limit (float): The limit for the linear relationship in the level dynamic.
        """

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

    def integration(self, step, init_storage, to_release, inflow, cday):
        """
        Simulates the behavior of a lake over a discretized time period.

        Args:
            step (int): Number of discrete steps in the simulation period.
            init_storage (float): Initial water storage in the lake (m³).
            to_release (float): Target water release rate (m³/s).
            inflow (float): Inflow rate into the lake (m³/s).
            cday (int): Current day of the year (1-365).

        Returns:
            tuple:
                - final_storage (float): Final water storage in the lake (m³).
                - mean_release (float): Mean water release rate over the simulation period (m³/s).

        Notes:
            - The simulation accounts for evaporation if the `evaporation` attribute is enabled.
            - Evaporation is calculated based on daily evaporation rates (`evap_rates`) and the lake's surface area.
            - The actual release is determined by the `actual_release` method, which considers constraints like storage capacity.
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
        Calculate the actual water release from the reservoir based on the desired release,
        current storage, and the day of the year, while ensuring it stays within the
        minimum and maximum allowable release limits.

        Args:
            to_release (float): The desired amount of water to release.
            storage (float): The current water storage in the reservoir.
            cday (int): The current day of the year (1-365).

        Returns:
            float: The actual water release, constrained by the minimum and maximum
            allowable release limits.
        """

        release_min = self.min_release(storage, cday)
        release_max = self.max_release(storage, cday)

        return min(release_max, max(release_min, to_release))

    def min_release(self, s, cday):
        """
        Calculate the minimum water release based on the current storage and day of the year.

        Args:
            s (float): The current storage volume in the reservoir.
            cday (int): The current day of the year (1-based index).

        Returns:
            float: The minimum release flow rate.

        Description:
            - The method determines the minimum release flow rate based on the reservoir's
              storage level and environmental flow requirements.
            - If the water level (h) is below or equal to the minimum level, the release is set to 0.
            - If the water level is between the minimum and maximum levels, the release is set to
              the minimum environmental flow for the given day.
            - If the water level exceeds the maximum level, the release is calculated using a
              rating curve based on the water level and predefined coefficients.
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
        Calculate the maximum water release from the dam based on the current storage and day of the year.

        Args:
            s (float): The current storage volume in the reservoir.
            cday (int): The current day of the year.

        Returns:
            float: The maximum release flow rate (e.g., in cubic meters per second).

        Notes:
            - If the water level corresponding to the storage is below or equal to the minimum level, 
              the release is zero.
            - If the water level is below or equal to the linear limit, the release is calculated 
              using a linear equation.
            - Otherwise, the release is calculated using a rating curve.
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
        Converts a given storage value to the corresponding water level.

        The conversion is based on the surface area of the lake and the minimum
        water level. The formula used is:
            level = (storage / surface) + min_level

        Args:
            storage (float): The volume of water storage in the lake.

        Returns:
            float: The calculated water level corresponding to the given storage.
        """
        return storage / self.surface + self.min_level

    def level_to_storage(self, h):
        """
        Converts the water level (h) to the corresponding storage volume.

        Args:
            h (float): The current water level.

        Returns:
            float: The storage volume calculated based on the water level.
        """
        return self.surface * (h - self.min_level)

    def level_to_surface(self, h):
        """
        Converts the water level (height) to the surface area of the lake.

        Parameters:
            h (float): The water level (height) in the lake.

        Returns:
            float: The surface area of the lake corresponding to the given water level.
        """
        return self.surface