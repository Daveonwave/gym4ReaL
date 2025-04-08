class SOCEstimator:
    """

    """
    def __init__(self,
                 capacity: float,
                 soc_max: float,
                 soc_min: float,
                 estimation_mode='CC'):
        """

        Args:
            capacity ():
            soc_max ():
            soc_min ():
            estimation_mode ():
        """
        self._estimation_mode = estimation_mode
        self._c_max = capacity
        self._soc = 0
        self._soc_max = soc_max
        self._soc_min = soc_min

    @property
    def c_max(self):
        return self._c_max

    @c_max.setter
    def c_max(self, q):
        self._c_max = q

    def reset_soc(self, v, v_max, v_min):
        """

        Args:
            v ():
            v_max ():
            v_min ():
        """
        self._soc = (v - v_min) / (v_max - v_min)
        return self._soc

    def compute_soc(self, soc_, i, dt):
        """
        CC = Coulomb Counting
        """        
        if self._estimation_mode == "CC":
            self._soc = soc_ + i / (self._c_max * 3600) * dt
            self._crop_soc()
        else:
            raise Exception("Required mode for SoC estimation not existing or not implemented yet.")

        return self._soc

    def _crop_soc(self):
        """

        """
        if self._soc < 0:
            self._soc = 0

        if self._soc > 1:
            self._soc = 1

    def get_feasible_current(self, soc_: float, dt: float):
        """
        Compute the maximum feasible current of the battery according to the soc.
        """
        i_max = (self._soc_max - soc_) / dt * self._c_max * 3600
        i_min = (self._soc_min - soc_) / dt * self._c_max * 3600
        return i_max, i_min
