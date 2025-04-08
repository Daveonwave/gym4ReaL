class ECMComponent:
    """
    Generic component of Thevenin equivalent circuits.

    Parameters
    ----------
    :param name: identifier of the component
    :type name: str

    Attributes
    ----------
    _v_series: collection of all the past component voltages
    _t_series: collection of all the past discrete steps of time

    Purpose
    -------
    This class builds a generic component of the Thevenin equivalent circuit and presents common attributes and
    collections that can be useful in each single element of the circuit (Resistor, RCParallel, V_OCV generator).
    """
    def __init__(self, name):
        self._name = name

        # Dependency variable of components
        self._temp = None
        self._soc = None
        self._soh = None

        # Collections related to the specific component
        self._v_series = []

    @property
    def name(self):
        return self._name

    @property
    def temp(self):
        return self._temp

    @property
    def soc(self):
        return self._soc

    @property
    def soh(self):
        return self._soh

    @temp.setter
    def temp(self, value: float):
        self._temp = value

    @soc.setter
    def soc(self, value: float):
        assert 0 <= value <= 1, \
            "The value of the State of Charge (SoC) passed to {} is wrong. " \
            "It has to be comprised between 0 and 1. The current value is {}.".format(self.name, value)
        self._soc = value

    @soh.setter
    def soh(self, value: float):
        assert 0 <= value <= 1, \
            "The value of the State of Health (SoH) passed to {} is wrong. " \
            "It has to be comprised between 0 and 1. The current value is {}.".format(self.name, value)
        self._soh = value

    def get_v_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int,\
                "Cannot retrieve voltage of {} at step K, since it has to be an integer".format(self._name)

            if len(self._v_series) > k:
                return self._v_series[k]
            else:
                raise IndexError("Voltage V of {} at step K not computed yet".format(self._name))
        return self._v_series

    def reset_data(self):
        self._v_series = []

    def init_component(self, v=0):
        self.update_v(v)

    def update_v(self, value: float):
        self._v_series.append(value)


