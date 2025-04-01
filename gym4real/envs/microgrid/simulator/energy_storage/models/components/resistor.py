from typing import Union

from ernestogym.ernesto.energy_storage.battery_models.parameters.units import Unit
from ernestogym.ernesto.energy_storage.battery_models.parameters import Scalar, LookupTableFunction
from .generic_component import ECMComponent
from copy import deepcopy


class Resistor(ECMComponent):
    """
    Resistor element of Thevenin equivalent circuits.

    Parameters
    ----------
    :param name: identifier of the resistor
    :type name: str

    :param resistance: value of the resistance (Ohm)
    :type resistance: float or int or pint.Quantity

    """
    def __init__(self,
                 name: str,
                 resistance: Union[Scalar, LookupTableFunction],
                 ):
        super().__init__(name)
        self._resistance = resistance
        self._nominal_resistance = deepcopy(resistance)

        # TODO: fix the unit through a yaml file
        self._r0_unit = Unit.OHM

        # Collections
        self._r0_series = []

    def __getattr__(self, item: str):
        if item in "resistor":
            return self._resistance
        else:
            raise AttributeError("No attribute {} in class Resistor.".format(item))

    @property
    def resistance(self):
        """
        Getter of the R0 value. Depending on the x_names (inputs of the function), we retrieve components attribute
        among {SoC, SoH, Temp}.
        If R0 is a scalar, we don't need to provide any input.
        """
        input_vars = {}

        if not isinstance(self._resistance, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._resistance.x_names}
            except:
                raise Exception("Cannot retrieve required input variables to compute resistance for {}!".format(self.name))

        return self._resistance.get_value(input_vars=input_vars)

    @property
    def nominal_resistance(self):
        """
        Getter of the R0 nominal value. Depending on the x_names (inputs of the function), we retrieve components attribute
        among {SoC, SoH, Temp}.
        If R0 is a scalar, we don't need to provide any input.
        """
        input_vars = {}

        if not isinstance(self._nominal_resistance, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._nominal_resistance.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute resistance for {}!".format(self.name))

        return self._nominal_resistance.get_value(input_vars=input_vars)

    @resistance.setter
    def resistance(self, new_value):
        self._resistance.set_value(new_value)

    def reset_data(self):
        self._v_series = []
        self._r0_series = []

    def init_component(self, v=None, r0=None):
        """
        Initialize R0 component at t=0
        """
        r0 = self.resistance if r0 is None else r0
        v = 0 if v is None else v

        super().init_component(v)
        self._update_r0_series(r0)

    def get_r0_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve resistance of {} at step K, since it has to be an integer".format(self._name)

            if len(self._r0_series) > k:
                return self._r0_series[k]
            else:
                raise IndexError("Resistance R0 of {} at step K not computed yet".format(self._name))
        return self._r0_series

    def compute_v(self, i):
        """
        Compute the resistor potential V_r0, given in input the electric current I=I_r0
        #TODO: we will use 'k' when there will be the decay of resistance
        """
        v_r0 = i * self.resistance
        return v_r0

    def compute_i(self, v_r0):
        """
        Compute the flowing electric current I_r0=I, given in input the resistor potential V_r0
        """
        i_r0 = v_r0 / self.resistance
        return i_r0

    def compute_dv(self, i, i_, dt):
        """
        Compute the derivative of (dv_r0/dt) using the backward finite differences approach.
        We consider (dr0/dt) constant, so we can erase the second term of the derivation by parts.

        Inputs:
        :param i: current at time t
        :param i_: current at previous sampling time t-dt
        :param dt: delta of time
        """
        dv_r0 = (i - i_) / dt * self.resistance
        return dv_r0

    def _update_r0_series(self, value: float):
        self._r0_series.append(value)

    def update_step_variables(self, r0, v_r0):
        """
        Aggiorno le liste delle variabili calcolate
        """
        self._update_r0_series(r0)
        self.update_v(v_r0)

