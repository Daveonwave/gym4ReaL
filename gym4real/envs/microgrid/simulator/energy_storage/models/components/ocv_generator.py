from typing import Union

from .generic_component import ECMComponent
from ernestogym.ernesto.energy_storage.battery_models.parameters import Scalar, LookupTableFunction


class OCVGenerator(ECMComponent):
    """
    Open Circuit Voltage (OCV) element for Thevenin equivalent circuits.

    Parameters
    ----------

    """
    def __init__(self,
                 name,
                 ocv_potential: Union[Scalar, LookupTableFunction],
                 ):
        super().__init__(name, )
        self._ocv_potential = ocv_potential

    @property
    def ocv_potential(self):
        input_vars = {}

        if not isinstance(self._ocv_potential, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._ocv_potential.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute resistance for {}!".format(self.name))

        return self._ocv_potential.get_value(input_vars=input_vars)

    @ocv_potential.setter
    def ocv_potential(self, new_value):
        self._ocv_potential.set_value(new_value)

    def init_component(self, v=None):
        """
        Initialize V_ocv component at t=0
        """
        v = 0 if v is None else v
        super().init_component(v)

    # def _compute_v(self):
    #     """
    #
    #     """
    #     v_ocv = 3.43 + 0.68 * self._soc - 0.68 * (self._soc ** 2) + 0.81 * (self._soc ** 3) - 0.31 * np.exp(-46 * self._soc)
    #     return v_ocv

