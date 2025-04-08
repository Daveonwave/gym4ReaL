from .generic_models import ThermalModel
from .parameters import Scalar, instantiate_variables


class R2CThermal(ThermalModel):
    """
    Scarpelli-Fioriti paper
    TODO: implement this class (which could be too dependent on cell physical factors)
    """
    def __init__(self, components_settings: dict, **kwargs):
        super().__init__(name='R2C_thermal')
        
        self._init_components = instantiate_variables(components_settings)
        self._c_term = self._init_components['c_term']
        self._r_cond = self._init_components['r_cond']
        self._r_conv = self._init_components['r_conv']
        self._dv_dT = self._init_components['dv_dT']
        self._soc = None

    @property
    def soc(self):
        return self._soc

    @property
    def c_term(self):
        input_vars = {}

        if not isinstance(self._c_term, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._c_term.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal capacity!")

        return self._c_term.get_value(input_vars=input_vars)

    @property
    def r_cond(self):
        input_vars = {}

        if not isinstance(self._r_cond, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._r_cond.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal conductive resistor!")

        return self._r_cond.get_value(input_vars=input_vars)

    @property
    def r_conv(self):
        input_vars = {}

        if not isinstance(self._r_conv, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._r_conv.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal convective resistor!")

        return self._r_conv.get_value(input_vars=input_vars)

    @property
    def dv_dT(self):
        input_vars = {}

        if not isinstance(self._dv_dT, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._dv_dT.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute entropic coefficient!")

        return self._dv_dT.get_value(input_vars=input_vars)

    def reset_model(self, **kwargs):
        self._temp_series = []
        self._heat_series = []

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 25 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if 'temperature' in kwargs else 298.15
        heat = kwargs['dissipated_heat'] if 'dissipated_heat' in kwargs else 0
        super()._update_temp(temp)
        super()._update_heat(heat)

    def load_battery_state(self, **kwargs):
        self._soc = kwargs['soc']

    def compute_temp(self, q, i, T_amb, dt, k=-1):
        """
        Compute the current temperature with equation described in the aforementioned paper
        Args:
            q (float): power dissipated adopted as heat
            i (float): actual current in the circuit
            T_amb (float): ambient temperature
            dt (float): delta of time from last update
            k (int): iteration
        """
        term_1 = self.c_term / dt * self.get_temp_series(k=k)
        term_2 = T_amb / (self.r_cond + self.r_conv)
        denominator = self.c_term / dt + 1 / (self.r_cond + self.r_conv) - self.dv_dT * i

        t_core = (term_1 + term_2 + q) / denominator

        t_surf = t_core + self.r_cond * (T_amb - t_core) / (self.r_cond + self.r_conv)

        return t_surf


