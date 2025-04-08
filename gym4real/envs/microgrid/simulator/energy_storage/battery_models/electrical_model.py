from .generic_models import ElectricalModel
from .ecm_components import Resistor
from .ecm_components import ResistorCapacitorParallel
from .ecm_components import OCVGenerator
from .parameters import instantiate_variables


class TheveninModel(ElectricalModel):
    def __init__(self,
                 components_settings: dict,
                 sign_convention='active',
                 **kwargs
                 ):
        super().__init__(name='Thevenin')
        self._sign_convention = sign_convention

        # TODO: to approximate multiple RC modules in series
        self.ns_cells_module = 0
        self.np_cells_module = 0
        self.ns_modules = 0
        self.np_modules = 0
        self.ns_cells_battery = 0
        self.np_cells_battery = 0

        self._init_components = instantiate_variables(components_settings)

        self.r0 = Resistor(name='R0', resistance=self._init_components['r0'])
        self.rc = ResistorCapacitorParallel(name='RC',
                                            resistance=self._init_components['r1'],
                                            capacity=self._init_components['c'])
        self.ocv_gen = OCVGenerator(name='OCV', ocv_potential=self._init_components['v_ocv'])

    def reset_model(self, **kwargs):
        self._v_load_series = []
        self._i_load_series = []
        self.r0.reset_data()
        self.rc.reset_data()
        self.ocv_gen.reset_data()

    def init_model(self, **kwargs):
        """
        Initialize the model at t=0
        """
        v = kwargs['voltage'] if kwargs['voltage'] else 0
        i = kwargs['current'] if kwargs['current'] else 0
        p = v * i

        self.update_v_load(v)
        self.update_i_load(i)
        self.update_power(p)

        r0 = kwargs['r0'] if 'r0' in kwargs else None
        r1 = kwargs['r1'] if 'r1' in kwargs else None
        c = kwargs['c'] if 'c' in kwargs else None
        v_r0 = kwargs['v_r0'] if 'v_r0' in kwargs else None
        v_rc = kwargs['v_rc'] if 'v_rc' in kwargs else None
        v_ocv = kwargs['v_ocv'] if 'v_ocv' in kwargs else 0

        self.r0.init_component(r0=r0, v=v_r0)
        self.rc.init_component(r1=r1, c=c, v_rc=v_rc)
        self.ocv_gen.init_component(v=v_ocv)

    def load_battery_state(self, temp=None, soc=None, soh=None):
        """
        Update the SoC and SoH for the current simulation step
        """
        for component in [self.r0, self.rc, self.ocv_gen]:
            if temp is not None:
                component.temp = temp
            if soc is not None:
                component.soc = soc
            if soh is not None:
                component.soh = soh

        # self.r0.soc(value=soc) -> I'll probably need to do this one day
        # self.rc.soc(value=soc)

    def step_voltage_driven(self, v_load, dt, k):
        """
        CV mode
        """
        # Solve the equation to get I
        r0 = self.r0.resistance
        r1 = self.rc.resistance
        c = self.rc.capacity
        v_ocv = self.ocv_gen.ocv_potential

        # Compute V_c with finite difference method
        term_1 = self.rc.get_v_series(k=-1) / dt
        term_2 = (v_ocv - v_load) / (r0 * c)
        denominator = 1/dt + 1/(r0 * c) + 1/(r1 * c)

        v_rc = (term_1 + term_2) / denominator
        i = (v_ocv - v_rc - v_load) / r0

        if self._sign_convention == "passive":
            i = -i

        # Compute V_r0
        v_r0 = self.r0.compute_v(i=i)

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i, i_r1=i_r1)

        # Compute power
        power = v_load * i

        # Update the collections of variables of ECM components
        self.r0.update_step_variables(r0=r0, v_r0=v_r0)
        self.rc.update_step_variables(r1=r1, c=c, v_rc=v_rc, i_r1=i_r1, i_c=i_c)
        self.ocv_gen.update_v(value=v_ocv)
        self.update_i_load(value=i)
        self.update_v_load(value=v_load)
        self.update_power(value=power)

        return v_load, i

    def step_current_driven(self, i_load, dt, k, p_load=None):
        """
        CC mode
        """
        # Solve the equation to get V
        r0 = self.r0.resistance
        r1 = self.rc.resistance
        c = self.rc.capacity
        v_ocv = self.ocv_gen.ocv_potential

        if self._sign_convention == 'passive':
            i_load = -i_load

        # Compute V_r0 and V_rc
        v_r0 = self.r0.compute_v(i=i_load)
        v_rc = (self.rc.get_v_series(k=-1) / dt + i_load / c) / (1/dt + 1 / (c*r1))

        # Compute V
        v = v_ocv - v_r0 - v_rc

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i_load, i_r1=i_r1)

        if p_load is not None:
            i_load = -i_load

        # Compute power
        if p_load is not None:
            power = p_load
        else:
            power = v * i_load
            if self._sign_convention == 'passive':
                power = -power

        # Update the collections of variables of ECM components
        self.r0.update_step_variables(r0=r0, v_r0=v_r0)
        self.rc.update_step_variables(r1=r1, c=c, v_rc=v_rc, i_r1=i_r1, i_c=i_c)
        self.ocv_gen.update_v(value=v_ocv)
        self.update_v_load(value=v)
        self.update_i_load(value=i_load)
        self.update_power(value=power)

        return v, i_load

    def step_power_driven(self, p_load, dt, k):
        """
        CP mode: to simplify the power driven case, we pose I = P / V(t-1), having a little shift in computed data
        """
        if self._sign_convention == 'passive':
            return self.step_current_driven(i_load=p_load / self._v_load_series[-1], dt=dt, k=k, p_load=p_load)
        else:
            return self.step_current_driven(i_load=p_load / self._v_load_series[-1], dt=dt, k=k, p_load=p_load)

    def compute_generated_heat(self, k=-1):
        """
        Compute the generated heat that can be used to feed the thermal model (when required).
        For Thevenin first order circuit it is: [P = V * I + V_rc * I_r1].

        Inputs:
        :param k: step for which compute the heat generation
        """
        # TODO: option about dissipated power computed with r0 only or r0 and r1
        return self.r0.get_r0_series(k=k) * self.get_i_series(k=k)**2 + \
            self.rc.get_r1_series(k=k) * self.rc.get_i_r1_series(k=k)**2
        # return self.r0.get_r0_series(k=k) * self.get_i_series(k=k) ** 2

    def get_internal_resistance(self, nominal:bool=False):
        if nominal:
            return self.r0.nominal_resistance
        else:
            return self.r0.resistance

    def get_polarization_resistance(self, nominal:bool=False):
        if nominal:
            return self.rc.nominal_resistance
        else:
            return self.rc.resistance

    def get_internal_capacity(self):
        return self.rc.capacity

    def get_results(self, **kwargs):
        """
        Returns a dictionary with all final results
        TODO: selection of results by label from config file?
        """
        k = kwargs['k'] if 'k' in kwargs else None

        return {'voltage': self.get_v_series(k=k),
                'current': self.get_i_series(k=k),
                'power': self.get_p_series(k=k),
                'v_oc': self.ocv_gen.get_v_series(k=k),
                'r0': self.r0.get_r0_series(k=k),
                'r1': self.rc.get_r1_series(k=k),
                'c': self.rc.get_c_series(k=k),
                'v_r0': self.r0.get_v_series(k=k),
                'v_rc': self.rc.get_v_series(k=k)
                }
