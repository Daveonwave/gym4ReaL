from .battery_models import *


class BatteryEnergyStorageSystem:
    """
    Class representing the battery abstraction.
    Here we select all the electrical, thermal and mathematical electrical to simulate the BESS behaviour.
    #TODO: can be done with multi-threading (one for each submodel)?
    """
    def __init__(self,
                 models_config: list,
                 battery_options: dict,
                 input_var: str,
                 check_soh_every=None,
                 **kwargs
                 ):
        """
        Args:
            models_config (list):
            battery_options (dict):
            input_var (str):
            check_soh_every (int, None):
        """
        self.models_settings = models_config
        self._load_var = input_var
        self._ground_data = kwargs["ground_data"] if "ground_data" in kwargs else None

        # Possible electrical to build
        self._electrical_model = None
        self._thermal_model = None
        self._aging_model = None
        self._soc_model = None
        self.models = []

        # Battery datasheet parameters
        self.nominal_capacity = battery_options['params']['nominal_capacity']
        self.nominal_dod = battery_options['params']['nominal_dod'] \
            if 'nominal_dod' in battery_options['params'].keys() else None
        self.nominal_lifetime = battery_options['params']['nominal_lifetime'] \
            if 'nominal_lifetime' in battery_options['params'].keys() else None
        self.nominal_voltage = battery_options['params']['nominal_voltage'] \
            if 'nominal_voltage' in battery_options['params'].keys() else None
        self.nominal_cost = battery_options['params']['nominal_cost'] \
            if 'nominal_cost' in battery_options['params'].keys() else None
        self.v_max = battery_options['params']['v_max']
        self.v_min = battery_options['params']['v_min']
        self.temp_ambient = battery_options['params']['temp_ambient'] \
            if 'temp_ambient' in battery_options['params'].keys() else None

        self.soc_min = battery_options['bounds']['soc']['low']
        self.soc_max = battery_options['bounds']['soc']['high']

        self._sign_convention = battery_options['sign_convention']
        self._reset_soc_every = battery_options['reset_soc_every'] if 'reset_soc_every' in battery_options['params'].keys() else None
        self._check_soh_every = check_soh_every if check_soh_every is not None else 1

        # Collection where will be stored the simulation variables
        self._c_max = battery_options['params']['nominal_capacity']
        self.soc_series = []
        self.soh_series = []
        # self.temp_series = []
        self.t_series = []

        # Instantiate models
        self._build_models()

    @property
    def load_var(self):
        return self._load_var

    @load_var.setter
    def load_var(self, var: str):
        self._load_var = var

    def get_v(self):
        return self._electrical_model.get_v_series(k=-1)

    def get_i(self):
        return self._electrical_model.get_i_series(k=-1)

    def get_p(self):
        return self._electrical_model.get_p_series(k=-1)

    def get_temp(self):
        return self._thermal_model.get_temp_series(k=-1)

    def get_internal_resistance(self, nominal:bool=False):
        return self._electrical_model.get_internal_resistance(nominal=nominal)

    def get_polarization_resistance(self, nominal:bool=False):
        return self._electrical_model.get_polarization_resistance(nominal=nominal)

    def get_internal_capacity(self):
        return self._electrical_model.get_internal_capacity()

    def get_c_max(self):
        return self._c_max

    def get_feasible_current(self, last_soc=None, dt=1):
        soc_ = self.soc_series[-1] if last_soc is None else last_soc
        return self._soc_model.get_feasible_current(soc_=soc_, dt=dt)

    def get_delta_degradation(self):
        return abs(self._aging_model.get_deg_series(-1) - self._aging_model.get_deg_series(-2))

    def _build_models(self):
        """
        Model instantiation depending on the 'type' reported in the model yaml file.
        In the same file is annotated also the 'class_name' of the model object to instantiate.

        Accepted 'types' are: ['electrical', 'thermal', 'aging'].
        """
        for model_config in self.models_settings:
            if model_config['type'] == 'electrical':
                components = model_config['components'] if 'components' in model_config.keys() else None
                kwargs = {
                    'alpha_fading': model_config['alpha_fading'] if 'alpha_fading' in model_config.keys() else None,
                    'beta_fading': model_config['beta_fading'] if 'beta_fading' in model_config.keys() else None
                }
                self._electrical_model = globals()[model_config['class_name']](components_settings=components,
                                                                               sign_convention=self._sign_convention,
                                                                               **kwargs)
                self.models.append(self._electrical_model)

            elif model_config['type'] == 'thermal':
                components = model_config['components'] if 'components' in model_config.keys() else None
                kwargs = {'ground_temps': self._ground_data['temperature'] if self._ground_data else None}
                self._thermal_model = globals()[model_config['class_name']](components_settings=components, **kwargs)
                self.models.append(self._thermal_model)

            elif model_config['type'] == 'aging':
                self._aging_model = globals()[model_config['class_name']](components_settings=model_config['components'],
                                                                          stress_models=model_config['stress_models'])
                self.models.append(self._aging_model)

            else:
                raise Exception("The 'type' of {} you are trying to instantiate is wrong!"\
                                .format(model_config['class_name']))

        # Instantiation of battery state estimators
        self._soc_model = SOCEstimator(capacity=self._c_max, soc_max=self.soc_max, soc_min=self.soc_min)

    def reset(self):
        """

        """
        self.soc_series = []
        self.soh_series = []
        self.t_series = []
        self._c_max = self.nominal_capacity

        for model in self.models:
            model.reset_model()

    def init(self, init_info: dict = {}):
        """
        Initialization of the battery simulation environment at t=0.
        """
        self.t_series.append(-1)
        self.soc_series.append(init_info['soc'])
        self.soh_series.append(init_info['soh'])
        self.temp_ambient = init_info['temp_ambient'] if 'temp_ambient' in init_info.keys() else None
        
        init_info['q_moved_charge'] = 0

        for model in self.models:
            model.load_battery_state(temp=init_info['temperature'],
                                     soc=init_info['soc'],
                                     soh=init_info['soh'])
            model.init_model(**init_info)

    def step(self, load: float, dt: float, k: int, t_amb: float = None):
        """

        Args:
            load ():
            dt ():
            k ():
        """
        v, i, soc = self._step_electrical(load=load, dt=dt)
        self.soc_series.append(soc)
        
        # Thermal model step if present
        if self._thermal_model is not None:
            t_amb = self.temp_ambient if t_amb is None else t_amb
            temp, heat = self._step_thermal(i=i, t_amb=t_amb, dt=dt)
            self._thermal_model.update(**{'temp':temp, 'heat':heat})
        else:
            temp = self._init_conditions['temperature']
                
        # Aging model step if present
        if self._aging_model is not None:
            soh = self._step_aging(k=k)
            self._c_max = soh * self.nominal_capacity
        else:
            soh = self._c_max / self.nominal_capacity
        self.soh_series.append(soh)
                
        # Forward SoC, SoH and temperature to models and their components
        for model in self.models:
            model.load_battery_state(temp=temp, soc=soc, soh=soh)
            
        # Reset the SoC estimation to avoid an error drift of the SoC estimation. 
        if self._reset_soc_every is not None and k % self._reset_soc_every == 0:
            self.soc_series[-1] = self._soc_model.reset_soc(v=v, v_max=self.v_max, v_min=self.v_min)

    def _step_electrical(self, load: float, dt: float):
        """
        Perform a step of the electrical model of the battery.

        Args:
            load (float): value of the load to apply to the battery.
            dt (float): delta of time between the current and the previous sample.

        Returns:
            tuple: voltage, current and state of charge of the battery.
        
        Raises:
            Exception: if the provided battery simulation mode doesn't exist or is just not implemented.
        """
        if self._load_var == 'current':
            v, _ = self._electrical_model.step_current_driven(i_load=load, dt=dt, k=-1)
            i = load
        elif self._load_var == 'voltage':
            _, i = self._electrical_model.step_voltage_driven(v_load=load, dt=dt, k=-1)
            v = load
        elif self._load_var == 'power':
            v, i = self._electrical_model.step_power_driven(p_load=load, dt=dt, k=-1)
        else:
            raise Exception("The provided battery simulation mode {} doesn't exist or is just not implemented!"
                            "Choose among the provided ones: Voltage, Current or Power.".format(self._load_var))
        
        self._c_max = self._electrical_model.compute_parameter_fading(self.nominal_capacity)
        self._soc_model.c_max = self._c_max
        soc = self._soc_model.compute_soc(soc_=self.soc_series[-1], i=i, dt=dt)
        return v, i, soc
        
    def _step_thermal(self, i: float, t_amb: float, dt: float):
        """
        Perform a step of the thermal model of the battery

        Args:
            i (float): current flowing through the battery.
            t_amb (float): ambient temperature.
            dt (float): delta of time between the current and the previous sample.
            ground_temp (float): temperature of the ground (used only with dummy model).
        
        Returns:
            tuple: temperature and heat generated by the battery.
        """
        heat = self._electrical_model.compute_generated_heat()
        temp = self._thermal_model.compute_temp(q=heat, i=i, T_amb=t_amb, dt=dt, k=-1)
        return temp, heat
    
    def _step_aging(self, k: int):
        """
        Perform a step of the aging model of the battery.

        Args:
            k (int): k-th iteration of the simulation.

        Returns:
            float: state of health of the battery.
        """
        if self._aging_model.name == 'Bolun':
            if k % self._check_soh_every == 0:
                return self.soh_series[0] - self._aging_model.compute_degradation(soc_history=self.soc_series,
                                                                                  temp_history=self._thermal_model.get_temp_series(),
                                                                                  elapsed_time=self.t_series[-1],
                                                                                  k=k)
            else:
                return self.soh_series[-1]
        
        # BOLUN DROPFLOW MODEL
        elif self._aging_model.name == 'BolunDropflow':
            return self.soh_series[0] - self._aging_model.compute_degradation(soc=self.soc_series[-1],
                                                                              temp=self._thermal_model.get_temp_series(k=-1),
                                                                              elapsed_time=self.t_series[-1],
                                                                              k=k,
                                                                              do_check=(k % self._check_soh_every == 0))        
        else:
            raise Exception("The provided aging model {} doesn't exist or is just not implemented!".format(self._aging_model.name))
    
    def get_snapshot(self):
        """
        Collect the status of the battery and its components at the current time step.
        Used to update the queues of the writer.
        """
        status_dict = {'time': self.t_series[-1], 'soc': self.soc_series[-1], 'soh': self.soh_series[-1], 'c_max': self._c_max}

        for model in self.models:
            status_dict.update(model.get_results(**{'k': -1}))

        return status_dict
    
    def build_results_table(self):
        """
        """
        final_dict = {'time': self.t_series, 'soc': self.soc_series, 'soh': self.soh_series}

        for model in self.models:
            final_dict.update(model.get_results())

        deg_dict = {}

        # Create results of degradation (sparser than other results)
        if self._aging_model is not None:
            deg_keys = ['iteration', 'cyclic_aging', 'calendar_aging', 'degradation']
            deg_dict = {key: value for key, value in final_dict.items() if key in deg_keys}
            for key in deg_keys:
                del final_dict[key]

        return {'operations': final_dict, 'aging': deg_dict}





