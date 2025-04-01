import numpy as np
from .dropflow import Dropflow
from enum import Enum

from ernestogym.ernesto.energy_storage.battery_models.generic_models import AgingModel
from ernestogym.ernesto.energy_storage.battery_models.aging import stress_functions


class BolunDropflowModel(AgingModel):
    """
    Bolun model (https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment)
    """
    def __init__(self,
                 components_settings: dict,
                 stress_models: dict,
                 ):
        """
        Args:
            components_settings ():
            stress_models ():
            init_soc ():
        """
        super().__init__(name='BolunDropflow')

        self._f_cyc = 0
        self._f_cyc_series = []
        self._f_cal_series = []
        self._k_iters = []
        
        self._soc_history = []
        self._temp_history = []
        
        self._mean_soc = 0
        self._mean_temp = 0

        self._alpha_sei = components_settings['SEI']['alpha_sei']
        self._beta_sei = components_settings['SEI']['beta_sei']

        # Collect and build stress factors for both calendar and cyclic aging
        self._calendar_factors = {key: stress_models[key] for key in components_settings['stress_factors']['calendar']}
        self._cyclic_factors = {key: stress_models[key] for key in components_settings['stress_factors']['cyclic']}

        # Stress models constants
        self._stress_models_params = stress_models

        # Fatigue analysis method
        self._cycle_counting_mode = "dropflow"
        self._dropflow = Dropflow()
            
    @property
    def collections_map(self):
        return {'calendar_aging': self.get_f_cal_series,
                'cyclic_aging': self.get_f_cyc_series,
                'degradation': self.get_deg_series,
                'aging_iteration': self.get_k_iter_series}

    def get_f_cal_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve calendar aging at step K, since it has to be an integer"

            if len(self._f_cal_series) > k:
                return self._f_cal_series[k]
            else:
                raise IndexError("Calendar aging at step K not computed yet")
        return self._f_cal_series

    def get_f_cyc_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve cyclic aging at step K, since it has to be an integer"

            if len(self._f_cyc_series) > k:
                return self._f_cyc_series[k]
            else:
                raise IndexError("Cyclic aging at step K not computed yet")
        return self._f_cyc_series
    
    def get_k_iter_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve aging iteration at step K, since it has to be an integer"

            if len(self._k_iters) > k:
                return self._k_iters[k]
            else:
                raise IndexError("Aging iteration at step K not computed yet")
        return self._k_iters

    def _update_f_cal_series(self, value: float):
        self._f_cal_series.append(value)

    def _update_f_cyc_series(self, value: float):
        self._f_cyc_series.append(value)
    
    def _update_k_iter_series(self, value: int):
        self._k_iters.append(value)
        
    def reset_model(self, **kwargs):
        self._f_cyc = 0
        self._f_cyc_series = []
        self._f_cal_series = []
        self._k_iters = []
        self._deg_series = []
        self._soc_history = []
        self._temp_history = []
        self._mean_soc = 0
        self._mean_temp = 0
        self._dropflow.reset()

    def init_model(self, **kwargs):
        self.update_deg(0)
        self._update_f_cyc_series(0)
        self._update_f_cal_series(0)
        self._update_k_iter_series(0)

    def compute_degradation(self, soc: float, temp: float, elapsed_time: float, k: int, do_check: bool = False):
        """
        Compute the aging of the battery

        Inputs:
        :param f_d:
        """
        self._dropflow.add_point(soc, k)
        
        self._soc_history.append(soc)
        self._temp_history.append(temp)
        self._mean_soc = (soc + (self._mean_soc * (len(self._soc_history) - 1))) / len(self._soc_history)
        self._mean_temp = (temp + (self._mean_temp * (len(self._temp_history) - 1))) / len(self._temp_history)
        
        if not do_check:
            return self._deg_series[-1]
        
        # Compute the calendar aging           
        f_cal = self._compute_calendar_aging(curr_time=elapsed_time)
        self._update_f_cal_series(f_cal)
        
        # Compute the cyclic aging through rainflow algorithm
        extracted_cycles = list(self._dropflow.extract_new_cycles(ignore_stopper=False))
        incomplete_f_cyc = 0
        
        half_cycles = []
                
        for cycle in extracted_cycles[-1::-1]:
            if cycle[2] != 0.5:
                break
            half_cycles.append(cycle)
            extracted_cycles.remove(cycle)
        
        # With Dropflow we retrieve tuples of (range, mean, count, i_start, i_end) associated to each cycle
        for rng, mean, count, i_start, i_end in half_cycles:
            incomplete_f_cyc += self._compute_cyclic_aging(cycle_type=count,
                                                           cycle_dod=rng,
                                                           avg_cycle_soc=mean,
                                                           avg_cycle_temp=np.mean(self._temp_history[i_start:i_end])
                                                           )
            
        for rng, mean, count, i_start, i_end in extracted_cycles:
            self._f_cyc += self._compute_cyclic_aging(cycle_type=count,
                                                      cycle_dod=rng,
                                                      avg_cycle_soc=mean,
                                                      avg_cycle_temp=np.mean(self._temp_history[i_start:i_end])
                                                      )
            
        self._update_f_cyc_series(self._f_cyc + incomplete_f_cyc)
        f_d =  f_cal + self._f_cyc + incomplete_f_cyc
        
        # Compute degradation considering the SEI film factor
        deg = np.clip(1 - self._alpha_sei * np.exp(-self._beta_sei * f_d) - (1 - self._alpha_sei) * np.exp(-f_d),
                    a_min=0., a_max=1.)
                
        self.update_deg(deg)
        self._update_k_iter_series(k)
        
        return deg

    def _compute_calendar_aging(self, curr_time):
        """
        Compute the calendar aging of the battery from the start of the simulation.

        Inputs:
        :param curr_time: wrt the start of the simulation
        :param avg_temp:
        :param avg_soc:
        """
        cal_aging = 1

        # For each defined stress model we get the stress function with relative parameters
        for factor in self._calendar_factors.keys():
            stress_func = getattr(stress_functions, factor + '_stress')
            sim_variables = {}

            if factor == 'time':
                sim_variables['t'] = curr_time

            elif factor == 'temperature':
                sim_variables['mean_temp'] = self._mean_temp

            elif factor == 'soc':
                sim_variables['soc'] = self._mean_soc
            else:
                raise KeyError("Stress factor {} shouldn't be used for computing calendar aging in Bolun model"
                               .format(factor))

            # Product of all calendar aging factors within the loop
            stress_value = stress_func(**self._stress_models_params[factor], **sim_variables)
            cal_aging = cal_aging * stress_value

        return cal_aging

    def _compute_cyclic_aging(self, cycle_type, cycle_dod, avg_cycle_temp, avg_cycle_soc):
        """
        Compute the cyclic aging of the battery.
        The parameters in input belong to a single cycle identify by means of the Rainflow algorithm.

        Inputs:
        :param cycle_type: value that can be 0.5 or 1.0 and  tells if a cycle is a half or a full cycle
        :param cycle_dod:
        :param avg_cycle_temp:
        :param avg_cycle_soc:
        """
        cyclic_aging = cycle_type
        
        # For each defined stress model we get the stress function with relative parameters
        for factor in self._cyclic_factors.keys():
            stress_func = getattr(stress_functions, factor + '_stress')
            sim_variables = {}

            if factor == 'dod_bolun':
                sim_variables['dod'] = cycle_dod

            elif factor == 'dod_quadratic':
                sim_variables['dod'] = cycle_dod

            elif factor == 'dod_exponential':
                sim_variables['dod'] = cycle_dod

            elif factor == 'temperature':
                sim_variables['mean_temp'] = avg_cycle_temp

            elif factor == 'soc':
                sim_variables['soc'] = avg_cycle_soc

            else:
                raise KeyError("Stress factor {} shouldn't be used for computing calendar aging in Bolun model"
                               .format(factor))

            # Product of all cyclic aging factors within the loop
            stress_value = stress_func(**self._stress_models_params[factor], **sim_variables)
            cyclic_aging = cyclic_aging * stress_value
            
        return cyclic_aging

    def get_results(self, **kwargs):
        """
        Returns a dictionary with all final results
        TODO: selection of results by label from config file?
        """
        results = {}
        k = kwargs['k'] if 'k' in kwargs else None
        var_names = kwargs['var_names'] if 'var_names' in kwargs else None
        
        for key, func in self.collections_map.items():
            if var_names is not None and key not in var_names:
                continue
            results[key] = func(k=k)
        
        return results
    
    def clear_collections(self, **kwargs):
        super().clear_collections(**kwargs)
        self._f_cyc_series = [self._f_cyc_series[-1]]
        self._f_cal_series = [self._f_cal_series[-1]]
        self._k_iters = [self._k_iters[-1]]