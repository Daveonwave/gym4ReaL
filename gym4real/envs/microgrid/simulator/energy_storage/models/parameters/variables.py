from typing import Union

import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from ernestogym.ernesto.energy_storage.preprocessing.utils import _validate_data_unit
import pandas as pd

params_csv_folder = 'ernestogym/ernesto/data/battery/params/'


class GenericVariable:
    """

    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def get_value(self, input_vars: dict):
        raise NotImplementedError

    def set_value(self, new_value):
        raise NotImplementedError


class Scalar(GenericVariable):
    """

    """

    def __init__(self, name: str, value: Union[int, float]):
        super().__init__(name)
        self._value = value

    def get_value(self, input_vars: dict = None):
        return self._value

    def set_value(self, new_value: float):
        self._value = new_value


class Function:
    """

    """
    pass


# TODO: understand how to handle Parametric Functions
class FunctionTerm:
    """

    """

    def __init__(self, variable, coefficient, operation, degree):
        self._variable = variable
        self._coefficient = coefficient
        self._operation = operation
        self._degree = degree


class ParametricFunction(GenericVariable):
    """

    """

    def __init__(self, name: str, function_terms: dict):
        super().__init__(name)
        self.function_terms = function_terms
        self._check_function_terms_format()

        self.input_vars = function_terms.keys()

    def _check_function_terms_format(self):
        pass

    def get_value(self, **params):
        result = 0
        for j, var in enumerate(self.input_vars):
            degrees = [deg for deg in range(len(self.coefficients[j]))]


class LookupTableFunction(GenericVariable):
    """

    """

    def __init__(self, name: str, y_values: list, x_names: list, x_values: list):
        super().__init__(name)
        self.y_values = y_values
        self.x_names = x_names
        self.x_values = x_values

        self._function = None
        self._backup_function = None

        if len(x_names) == 1:
            self._function = interp1d(x_values[0], y_values, fill_value='extrapolate')

        elif len(x_names) > 1:
            x_points = [[l[i] for l in self.x_values] for i in range(len(self.x_values[0]))]
            self._function = LinearNDInterpolator(points=np.array(x_points, dtype=np.float32),
                                                  values=np.array(self.y_values, dtype=np.float32))
            self._backup_function = NearestNDInterpolator(x=np.array(x_points, dtype=np.float32),
                                                          y=np.array(self.y_values, dtype=np.float32))
        else:
            raise Exception("Too many variables to interpolate, not implemented yet!")

    def get_value(self, input_vars: dict):
        """
        Retrieve the result of the interpolation function from the lookup table.
        """
        input_values = []

        for expected_input, given_input in zip(self.x_names, input_vars.keys()):

            if expected_input != given_input:
                raise Exception("Given inputs aren't correct for the computation of {}! Required inputs are {}.".format(
                    self.name, self.x_names))

            input_values.append(input_vars[given_input])

        if isinstance(self._function, interp1d):
            return float(self._function(*input_values))

        elif isinstance(self._function, LinearNDInterpolator):
            res = float(self._function(*input_values))
            if np.isnan(res):
                res = float(self._backup_function(*input_values))
            return res

        else:
            raise Exception("Given inputs list has a wrong dimension for the computation of {}".format(self.name))

    def get_y_values(self):
        """
        Get y_values from which is extracted the result of the interpolation function.
        """
        return self._function.values

    def set_value(self, new_values: np.ndarray):
        """
        Set the values of the lookup table
        """
        self._function.values = new_values
        # raise AttributeError("Is impossible to modify the values within the lookup table of the parameter {}".
        #                      format(self.name))

    def render(self):
        data_list = self.x_values.copy()
        names_list = self.x_names.copy()
        data_list.append(self.y_values)
        names_list.append(self.name)
        table = pd.DataFrame(data={name: values for name, values in zip(names_list, data_list)})
        print(table)


def instantiate_variables(var_dict: dict) -> dict:
    """
    # TODO: cambiare configurazione dati in ingresso (esempio: LookupTable passata con un csv)
    """
    instantiated_vars = {}

    for var in var_dict.keys():

        if var_dict[var]['selected_type'] == "scalar":
            instantiated_vars[var] = Scalar(name=var, value=var_dict[var]['scalar'])

        elif var_dict[var]['selected_type'] == "function":
            instantiated_vars[var] = Function()  # TODO: implement

        elif var_dict[var]['selected_type'] == "lookup":
            # Hardcoded lookup table
            if 'table' not in var_dict[var]['lookup'].keys():
                instantiated_vars[var] = LookupTableFunction(
                    name=var,
                    y_values=var_dict[var]['lookup']['output'],
                    x_names=var_dict[var]['lookup']['inputs'].keys(),
                    x_values=[var_dict[var]['lookup']['inputs'][key] for key in
                              var_dict[var]['lookup']['inputs'].keys()]
                )
            # Csv lookup table
            else:
                table = pd.read_csv(params_csv_folder + var_dict[var]['lookup']['table'])
                instantiated_vars[var] = LookupTableFunction(
                    name=var_dict[var]['lookup']['output']['label'],
                    y_values=_validate_data_unit(data_list=table[var_dict[var]['lookup']['output']['label']].tolist(),
                                                 var_name=var_dict[var]['lookup']['output']['var'],
                                                 unit=var_dict[var]['lookup']['output']['unit']),
                    x_names=[var['label'] for var in var_dict[var]['lookup']['inputs']],
                    x_values=[_validate_data_unit(data_list=table[var['label']].tolist(),
                                                  var_name=var['var'],
                                                  unit=var['unit'])
                              for var in var_dict[var]['lookup']['inputs']]
                )

        else:
            raise Exception("The chosen 'type' for the variable '{}' is wrong or nonexistent! Try to select another"
                            " option among this list: ['scalar', 'function', 'lookup'].".format(var))

    return instantiated_vars
