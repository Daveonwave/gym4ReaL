import time
import yaml
import logging
import os
import pint.util
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.interpolate import interp1d

logger = logging.getLogger('DT_ernesto')


def load_data_from_csv(csv_file: Path, vars_to_retrieve: [dict], **kwargs):
    """
    Function to preprocess preprocessing that need to be read from a csv table.

    Args:
        csv_file (pathlib.Path): file path of the csv which we want to retrieve preprocessing from
        vars_to_retrieve (list(dict)): variables to retrieve from csv file
    """
    # Check file existence
    if not os.path.isfile(csv_file):
        raise FileNotFoundError("The specified file '{}' doesn't not exist.".format(csv_file))

    df = None
    try:
        df = pd.read_csv(csv_file, encoding='unicode_escape')

        if kwargs['iterations']:
            df = df.iloc[:kwargs['iterations']]
    except IOError:
        logger.error("The specified file '{}' cannot be imported as a Pandas Dataframe.".format(csv_file))

    # Retrieve and convert timestamps to list of seconds (format: YYYY/MM/DD hh:mm:ss)
    if kwargs['time_format'] == 'seconds':
        timestamps = df['Time']
    else:
        timestamps = pd.to_datetime(df['Time'], format="%Y/%m/%d %H:%M:%S").values.astype(float) // 10 ** 9
    vars_data = {}

    # We first check if the variable column label exists
    for var in vars_to_retrieve:
        if var['label'] not in df.columns:
            raise NameError("Label {} is not present among df columns [{}]".format(var['label'], df.columns))
        else:
            vars_data[var['var']] = _validate_data_unit(df[var['label']].values.tolist(), var['var'], var['unit'])

    return timestamps.tolist(), vars_data


def sync_data_with_step(times: list, data: dict, sim_step: float, interp: bool = False):
    """
    # TODO: to be removed
    Augmentation or reduction of the ground dataset in order to adapt it to the specified simulator timestep.
    If the simulator timestamp is smaller than the time delta, we need to replicate the previous values or interpolate
    data to coherently extend the dataset.
    If the simulator timestamp is bigger, instead, we need to skip some input data and/or interpolate if necessary.

    Args:
        times ():
        data ():
        sim_step ():
        interp ():
    """
    if interp:
        logger.error("Interpolation of ground data has not been implemented yet! "
                     "The values will be just replicated by existing data!")
        interp = False

    sync_times = [times[0]]
    sync_data = {key: [data[key][0]] for key in data.keys()}

    i = 1
    while i < len(times):
        dt = times[i] - times[i - 1]

        # If sim_step is smaller than dt, we perform data augmentation by duplicating data in between
        if sim_step < dt:
            # Compute the floor of the integer division and extend the new times list
            floor = int(dt // sim_step)
            sync_times.extend([times[i - 1] + j * sim_step for j in range(1, floor)])

            if not interp:
                aug_data = {key: [data[key][i - 1]] * (floor - 1) for key in sync_data.keys()}
            else:
                raise NotImplementedError()

            # Extend the new data dictionary with repeated data
            [sync_data[key].extend(aug_data[key]) for key in sync_data.keys()]

            sync_times.append(times[i])
            [sync_data[key].append(data[key][i]) for key in sync_data.keys()]
            i += 1

        elif sim_step == dt:
            sync_times.append(times[i])
            [sync_data[key].append(data[key][i]) for key in sync_data.keys()]
            i += 1

        # If sim_step is greater than dt, we perform data reduction by deleting data in between
        else:
            # Deleting all the skipped data
            [data[key].pop(i) for key in sync_data.keys()]
            times.pop(i)

    return sync_times, sync_data


def _validate_data_unit(data_list, var_name, unit):
    """
    # TODO: to be removed
    Function to validate and adapt preprocessing unit to internal simulator units.

    Args:
        data_list (list): list with values of a preprocessing stream
        var_name (str): name of the variable
        unit (str): unit of the variable
    """
    # Unit employed is already compliant with internal simulator units
    if unit == internal_units[var_name][1]:
        return data_list

    try:
        tmp_data = data_list * ureg.parse_units(unit)
        transformed_data = tmp_data.to(internal_units[var_name][2])
        logger.info("Ground variable '{}' has been converted from [{}] to [{}]"
                    .format(var_name, unit, internal_units[var_name][1]))

    except pint.PintError as e:
        logger.error("UnitError on '{}': ".format(var_name), e)
        exit(1)

    return transformed_data.magnitude.tolist()


def validate_yaml_parameters(param_dict) -> dict:
    """
    Function to validate and adapt units of provided parameters to internal simulator units.

    Args:
        param_dict (dict): dictionary of parameters (read by for example yaml config file)
    """
    transformed_dict = {}

    for key in param_dict.keys():
        param = param_dict[key]

        # Check if the parameter has a unit measure with a dictionary structure
        if isinstance(param, dict):
            # Parameter unit measure is not compliant with internal simulator units
            if param['unit'] != internal_units[param['var']][1]:
                try:
                    tmp_param = param['value'] * ureg.parse_units(param['unit'])
                    transformed_dict[key] = tmp_param.to(internal_units[param['var']][2]).magnitude

                except pint.PintError as e:
                    logger.error("UnitError on '{}': ".format(param['var']), e)
                    exit(1)
            else:
                transformed_dict[key] = param['value']
        else:
            transformed_dict[key] = param_dict[key]

    return transformed_dict


def read_yaml(yaml_file: str):
    with open(yaml_file, 'r') as fin:
        params = yaml.safe_load(fin)
    return params