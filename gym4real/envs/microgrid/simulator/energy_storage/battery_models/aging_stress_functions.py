import numpy as np


def temperature_stress(k_temp, mean_temp, temp_ref):
    """
    Stress function caused by the temperature, to be computed in Kelvin.
    This can be used only for batteries operating above 15°C.
    
    Inputs:
    :param k_temp: temperature stress coefficient
    :param mean_temp: current battery temperature
    :param temp_ref: ambient temperature
    """
    return np.exp(np.float32(k_temp) * (mean_temp - temp_ref) * (temp_ref / mean_temp))


def soc_stress(k_soc, soc, soc_ref):
    """
    Stress function caused by the SoC (State of Charge).
    
    Inputs:
    :param k_soc: soc stress coefficient
    :param soc: current battery soc
    :param soc_ref: reference soc level, usually around 0.4 to 0.5
    """
    return np.exp(k_soc * (soc - soc_ref))


def time_stress(k_t, t):
    """
    Stress function of calendar elapsed time

    Inputs:
    :param k_t: time stress coefficient
    :param t: current battery age
    """
    return np.float32(k_t) * t


def dod_bolun_stress(dod, k_delta1, k_delta2, k_delta3):
    """
    Stress function caused by DoD (Depth of Discharge) presented in Bolun's paper.
    This is more accurate to fit with LMO batteries.
    
    Inputs:
    :param dod: current depth of discharge
    :param k_delta1: first dod stress coefficient
    :param k_delta2: second dod stress coefficient
    :param k_delta3: third dod stress coefficient
    """
    return (np.float32(k_delta1) * dod ** np.float32(k_delta2) + np.float32(k_delta3)) ** (-1)


def dod_exponential_stress():
    """
    Stress function caused by DoD (Depth of Discharge) in alternative to Bolun's DoD stess.
    """
    pass


def dod_quadratic_stress():
    """
    Stress function caused by DoD (Depth of Discharge) in alternative to Bolun's DoD stess.
    """
    pass

