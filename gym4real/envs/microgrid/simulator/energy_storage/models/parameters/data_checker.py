import pint.util
from pint import UnitRegistry
ureg = UnitRegistry()


def check_data_unit(param, unit: ureg.Unit):
    """

    """
    # Param is Quantity
    if isinstance(param, pint.Quantity):
        new_magnitude = param.magnitude

        # Check the magnitude: it has to be a float value
        if not isinstance(param.magnitude, float):
            new_magnitude = float(param.magnitude)

        # Check unit
        if str(param.units) != str(unit):
            # Same dimensionality: convert it to the required unit
            if param.dimensionality == unit.dimensionality:
                return (new_magnitude * param.units).to(unit)
            # Different dimensionality
            else:
                raise Exception("Provided preprocessing {} has a different unit measure from required one ({})"
                                .format(param.units, unit))
        return new_magnitude * param.units

    # Create the Quantity variable if the param is int, float or numeric string
    elif isinstance(param, float) or isinstance(param, int) or isinstance(param, str):
        return craft_data_unit(param, unit)

    # Param is something else
    else:
        raise Exception(
            "Provided preprocessing {} has a wrong format. You can pass ".format(param) + \
            "'Pint.Quantity' objects in accordance with the required unit measure"
        )


def craft_data_unit(param, unit: ureg.Unit):
    """

    """
    # Param is float or int: convert it to float quantity
    if isinstance(param, float) or isinstance(param, int):
        return ureg.Quantity(float(param), unit)

    elif isinstance(param, str) and param.isnumeric():
        return ureg.Quantity(float(param), unit)

    elif isinstance(param, pint.Quantity):
        return ureg.Quantity(float(param.magnitude), unit)

    # Param is something else
    else:
        raise Exception(
            "Provided preprocessing {} has a wrong format. You can pass 'int', 'float', 'pint.Quantity' or even 'str' "
            "values".format(param)
        )


