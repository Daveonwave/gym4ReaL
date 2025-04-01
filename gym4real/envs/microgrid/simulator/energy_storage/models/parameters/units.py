from pint import UnitRegistry
ureg = UnitRegistry()


class Unit:
    # Resistor
    OHM = ureg.ohm
    # Current
    AMPERE = ureg.ampere
    # Voltage
    VOLT = ureg.volt
    # Capacitor
    FARADAY = ureg.faraday
    # Power
    WATT = ureg.watt

    # Hour
    HOUR = ureg.hour
    # Minute
    MINUTE = ureg.minute
    # Second
    SECOND = ureg.second

    # Celsius
    CELSIUS = ureg.degC

    # Kelvin
    KELVIN = ureg.kelvin


    # TODO: posso fare una classe che legge da yaml per decidere quali unita di misura utilizzare.
    #       In quel caso portei fare dei dizionari come attributi della classe

    def __init__(self, **kwargs):
        self.resistance = kwargs['ohm']