def dsr(readings:dict):
    """
    Compute the demand supply ratio (DSR).
    It is computed as the ratio between the sum of the demand and the sum of the basedemand multiplied by
    the current pattern multiplier.
    The variable that can be used provided by EPANET is the demand deficit.

    Args:
        readings (dict): 

    Returns:
        float: 
    """ 
    supplies = []
    expected_demand = []
    
    for key, value in readings.items():
        if key.startswith('J'):
            supplies.append(value['demand'])
            expected_demand.append(value['basedemand']* value['multiplier'])

    total_supplied = sum(supplies)
    total_expected = sum(expected_demand)
    
    if total_expected > 0:
        dsr = total_supplied / total_expected if total_supplied / total_expected <= 1 else float(1)
    else:
        dsr = float(0)
    return dsr


def deficit(readings:dict):
    """
    Compute the demand deficit.
    It is computed as the sum of the demand deficit for each junction.
    The variable that can be used provided by EPANET is the demand deficit.
    The demand deficit is the difference between the expected demand and actual demand.
    
    Args:
        readings (dict): 

    Returns:
        float: 
    """
    deficit = 0
    
    for key, value in readings.items():
        if key.startswith('J'):
            deficit += value['demand_deficit'] if value['demand_deficit'] > 0 else 0
            
    return deficit


def overflow(readings:dict, risk_percentage=0.95):
    """
    Check if there is an overflow problem in the tanks. We have an overflow if after one hour we the tank is
    still at the maximum level.
    :return: penalty value
    """
    overflow = 0

    for key, value in readings.items():
        if key.startswith('T'):
            if value['pressure'] > value['maxlevel'] * risk_percentage:
                out_bound = value['pressure'] - (value['maxlevel'] * risk_percentage)
                # Normalization of the out_bound pressure
                overflow += out_bound / ((1 - risk_percentage) * value['maxlevel'])
    
    return overflow
                
        
def check_pumps_flow(self):
    """
    TODO: to implement and substitute to update penalty
    """
    total_flow = 0
    lowest_flow = 0
    highest_flow = 1000      # retrieved with empirical experiments

    for pump in self.action_vars:
        for sensor in self.sensor_plcs:
            if pump in self.readings[sensor._name].keys():
                total_flow += self.readings[sensor._name][pump]['flow']

    # we return as penalty the max-min normalized flow
    return (total_flow - lowest_flow) / (highest_flow - lowest_flow)

def check_pumps_updates(self, step_updates:dict, simple=True):
    """
    Check whether pumps status is updated too frequently.
    It looks at the previous simulation step and collects the state of pumps: if there was an update, the method
    compute an incremental penalty in order to avoid too frequent and adjacent updates.
    """
    pumps_update_penalty = 0

    if simple:
        for pump in self.action_vars:
            if step_updates[pump] > 0:
                pumps_update_penalty += 1

    return pumps_update_penalty