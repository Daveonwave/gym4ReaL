from pathlib import Path
import pandas as pd
from collections import defaultdict
from gym4real.envs.wds.simulator.wn import WaterNetwork

# TODO: discuss the hierarchy (can differ between sensor and actuators).
# Used when we have concurrent attacks to decide which one has to be applied before.
attackers_hierarchy = ['NetworkDelay', 'DOS', 'MITM']


class GenericPLC:
    def __init__(self, name, wn: WaterNetwork, plc_variables):
        self.name = name
        self._wn = wn

        self._owned_info = plc_variables
        self._variables = []
        if 'nodes' in self._owned_info:
            self._variables.extend([key for key, item in self._owned_info['nodes'].items()])
        if 'links' in self._owned_info:
            self._variables.extend([key for key, item in self._owned_info['links'].items()])
        
        self._attackers = None
        self._active_attackers = []

    @property
    def variables(self):
        return self._variables

    def reset_collected_data(self):
        pass

    def is_on_eval(self, flag: bool):
        self.on_eval = flag

    def set_attackers(self, attackers:list):
        """
        Set the attackers as a list of attacker objects
        """
        self._attackers = attackers

    def apply(self, **kwargs):
        pass

    def _check_for_ongoing_attacks(self, curr_time):
        """
        Check if there are ongoing attacks and return 1 in that case. Moreover, it extracts active attackers to apply
        the injection of the attack.
        """
        is_attack_ongoing = False

        if self._attackers:
            curr_attackers = [attacker for attacker in self._attackers
                              if attacker.event_start <= curr_time < attacker.event_end]
            if curr_attackers:
                self._active_attackers = []
                # Order the attackers following a predefined hierarchy
                for item in attackers_hierarchy:
                    self._active_attackers.extend([attacker for attacker in curr_attackers
                                                  if type(attacker).__name__ == item])
                is_attack_ongoing = True

        return is_attack_ongoing


class SensorPLC(GenericPLC):

    def init_readings(self, var, prop):
        if var in self._owned_info['nodes']:
            return getattr(self._wn.nodes[var], prop)
        else:
            raise Exception("Variable {} is not controlled by {}".format(var, self.name))

    def apply(self):
        """
        Reads data from the physical process
        """
        readings = {}
        non_altered_readings = {}
        is_attacked = False
        
        # time used to print results
        elapsed_time = self._wn.times[-1]

        # TODO: list of data passed as parameter in config file
        for object_type in self._owned_info.keys():
            for var in self._owned_info[object_type].keys():
                if object_type == 'nodes':
                    readings[var] = self._wn._get_node(var)
                elif object_type == 'links':
                    readings[var] = self._wn._get_link(var)
                else:
                    raise Exception("Object type {} is not supported".format(object_type))

        # Apply attacks
        if self._check_for_ongoing_attacks(elapsed_time):
            non_altered_readings = readings.copy()
            is_attacked = True
            for attacker in self._active_attackers:
                readings = attacker.apply_attack(readings)
                
        readings['under_attack'] = 1 if is_attacked else 0
        return readings, non_altered_readings


class ActuatorPLC(GenericPLC):

    def __init__(self, name, wn: WaterNetwork, plc_variables, n_actions):
        super().__init__(name, wn, plc_variables)
        self.n_actions = n_actions

    def apply(self, action, curr_time):
        """
        Reads data from agent's action space and apply them into the physical process
        """
        pump_actuation = {pump_id: 0 for pump_id in self._owned_info['links'].keys()}

        # Action translated in binary value with one bit for each pump status
        bin_action = '{0:0{width}b}'.format(action, width=len(self._owned_info['links'].keys()))

        for i, key in enumerate(pump_actuation.keys()):
            pump_actuation[key] = int(bin_action[i])
        
        # Apply attacks
        if self._check_for_ongoing_attacks(curr_time):
            for attacker in self._attackers:
                pump_actuation = attacker.apply_attack(pump_actuation)

        # Update pump status
        return self._wn.update_pumps(new_status=pump_actuation)

