from pathlib import Path
import pandas as pd
from gym4real.envs.wds.simulator.wds import WaterDistributionSystem

# TODO: discuss the hierarchy (can differ between sensor and actuators).
# Used when we have concurrent attacks to decide which one has to be applied before.
attackers_hierarchy = ['NetworkDelay', 'DOS', 'MITM']


class GenericPLC:
    def __init__(self, name, wn: WaterDistributionSystem, plc_variables):
        """
        :param name:
        :param wn:
        :param plc_variables:
        :param attackers:
        """
        self.name = name
        self.wn = wn

        self.owned_vars = plc_variables
        self.attackers = None
        self.active_attackers = []

        self.elapsed_time = 0
        self.ongoing_attack_flag = 0

        self.on_eval = False
        self.ground_readings = {}
        self.altered_readings = {}

    def get_own_vars(self):
        # initialize self.owned_vars through a config file
        pass

    def reset_collected_data(self):
        pass

    def is_on_eval(self, flag: bool):
        self.on_eval = flag

    def set_attackers(self, attackers=None):
        """
        Set the attackers as a list of attacker objects
        """
        self.attackers = attackers

        #if self.on_eval:
        for object_type in self.owned_vars.keys():
            for var in self.owned_vars[object_type].keys():
                self.altered_readings[var] = {}
                for prop in self.owned_vars[object_type][var]:
                    self.altered_readings[var][prop] = []
        #else:
        #    self.altered_readings = {}

    def apply(self, var_list):
        pass

    def save_ground_data(self, output_file, seed):
        """

        """
        Path(output_file).mkdir(parents=True, exist_ok=True)
        dict_for_df = {(outerKey, innerKey): values for outerKey, innerDict in self.ground_readings.items()
                       for innerKey, values in innerDict.items()}
        df_ground = pd.DataFrame(dict_for_df, index=self.wn.times)
        csv_filename = self.name + "_ground-" + str(seed) + ".csv"
        df_ground.to_csv(output_file / csv_filename)

    def save_altered_data(self, output_file, seed):
        """

        """
        if self.altered_readings:
            Path(output_file).mkdir(parents=True, exist_ok=True)
            dict_for_df = {(outerKey, innerKey): values for outerKey, innerDict in self.altered_readings.items()
                           for innerKey, values in innerDict.items()}
            df_altered = pd.DataFrame(dict_for_df, index=self.wn.times)
            csv_filename = self.name + "_altered-" + str(seed) + ".csv"
            df_altered.to_csv(output_file / csv_filename)

    def check_for_ongoing_attacks(self):
        """
        Check if there are ongoing attacks and return 1 in that case. Moreover, it extracts active attackers to apply
        the injection of the attack.
        """
        self.ongoing_attack_flag = 0

        if self.attackers:
            curr_attackers = [attacker for attacker in self.attackers
                              if attacker.event_start <= self.elapsed_time < attacker.event_end]
            if curr_attackers:
                self.active_attackers = []
                # Order the attackers following a predefined hierarchy
                for item in attackers_hierarchy:
                    self.active_attackers.extend([attacker for attacker in curr_attackers
                                                  if type(attacker).__name__ == item])
                # print(self.name, [type(att).__name__ for att in self.active_attackers])
                self.ongoing_attack_flag = 1

        return self.ongoing_attack_flag


class SensorPLC(GenericPLC):

    def init_readings(self, var, prop):
        """

        """
        #if self.on_eval:
        #    self.reset_collected_data()
        #else:
        #    self.ground_readings = {}

        if var in self.owned_vars['nodes']:
            return getattr(self.wn.nodes[var], prop)
        else:
            raise Exception("Variable {} is not controlled by {}".format(var, self.name))

    def reset_collected_data(self):
        """

        """
        for object_type in self.owned_vars.keys():
            for var in self.owned_vars[object_type].keys():
                self.ground_readings[var] = {}
                for prop in self.owned_vars[object_type][var]:
                    self.ground_readings[var][prop] = []

        if self.altered_readings:
            for object_type in self.owned_vars.keys():
                for var in self.owned_vars[object_type].keys():
                    self.altered_readings[var] = {}
                    for prop in self.owned_vars[object_type][var]:
                        self.altered_readings[var][prop] = []

    def apply(self, var_list=None):
        """
        Reads data from the physical process
        """
        readings = {}
        # time used to print results
        self.elapsed_time = self.wn.times[-1]

        # TODO: list of data passed as parameter in config file
        for object_type in self.owned_vars.keys():
            for var in self.owned_vars[object_type].keys():
                readings[var] = {}
                for prop in self.owned_vars[object_type][var]:
                    readings[var][prop] = getattr(self.wn, object_type)[var].results[prop][-1]
                    self.ground_readings[var][prop].append(readings[var][prop])
                    #if self.ground_readings:
                    #    self.ground_readings[var][prop].append(readings[var][prop])

        # Apply attacks
        if self.check_for_ongoing_attacks():
            for attacker in self.active_attackers:
                readings = attacker.apply_attack(readings)

        # Update altered readings
        if self.altered_readings:
            for var in readings.keys():
                #print(var)
                for prop in readings[var].keys():
                    self.altered_readings[var][prop].append(readings[var][prop])

        readings['under_attack'] = self.ongoing_attack_flag
        #print([attacker.old_readings for attacker in self.active_attackers if attacker.name == 'attack3'])
        return readings


class ActuatorPLC(GenericPLC):

    def __init__(self, name, wn: WaterDistributionSystem, plc_variables, n_actions):
        super().__init__(name, wn, plc_variables)
        self.n_actions = n_actions

    def apply(self, action):
        """
        Reads data from agent's action space and apply them into the physical process
        """
        new_status_dict = {pump_id: 0 for pump_id in self.owned_vars['links'].keys()}

        # Action translated in binary value with one bit for each pump status
        if self.n_actions == 4:
            bin_action = '{0:0{width}b}'.format(action[0], width=len(self.owned_vars['links'].keys()))
        elif self.n_actions == 3:
            if action == 0:
                bin_action = "00"
            elif action == 1:
                bin_action = "10"
            else:
                bin_action = "11"
        else:
            raise Exception("Number of selected actions not supported!")

        for i, key in enumerate(new_status_dict.keys()):
            new_status_dict[key] = int(bin_action[i])

        # Apply attacks
        if self.check_for_ongoing_attacks():
            for attacker in self.attackers:
                new_status_dict = attacker.apply_attack(new_status_dict)

        # TODO: understand how to handle attacks from actuators perspective

        # Update pump status
        return self.wn.update_pumps(new_status=new_status_dict)

