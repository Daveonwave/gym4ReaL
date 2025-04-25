import pandas as pd
from collections import defaultdict
from .epynet import utils
from .epynet.network import Network


class WaterNetwork(Network):
    """
    Class of the network inherited from Epynet.Network
    """
    def __init__(self, inpfile: str):
        super().__init__(inputfile=inpfile)
        self.times = []
        self.duration = 0
        self.hydraulic_step = 0
        self.pattern_step = 0

    def set_time_params(self, duration=None, hydraulic_step=None, pattern_step=None, report_step=None, start_time=None,
                        rule_step=None):
        """
        Set the time parameters before the simulation (unit: seconds)
        :param duration: EN_DURATION
        :param hydraulic_step: EN_HYDSTEP
        :param pattern_step: EN_PATTERNSTEP
        :param report_step: EN_REPORTSTEP
        :param start_time: EN_STARTTIME
        :param rule_step: EN_RULESTEP
        """
        self.duration = duration if duration is not None else self.duration
        self.hydraulic_step = hydraulic_step if hydraulic_step is not None else self.hydraulic_step
        self.pattern_step = pattern_step if pattern_step is not None else self.pattern_step
        
        if duration is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_DURATION'), duration)
        if hydraulic_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_HYDSTEP'), hydraulic_step)
        if pattern_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_PATTERNSTEP'), pattern_step)
        if report_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_REPORTSTEP'), report_step)
        if start_time is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_STARTTIME'), start_time)
        if rule_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_RULESTEP'), rule_step)

    def set_demand_pattern(self, uid: str, values=None, junctions=None):
        """
        Set a base-demand pattern for junctions if exists, otherwise it creates and set a new pattern
        :param uid: pattern id
        :param values: list of multipliers, None if already existing
        :param junctions: list of junction objects to which we want to set the pattern
        """
        if values is None:
            if uid not in self.patterns.uid:
                raise KeyError("Chosen pattern id doesn't exist")
        else:
            if uid in self.patterns.uid:
                self.patterns[uid].values = values
            else:
                self.add_pattern(uid, values)
        if junctions:
            for junc in junctions:
                junc.pattern = uid

    def demand_model_summary(self):
        """
        Print information related to the current demand saved_models
        """
        dm_type, pmin, preq, pexp = self.ep.ENgetdemandmodel()
        if dm_type == 0:
            print("Running a demand driven analysis...")
        else:
            print("Running a pressure driven analysis...")
            print("-> Minimum pressure: {:.2f}".format(pmin))
            print("-> Required pressure: {:.2f}".format(preq))
            print("-> Exponential pressure: {:.2f}".format(pexp))
            
    def init_simulation(self, interactive=False):
        """
         Initialize the network simulation
        """
        self.interactive = interactive
        self.reset()
        self.times = []
        self.ep.ENopenH()
        self.ep.ENinitH(flag=0)

    def run(self, interactive=False, status_dict=None):
        """
        Run method wrapper to set the interactivity option (and others in the future related to RL)
        :param interactive: to update the actuators with own values
        :param status_dict: dictionary with predefined updates (just to test, it will be removed)
        TODO: remove status_dict
        """
        if self.solved:
            self.reset()

        global actuators_update_dict
        if status_dict and interactive:
            actuators_update_dict = status_dict
            self.interactive = interactive
        else:
            self.interactive = False
        
        self.init_simulation(interactive=self.interactive)
        curr_time = 0
        timestep = 1

        # Timestep becomes 0 at the last hydraulic step
        while timestep > 0:
            timestep = self.simulate_step(curr_time=curr_time)
            curr_time += timestep

        self.ep.ENcloseH()
        self.solved = True

    def simulate_step(self, curr_time):
        """
        Simulation of one step from the given time
        :param curr_time: current simulation time
        :return: time until the next event, if 0 the simulation is going to end
        """
        # uids = ['P78', 'P79']
        self.ep.ENrunH()
        timestep = self.ep.ENnextH()

        # Append new values to reports
        self.times.append(curr_time)
        self.load_attributes(curr_time)
        return timestep

    def update_pumps(self, new_status):
        """
        Set actuators (pumps and valves) status to a new current state
        :param new_status: dictionary of pumps with next value for their status
        """
        step_updates = {}

        for uid in new_status.keys():
            if self.links[uid].status != new_status[uid]:
                step_updates[uid] = 1
            else:
                step_updates[uid] = 0
            self.links[uid].status = new_status[uid]

        return step_updates

    def get_network_state(self):
        """
        Retrieve the current values of the network in the form a pandas series of dictionaries.
        The collected values are referred to:
            - tanks: {pressure}
            - junctions: {pressure}
            - pumps: {status, flow}
            - valves: {status, flow}
        :return: the series with the above enlisted values
        """
        network_state = pd.Series()
        for uid in self.tanks.results.index.append(self.junctions.results.index):
            nodes_dict = {key: self.nodes[uid].results[key][-1] for key in ['pressure']}
            network_state[uid] = nodes_dict

        if self.valves:
            for uid in self.pumps.results.index.append(self.valves.results.index):
                links_dict = {key: self.links[uid].results[key][-1] for key in ['status', 'flow']}
                network_state[uid] = links_dict
        else:
            for uid in self.pumps.results.index:
                links_dict = {key: self.pumps[uid].results[key][-1] for key in ['status', 'flow']}
                network_state[uid] = links_dict
        return network_state
    
    def get_snapshot(self, current_pattern_step=None):
        state = defaultdict(dict)
          
        for link in self.pumps:
            state[link.uid] = self._get_link(link.uid)
        for node in self.nodes:
            state[node.uid] = self._get_node(node.uid)
        
        return state
    
    def _get_link(self, uid:str):
        """
        Get the components of the network
        :return: list of components
        """
        data = {}
        for prop in self.links[uid].properties:
            data[prop] = self.links[uid].results[prop][-1] if prop in self.links[uid].results else []
        for prop in self.links[uid].static_properties:
            data[prop] = getattr(self.links[uid], prop) if prop in self.links[uid].static_properties else []
        return data   
    
    def _get_node(self, uid:str):
        """
        Get the components of the network
        :return: list of components
        """
        data = {}
        for prop in self.nodes[uid].properties:
            data[prop] = self.nodes[uid].results[prop][-1] if prop in self.nodes[uid].results else []
        for prop in self.nodes[uid].static_properties:
            data[prop] = getattr(self.nodes[uid], prop) if prop in self.nodes[uid].static_properties else []
            if uid.startswith('J'):
                data['multiplier'] = self.nodes[uid].pattern.values[(self.duration // self.pattern_step) % len(self.nodes[uid].pattern.values)]
        return data
        