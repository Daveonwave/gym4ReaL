import pandas as pd
import datetime
from pathlib import Path
from tqdm import tqdm
from .epynet import utils
from epynet.network import Network


class WaterDistributionSystem(Network):
    """
    Class of the network inherited from Epynet.Network
    """
    def __init__(self, inpfile: str):
        super().__init__(inputfile=inpfile)
        self.times = []
        self.sensor_plcs = []

        # Interactive flag can be set in run() or in init_simulation() if you want to build manually the step-by-step
        self.network_state = pd.Series()

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

    def create_df_reports(self, do_create_nodes_report=False, do_create_links_report=False):
        """
        Create nodes and links report dataframes - 3 level dataframe
        How to access: df['node', 'id', 'property'] -> column
        TODO: create a unique 4 level dataframe with 0 level distinguishing between node and link
        """
        if self.df_nodes_report is not None:
            del self.df_nodes_report
        if self.df_links_report is not None:
            del self.df_links_report

        if do_create_nodes_report:
            tanks_ids = [uid for uid in self.tanks.uid]
            junctions_ids = [uid for uid in self.junctions.uid]
            tanks_iterables = [['tanks'], tanks_ids, ['head', 'pressure']]
            junct_iterables = [['junctions'], junctions_ids,
                               ['head', 'pressure', 'basedemand', 'demand', 'demand_deficit']]
            tanks_indices = pd.MultiIndex.from_product(iterables=tanks_iterables, names=["node", "id", "properties"])
            junctions_indices = pd.MultiIndex.from_product(iterables=junct_iterables, names=["node", "id", "properties"])

            # We use timestamp as index for both nodes and links dataframes
            times = [datetime.timedelta(seconds=time) for time in self.times]

            # Nodes dataframes creation
            df_tanks = pd.DataFrame(columns=tanks_indices, index=times)
            df_junctions = pd.DataFrame(columns=junctions_indices, index=times)

            # Dataframe filling
            for i, j in zip(df_tanks.columns.get_level_values(1), df_tanks.columns.get_level_values(2)):
                df_tanks['tanks', i, j] = self.tanks.results[i][j]
            for i, j in zip(df_junctions.columns.get_level_values(1), df_junctions.columns.get_level_values(2)):
                df_junctions['junctions', i, j] = self.junctions.results[i][j]

            self.df_nodes_report = pd.concat([df_tanks, df_junctions], axis=1)

        if do_create_links_report:
            # We can assume that there is always at least one pump in each network, since would be pointless to study a
            # wds without this kind of links.
            pumps_ids = [uid for uid in self.pumps.uid]
            pumps_iterables = [['pumps'], pumps_ids, ['flow', 'energy', 'status']]
            pumps_indices = pd.MultiIndex.from_product(iterables=pumps_iterables, names=["link", "id", "properties"])

            # We use timestamp as index for both nodes and links dataframes
            times = [datetime.timedelta(seconds=time) for time in self.times]
            df_pumps = pd.DataFrame(columns=pumps_indices, index=times)

            # Pump dataframe filling and columns renaming
            for i, j in zip(df_pumps.columns.get_level_values(1), df_pumps.columns.get_level_values(2)):
                df_pumps['pumps', i, j] = self.pumps.results[i][j]

            self.df_links_report = df_pumps

            # We cannot do the same assumption for valves, as we can see in "anytown" network
            if self.valves:
                valves_ids = [uid for uid in self.valves.uid]
                valves_iterables = [['valves'], valves_ids, ['velocity', 'flow', 'status']]
                valves_indices = pd.MultiIndex.from_product(iterables=valves_iterables, names=["link", "id", "properties"])

                df_valves = pd.DataFrame(columns=valves_indices, index=times)

                # Valves dataframe filling and columns renaming
                for i, j in zip(df_valves.columns.get_level_values(1), df_valves.columns.get_level_values(2)):
                    df_valves['valves', i, j] = self.valves.results[i][j]

                self.df_links_report = pd.concat([df_pumps, df_valves], axis=1)
