import pandas as pd
import numpy as np
import random
import yaml
from pathlib import Path
from gymnasium import Env, spaces
from . import objFunction
from .simulator.generic_plc import SensorPLC, ActuatorPLC
from physical_process import WaterDistributionNetwork
from network_attacks.generic_attacker import MITM, DOS, NetworkDelay

demand_pattern_folder = Path("/data/davidesal/epynet_cps/demand_patterns")


class WaterDistributionNetworkEnv(Env):
    """
    Custom Environment that follows gym interface for the Water Distribution System
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self,
                 settings
                 #env_config_file,
                 #demand_patterns_file,
                 #attacks_train_file=None,
                 #attacks_test_file=None,
                 #logger=None
                 ):
        super().__init__()

        #with open(env_config_file, 'r') as fin:
        #    env_config = yaml.safe_load(fin)
        #with open(demand_patterns_file, 'r') as fin:
        #    demand_patterns_config = yaml.safe_load(fin)

        self.town = settings['town']
        self.town_name = settings['town_name']
        self.state_vars = settings['state_vars']
        self.action_vars = settings['action_vars']

        self.duration = settings['duration']
        self.hyd_step = settings['hyd_step']
        self.pattern_step = settings['pattern_step']

        # Demand patterns - NEW
        # Dictionary of demand patterns with different conditions and chances
        self.patterns_train = settings["demand_patterns_config"]['train']
        self.patterns_test_csv = settings["demand_patterns_config"]['test']
        self.training_conditions = []

        # Random seed set in config file
        if settings["demand_patterns_config"]['seed']:
            random.seed(settings["demand_patterns_config"]['seed'])

        self.demand_moving_average = None
        self.test_weeks = settings['test_weeks']
        self.curr_seed = None
        self.on_eval = False

        # Physical process of the environment
        self.wn = WaterDistributionNetwork(self.town + '.inp')
        self.wn.set_time_params(duration=self.duration, hydraulic_step=self.hyd_step, pattern_step=self.pattern_step)

        # Simulation variables
        self.curr_time = None
        self.timestep = None
        self.readings = None

        # Reward weights
        self.w_dsr, self.w_overflow, self.w_flow, self.w_pump_updates = [env_config['weights'][w] for w
                                                                         in env_config['weights'].keys()]

        # Result variables
        self.done = False
        self.total_updates = 0
        self.total_supplies = []
        self.total_demands = []
        self.dsr = 0
        self.total_overflow = []
        self.total_flow = []
        self.total_updates_list = []
        self.step_dsr_list = []
        self.avg_step_dsr = 0
        self.dsr_per_junction = 0

        # Initialize ICS component
        self.plcs_config = env_config['plcs']
        self.sensor_plcs = []
        self.actuator_plcs = []

        # Attackers
        self.attackers_train_file = attacks_train_file
        self.attackers_test_file = attacks_test_file
        self.attackers = []
        self.attackers_generator = None

        # Two possible values for each pump: 2 ^ n_pumps
        #   -> 3 ACTIONS because pumps are overlapped
        self.action_space = spaces.Discrete(env_config['action_space'])

        # Bounds for observation space
        lows = np.array([self.state_vars[key]['bounds']['min'] for key in self.state_vars.keys()])
        highs = np.array([self.state_vars[key]['bounds']['max'] for key in self.state_vars.keys()])

        print(lows, highs)

        # Observation space
        self.observation_space = spaces.Box(low=lows, high=highs, shape=(len(self.state_vars),))

        print(type(self.action_space), type(self.observation_space))

        self.build_ics_devices()

    def reset(self):
        """
        Called at the beginning of each episode
        :param state:
        :return:
        """
        self.wn.reset()

        self.curr_time = 0
        self.timestep = 1
        self.readings = {}

        self.wn.solved = False
        self.done = False

        self.total_supplies = []
        self.total_demands = []
        self.total_updates = 0
        self.total_overflow = []
        self.total_flow = []
        self.total_updates_list = []
        self.step_dsr_list = []
        self.dsr = 0
        self.avg_step_dsr = 0
        self.dsr_per_junction = 0

        self.attackers = []

        self._reset_week()

        # Link attackers to relative PLCs
        if self.attackers:
            for sensor in self.sensor_plcs:
                sensor.set_attackers([attacker for attacker in self.attackers if attacker.target == sensor.name])
            for actuator in self.actuator_plcs:
                actuator.set_attackers([attacker for attacker in self.attackers if attacker.target == actuator.name])

        self.wn.init_simulation()

        state = self.build_current_state(readings=[], reset=True)
        return np.array(state).astype(np.float32)

    def _reset_week(self):
        """

        """
        junc_demands = []
        col = None

        if self.on_eval:
            # Build demand patterns features
            if self.patterns_test_csv:
                junc_demands = pd.read_csv(demand_pattern_folder / self.patterns_test_csv)

                # Check if have been set a given seed for test, otherwise uses random columns
                if self.curr_seed is not None and self.curr_seed < len(junc_demands.columns):
                    col = junc_demands.columns.values[self.curr_seed]
                else:
                    col = random.choice(junc_demands.columns.values)
                # print("col: ", col)
                self.wn.set_demand_pattern('junc_demand', junc_demands[col], self.wn.junctions)

            # Build attackers instances
            if self.attackers_test_file:
                with open(self.attackers_test_file, 'r') as fin:
                    attackers = yaml.safe_load(fin)
                for att in attackers:
                    self.build_attacker(att)
        else:
            # Build demand patterns features
            if self.patterns_train:
                # Here we choose the different type of pattern demand
                pattern_types = [pattern['type'] for pattern in self.patterns_train]
                pattern_weights = [pattern['chance'] for pattern in self.patterns_train]
                chosen_pattern = random.choices(population=pattern_types, weights=pattern_weights, k=1)

                # Here we choose a random week from a random interval of the picked pattern type
                pattern_train_csv = random.choice(
                    self.patterns_train[pattern_types.index(chosen_pattern[0])]['pattern_files']
                )
                junc_demands = pd.read_csv(demand_pattern_folder / self.town_name / pattern_train_csv)
                col = random.choice(junc_demands.columns.values)
                # print("col: ", col)
                self.training_conditions.append({'type': chosen_pattern, 'file': pattern_train_csv, 'col': col})
                self.wn.set_demand_pattern('junc_demand', junc_demands[col], self.wn.junctions)
                # TODO: randomize attacks -> think about that

            # Build attackers instances
            if self.attackers_train_file:
                with open(self.attackers_train_file, 'r') as fin:
                    attackers_config = yaml.safe_load(fin)

                # Build scheduled attackers
                if attackers_config['scheduled_attackers']:
                    week = random.choice(list(attackers_config['scheduled_attackers'].keys()))
                    # Check if the attacker file exists or if it is a week without attacks
                    if attackers_config['scheduled_attackers'][week]:
                        for att in attackers_config['scheduled_attackers'][week]:
                            self.build_attacker(att)

                # Build randomized attackers
                """
                if attackers_config['randomized_attackers']:
                    if not self.attackers_generator:
                        self.attackers_generator = AttackerGenerator()
                    self.attackers_generator.parse_configuration(attackers_config['randomized_attacks'])
    
                    CONTINUE FROM HERE FOR TRAIN RANDOMIZATIONs      
                """
        # Create moving average values
        if 'demand_SMA' in self.state_vars.keys():
            self.demand_moving_average = junc_demands[col].rolling(window=6, min_periods=1).mean()
            # self.demand_moving_average = junc_demands[col].ewm(alpha=0.1, adjust=False).mean()

        self.curr_seed = col

    def step(self, action):
        """
        :param action:
        :return:
        """
        pump_updates = {}

        # Actuator PLCs apply the action into the physical process
        for plc in self.actuator_plcs:
            # Return a dict with 1 if the actuator has been updated, 0 otherwise
            pump_updates.update(plc.apply(action))
            self.total_updates += sum(pump_updates.values())

        # Simulate the next hydraulic step
        # TODO: understand if we want to simulate also intermediate steps (not as DHALSIM)
        self.timestep = self.wn.simulate_step(self.curr_time)
        self.curr_time += self.timestep

        # in case if we want to skip unwanted steps
        """
        while self.curr_time % self.hyd_step != 0 and self.timestep != 0:
            self.timestep = self.wn.simulate_step(self.curr_time)
            self.curr_time += self.timestep
        """

        for sensor in self.sensor_plcs:
            self.readings[sensor.name] = sensor.apply()

        # Retrieve current state and reward from the chosen action
        state = np.array(self.build_current_state(readings=self.readings)).astype(np.float32)
        reward = self.compute_reward(pump_updates)

        if self.timestep == 0:
            self.done = True
            self.wn.solved = True
            self.dsr = self.evaluate()
            self.logger.log_results(self.curr_seed, self.dsr, self.total_updates)
            self.curr_seed = None

        return state, reward, self.done, {}

    def build_ics_devices(self):
        """
        Create instances of actuators and sensors
        """
        self.sensor_plcs = [SensorPLC(name=sensor['name'],
                                      wn=self.wn,
                                      plc_variables=sensor['vars'])
                            for sensor in self.plcs_config if sensor['type'] == 'sensor']
        self.actuator_plcs = [ActuatorPLC(name=sensor['name'],
                                          wn=self.wn,
                                          plc_variables=sensor['vars'],
                                          n_actions=self.action_space.n)
                              for sensor in self.plcs_config if sensor['type'] == 'actuator']

    def build_attacker(self, attacker):
        """
        Builds attacks in two ways depending on if it is a train or test episode
        :param attacker:
        """
        evil_instance = globals()[attacker['type']]
        #print(attacker)
        self.attackers.append(evil_instance(attacker['name'], attacker['target'], attacker['trigger']['start'],
                                            attacker['trigger']['end'], attacker['tags']))

    def build_current_state(self, readings, reset=False):
        """
        Build current state list, which can be used as input of the nn saved_models
        :param readings:
        :param reset:
        :return:
        """
        state = []

        # Initial state acquire from PLCs (at least for tanks)
        if reset:
            for var in self.state_vars:
                if var == 'demand_SMA':
                    state.append(self.demand_moving_average.iloc[0])
                elif var.startswith('J'):
                    state.append(0)
                else:
                    for sensor in self.sensor_plcs:
                        if var in sensor.owned_vars['nodes']:
                            state.append(sensor.init_readings(var, 'pressure'))
        else:
            seconds_per_day = 3600 * 24
            days_per_week = 7
            current_hour = (self.curr_time % (seconds_per_day * days_per_week)) // 3600

            # Retrieve from reading vars specified in the agent observation space
            for var in self.state_vars:
                if var == 'demand_SMA':
                    state.append(self.demand_moving_average.iloc[current_hour])
                elif var == 'under_attack':
                    attack_flag = 0
                    # Checks if one of the sensor plc is compromised by an ongoing attack
                    for sensor in self.sensor_plcs:
                        if readings[sensor.name]['under_attack']:
                            attack_flag = 1
                            # TODO: implement attack localization
                            break
                    state.append(attack_flag)
                else:
                    for sensor in self.sensor_plcs:
                        if var in readings[sensor.name].keys():
                            state.append(readings[sensor.name][var]['pressure'])
        return state

    def check_overflow(self, simple=True):
        """
        Check if there is an overflow problem in the tanks. We have an overflow if after one hour we the tank is
        still at the maximum level.
        :return: penalty value
        """
        risk_percentage = 0.9
        overflow_penalty = 0

        for sensor in self.readings.keys():
            tanks = dict((key, self.readings[sensor][key]) for key in self.readings[sensor].keys()
                         if key.startswith('T'))
            for tank in tanks.keys():
                if tanks[tank]['pressure'] > self.state_vars[tank]['bounds']['max'] * risk_percentage:
                    out_bound = tanks[tank]['pressure'] - (self.state_vars[tank]['bounds']['max'] * risk_percentage)
                    # Normalization of the out_bound pressure
                    multiplier = out_bound / ((1 - risk_percentage) * self.state_vars[tank]['bounds']['max'])
                    overflow_penalty = self.overflow_penalty_coefficient * multiplier

        if simple:
            return overflow_penalty

    def check_pumps_flow(self):
        """
        TODO: to implement and substitute to update penalty
        """
        total_flow = 0
        lowest_flow = 0
        highest_flow = 1000      # retrieved with empirical experiments

        for pump in self.action_vars:
            for sensor in self.sensor_plcs:
                if pump in self.readings[sensor.name].keys():
                    total_flow += self.readings[sensor.name][pump]['flow']

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

    def compute_reward(self, step_pump_updates:dict):
        """
        Compute the reward for the current step. It depends on the step_DSR

        :param step_pump_updates:
        :return:
        """
        overflow_penalty = self.check_overflow(simple=True)
        flow_penalty = self.check_pumps_flow()
        update_penalty = self.check_pumps_updates(step_pump_updates, simple=True)

        self.total_overflow.append(overflow_penalty)
        self.total_flow.append(flow_penalty)

        # DSR computation
        supplies = []
        base_demands = []

        for sensor in self.readings.keys():
            # Filter keys of readings belonging to junction properties
            junctions = dict((key, self.readings[sensor][key]) for key in self.readings[sensor].keys()
                             if key.startswith('J'))
            supplies.extend([junctions[var]['demand'] for var in junctions.keys()])
            base_demands.extend([junctions[var]['basedemand'] for var in junctions.keys()])

        dsr_ratio = objFunction.step_supply_demand_ratio(supplies=supplies, demands=base_demands)

        self.total_supplies.append(supplies)
        self.total_demands.append(base_demands)
        self.step_dsr_list.append(dsr_ratio)
        self.total_updates_list.append(update_penalty)

        #print("Step_DSR: ", self.step_dsr[-1])

        # Total reward computation
        reward = dsr_ratio * self.w_dsr - \
                 overflow_penalty * self.w_overflow - \
                 flow_penalty * self.w_flow - \
                 update_penalty * self.w_pump_updates

        return reward

    def evaluate(self):
        """
        Evaluate the model at the end of the episode.

        :return: total DSR computed across the entire timeframe
        """
        total_dsr = objFunction.supply_demand_ratio(supplies=self.total_supplies, demands=self.total_demands)
        self.avg_step_dsr = objFunction.average_step_supply_demand_ratio(self.step_dsr_list)
        self.dsr_per_junction = objFunction.supply_demand_per_junction_ratio(supplies=self.total_supplies,
                                                                             demands=self.total_demands)

        # print("Total_dsr: ", total_dsr)
        # print("Avg_step_dsr: ", self.avg_step_dsr)
        # print("Per_junction_dsr: ", self.dsr_per_junction)
        return total_dsr
