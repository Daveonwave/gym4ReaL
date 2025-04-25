import json
import numpy as np


def read_json(file_path):
    """
    Read a JSON file and return its content as a dictionary.
    :param file_path: Path to the JSON file.
    :return: Dictionary containing the JSON data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class AttackScheduler:
    def __init__(self, data_config: dict):
        self._data_path = data_config['dataset_path']
        self._data_config = data_config
        self._current_attacks = []

    @property
    def attacks(self):
        return self._current_attacks
    
    @property
    def __len__(self):
        return len(self._current_attacks)
    
    def __getitem__(self, idx):
        
        return self._current_attacks[idx]
    
    def draw_attacks(self, is_evaluation: bool = False):
        """
        Draw pattern for the current profile
        :param is_evaluation: if True, draw pattern for evaluation
        """
        self._current_attacks = []
        
        if is_evaluation:   # Draw pattern for testing
            current_attacks = read_json(self._data_path + self._data_config['test'])
        else:   # Draw pattern for training
            dataset = read_json(self._data_path + self._data_config['train'])
            current_attacks = dataset[np.random.choice(range(len(dataset)), 1)[0]]
        
        # Instantiate the attacks
        for attack in current_attacks:
            attack_instance = globals()[attack['type']]
            self._current_attacks.append(attack_instance(attack['name'], 
                                                         attack['target'], 
                                                         attack['trigger']['start'],
                                                         attack['trigger']['end'], 
                                                         attack['tags']))   


class GenericAttacker:
    def __init__(self, name, target, event_start, event_end, tags=None):
        self.name = name
        self.target = target
        self.event_start = event_start
        self.event_end = event_end
        self.tags = tags

    def apply_attack(self, variables):
        pass


class MITM(GenericAttacker):
    def apply_attack(self, variables):
        """
        Inject the fake value for each tag declared in tags dictionary
        """
        for var in self.tags:
            variables[var['tag']][var['property']] = var['value']
        return variables


class DOS(GenericAttacker):
    def __init__(self, name, target, event_start, event_end, tags=None):
        super().__init__(name, target, event_start, event_end, tags)
        self.old_readings = {}

    def apply_attack(self, readings):
        if not self.old_readings:
            self.old_readings = readings.copy()
        return self.old_readings.copy()


class NetworkDelay(GenericAttacker):
    def apply_attack(self, variables):
        return variables



