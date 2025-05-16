import pandas as pd
import numpy as np


class WaterDemandPattern:
    """
    Water demand pattern
    --------------------
    
    Demand patterns are multiple dataset representing different conditions: normal, stressful and extreme.
    Each type as a different probability of occurrence.
    Selected the type, the pattern file is randomly selected from the relative ones.
    Then, the pattern is a randomly selected column from the file.
    The column represent a week of demand data, with a time step of 1 hour.
    Each file continains 153 columns, each one representing a week of demand data, for a total of a year of data.
    The pattern is used to set the demand for each junction in the network.
    """
    def __init__(self, 
                 data_config: dict,
                 pattern_step: int, 
                 event_probs: dict,
                 **kwargs):
        self._data_path = data_config['dataset_path']
        self._data_config = data_config
        self._pattern_step = pattern_step
        self._event_probs = event_probs

        # To track the current profile
        self._current_pattern: pd.Series = None
        self._current_pattern_type: str = None
        
        self._moving_average = None
        self._exp_moving_average = None

    @property
    def pattern(self):
        return self._current_pattern
    
    @property
    def pattern_type(self):
        return self._current_pattern_type
    
    @property
    def moving_average(self):
        return self._moving_average
    
    @property
    def exp_moving_average(self):
        return self._exp_moving_average
    
    def __len__(self):
        return len(self._current_pattern)
    
    def __getitem__(self, time):
        """
        Get demand by index
        :param idx:
        """
        time = time % (len(self) * self.timestep)
        idx = int(time // self.timestep)
        
        assert 0 <= idx <= len(self), "k must be between 0 and the total length of the demand history!"
        return self._current_pattern[idx]
    
    def draw_pattern(self, is_evaluation: bool = False):
        """
        Draw pattern for the current profile
        :param is_evaluation: if True, draw pattern for evaluation
        """
        if is_evaluation:   # Draw pattern for testing
            df = pd.read_csv(self._data_path + self._data_config['test'], index_col=None)
            self._current_pattern = df[np.random.choice(df.columns.values, 1)[0]]
            
        else:   # Draw pattern for training
            # Randomly select a pattern type based on the event probabilities
            self._current_pattern_type = np.random.choice(a=list(self._event_probs.keys()), 
                                                          p=list(self._event_probs.values()), 
                                                          size=1)[0]
            csv_file = np.random.choice(a=self._data_config['train'][self._current_pattern_type], size=1)[0]
            df = pd.read_csv(self._data_path + str(csv_file), index_col=0)
            # Randomly select a pattern from the chosen dataframe
            self._current_pattern = df[np.random.choice(df.columns.values, 1)[0]]
    
    def set_moving_average(self, window_size: int, total_basedemand: float):
        """
        Calculate the moving average of the demand pattern
        :param window_size: size of the moving average window
        :param total_basedemand: total basedemand of the network
        :return: moving average of the demand pattern
        """
        if self._moving_average is None:
            self._moving_average = self._current_pattern.rolling(window=window_size, min_periods=1).mean()
            
    def set_exp_moving_average(self, window_size: int, total_basedemand: float):
        """
        Calculate the exponential moving average of the demand pattern
        :param window_size: size of the moving average window
        :param total_basedemand: total basedemand of the network
        :return: exponential moving average of the demand pattern
        """
        if self._exp_moving_average is None:
            self._exp_moving_average = self._current_pattern.ewm(span=window_size, min_periods=1).mean()




