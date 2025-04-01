import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class EnergyDemand:
    """
    Energy demand
    """
    def __init__(self, data: pd.DataFrame, 
                 timestep: int, 
                 data_usage: str = 'end',
                 **kwargs):
        assert data_usage in ['end', 'circular'], "'data_usage' of demand must be 'end' or 'circular'."

        self.timestep = timestep
        self._history = data.drop(columns=['delta_time'])

        # To track the current profile during training
        self._current_profile = None

        # Max demand value
        self.max_demand = max([max(values) for key, values in self._history.items()])
        self.min_demand = min([min(values) for key, values in self._history.items()])

    @property
    def history(self):
        return self._history[self._current_profile]

    @property
    def labels(self):
        return self._history.columns.to_list()

    @property
    def profile(self):
        return self._current_profile

    @profile.setter
    def profile(self, profile_id: str):
        assert str(profile_id) in self.labels, \
            "'profile_id' of demand must be a label within the columns of the dataframe."
        self._current_profile = profile_id

    def __len__(self):
        return self._history.shape[0]

    def __getitem__(self, idx):
        """
        Get demand by index
        :param idx:
        """
        assert 0 <= idx <= len(self), "k must be between 0 and the total length of the demand history!"
        return None, None, self._history[self._current_profile][idx]

    def get_idx_from_times(self, time: int) -> int:
        """
        Get index of demand history for given time.
        :param time: time of demand history
        :return: index of demand history
        """
        time = time % (len(self) * self.timestep)
        idx = int(time // self.timestep)
        return idx

    def is_run_out_of_data(self):
        """
        Check if demand history is out-of-data.
        """
        return False



