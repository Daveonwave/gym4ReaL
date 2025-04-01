import pandas as pd


class AmbientTemperature:
    def __init__(self, data: pd.DataFrame, timestep: int, data_usage: str = 'end'):
        assert data_usage in ['end', 'circular'], "'data_usage' of generation must be 'end' or 'circular'."

        self.timestep = timestep
        self._timestamps = data['timestamp'].to_numpy()
        self._times = data['delta_time'].to_numpy()
        self._delta_times = self._times - self._times[0]
        self._history = data['temp_amb'].to_numpy()

        # Variables used to handle terminating conditions
        self._data_usage = data_usage
        self._first_idx = None
        self._last_idx = None

        # Max value for reward normalization
        self.max_temp = self._history.max()
        self.min_temp = self._history.min()

    @property
    def history(self):
        return self._history

    def __len__(self):
        return self._timestamps.shape[0]

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index {idx} out of range for energy generation"
        return self._timestamps[idx], self._times[idx], self._history[idx]

    def get_idx_from_times(self, time: int):
        """
        Get index of generation history for given time.
        :param time: time of generation history
        """
        if self._data_usage is None:
            assert self._delta_times[0] < time < self._delta_times[-1], \
                "Cannot be retrieve an index of the generation exceeding the time's range at the first iteration!"

        time = time % self._delta_times[-1]
        idx = int(time // self.timestep)

        if self._first_idx is None:
            self._first_idx = idx

        self._last_idx = idx
        return idx

    def is_run_out_of_data(self):
        """
        Check if generation history is out-of-data.
        """
        if self._data_usage == 'end':
            if self._last_idx == len(self) - 1:
                print("Ambient temperature history is run out-of-data: end of dataset reached.")
                return True
        else:
            if self._last_idx == self._first_idx - 1:
                print("Ambient temperature history is run out-of-data: circular termination reached.")
                return True

        return False


class DummyAmbientTemperature:
    """
    Dummy generator for testing purposes with fixed energy generation.
    """
    def __init__(self, temp_value: float):
        self._temp_value = temp_value

    @property
    def history(self):
        return self._temp_value

    def __len__(self):
        raise AttributeError("The length of ambient temperature is undefined since 't_amb' is fixed.")

    def __getitem__(self, idx=None):
        return None, None, self._temp_value

    def get_idx_from_times(self, time: int = None) -> int:
        """
        Get index of generation history for given time.
        :param time: time of generation history
        """
        return 0

    def is_run_out_of_data(self):
        """
        Check if generation history is out-of-data.
        """
        return False
