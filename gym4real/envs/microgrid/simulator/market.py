import numpy as np
import pandas as pd


class EnergyMarket:
    """
    Class to represent an energy market.
    """
    def __init__(self, data: pd.DataFrame, timestep: int, data_usage: str = 'end', spread_factor: float = 1.0):
        assert data_usage in ['end', 'circular'], "'data_usage' of market must be 'end' or 'circular'."

        self.timestep = timestep
        self._ask = data['ask'].to_numpy()
        self._bid = data['bid'].to_numpy()
        
        self._ask *= spread_factor
        self._bid *= spread_factor
                
        self._timestamps = data['timestamp'].to_numpy()
        self._times = data['delta_time'].to_numpy()
        self._delta_times = self._times - self._times[0]

        # Variables used to handle terminating conditions
        self._data_usage = data_usage
        self._first_idx = None
        self._last_idx = None

        # Max values for normalization
        self.max_ask = self._ask.max()
        self.max_bid = self._bid.max()

    @property
    def ask(self):
        return self._ask

    @property
    def bid(self):
        return self._bid

    def __len__(self):
        return self._timestamps.shape[0]

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index {idx} out of range for energy market"
        return self._timestamps[idx], self._times[idx], self._ask[idx], self._bid[idx]

    def get_idx_from_times(self, time: int) -> int:
        """
        Get index of market history for given time.
        :param time: time of market history
        """
        if self._data_usage is None:
            assert self._delta_times[0] < time < self._delta_times[-1], \
                "Cannot be retrieve an index of market exceeding the time's range at the first iteration!"

        time = time % self._delta_times[-1]
        idx = int(time // self.timestep)

        if self._first_idx is None:
            self._first_idx = idx

        self._last_idx = idx
        return idx

    def is_run_out_of_data(self):
        """
        Check if market history is out-of-data.
        """
        if self._data_usage == 'end':
            if self._last_idx == len(self) - 1:
                print("Market history is run out-of-data: end of dataset reached.")
                return True
        else:
            if self._last_idx == self._first_idx - 1:
                print("Market history is run out-of-data: circular termination reached.")
                return True

        return False


class DummyMarket:
    """
    Simpler version of EnergyMarket with fixed ask and bid values.
    """
    def __init__(self, ask: float, bid: float, spread_factor: float = 1.0):
        self._ask = ask
        self._bid = bid
        
        self._ask *= spread_factor
        self._bid *= spread_factor
        
        self.max_ask = ask
        self.max_bid = bid

    @property
    def ask(self):
        return self._ask

    @property
    def bid(self):
        return self._bid

    def __len__(self):
        raise AttributeError("The length of a dummy market is undefined since 'ask' and 'bid' are fixed.")

    def __getitem__(self, idx=None):
        return None, None, self._ask, self._bid

    def get_idx_from_times(self, time: int = None) -> int:
        """
        Get index of market history for given time. Useless in dummy market.
        """
        return 0

    def is_run_out_of_data(self):
        """
        Check if market history is out-of-data.
        """
        return False
