from typing import SupportsFloat, Any
import pandas as pd
import numpy as np
import os
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.core import ActType, ObsType, RenderFrame
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def _compute_features(data, offset_delta, number_of_deltas, sinusoidal_transformation, start_trading, end_trading):
    data['Mid-Price-1'] = data['Mid-Price'].shift(offset_delta)
    data['delta_mid_0'] = (data['Mid-Price'] - data['Mid-Price-1']) / data['Mid-Price-1']
    for i in range(1, number_of_deltas):
        data[f'delta_mid_{i}'] = data['delta_mid_0'].shift(i * offset_delta)
    data['DayOfTheWeek'] = data['datetime'].dt.dayofweek
    data['DayOfTheWeek_sin'] = np.sin(2 * np.pi * data['DayOfTheWeek'] / 5)
    data['DayOfTheWeek_cos'] = np.cos(2 * np.pi * data['DayOfTheWeek'] / 5)
    data['Day'] = data['datetime'].dt.day
    if sinusoidal_transformation:
        data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
        data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

    data['Timestamp'] = data['datetime'].dt.hour * 60 + data['datetime'].dt.minute
    data['Timestamp'] = (data['Timestamp'] - (start_trading * 60)) / ((60 * end_trading) - (60 * start_trading))
    if sinusoidal_transformation:
        data['Timestamp_sin'] = np.sin(2 * np.pi * data['Timestamp'])
        data['Timestamp_cos'] = np.cos(2 * np.pi * data['Timestamp'])

    data['Month'] = data['datetime'].dt.month
    if sinusoidal_transformation:
        data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    return data


class TradingEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 settings: dict[str, Any],
                 scaler=None,
                 seed=None):
        self._years = settings['years']
        self._persistence = settings['persistence']
        self._data_directory = settings['data_directory']

        # Open and close our
        self._trading_open = settings['trading_open']
        self._trading_close = settings['trading_close']
        self._offset_delta = settings['offset_delta']

        self._sinusoidal_transformation = settings['sinusoidal_transformation']

        # build dataset or load it
        if not f"dataset_1_min_{self._years}.csv" in os.listdir(self._data_directory):
            self._data = self._load_data(self._data_directory, self._years, settings['ffill_limit_data'])
        else:
            self._data = pd.read_csv(os.path.join(self._data_directory, f"dataset_1_min_{self._years}.csv"),
                                     index_col=0, parse_dates=['datetime'], date_format='%Y-%m-%d %H:%M:%S')
            self._data['datetime'] = pd.to_datetime(self._data['datetime'])

        # Remove anomalous days (e.g. covid etc.)
        if len(settings['anomalous_days']) > 0:
            self._data = self._clean_data(settings['anomalous_days'])

        # Compute features
        self._number_of_deltas = settings['number_of_deltas']
        self._data = _compute_features(self._data, self._offset_delta, self._number_of_deltas,
                                            self._sinusoidal_transformation, self._trading_open, self._trading_close)
        if settings['fillna_features'] is True:
            self._data = self._data.fillna(0)

        np.random.seed(seed)

        # State space dimension: Number of deltas + (Timestamp + Month + Day of the Week + Day) + Agent Position, where the temporal features are transformed using sine/cosine if requested
        self._use_month = settings['use_month']
        self._use_day_of_week = settings['use_day_of_week']
        self._use_day = settings['use_day']
        self.state_dim = self._number_of_deltas + (
                    1 + int(self._use_month) + int(self._use_day_of_week) + int(self._use_day)) * (
                             2 if self._sinusoidal_transformation else 1) + 1

        if scaler is None:
            self._scaler = MinMaxScaler()
            self._data[self.get_state()[0: self._number_of_deltas]] = self._scaler.fit_transform(
                self._data[self.get_state()[0: self._number_of_deltas]])
        else:
            self._scaler = scaler
            self._data[self.get_state()[0: self._number_of_deltas]] = self._scaler.transform(
                self._data[self.get_state()[0: self._number_of_deltas]])

        self.observation_space = Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float64)
        self.action_space = Discrete(3)

        self._sequential = settings['sequential']
        self._day = -1
        self._state_data = self._data[self.get_state()[0: -1]].to_numpy()
        self._mid_prices = self._data['Mid-Price'].to_numpy()
        self._day_numbers = self._data['day_number'].to_numpy()
        self._datetimes = self._data['datetime'].to_numpy()

        self._capital = settings['capital']
        self._fees = settings['fees']

        self.LONG = 2
        self.FLAT = 1
        self.SHORT = 0

    def get_scaler(self):
        return self._scaler

    def _load_data(self, data_directory, years, ffill_limit):
        columns = ["Datetime", "Bid", "Ask", "Volume"]
        months_list = [f"{i:02d}" for i in range(1, 13)]
        years_list = [str(year) for year in years]
        prefix = [f for f in os.listdir(data_directory) if not f.startswith(".") and f.endswith(".csv")][0].split("2")[
            0]
        dfs = []
        for year in years_list:
            for month in months_list:
                print(year, month)
                dfs.append(
                    pd.read_csv(os.path.join(data_directory, f"{prefix}{year}{month}.csv"), names=columns, header=None))
        data = pd.concat(dfs, ignore_index=True)

        data['Datetime'] = pd.to_datetime(data['Datetime'], format="%Y%m%d %H%M%S%f")
        data = data.set_index('Datetime').resample("1min").last()
        full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1min')
        data = data.reindex(full_index)
        data = data.rename_axis('datetime').reset_index()
        data['Weekday'] = pd.to_datetime(data['datetime']).dt.dayofweek
        data['Time'] = pd.to_datetime(data['datetime']).dt.time

        # Remove Weekend and data from outside the considered interval
        data = data[~data['Weekday'].isin([5, 6]) &
                    (data['Time'] >= pd.to_datetime(f'{self._trading_open}:00:00').time()) &
                    (data['Time'] < pd.to_datetime(f'{self._trading_close}:00:00').time())]

        data['day'] = data['datetime'].dt.date
        self.unique_days = pd.Series(sorted(data['day'].unique()))
        # Create a mapping from date -> integer (0, 1, ..., N)
        day_to_number = {day: i for i, day in enumerate(self.unique_days)}
        # Map the day column to the corresponding number
        data['day_number'] = data['day'].map(day_to_number)
        data['Mid-Price'] = (data['Ask'] + data['Bid']) / 2
        data['Timestamp'] = data['datetime'].dt.hour * 60 + data['datetime'].dt.minute
        data = data.ffill(limit=ffill_limit)
        # Remove days with nan
        data['date'] = data['datetime'].dt.date
        days_with_nan = data[data['Mid-Price'].isna()]['date'].unique()
        print("Removed days due to NaN in 'mid_price':", days_with_nan)
        data = data[~data['date'].isin(days_with_nan)]
        # remove not full days
        samples_per_day = pd.to_datetime(data['datetime']).dt.date.value_counts().sort_index()
        shorter_days = samples_per_day[samples_per_day != ((self._trading_close - self._trading_open) * 60)].index
        print("Removed days since shorter", shorter_days)
        data = data[~data['date'].isin(shorter_days)]

        data.to_csv(os.path.join(data_directory, f"dataset_1_min_{years}.csv"))
        return data

    def get_state(self):
        return ( [f"delta_mid_{i}" for i in range(0, self._number_of_deltas)]
                + (['Month'] if self._use_month else ['Month_sin','Month_cos'] if self._use_month and self._sinusoidal_transformation else [])
                + (['Day'] if self._use_day else ['Day_sin','Day_cos'] if self._use_day and self._sinusoidal_transformation else [])
                + (['DayOfWeek'] if self._use_day_of_week else ['DayOfWeek_sin','DayofWeek_cos'] if self._use_day_of_week and self._sinusoidal_transformation else [])
                + (['Timestamp_sin', 'Timestamp_cos'] if self._sinusoidal_transformation else ['Timestamp'])
                + (['Position']))
                # ['DayOfTheWeek_sin', 'DayOfTheWeek_cos', 'Day_sin', 'Day_cos', 'Timestamp_sin', 'Timestamp_cos', 'Month_cos', 'Month_cos', 'Position']

    def reset(
            self,
            day_to_set = None,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:


        if day_to_set is not None:
            self._day = day_to_set
        else:

            if self._sequential is False:
                self._day = np.random.choice(self._data['day_number'].unique())

            else:
                if self._day == -1:
                    self._day = self._data['day_number'].unique().min()
                else:
                    p = np.where(self._data['day_number'].unique() == self._day)[0][0]
                    try:
                        self._day = self._data['day_number'].unique()[p + 1]
                    except:  # reset day
                        self._day = self._data['day_number'].unique().min()
        self._index = self._data.index.get_loc(self._data[self._data['day_number'] == self._day].index[0])
        #print(self._data.iloc[self._index]['datetime'])
        obs = np.append(self._state_data[self._index],
                        self.FLAT)  # np.append(self._data.iloc[self.index][self.get_state()[:-1]], [self.FLAT])
        self._state = obs
        return obs, {}

    def get_days_list(self):
        return list(self._data['day_number'].unique())

    def get_trading_day_num(self):
        return len(self._data['day_number'].unique())

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert action in self.action_space

        obs = self._state
        next_index = self._index + self._persistence
        next_obs = self._state_data[next_index]
        reward = self._capital * (action - 1) * (self._mid_prices[next_index] - self._mid_prices[self._index]) - (
            self._fees) * abs((action - 1) - (obs[self.get_state().index('Position')] - 1))
        self._index = next_index
        info = {'datetime': self._datetimes[self._index]}
        next_obs = np.append(next_obs, [action])
        self._state = next_obs

        if (next_index + 2 * self._persistence >= self._state_data.shape[0] or
                self._day_numbers[next_index + 2 * self._persistence] != self._day_numbers[next_index]):

            next_obs = self._state_data[next_index + self._persistence]
            next_obs = np.append(next_obs, [self.FLAT])
            reward -= self._fees * abs(0 - (obs[self.get_state().index('Position')] - 1))  # force close the position
            done = True

        else:

            done = False

        return next_obs, reward, done, done, info

    def _clean_data(self, anomalous_days):
        dates_to_remove = pd.to_datetime(anomalous_days).date
        return self._data[~self._data['datetime'].dt.date.isin(dates_to_remove)]
