# TradingEnv

This document describes the `TradingEnv` environment available in the `Gym4ReaL` library to learn a trading strategy using Reinforcement Learning (RL) techniques. This environment is built on the [Gymnasium](https://gymnasium.farama.org) interface, enabling efficient training and evaluation of RL agents.

## Overview
The `TradingEnv` simulates a Forex (Foreign Exchange) market where a Reinforcement Learning (RL) agent can learn how to trade.
The environment is built using historical market data, specifically for the EUR/USD currency pair.

## Conda Usage

```bash
conda create -n env-name python=3.12
```

## Installation
To install the general and environment-specific requirements, run:
```bash
pip install -r requirements.txt
pip install -r gym4real/envs/trading/requirements.txt
```

## Trading Environment

The setting described in the `TradingEnv` is the following:

- **Observation Space:** Composed of 60 delta-mid prices at time _t_, the angular position of the current time over the trading period and the agent position. The observation space can be customized by changing the configuration files.
- **Action Space:** 3 discrete actions {s, f ,l}, where _s_ is short-sell which corresponds to betting that the price will go down, _f_ is flat - no exposure -, and _l_ long which corresponds to betting that the price will go up. Each action corresponds to a fixed amount of capital of 100k€.
- **Goal:** Maximize profit-and-loss (P&L) at the net of transaction costs.


## Usage

To use this environment:

1. Download the `Gym4ReaL` library and install its dependencies.
2. Register and instantiate the `TradingEnv` environment using Gym's API.
3. Download EUR/USD data from [HistData](https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/EURUSD), from 2019 to 2022. 
4. Move the downloaded `.csv` in the folder `gym4real/data/trading/forex`.
5. Train and evaluate your custom or off-the-shelf RL agent implementation.

Example:

```python
import gymnasium as gym
from gym4real.envs.trading.utils import parameter_generator

params_train = parameter_generator(world_options='gym4real/envs/trading/world_train.yaml')
env = gym.make("gym4real/TradingEnv-v0", **{'settings': params_train, 'seed': 1234})
obs,info = env.reset()
done = False

while not done:
  action = env.action_space.sample()
  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
```

## Configuration

Environment parameters (e.g., years, number of delta-mid prices, calendar features) can be set by modifying the `gym4real/envs/trading/world_train.yaml`, `gym4real/envs/trading/world_validation.yaml` and `gym4real/envs/trading/world_test.yaml` files.
It is important to set the **same configuration across all the three files**, to avoid inconsistencies and potential runtime errors.


---

## Reproducibility

In order to reproduce the results, open the notebook in `examples/trading/benchmarks.ipynb` and run the whole notebook. It is possible to retrain the models inside the notebooks or to use pre-trained ones available in `trained_model` folder.

To obtain the trained models presented in the paper, launch this command from the main directory:

```bash
python gym4real/algorithms/trading/[ppo|dqn].py
```


