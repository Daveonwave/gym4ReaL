# MicrogridEnv

This document describes the `MicrogridEnv` environment available in the `Gym4ReaL` library for energy management within a microgrid adopting reinforcement learning (RL) techniques. This environment is built on [ErNESTO-DT](https://github.com/Daveonwave/ErNESTO-DT) simulator and the [Gymnasium](https://gymnasium.farama.org) interface, enabling efficient training and evaluation of RL agents.

## Overview

The `MicrogridEnv` simulates a microgrid controller tasked with optimize energy management within a local network. The simulation includes:

- An accurate battery digital twin simulator comprising an electrial, a thermal, and a degradation model.
- Time-series of exogenous signals (demand, generation, market, and ambient temperature).
- Configurable yaml files to adapt the problem to different battery architectures.

## Conda Usage

```bash
conda create -n env-name python=3.12
```

## Installation

To install the general and environment-specific requirements, run:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/microgrid/requirements.txt
```

## Microgrid Environment

The setting described in the `MicrogridEnv` is the following:

- **Observation Space:** Signals of estimated demand and generations, signals of the energy price, internal battery state (state of charge and temperature), time variables.
- **Action Space:** Percentage of net power, computed subtracting energy consumption from generation, to store (retrieve) to (from) the battery system.
- **Goal:** Maximize the revenue in trading with the energy market, limiting the degradation costs derived from battery usage.

## Usage

To use this environment:

1. Download the `Gym4ReaL` library and install its dependencies.
2. Register and instantiate the `MicrogridEnv` environment using Gym's API.
3. Train and evaluate your custom or off-the-shelf RL agent implementation.

Example:

```python
import gymnasium as gym
from gym4real.envs.microgrid.utils import parameter_generator

params = parameter_generator()
env = gym.make('gym4real/microgrid-v0', **{'settings': params})
obs,info = env.reset()
done = False

while not done:
  action = env.action_space.sample()
  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
```

## Configuration

Simulator parameters (e.g., battery model parameters, state of charge range, end-of-life state of health, battery sizing) can be set via files contained in `gym4real/envs/microgrid/simulator/configuration/` folder.
Environment parameters (e.g., observation space, environment timestep) can be set by modifying the `gym4real/envs/microgrid/world_train.yaml` file.

---

## Reproducibility

In order to reproduce the results, open the notebooks in `examples/microgrid` folder and run the `benchmarks.ipynb` notebook which employs the already trained model contained within the `trained_models/` folder.

To train an RL agent from scratch, launch the following command from the main directory selecting the environment and the training parameters:

```bash
python gym4real/algorithms/microgrid/ppo.py
```

For a tutorial for training your own RL algorithm refer to `examples/microgrid/training-tutorial.ipynb`.
