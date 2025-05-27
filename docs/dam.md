# DamEnv

This document describes the `DamEnv` environment available in the `Gym4Real` library for water management in a reservoir connected to a dam, using RL techniques. This environment is built on the [Gymnasium](https://gymnasium.farama.org) interface, enabling efficient training and evaluation of RL agents.

## Overview

The `DamEnv` environment simulates a controller of a dam with the aim of meeting the downstream water demand, while avoiding overflows or water starvation. The simulation includes:

- A lake simulator that computes the updates of the water level at each step.
- Time-series of water demand and inflow.
- Time-series of the water level that can be used for imitation learning.
- Configurable yaml files to customize the environment and reservoir parameters.

## Conda usage

```bash
conda create -n env-name python=3.12
```

## Installation

To install the general and environment-specific requirements, run:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/dam/requirements.txt

```

## Dam Environment

The setting described in the `DamEnv` is the following:

- **Observation Space:** Signal of estimated demand, signal of the water level, time variables.
- **Action Space:** Amount of water to release per unit of time .
- **Goal:** Minimize the demand that is not met, while avoiding floods or water starvation

## Usage

To use this environment:

1. Download the `Gym4ReaL` library and install its dependencies.
2. Register and instantiate the `DamEnv` environment using Gym's API.
3. Train and evaluate your custom or off-the-shelf RL agent implementation.

Example:

```python
import gymnasium as gym
from gym4real.envs.dam.utils import parameter_generator

params = parameter_generator()
env = gym.make('gym4real/dam-v0', settings=params)
obs,info = env.reset()
done = False

while not done:
  action = env.action_space.sample()
  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
```

## Configuration

Lake simulator parameters (e.g. surface area, minimum and maximum water level) can be set by modifying the `gym4real/envs/dam/lake.yaml` file.
Environment parameters (e.g., observation space, rewards weights) can be set by modifying the `gym4real/envs/dam/world_train.yaml` file.

---

## Reproducibility

For a tutorial for training your own RL algorithm refer to `examples/dam/training-tutorial.ipynb`.
To obtain the trained models presented in the paper launch this command from the main directory.

```bash
python gym4real/algorithms/dam/ppo_skrl.py
```

To reproduce the results, open the notebook in `examples/dam/benchmarks.ipynb` and run the whole notebook.
