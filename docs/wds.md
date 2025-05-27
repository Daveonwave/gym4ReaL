# WDSEnv

This document describes the `WDSEnv` environment available in the `Gym4ReaL` library addressing a water management problem within a water distribution system adopting reinforcement learning (RL) techniques. This environment is built on [Epynet](https://github.com/Vitens/epynet), a Python wrapper of the [Epanet](https://www.epa.gov/water-research/epanet) hydraulic simulator, and the [Gymnasium](https://gymnasium.farama.org) interface, enabling efficient training and evaluation of RL agents.

## Overview

The `WDSEnv` simulates a water distribution system tasked with the maintainance of resilience within a water network. The simulation includes:

- An accurate hydraulic network simulator (Epanet).
- Time-series of water demand profiles generated with the [STREaM]() framework.
- Configurable yaml files to adapt the problem to different water networks.

## Conda Usage

```bash
conda create -n env-name python=3.12
```

To use `WDSEnv` on MacOS with Apple silicon you need to create an env compatible with Intel x64 cpu. To do so, run the command:

```bash
conda create --platform osx-64 --name env-name python=3.12
```

## Installation

To install the general and environment-specific requirements, run:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/wds/requirements.txt
```

## Water Network Environment

The setting described in the `WDSEnv` is the following:

- **Observation Space:** Tank levels, junction pressures, estimate of the demand (simple or exponentially weighted moving average), time variables.
- **Action Space:** Boolean signals to open/close network pumps.
- **Goal:** Maximize the network resilience, i.e., maximize the demand-satisfaction ratio (DSR) while minimizing the tank overflow risk.

## Usage

To use this environment:

1. Download the `Gym4ReaL` library and install its dependencies.
2. Register and instantiate the `WDSEnv` environment using Gym's API.
3. Train and evaluate your custom or off-the-shelf RL agent implementation.

Example:

```python
import gymnasium as gym
from gym4real.envs.wds.utils import parameter_generator

params = parameter_generator()
env = gym.make('gym4real/wds-v0', **{'settings': params})
obs,info = env.reset()
done = False

while not done:
  action = env.action_space.sample()
  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
```

## Configuration

Simulator parameters (editable in the `.inp` network file) can be set via `gym4real/data/wds/towns/[name_of_town].inp`.
Environment parameters (e.g., observation space, environment timestep) can be set modifing the `gym4real/envs/wds/world_[name_of_town].yaml` file.

---

## Reproducibility

In order to reproduce the results, open the notebooks in `examples/wds` folder and run the `benchmarks.ipynb` notebook which employs the already trained model contained within the `trained_models/` folder.

To train an RL agent from scratch, launch the following command from the main directory selecting the environment and the training parameters:

```bash
python gym4real/algorithms/wds/dqn.py
```

For a tutorial for training your own RL algorithm refer to `examples/wds/training-tutorial.ipynb`.
