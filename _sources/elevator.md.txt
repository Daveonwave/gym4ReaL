# ElevatorEnv

This document describes the `ElevatorEnv` environment available in the `Gym4ReaL` library to solve an optimal dispatching problem adopting reinforcement learning (RL) techniques. This environment is built on the [Gymnasium](https://gymnasium.farama.org) interface, enabling efficient training and evaluation of RL agents.

## Overview

The `ElevatorEnv` simulates an elevator controller tasked with optimal passengers dispatching during a _peak-down traffic_ period. The simulation includes:

- A lightweight elevator simulator.
- A arrival distribution generator based on a Poisson process.
- Configurable yaml files to customize the environment parameters.

Notably, `ElevatorEnv` is the "less" realistic environment of `Gym4ReaL` library, designed for enabling tabular and provably efficient RL algorithms, difficult to find within real-world problems.

## Conda Usage

```bash
conda create -n env-name python=3.12
```

## Installation

To install the general and environment-specific requirements, run:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/elevator/requirements.txt
```

## Elevator Environment

The setting described in the `ElevatorEnv` is the following:

- **Observation Space:** Elevator position and occupancy, queues status for each floor, arrivals at each floor.
- **Action Space:** 3 discrete actions to move the elevator up, down, and open the doors.
- **Goal:** Minimize the global waiting time of passengers.

## Usage

To use this environment:

1. Download the `Gym4ReaL` library and install its dependencies.
2. Register and instantiate the `ElevatorEnv` environment using Gym's API.
3. Train and evaluate your custom or off-the-shelf RL agent implementation.

Example:

```python
import gymnasium as gym
from gym4real.envs.elevator.utils import parameter_generator

params = parameter_generator()
env = gym.make('gym4real/elevator-v0', **{'settings': params})
obs,info = env.reset()
done = False

while not done:
  action = env.action_space.sample()
  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
```

## Configuration

Simulator parameters (e.g., number and height of floors, arrival rates, elevator's movement speed) can be set via `gym4real/envs/elevator/world.yaml` file.

---

## Reproducibility

In order to reproduce the results, open the notebooks in `examples/elevator` folder and run the `benchmarks.ipynb` notebook which employs the already trained model contained within the `trained_models/` folder.

To train an RL agent from scratch, launch the following command from the main directory selecting the environment and the training parameters:

```bash
python gym4real/algorithms/elevator/[qlearning|sarsa].py
```

For a tutorial for training your own RL algorithm refer to `examples/elevator/training-tutorial.ipynb`.
