# RoboFeederEnv

This document describes the RoboFeeder environments available in the `Gym4ReaL` library for robotic grasping tasks using reinforcement learning. These environments are designed for use with the MuJoCo simulator and the [Gymnasium](https://gymnasium.farama.org) interface, enabling efficient training and evaluation of robotic agents.

## Overview

The RoboFeeder environments simulate a robotic arm tasked with picking objects from a box. The simulation includes:

- A Staubli TX2-60 robot with a simple gripper.
- A box containing multiple objects to pick.
- A virtual overhead camera for observation.
- Optional object detection features.

The environments are compatible with GPU acceleration for training.

## Conda Usage

```bash
conda create -n env-name python=3.12
```

## Installation

To install the general and environment-specific requirements,if gpus are not available run:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/robofeeder/requirements.txt
```

Otherwise:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/robofeeder/requirements-gpus.txt
```

## RoboFeeder Environments

The following environments are implemented in `Gym4ReaL`:

### `RoboFeeder-v0`

- **Observation Space:** Cropped RGB image from the overhead camera, with the help of a pretrianed object dection neural network.
- **Action Space:** Cartesian coordinates for the robot end-effector.
- **Goal:** Pick an object based on visual input.

### `RoboFeeder-v1`

- **Observation Space:** Full RGB image from the overhead camera.
- **Action Space:** Cartesian coordinates for the robot end-effector.
- **Goal:** Pick an object based on visual input.

### `RoboFeeder-v2`

- **Observation Space:** List of segmented images for all objects in the box.
- **Action Space:** Index of the selected object and pick coordinates.
- **Goal:** Select and pick graspable object from the robot workspace.

## Usage

To use these environments:

1. Download the `Gym4ReaL` library and install its dependencies.
2. Register and instantiate the desired RoboFeeder environment using Gym's API.
3. Train and evaluate your custom or off-the-shelf RL agent implementation.

Example:

```python
import gymnasium as gym

env = gym.make('gym4real/robofeeder-picking-v0')
obs,info = env.reset()
done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

## Configuration

Environment parameters (e.g., camera resolution, robot workspace limits, number of objects) can be set via the `configuration.yaml` file in the project root.

---

## Reproducibility

In order to reproduce the results, open the notebooks in `examples/robofeeder/benchmarks` folder and run the whole notebook depending on the selected environment.

To reproduce the results from scratch, launch this command from the main directory selecting the environment and the training parameters:

```bash
git clone https://github.com/Daveonwave/gym4ReaL.git
cd gym4Real
python gym4real/algorithms/robofeeder/ppo.py --env gym4real/robofeeder-planning --episodes 1000 --batch-size 64 --learning-rate 0.0003
```

You can adjust the arguments (`--env`, `--episodes`, `--batch-size`, `--learning-rate`, etc.) as needed. See `ppo.py --help` for all available options.
