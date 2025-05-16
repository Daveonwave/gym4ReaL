# RoboFeederEnv

This document describes the RoboFeeder environments available in the `gym4ReaL` library for robotic grasping tasks using reinforcement learning. These environments are designed for use with the MuJoCo simulator and the OpenAI Gym interface, enabling efficient training and evaluation of robotic agents.

## Overview

The RoboFeeder environments simulate a robotic arm tasked with picking objects from a box. The simulation includes:

- A Staubli TX2-60 robot with a simple gripper.
- A box containing multiple objects to pick.
- A virtual overhead camera for observation.
- Optional image segmentation and object orientation features.

The environments are compatible with ROS2 Humble and leverage GPU acceleration for training.

## Conda Usage

```bash
conda create -n env-name python=3.12
```

## Installation

To install the general and environment-specific requirements, run:

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/robofeeder/requirements.txt
```

## RoboFeeder Environments

The following environments are implemented in `gym4ReaL`:

### `RoboFeeder-v0`

- **Observation Space:** RGB image from the overhead camera.
- **Action Space:** Cartesian coordinates for the robot end-effector.
- **Goal:** Pick an object based on visual input.

### `RoboFeeder-v1`

- **Observation Space:** Segmented image of the target object (using a pre-trained segmentation model).
- **Action Space:** Cartesian coordinates for the robot end-effector.
- **Goal:** Pick the object using segmentation to localize the target.

### `RoboFeeder-v2`

- **Observation Space:** List of segmented images for all objects in the box.
- **Action Space:** Index of the selected object and pick coordinates.
- **Goal:** Select and pick the correctly oriented object.

## Usage

To use these environments:

1. Install the `gym4ReaL` library and its dependencies.
2. Register and instantiate the desired RoboFeeder environment using Gym's API.
3. Train or evaluate your reinforcement learning agent as needed.

Example:

```python
import gym
import gym4real

env = gym.make('RoboFeeder-v0')
obs,info = env.reset()
done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

## Configuration

Environment parameters (e.g., camera resolution, robot workspace limits, number of objects) can be set via the `configuration.yaml` file in the project root.

---

For more details, refer to the official `gym4ReaL` documentation and code examples.
