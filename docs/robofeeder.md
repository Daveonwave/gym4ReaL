# Mujoco_RobotGrasping_DeepRL
This repository contains all the necessary files to train a deep reinforcement learning agent to perform robot grasping in the MuJoCo simulator. The environment is composed of the robot, a box containing all the objects to pick, and a hidden virtual camera above the items to keep track of their current status. The box contains a vibrating plate that permits the reorganization of the position and of the orientation of the objects inside

<div align="center">
<img src="/media/result.gif" width="450">
</div>

The environment is based on the [OpenAI Gym](https://gym.openai.com/) framework and the [MuJoCo](http://www.mujoco.org/) simulator. The robot used is [staubli TX2-60](https://www.staubli.com/tw/en/robotics/products/industrial-robots/tx2-60.html)
equipped with a simple picking hand. The motion of the robot is controlled by the [ROS2 Humble](https://docs.ros.org/en/humble/index.html) with [pyMoveit2](https://github.com/AndrejOrsula/pymoveit2) python interface.

The agent is trained using the [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/) library, for some environments, some pre-trained agents are used as part of the new agent, [onnxruntime](https://onnxruntime.ai/) is used to run the pre-trained agent.

<div align="center">
<img src="/media/architecture.png" width="500">
</div>


## 1) Setup :toolbox:
<div>
    <details>
<summary> Tap here to show the Setup Instruction </summary>

All the code is tested on a machine running Ubuntu 22.04.2 LTS* with Python 3.10.12. Parallelization is available for the environment using [Vectorized Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html), 
training of the agent is done by exploiting Nvidia GPU acceleration with [CUDA](https://developer.nvidia.com/cuda-zone).

*Xorg Desktop Environment is used because of the better compatibility with MuJoCo viewer


The first step is to clone this repository:
```
git clone https://github.com/Giuseppe-Calcagno/Mujoco_RobotGrasping_DeepRL.git
```

### 1.1) ROS 
The ROS environment is used to control the robot, the code is tested on ROS Humble. Installation is available following the  [official guide](https://docs.ros.org/en/humble/Installation.html).

The full environment requires the installation of:
- ros-humble-desktop
- ros-dev-tools

Additionally, the following packages are required:
- ros-humble-joint-state-broadcaster
- ros-humble-joint-trajectory-controller

For Debian Linux machines :
```
sudo apt install ros-humble-joint-trajectory-controller
sudo apt install ros-humble-joint-state-broadcaster
```

Before utilizing this package, remember to source the ROS2 as suggested in the guideline.

Opening a terminal in `rosWS` folder, the initialization of [rosdep](http://wiki.ros.org/rosdep) is needed:

```
sudo rosdep init
rosdep update
```

Then, start the installation of all the other ROS dependencies:
```
rosdep install -y -r -i --rosdistro ${ROS_DISTRO} --from-paths .
```

At this point, the build of the project should be successful:
```
colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release"
```

Remember to source ROS workspace setup to re-call it from other directories
```
source {YOUR-PATH}/Mujoco_RobotGrasping_DeepRL/rosWS/install/setup.bash
```

Now ROS should work properly, to test if all works fine, just run to test the robot planning capability:
```
LC_NUMERIC=en_US.UTF-8 ros2 launch staubli_moveit3 demo.launch.py
```

### 1.2) MuJoCo and OpenAI Gym
Opening a terminal in the `environment` folder, the first thing to do is install the dependencies
using PIP, a package manager for Python packages

```
pip3 install -r requirement.txt 
```
At this point, the simulator and the environment are ready to be used.
To test the simulator, run the notebook in ```test\simulator_test.ipynb```

### 1.3) Stable Baselines
[Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) is a set of reliable implementations of reinforcement learning algorithms in PyTorch.
While not strictly essential, this library is employed for both agent training and the implementation of the parallel agent version.


The installation steps are available following the [official guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).
</div>

---

## 2) Usage :mechanical_arm:
<div>
    <details>
<summary> Tap here to show the Usage Instruction </summary>

### Environments
Opening the project in the `environment` folder, 3 different Gym environments are available, each one with a different observation space and goal:

#### Env 0 
This environment represents the baseline of the project. In it, the action space is represented by the cartesian position of the end effector (that is needed to perform a pick action on the object)
and the observation space is represented by the image of the camera


Sample of the observation space:
<div align="center">
<img src="/media/env0.png" height="260">
</div>

#### Env 1
This environment is provided with a pre-trained image segmentation model. In it, the action space is represented by the cartesian position of the end effector and the observation space is represented by the segmented image of the object. The goal is to provide the offset from the center of the cropped image to perform a pick action on the object


Sample of the observation space:
<div align="center">
<img src="/media/env1.png" height="260">
</div>

#### Env 2
This environment is provided with a pre-trained image segmentation model and an agent trained with env_1. In it, the observation space is represented by a list of segmentation images of objects.
The goal is to provide the cropped image index to perform a pick action on the correctly oriented object


Sample of the observation space:
<div align="center">
<img src="/media/env2.png" height="260">
</div>


### Utilities
---
#### Onnxruntime
- in the `utility` folder, the `stableBaseline_to_onxx` notebook contains the code to convert a stable baseline agent to an onnx model
- in the `utility/objDetectionNetwork` folder, the `TendorFlow_to_onnx` notebook contains the code to convert an onnx model to a TensorFlow model


- A pre-trained onnx model of image segmentation is available in the `utility/objDetectionNetwork/objDetection.onnx`, it is used to run the agent in env1 and env2
- A pre-trained onnx model of env1 is available in the `utility/PPO_Stoc`, it is used to run the agent in the env2 

---
#### Synthetic database
in the `utility` folder, the `createDataset` notebook contains the code to generate a synthetic database of images from env0 with a CSV of labels containing the following information for each object:
- position of the object in pixel coordinates
- orientation of the object in radiants
- a flag to indicate if the object is oriented correctly or not
A dataset of this type is used to train the image segmentation model

---
#### Configuration file
The `configuration.yaml` file contains all the parameters to configure the environment, including:
- The dimensions of the image from the simulator camera
- Robot work area Limits (in pixel coordinates)
- Parameters of the robot motion
- The number of objects in the box
- The orientation of the objects in the box
- Simulation parameters: Real-Time and GUI 

---
#### Others 
Also, you can find the following files: 
- folder ```performnace ``` containing performance measurements of the agent during pick simulation (running on 11th Gen Intel® Core™ i5-11300H @ 3.10GHz × 8, 16GB RAM, Nvidia GeForce GTX 1650 Mobile)
- folder ```staubli``` containing the robot model and the environment model 
- folder ```test``` where the agent can be tested, checking the observation image and the reward function
- a notebook ```sb_PPO.ipynb``` as a sample of the training of the agent with the PPO algorithm
</div>
