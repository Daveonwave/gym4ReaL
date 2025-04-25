import os 
os.chdir('/usr/src/data')
import sys
sys.path.append('/usr/src/data')

import Env_0 as base_env

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3.common.env_checker import check_env
from planner import PlanningClass


import robot_simulator as simulator
import matplotlib.pyplot as plt

config_file = "/usr/src/data/configuration.yaml"
simulator = simulator.robot_simulator(config_file)
simulator.reset()

from scipy.spatial.transform import Rotation
import numpy as np


def normalizeAngle(angle):
    if(angle>np.pi):angle -=np.pi
    elif(angle<0):angle += np.pi
    return angle

result = 0

while not result==1:

    coords = simulator.data.site(1).xpos.copy()
    coords[2] = coords[2] - 0.11
    print("Initial coordinates:", coords)
    rot = Rotation.from_quat(simulator.data.qpos[3:7*(1)].copy()).as_euler('xyz')
    rot = normalizeAngle(2.35+rot[0])
    print("Initial rotation:", rot)

    image, result = simulator.simulate_pick(coords,rot)

    # Plot the image returned by the simulator
    plt.imshow(image)

    #1 if the pick was successful, 0 otherwise
    print("Result of the Simulator:",result)

# def compute_camera_matrix():
#   model = env.simulator.model
#   data = env.simulator.data
#   renderer = env.simulator.renderer
#   renderer.update_scene(data,camera="top_down")
#   pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
#   z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
#   y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
#   rot = np.vstack((np.cross(y, z), y, z))
#   fov = model.cam('top_down').fovy[0]

#   # Translation matrix (4x4).
#   translation = np.eye(4)
#   translation[0:3, 3] = -pos

#   # Rotation matrix (4x4).
#   rotation = np.eye(4)
#   rotation[0:3, 0:3] = rot

#   # Focal transformation matrix (3x4).
#   focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
#   focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

#   # Image matrix (3x3).
#   image = np.eye(3)
#   image[0, 2] = (renderer.width - 1) / 2.0
#   image[1, 2] = (renderer.height - 1) / 2.0
#   return image @ focal @ rotation @ translation

# def word2pixel(xyz_global):
#     cam_matrix = compute_camera_matrix()
#     # Camera matrices multiply homogenous [x, y, z, 1] vectors.
#     corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
#     corners_homogeneous[:3, :] = xyz_global
#     # Get the camera matrix.
#     xs, ys, s = cam_matrix @ corners_homogeneous
#     # x and y are in the pixel coordinate system.
#     x = xs / s
#     y = ys / s
#     return x,y


# def normalizeAngle(angle):
#     if(angle>np.pi):angle -=np.pi
#     elif(angle<0):angle += np.pi
#     return angle


# def get_objs(): # list(x,y,rot)
#     initialObjPos=[]
#     for i in range(env.simulator.constants["NUMBER_OF_OBJECTS"]):
#         # get the position of the obj from the site in the right position (+1 due to the target (goal) site)
#         coords = env.simulator.data.site(1+i).xpos.copy()
#         pixelCoord = word2pixel(np.array([coords[0],coords[1],coords[2]]).reshape(3,1))
#         rot = Rotation.from_quat(env.simulator.data.qpos[0+7*i+3:7*(i+1)].copy()).as_euler('xyz')
#         rot = normalizeAngle(2.35+rot[0])
#         initialObjPos.append((pixelCoord[0][0],pixelCoord[1][0],rot))
#     return initialObjPos


# def showObs(obs):
#     plt.imshow(obs.transpose(1, 2, 0))
#     plt.savefig('observation_image.png')
#     plt.close()

# def printobs():
#     plt.figure(figsize=(10, 5))
#     for i in range(env.IMAGE_NUM):
#         ax = plt.subplot(1, env.IMAGE_NUM, i + 1)
#         ax.imshow(env.current_obs[i],cmap=plt.cm.gray)
#         ax.title.set_text("Action "+str(i)+" - "+str(env.obsCenter[i]))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.savefig('observation_image.png')


# env = base_env.robotEnv("/usr/src/data/configuration.yaml")

# # perform a random action
# action = env.action_space.sample()
# action = np.array([-0.43037541 , 0.2436779  , 0.20252  ])
# obs,rew,done,_,_ = env.step(action)

# print("done:",done) 
# print("reward:",rew,"\n")

# #if(done): env.reset()

# # print the obs after the action
# showObs(env.current_obs)

# from ikpy.chain import Chain
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the robot chain from a URDF file
# robot_chain = Chain.from_urdf_file("/usr/src/data/staubli/urdf/tx2_60.urdf")


# # Print the links to know how many there are and their names
# for i, link in enumerate(robot_chain.links):
#     print(i, link.name)



# planner = PlanningClass()

# traj,index = planner.planFunction(initial_joint_position=None)

# def plot_trajectory_with_chain(plan, save_file="trajectory_plot.png"):
#     """
#     Plots the robot's trajectory in 3D for each waypoint and saves the plot.
#     """
#     # Create a 3D plot for the entire trajectory
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Initialize lists to store end-effector positions
#     x_positions = []
#     y_positions = []
#     z_positions = []

#     # Iterate through each waypoint in the trajectory
#     for i, joint_angles in enumerate(plan):
#         # Get the end effector position for the current joint configuration
#         end_effector_position, _ = planner.get_end_effector_position_and_orientation(joint_angles)
#         x_positions.append(end_effector_position[0])
#         y_positions.append(end_effector_position[1])
#         z_positions.append(end_effector_position[2])
#         print(f"Waypoint {i + 1}:")
#         print(f"End Effector Position: {end_effector_position}")

#     # Plot the trajectory of the end effector
#     ax.plot(x_positions, y_positions, z_positions, marker='o', label="End Effector Trajectory")

#     # Set plot labels and title
#     ax.set_title("End Effector Trajectory in 3D")
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_zlabel('Z Position')
#     ax.legend()

#     # Save the plot to a file
#     plt.savefig(save_file, dpi=300)  # Save as PNG with high resolution
#     plt.close(fig)  # Close the figure to avoid GUI opening

# # Example usage
# plot_trajectory_with_chain(traj, save_file="trajectory_plot.png")

# print(index)

# # for i in range(len(traj)):
# #     #robot_chain.plot(traj[i], ax=None)
# #     # Get the end effector position and orientation
# #     end_effector_position, end_effector_orientation = planner.get_end_effector_position_and_orientation(traj[i])
# #     print(f"Waypoint {i + 1}:")
# #     print(f"End Effector Position: {end_effector_position}")
# #     print(f"End Effector Orientation: {end_effector_orientation}")
# #     print("-----")
