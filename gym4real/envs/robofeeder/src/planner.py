import os
from ikpy.chain import Chain
from ikpy.link import URDFLink
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

class PlanningClass:

    def __init__(self):
        # Load the robot arm chain from URDF or manually

        # Get the directory this script is in
        current_dir = os.path.dirname(__file__)

        # Build the path to staubli/urdf relative to this file
        urdf_path = os.path.join(current_dir, "staubli", "urdf/")
        self.robot_chain = Chain.from_urdf_file(urdf_path + "tx2_60.urdf", active_links_mask = [False, True, True, True, True, True, True, False, False])
        self.above_offset = 0.25
        self.final_position = [-0.0882, 0.4929, 0.4596]
        self.num_interpolated_points = 5  # Number of interpolated points between each waypoint pair

    def make_transform_matrix(self,position, quat):
        """
        Creates a 4x4 transformation matrix from a given position and quaternion.
        This method converts a quaternion into a 3x3 rotation matrix and combines it
        with a position vector to form a 4x4 transformation matrix. The resulting
        matrix can be used to represent the pose of an object in 3D space.
        Args:
            position (array-like): A 3-element array or list representing the 
                position [x, y, z] in 3D space.
            quat (array-like): A 4-element array or list representing the quaternion 
                [x, y, z, w], where w is the scalar part.
        Returns:
            numpy.ndarray: A 4x4 transformation matrix combining the rotation and 
            translation.
        """
        # Convert quaternion to 3x3 rotation matrix
        rot = R.from_quat(quat).as_matrix()  # quat = [x, y, z, w]
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = position
        return transform
    
    def extract_axis_from_quat(self, quat, axis="Z"):
        """
        Extracts a specific axis vector from a quaternion.

        This method converts a quaternion into a rotation matrix and retrieves the 
        specified axis vector (X, Y, or Z) from the matrix.

        Args:
            quat (array-like): The quaternion in the format [x, y, z, w].
            axis (str): The axis to extract ('X', 'Y', or 'Z'). Defaults to 'Z'.

        Returns:
            numpy.ndarray: The extracted axis vector as a 3-element array.

        Raises:
            ValueError: If an invalid axis is specified.
        """
        r = R.from_quat(quat)  # x, y, z, w format
        rot_matrix = r.as_matrix()

        if axis == "X":
            return rot_matrix[:, 0]
        elif axis == "Y":
            return rot_matrix[:, 1]
        elif axis == "Z":
            return rot_matrix[:, 2]
        else:
            raise ValueError("Invalid axis")

    def compute_ik(self, position, quat, axis="Z", initial_j_position=None):
        """
        Computes the inverse kinematics (IK) solution for a given target position and orientation.

        This method calculates the joint positions required for the robot to reach a specified 
        position and orientation in space. The orientation is determined based on the specified 
        axis and the quaternion provided.

        Args:
            position (list or tuple): The target position in 3D space as [x, y, z].
            quat (list or tuple): The target orientation represented as a quaternion [x, y, z, w].
            axis (str, optional): The axis to align the orientation with. Defaults to "Z".
            initial_j_position (list or tuple, optional): The initial joint positions to seed the 
                IK solver. Defaults to None.

        Returns:
            list: A list of joint positions representing the IK solution.
        """
        direction = self.extract_axis_from_quat(quat, axis=axis)

        ik_solution = self.robot_chain.inverse_kinematics(
            target_position=position,
            target_orientation=direction,
            orientation_mode=axis,
            initial_position=initial_j_position
        )
        return ik_solution


    def cubic_spline_interpolation(self, waypoints, num_points=50):
        """
        Interpolates joint positions using cubic splines between every consecutive pair of waypoints,
        except when i == 1, where linear interpolation is used instead.

        Args:
            waypoints (list or np.ndarray): List or array of joint positions.
            num_points (int): Number of interpolated points between each pair of waypoints.

        Returns:
            np.ndarray: Interpolated joint positions.
        """
        all_interpolated_points = []

        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_end = waypoints[i + 1]

            t_interp = np.linspace(0, 1, num_points)

            if i == 1 or i ==2:
                # Linear interpolation
                interpolated_points = np.array([
                    wp_start + (wp_end - wp_start) * t for t in t_interp
                ])
            else:
                # Cubic spline interpolation
                t = np.array([0, 1])
                cs = CubicSpline(t, np.array([wp_start, wp_end]).T, axis=1)
                interpolated_points = cs(t_interp).T

            all_interpolated_points.extend(interpolated_points)

        return np.array(all_interpolated_points)


    def planFunction(self, initial_joint_position=None, obj_position=[0.0882 , 0.0  ,0.4596]):
        """
        Plans a trajectory for the robot arm to pick an object and place it at a final position.
        This function computes a sequence of waypoints for the robot arm to follow, including
        pre-pick, pick, post-pick, and place positions. It uses inverse kinematics (IK) to calculate
        joint configurations for each waypoint and interpolates between them using cubic splines
        to generate a smooth trajectory.
        Args:
            initial_joint_position (list or None): The initial joint positions of the robot arm.
                If None, the initial joint positions are set to zeros. Otherwise, the provided
                joint positions are used, with additional zeros added for the base link, flange,
                and gripper.
            obj_position (list): The 3D position of the object to be picked, specified as
                [x, y, z]. Defaults to [0.0882, 0.0, 0.4596].
        Returns:
            tuple: A tuple containing:
                - plan (numpy.ndarray): The planned trajectory, including interpolated points.
                - indices (list): Indices of the key waypoints in the trajectory.
        """
        if initial_joint_position is None:
            initial_joint_position = np.zeros(len(self.robot_chain.links))
        else:
            #add as 0 the position of baselink ,  flange and gripper
            initial_joint_position = [0.0] + initial_joint_position + [0.0, 0.0]
            initial_joint_position = np.array(initial_joint_position)

        waypoints = []
        indices = []

        waypoints.append(initial_joint_position)

        quat = [1, 0, 0, 0]  # Assuming no rotation for the object
        # 0 - Pre-pick: hover above the object
        pre_pick_pos = [obj_position[0], obj_position[1], obj_position[2] + self.above_offset]
        wp0 = self.compute_ik(pre_pick_pos, quat, initial_j_position=initial_joint_position)
        waypoints.append(wp0)
        indices.append(self.num_interpolated_points)

        # 1 - Pick: go to object
        wp1 = self.compute_ik(obj_position, quat, initial_j_position=list(wp0))
        waypoints.append(wp1)
        indices.append((len(waypoints)-1)*self.num_interpolated_points)

        # 2 - Post-pick: move back up (reusing wp0 as the "hover above" position)
        waypoints.append(wp0)  
        indices.append((len(waypoints)-1)*self.num_interpolated_points)

        # 3 - Place: move to final drop location
        wp3 = self.compute_ik(self.final_position, quat, initial_j_position=list(wp0))
        waypoints.append(wp3)
        indices.append((len(waypoints)-1)*self.num_interpolated_points)

        # Interpolate between all consecutive waypoints using cubic splines

        interpolated_trajectory = self.cubic_spline_interpolation(waypoints, num_points=self.num_interpolated_points)

        # Add interpolated points to waypoints (all intermediate points)
        waypoints = interpolated_trajectory.tolist()

        # Final plan (including interpolated points)
        plan = np.array(waypoints)
        
        return plan, indices

    def forward_kinematics(self, joint_angles):
        """
        Compute the forward kinematics for the robot given a set of joint angles.

        Args:
            joint_angles (list or np.ndarray): Joint angles of the robot.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the end effector's position and orientation.
        """
        # Use the robot chain's forward kinematics method to get the transformation matrix
        transform_matrix = self.robot_chain.forward_kinematics(joint_angles)

        return transform_matrix

    def get_end_effector_position_and_orientation(self, joint_angles):
        """
        Get the position and orientation of the end effector from the forward kinematics.

        Args:
            joint_angles (list or np.ndarray): Joint angles of the robot.

        Returns:
            position (np.ndarray): A 3D vector representing the position of the end effector.
            orientation (np.ndarray): A 3x3 rotation matrix representing the orientation of the end effector.
        """
        # Get the transformation matrix
        transform_matrix = self.forward_kinematics(joint_angles)
        
        # Extract the position (translation part)
        position = transform_matrix[:3, 3]
        
        # Extract the orientation (rotation part)
        orientation = transform_matrix[:3, :3]

        return position, orientation