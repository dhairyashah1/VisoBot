"""
This is a debug file
"""

import time
import numpy as np
import os
import pybullet as p
from stretch import Robot
from stretch import *

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

p.setGravity(0, 0, -9.81)  # Set gravity

def initAxis(center, quater):
    rot_mat = p.getMatrixFromQuaternion(quater)
    rotmat = np.array(rot_mat).reshape((3, 3))
    p.addUserDebugLine(lineFromXYZ=center, lineToXYZ=center + rotmat[:3, 0] * 0.1, lineColorRGB=[1, 0, 0], lineWidth=10)
    p.addUserDebugLine(lineFromXYZ=center, lineToXYZ=center + rotmat[:3, 1] * 0.1, lineColorRGB=[0, 1, 0], lineWidth=10)
    p.addUserDebugLine(lineFromXYZ=center, lineToXYZ=center + rotmat[:3, 2] * 0.1, lineColorRGB=[0, 0, 1], lineWidth=10)


root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

################ Plane Environment ################
plane_id = p.loadURDF(os.path.join(root_dir, "resource/urdf/plane.urdf"), [0, 0, 0])
p.changeVisualShape(plane_id, -1)

################ Load the robot ################
mobot_urdf_file = os.path.join(root_dir, "resource/urdf/stretch/stretch.urdf")
# mobot = Robot(pybullet_api=p, start_pos=[0.0, 0.0, 0.0], urdf_file=mobot_urdf_file)
mobot = init_scene(p)

# Let the simulation settle
for _ in range(500):
    p.stepSimulation()

# ################ Load the object to be lifted (e.g., a bowl) ################
# bowl_position = [-0.7, -0.15, 0.3]  # Adjust based on the scene
# bowl_orientation = p.getQuaternionFromEuler([0, 0, 0])
# bowl_scaling = 0.1
# bowl_id = p.loadURDF(os.path.join(root_dir, "resource/urdf/obj_libs/bowls/b1/model.urdf"),
#                      basePosition=bowl_position,
#                      baseOrientation=bowl_orientation,
#                      globalScaling=bowl_scaling,
#                      useFixedBase=False)

################ Movement Helper ################
# def move_joint(robot, joint_index, target_pos, step_size=0.1, max_steps=500):
#     """
#     Moves the specified joint of the robot to the target position in small increments.
#     """
#     current_pos = p.getJointState(robot.robotId, joint_index)[0]
#     step = 0
#     while abs(current_pos - target_pos) > step_size and step < max_steps:
#         new_pos = current_pos + step_size * np.sign(target_pos - current_pos)
#         p.setJointMotorControl2(robot.robotId, joint_index, p.POSITION_CONTROL, targetPosition=new_pos)
#         p.stepSimulation()
#         current_pos = p.getJointState(robot.robotId, joint_index)[0]
#         step += 1
#         time.sleep(0.01)
#     print(f"Joint {joint_index} reached target position {target_pos}")

# # ################ Robot Control ################
# # # Example of controlling arm and gripper (Assuming the robot has specific joint indexes for these)
# # ARM_JOINT_INDEX = 4  # This should be set to the actual joint index of the robot's arm
# # GRIPPER_JOINT_INDEX = 6  # Replace with the actual index for the gripper

# # # Move arm down to the bowl position (assume a certain joint value corresponds to lowering the arm)
# # print("Moving arm to bowl position...")
# # move_joint(mobot, ARM_JOINT_INDEX, target_pos=0.1)  # Replace `0.3` with a suitable value for your arm

# # # Close gripper to grasp the object (gripper index might need fine-tuning)
# # print("Closing gripper to grasp the object...")
# # move_joint(mobot, GRIPPER_JOINT_INDEX, target_pos=0.02)  # Replace with actual gripper closed position

# # # Lift the arm after grasping
# # print("Lifting the object...")
# # move_joint(mobot, ARM_JOINT_INDEX, target_pos=0.6)  # Lift the arm to a higher position

# # # Open the gripper to release the object (optional)
# # print("Opening the gripper to release the object...")
# # move_joint(mobot, GRIPPER_JOINT_INDEX, target_pos=0.05)  # Replace with actual gripper open position

# def move_end_effector_to_position(robot, target_position, target_orientation=None, max_steps=500):
#     """
#     Moves the end effector of the robot to the specified target position and orientation using inverse kinematics.
    
#     :param robot: The robot object
#     :param target_position: The desired position of the end effector (x, y, z)
#     :param target_orientation: The desired orientation of the end effector as a quaternion (optional)
#     :param max_steps: Maximum steps to perform the movement
#     """
#     # Get the joint indices for the end effector
#     end_effector_index = 16  # Replace with the actual index for the end effector if needed
    
#     # If orientation is not provided, set it to the current orientation of the end effector
#     if target_orientation is None:
#         current_orientation = p.getLinkState(robot.robotId, end_effector_index)[1]
#         target_orientation = current_orientation

#     # Use inverse kinematics to calculate the joint angles
#     joint_angles = p.calculateInverseKinematics(
#         robot.robotId, 
#         end_effector_index, 
#         target_position, 
#         target_orientation
#     )

#     # Only move joints 8 to 16
#     for joint_index in range(8, 15):  # This assumes your robot has 20 joints total
#         # Set the target angle from the calculated angles
#         target_angle = joint_angles[joint_index]  # This index might be different, so ensure it matches your robot's configuration
#         move_joint(robot, joint_index, target_angle, max_steps=max_steps)

# # Example usage:
# target_position = [0, 0.2, -0.6]  # Adjust based on your scene - here the 2nd being more negative moves the arm up (y, z, x)
# target_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Adjust orientation if needed

# print("Moving end effector to target position...")
# move_end_effector_to_position(mobot, target_position, target_orientation)

def move_arm_to_position(robot, target_position, max_steps=500, max_velocity=0.05):
    """
    Moves the arm on the mast to the specified target position using prismatic joints.
    
    :param robot: The robot object
    :param target_position: The desired position of the end effector (x, y, z)
    :param max_steps: Maximum steps to perform the movement
    """
    # Joint indices for the arm
    arm_joint_indices = [8, 10, 11, 12, 13, 14, 16]  # Replace with actual indices if different

    joint_angle_indices = [5, 6, 7, 8, 9, 10, 11]

    # Use inverse kinematics to calculate the joint angles
    joint_angles = p.calculateInverseKinematics(
        robot.robotId, 
        19,  # End effector index
        target_position
    )

    for joint_index in arm_joint_indices:
        joint_info = p.getJointInfo(robot.robotId, joint_index)
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]
        print(f'Joint {joint_index} limits: {joint_lower_limit} to {joint_upper_limit}')

    print('joint_angles: ', joint_angles)
    current_position = p.getLinkState(robot.robotId, 19)[0]
    print(f'Current end effector position: {current_position}')
    print(f'Target end effector position: {target_position}')

    # Move each joint to the calculated angle
    for i in range(7):
        print('joint_index: ', arm_joint_indices[i])
        print(f'Moving joint {arm_joint_indices[i]} to angle {joint_angles[joint_angle_indices[i]]}')
        p.setJointMotorControl2(
            bodyIndex=robot.robotId,
            jointIndex=arm_joint_indices[i],
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_angles[joint_angle_indices[i]],
            force=100,
            maxVelocity=max_velocity
        )

    # Step the simulation to move the joints
    for _ in range(max_steps):
        p.stepSimulation()

# Example usage:
# target_position = [-0.009398931636435438, -0.5, 0.8]  # this position is absolute, not relative to the current position of robot
# move_arm_to_position(mobot, target_position)

# target_position = [-0.009398931636435438, -0.2, 0.5]  # this position is absolute, not relative to the current position of robot
# move_arm_to_position(mobot, target_position)

def move_robot_arm_to_position(robot, target_position, max_steps=500, max_velocity=0.05):
    """
    Moves the robot and arm on the mast to the specified target position while dynamically avoiding obstacles.
    
    :param robot: The robot object
    :param target_position: The desired position of the end effector (x, y, z)
    :param max_steps: Maximum steps to perform the movement
    """
    robot_joint_indices = [1, 2, 3, 8, 10, 11, 12, 13, 14, 16]  # Indices of the robot's joints
    joint_angle_indices = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]

    # Calculate the initial position for the end effector
    current_position = np.array(p.getLinkState(robot.robotId, 19)[0])
    target_position = np.array(target_position)
    
    # Step size for incremental movement
    step_size = (target_position - current_position) / max_steps

    for _ in range(max_steps):
        # Move to the next incremental position
        current_position += step_size

        # Calculate joint angles for the current position
        joint_angles = p.calculateInverseKinematics(robot.robotId, 19, current_position)

        # Move each joint to the calculated angle
        for i in range(len(robot_joint_indices)):
            p.setJointMotorControl2(
                bodyIndex=robot.robotId,
                jointIndex=robot_joint_indices[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[joint_angle_indices[i]],
                force=100,
                maxVelocity=max_velocity
            )

        # Step the simulation
        p.stepSimulation()

        # Check for collisions with obstacles
        closest_points = p.getClosestPoints(robot.robotId, -1, distance=0.02)
        if closest_points:  # If any collision is detected
            # Adjust current position to avoid collision
            for point in closest_points:
                # Calculate the normal vector for adjustment
                normal_vector = np.array(point[8]) - current_position  # point[8] is the position of the obstacle
                adjustment = normal_vector / np.linalg.norm(normal_vector) * 0.05  # Adjust by a small factor
                current_position += adjustment  # Move away from the obstacle

            # Recalculate joint angles for the adjusted position
            joint_angles = p.calculateInverseKinematics(robot.robotId, 19, current_position)

# Example usage:
target_position = [2.8, -0.3, 0.85]
move_robot_arm_to_position(mobot, target_position)

################ Main Simulation Loop ################
# Continue simulation for debugging and watching behavior
while True:
    p.stepSimulation()
    time.sleep(1 / 240.0)
