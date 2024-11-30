import pybullet as p
import numpy as np
import time
from utils.tools import *

def follow_waypoints(p, mobot, waypoints):
    """
    Moves the robot's base through the specified waypoints using inverse kinematics.
    """

    for waypoint in waypoints:
        target_pos = [waypoint[0], waypoint[1], 0.03]  # z= 0.03 in stretch.py

        # until the robot reaches the current waypoint
        while True:
            joint_positions = p.calculateInverseKinematics(
                mobot.robotId,
                4,  # base
                target_pos
            )

            # control robot's base joints
            p.setJointMotorControlArray(
                mobot.robotId,
                [1, 2, 3],
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_positions[:3],
                forces=[7, 7, 7]
            )

            # Get the robot's current position
            current_pos, _, _ = get_robot_base_pose(p, mobot.robotId)

            # Check if the robot is close enough to the target waypoint
            distance_to_target = np.linalg.norm(np.array(current_pos[:2]) - np.array(waypoint))
            if distance_to_target <= 0.25:
                print(f"Reached waypoint: {waypoint}")
                break  # Move to the next waypoint

            # Wait for a short duration before checking again
            time.sleep(1./240.)

    print("All waypoints reached.")

def stop_robot(p, mobot):
    """
    Stops the robot
    """

    # control robot's base joints
    # p.resetBaseVelocity(
    #     mobot.robotId,
    #     [0, 0, 0],
    #     [0, 0, 0]
    # )

    # p.setJointMotorControlArray(
    #     mobot.robotId,
    #     [1, 2, 3],
    #     controlMode=p.VELOCITY_CONTROL,
    #     # targetPositions=joint_positions[:3],
    #     forces=[0, 0, 0]
    # )

    while True:
        p.stepSimulation()
        time.sleep(1 / 240.0)