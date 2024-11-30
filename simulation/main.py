import time
import numpy as np
import pickle
import sys
import os
import pybullet as p
from stretch import *
from utils.tools import *
from grid import StaticGrid
from global_planner import *
from move import follow_waypoints, stop_robot
from trajectory import smooth_trajectory_cubic

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setGravity(0, 0, -9.81)

mobot = init_scene(p, mug_random=False)
object_ids = get_object_ids()

# Init static grid
# grid_size = (51, 51)
# cell_size = 0.2 # 20 cm
grid_size = (201, 201)
cell_size = 0.05
inflation_radius = 0.35

path_type = "random"
grid = StaticGrid(grid_size=grid_size, cell_size=cell_size, inflation_radius=inflation_radius)

grid.update_grid_with_objects(object_ids=object_ids)
# grid.mark_obstacle_without_inflation(get_cabinet_id()) # Attempt to mark obstacle
# print("Grid initialized!")
# grid.print_grid()

# start_coordinate = (-0.8, 0)
# end_coordinate = (1.45, -1.68)

start_idx = grid.world_to_grid(-0.8, 0)
# [0.27, -0.71, 0.92] [-1.70, -3.70, 0.46] [1.45, -1.68, 0.59]
# end_idx = grid.world_to_grid(-1.70, -3.70) # prof position
# [3.84, 0.05,  0.42] drawer position
end_idx = grid.world_to_grid(2.54, 0.05)

print(start_idx)
print(end_idx)

grid.print_grid()

path = astar(grid=grid, start=start_idx, end=end_idx, path_type= path_type)
print(path)

# Note: Always mark custom cells AFTER running astar...
grid.mark_custom_cell(start_idx, 3)
grid.mark_custom_cell(end_idx, 3)
grid.mark_custom_cells(path, 2)

grid.mark_custom_cell(start_idx, 3)
grid.mark_custom_cell(end_idx, 3)

grid.print_grid()

# get path in real-world coordinates
waypoints = [grid.grid_to_world(cell[0], cell[1]) for cell in path]

print(waypoints)

smoothed_waypoints = smooth_trajectory_cubic(waypoints, num_points=200)

# Todo trajectory smoothing
    
forward=0
turn=0
speed=10
up=0
stretch=0
gripper_open=0
roll=0
yaw=0

mobot.get_observation()

total_driving_distance = 0
previous_position, _, _ = get_robot_base_pose(p, mobot.robotId)
current_position = previous_position

constraint = None

navi_flag = False
grasp_flag = False
total_driving_distance - 0
previous_pos, _, _ = get_robot_base_pose(p, mobot.robotId)

while (1):
    time.sleep(1./240.)

    mobot.get_observation()

    # follow_waypoints(p, mobot, smoothed_waypoints)

    ## follow_waypoints -
    for waypoint in smoothed_waypoints:
        target_pos = [waypoint[0], waypoint[1], 0.03]  # z= 0.03 in stretch.py

        # until the robot reaches the current waypoint
        while True:
            joint_positions = p.calculateInverseKinematics(
                mobot.robotId,
                4,  # base
                target_pos
            )

            # control robot's base joints
            # p.setJointMotorControlArray(
            #     mobot.robotId,
            #     [1, 2, 3],
            #     controlMode=p.POSITION_CONTROL,
            #     targetPositions=joint_positions[:3],
            #     forces=[7, 7, 7]
            # )

            for motorId in [1, 2, 3]:
                p.setJointMotorControl2(
                mobot.robotId,
                motorId,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[motorId - 1],
                targetVelocity=0.001,
                force=5
            )


            # Get the robot's current position
            current_pos, _, _ = get_robot_base_pose(p, mobot.robotId)

            # Update total driving distance
            total_driving_distance += np.linalg.norm(np.array(current_pos) - np.array(previous_pos))
            previous_pos = current_pos

            # Check if the robot is close enough to the target waypoint
            distance_to_target = np.linalg.norm(np.array(current_pos[:2]) - np.array(waypoint))
            if distance_to_target <= 0.15:
                print(f"Reached waypoint: {waypoint}")
                break  # Move to the next waypoint

            # Wait for a short duration before checking again
            time.sleep(1./240.)

    ## old
    # current_position, _, _ = get_robot_base_pose(p, mobot.robotId)
    # total_driving_distance += np.linalg.norm(np.array(current_position) - np.array(previous_position))
    # previous_position = current_position

    if navi_flag == False:
        if current_position[0] > 1.6 and current_position[1] > -0.35:
            print("Reached the goal region!")
            print("Total driving distance: ", total_driving_distance)
            navi_flag = True
            # break
            # stop_robot(p, mobot)
        else:
            # pass
            print("Current driving distance: ", total_driving_distance)
            print("Current position: ", current_position)
            # stop_robot(p, mobot)
            # pass
            break
    else:
        # pass
        print("Reached the goal region! Total driving distance: ", total_driving_distance)
        # stop_robot(p, mobot)


    # if grasp_flag == False:
    #     mug_position = get_mug_pose(p)
    #     #print("Mug position: ", mug_position)

    #     if mug_position[0] > 3.3 and mug_position[0] < 3.5 \
    #         and mug_position[1] > -0.17 and mug_position[1] < 0.25 \
    #         and mug_position[2] > 0.71 and mug_position[2] < 0.75:
    #         print("Mug is in the drawer!")
    #         grasp_flag = True
    # else:
    #     print("Mug is in the drawer!")

for motorId in [1, 2, 3]:
    p.setJointMotorControl2(
        mobot.robotId,
        motorId,
        controlMode=p.POSITION_CONTROL,
        targetPosition=joint_positions[motorId - 1],
        targetVelocity=0,
        force=0
    )

while True:
    p.stepSimulation()
    time.sleep(1 / 240.0)
    # ee_position, _, _ = get_robot_ee_pose(p, mobot.robotId)
    #print("End-effector position: ", ee_position)
