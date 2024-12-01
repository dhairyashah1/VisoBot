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
from motion_planner_simple_ik import EndEffectorController

####################################    INITIALIZATION  #################################################
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setGravity(0, 0, -9.81)

mobot = init_scene(p, mug_random=False)
object_ids = get_object_ids()

NAV = 1
NAV_COMPLETE = 0
MANIP = 1
MANIP_COMPLETE = 0
GRASP = 1
GRASP_COMPLETE = 0

# Init static grid
grid_size = (201, 201)   # (51, 51)
cell_size = 0.05         # 0.2 #= 20 cm
inflation_radius = 0.2 #0.2

# path_type = "random" #shortest

#navigation endpoints
spawn_pos = [-0.8, 0]
# end_room_pos = [3.6, 0.9] # [2.54, 0.05] #old right of box
# end_room_pos = [3.6, 0.9] # [2.54, 0.05] #old right of box
end_room_pos = [3.3, 0.25] # 3.3, 0.25[2.54, 0.05] #old right of box

# manipulation coordinates
mani_pos_1 =  [0.27, -0.71, 0.92]
mani_pos_2 = [-1.70, -3.70, 0.46]
mani_pos_3 = [1.45, -1.68, 0.59]

nav_mani_pos_1 = [-0.26,-0.5, 0.08]
nav_mani_pos_2 = [-1.185, -3.195, 0.08] # tricky - reduce speed
nav_mani_pos_3 = [1.95, -2.0, 0.08 ]

mug1_pos = [0.25, -0.93, 1.53]
trashbin_pos = [-1.1, -4.01, 0.48]

drawer_pos = [3.84, 0.05,  0.42]
# mug2_pos = [drawer_pos[0]-0.15, drawer_pos[1], 0.75] # 1.5]
mug2_pos = [3.689974353354115, 0.050003517533320364, 0.9] #0.8347266461268507]

cup_pos = [-0.26,-0.5, 0.08]

# FINAL VIDEO
# 1)
# NAV - navi_mani_pos_1
# MANI - mug1_pos
# GRASP
#

# 2)
# NAV - end_room_pos
# MANI - mug2_pos
# GRASP
########################################################################################################

mobot.get_observation()

total_driving_distance = 0
previous_pos = 0
current_pos = 0
previous_pos, _, _ = get_robot_base_pose(p, mobot.robotId)
current_pos = previous_pos

constraint = None

navi_flag = False
grasp_flag = False
# previous_pos, _, _ = get_robot_base_pose(p, mobot.robotId)

def manipulate_simple_ik(p, mobot, target_position = [0.27, -0.89, 0.91]):

    global MANIP_COMPLETE
    # Define the joint indices and limits
    joint_indices = [3, 8, 10, 11, 12, 13, 14, 16]
    joint_limits = {}

    # Print joint information
    for i in joint_indices:
        joint_info = p.getJointInfo(mobot.robotId, i)
        joint_name = joint_info[1].decode('utf-8')  # Joint name is the second element
        joint_type = joint_info[2]  # Joint type is the third element
        joint_index = joint_info[0]  # Joint index is the first element
        joint_lower_limit = joint_info[8]  # Lower limit of the joint
        joint_upper_limit = joint_info[9]  # Upper limit of the joint

        # Print the joint information
        # print(f"Joint Index: {joint_index}, Joint Name: {joint_name}, Joint Type: {joint_type}, "
        #     f"Lower Limit: {joint_lower_limit}, Upper Limit: {joint_upper_limit}")

        joint_limits[joint_index] = (joint_lower_limit, joint_upper_limit)

    # Create the EndEffectorController
    end_effector_index = 18  # Replace with the actual end effector index
    controller = EndEffectorController(mobot, mobot.robotId, end_effector_index, joint_indices, joint_limits)

    time.sleep(10)
    # controller.orient_base_to_match_arm_orientation(target_position)
    controller.move_to_position(target_position, max_velocity=0.01) #ORIGINAL 0.05
    arm_control(mobot, p, up=0, stretch=0, roll=0, yaw=0)
    base_control(mobot, p, forward=0, turn=0)
    motion_planning_test(p, mobot.robotId, target_position)

    MANIP_COMPLETE = 1
    # Keep the simulation running for debugging
    # while True:
    #     p.stepSimulation()
    #     time.sleep(1 / 240.0)


def generate_waypoints(p, mobot, start_pos = spawn_pos, target_pos = end_room_pos):
    grid = StaticGrid(grid_size=grid_size, cell_size=cell_size, inflation_radius=inflation_radius)

    grid.update_grid_with_objects(object_ids=object_ids)
    # grid.mark_obstacle_without_inflation(get_cabinet_id()) # Attempt to mark obstacle
    print("Grid initialized!")

    start_idx = grid.world_to_grid(start_pos[0], start_pos[1]) # spawn position
    end_idx = grid.world_to_grid(target_pos[0], target_pos[1])

    print("Start grid index = ", start_idx)
    print("End grid index = ", end_idx)

    path = astar(grid=grid, start=start_idx, end=end_idx)
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
    return smoothed_waypoints

def follow_waypoints(p, mobot, smoothed_waypoints):

    global current_pos
    global previous_pos
    global total_driving_distance
    global NAV_COMPLETE
    # follow_waypoints(p, mobot, smoothed_waypoints)
    total_driving_distance = 0
    previous_pos, _, _ = get_robot_base_pose(p, mobot.robotId)
    current_pos = previous_pos
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
                targetVelocity=0.001, # 0.001
                force=5
            )

            # Get the robot's current position
            current_pos, _, _ = get_robot_base_pose(p, mobot.robotId)

            # Update total driving distance
            total_driving_distance += np.linalg.norm(np.array(current_pos) - np.array(previous_pos))
            previous_pos = current_pos

            # Check if the robot is close enough to the target waypoint
            distance_to_target = np.linalg.norm(np.array(current_pos[:2]) - np.array(waypoint))
            if distance_to_target <= 0.13:
                print(f"Reached waypoint: {waypoint}")
                break  # Move to the next waypoint

            # Wait for a short duration before checking again
            time.sleep(1./240.)

    print("All waypoints reached.")
    print("====================================================")
    print("Navigation target reached ===>")
    print("----------------------------------------------------")
    print(f"Total driving distance: {total_driving_distance} metres")
    print("----------------------------------------------------")
    print(f"Current robot position:{current_pos}")
    print("====================================================")

    NAV_COMPLETE = 1
    # Stop the robot
    base_control(mobot, p, forward=0, turn=0)
    # for motorId in [1, 2, 3]:
    #     p.setJointMotorControl2(
    #         mobot.robotId,
    #         motorId,
    #         controlMode=p.POSITION_CONTROL,
    #         targetPosition=joint_positions[motorId - 1],
    #         targetVelocity=0,
    #         force=80
    # )

def grasp_attach(p, mobot):
    # gripper open
    global constraint
    constraint = attach(21, mobot.robotId, 18)

def grasp_detach(p, mobot):
    # gripper close
    global constraint
    detach(constraint)
    constraint = None

# driver code
# total_driving_distance = 0
smoothed_waypoints = generate_waypoints(p, mobot, target_pos=end_room_pos)
while (1):
    time.sleep(1./240.)

    mug_position = get_mug_pose(p)
    print(f"Mug position: {mug_position}")

    if NAV and not(NAV_COMPLETE):
        follow_waypoints(p, mobot, smoothed_waypoints)

        # if navi_flag == False:
        #     if current_pos[0] > 1.6 and current_pos[1] > -0.35:
        #         print("Reached the goal region!")
        #         print("Total driving distance: ", total_driving_distance)
        #         navi_flag = True
        #         break
        #         # stop_robot(p, mobot)
        #     else:
        #         # pass
        #         print("Current driving distance: ", total_driving_distance)
        #         print("Current position: ", current_pos)
        #         # stop_robot(p, mobot)
        #         # pass
        #         # break
        # else:
        #     # pass
        #     print("Reached the goal region! Total driving distance: ", total_driving_distance)
        #     break
        #     # stop_robot(p, mobot)

    elif MANIP and NAV_COMPLETE and not(MANIP_COMPLETE):
        mug_position = get_mug_pose(p)
        print(f"Mug position: {mug_position}")
        manipulate_simple_ik(p, mobot, mug2_pos)
        MANIP_COMPLETE = 1
        # ee_position, _, _ = get_robot_ee_pose(p, mobot.robotId)
        # print(f"End-Effector Position: {ee_position}")
    elif GRASP and NAV_COMPLETE and MANIP_COMPLETE:
        grasp_attach(p, mobot)
        print("Grasp Done")


while True:
    p.stepSimulation()
    time.sleep(1 / 240.0)

    # if MANIP and :
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


    # ee_position, _, _ = get_robot_ee_pose(p, mobot.robotId)
    #print("End-effector position: ", ee_position)
