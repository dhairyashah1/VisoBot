"""
This is the main driver code for navigation, manipulation and grasping to control the robot and execute end-to-end tasks.
"""

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
# Robot object and environment settings
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setGravity(0, 0, -9.81)

mobot = init_scene(p, mug_random=False)
object_ids = get_object_ids()
mug1_id = get_mug1_id()

##########################################################################################################
TASK_NUM = 2  # EITHER 1 (mug1 pickup and place on bed) or 2 (mug2 pickup and place in drawer) or 3 (random navigation and manipulation)
##########################################################################################################

# State Machine Flags - Enable
NAV = 1                           # Navigation
NAV_COMPLETE = 0
MANIP = 1                         # Manipulation
MANIP_COMPLETE = 0
GRASP = 1                         # Grasp
GRASP_COMPLETE = 0
END = 0

# Init static grid for mapping
grid_size = (201, 201)             # (51, 51)
cell_size = 0.05                   # 0.2 #= 20 cm
inflation_radius = 0.2 #0.2

# path_type = "random" #shortest

#navigation endpoints
spawn_pos = [-0.8, 0]

##########################################################################################################
# manipulation coordinates
mani_pos_1 =  [0.27, -0.71, 0.92]
mani_pos_2 = [-1.70, -3.70, 0.46]
mani_pos_3 = [1.45, -1.68, 0.59]

# navigation coordinates for above mentioned manipulation coordinates
nav_mani_pos_1 = [-0.26,-0.5, 0.08]
nav_mani_pos_2 = [-1.185, -3.195, 0.08] # tricky - reduce speed
nav_mani_pos_3 = [1.95, -2.0, 0.08 ]
##########################################################################################################

# TASK 1 - Pickup mug_1 and take it to the bed #########################################################
# NAV - nav_overall_pos_1 = [-0.5, -0.9, 0.92]
# MANI - mug1_pos
# GRASP PICKUP
# MANI - current base position
# GRASP PLACE
# MANI - (3.0560903754156827, -2.6100665989264673, 0.08579989843587887)
nav_overall_pos_1 = [-0.5, -0.9, 0.92]
mug1_pos = [0.27, -0.94, 0.898] #1.53]
bed_side_pos = [3.05, -2.71, 0.085]
##########################################################################################################

# TASK 2 - Pickup mug_2 on top of the drawer in the end_room and place it in the open drawer #############
# NAV - end_room_pos = [3.55, 0.31]
# MANI - mug2_pos = [3.689974353354115, 0.050003517533320364, 0.95]
# GRASP PICKUP -
# MANI - [(current_pos[0]-0.23, current_pos[1])]
# GRASP PLACE (release)
end_room_pos = [3.55, 0.31]
drawer_pos = [3.84, 0.05,  0.42]
open_drawer_pos = [3.60, 0.05, 0.42]
mug2_pos = [3.689974353354115, 0.050003517533320364, 0.95] #0.8347266461268507]
##########################################################################################################

# Grid setup
grid = StaticGrid(grid_size=grid_size, cell_size=cell_size, inflation_radius=inflation_radius)

grid.update_grid_with_objects(object_ids=object_ids)
# grid.mark_obstacle_without_inflation(get_cabinet_id()) # Attempt to mark obstacle
print("Grid initialized!")

mobot.get_observation()

total_driving_distance = 0
previous_pos = 0
current_pos = 0
previous_pos, _, _ = get_robot_base_pose(p, mobot.robotId)
current_pos = previous_pos

constraint = None

navi_flag = False
grasp_flag = False

##########################################################################################################
def nearest_random_manipulation_coordinate(coordinate, radius=0.4):
    """
    Generates a random eligible co-ordinate of the world near the given coordinate in extent of given radius / distance
    """
    x, y = coordinate
    z = random.uniform(0.46, 1)

    # Generate random distance within radius
    # distance = random.uniform(0, radius)

    # Generate point at that radius
    distance = radius

    # Generate random angle between 0 and 2π
    angle = random.uniform(0, 2 * math.pi)

    # Calculate new x and y coordinates based on distance and angle
    new_x = x + distance * math.cos(angle)
    new_y = y + distance * math.sin(angle)
    return [new_x, new_y, z]

##########################################################################################################
if TASK_NUM == 3:
    random_coordinates = grid.get_random_points(num_points=1)

    # TASK 3 - Random location navigation and manipulation
    grasp_id = 0  # Change this if needed
    navi_location = random_coordinates[0]
    print(f"Generated Random Navigation Location = {navi_location}")
    mani_location = nearest_random_manipulation_coordinate(navi_location)
    print(f"Generated Random Manipulation Location = {mani_location}")
##########################################################################################################

# Miscellaneous points
cup_pos = [-0.26,-0.5, 0.08]
trashbin_pos = [-1.1, -4.01, 0.48]

########################################################################################################

def manipulate_simple_ik(p, mobot, target_position = [0.27, -0.89, 0.91]):
    """
    This function uses Inverse kinematics to generate waypoints and control the arm for manipulation operation
    """
    global MANIP_COMPLETE
    # Define the joint indices and limits
    # joint_indices = [1, 2, 3, 8, 10, 11, 12, 13, 14, 16] # added 1, 2
    joint_indices = [3, 8, 10, 11, 12, 13, 14, 16] # added 1, 2
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

    time.sleep(5)
    # controller.orient_base_to_match_arm_orientation(target_position)
    controller.move_to_position(target_position, max_velocity=0.05) #ORIGINAL 0.05
    arm_control(mobot, p, up=0, stretch=0, roll=0, yaw=0)
    base_control(mobot, p, forward=0, turn=0)
    motion_planning_test(p, mobot.robotId, target_position)

    MANIP_COMPLETE = 1


def generate_waypoints(p, mobot, start_pos = spawn_pos, target_pos = end_room_pos):
    """
    This function generates waypoints for robot navigation as per the target position and start position
    """
    global grid

    start_idx = grid.world_to_grid(start_pos[0], start_pos[1]) # spawn position
    end_idx = grid.world_to_grid(target_pos[0], target_pos[1])

    print("Start grid index = ", start_idx)
    print("End grid index = ", end_idx)

    path = astar(grid=grid, start=start_idx, end=end_idx)
    print(path)

    # Note: Always mark custom cells AFTER running astar - for debugging purposes
    # grid.mark_custom_cell(start_idx, 3)
    # grid.mark_custom_cell(end_idx, 3)
    # grid.mark_custom_cells(path, 2)
    # grid.mark_custom_cell(start_idx, 3)
    # grid.mark_custom_cell(end_idx, 3)
    grid.print_grid()

    # get path in real-world coordinates
    waypoints = [grid.grid_to_world(cell[0], cell[1]) for cell in path]
    # print(waypoints)
    smoothed_waypoints = smooth_trajectory_cubic(waypoints, num_points=200)
    return smoothed_waypoints

########################################################################################################

def follow_waypoints(p, mobot, smoothed_waypoints):
    """
    This function makes the robot follow the supplied waypoints
    """
    global current_pos
    global previous_pos
    global total_driving_distance
    global NAV_COMPLETE

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
                targetVelocity=0.002, # 0.001
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

    # Alternate way to stop the robot
    # for motorId in [1, 2, 3]:
    #     p.setJointMotorControl2(
    #         mobot.robotId,
    #         motorId,
    #         controlMode=p.POSITION_CONTROL,
    #         targetPosition=joint_positions[motorId - 1],
    #         targetVelocity=0,
    #         force=80
    # )

########################################################################################################

def grasp_attach(p, mobot, obj_id=21):
    """
    This function attaches object mentioned to the gripper
    """
    global constraint
    constraint = attach(obj_id, mobot.robotId, 18)

########################################################################################################

def grasp_detach(p, mobot):
    """
    This function detaches object mentioned to the gripper
    """
    global constraint
    detach(constraint)
    constraint = None

########################################################################################################

# DRIVER CODE ###########################################################################################

if TASK_NUM == 1:
    smoothed_waypoints = generate_waypoints(p, mobot, target_pos=nav_overall_pos_1)      ### 1
elif TASK_NUM == 2:
    smoothed_waypoints = generate_waypoints(p, mobot, target_pos=end_room_pos)       ### 2
else:
    smoothed_waypoints = generate_waypoints(p, mobot, target_pos=navi_location)       ### some other task

########################################################################################################

while (1):
    time.sleep(1./240.)

    # mug_position = get_mug_pose(p)
    # print(f"Mug position: {mug_position}")

    # NAVIGATION ##############################################################################################
    if NAV and not(NAV_COMPLETE):
        follow_waypoints(p, mobot, smoothed_waypoints)
        print("==== Navigation complete =================")

    # MANIPULATION ############################################################################################
    elif MANIP and NAV_COMPLETE and not(MANIP_COMPLETE):
        mug_position = get_mug_pose(p)
        # print(f"Mug position: {mug_position}")
        if TASK_NUM == 1:
            manipulate_simple_ik(p, mobot, mug1_pos)        ## 1
        elif TASK_NUM == 2:
            manipulate_simple_ik(p, mobot, mug2_pos)       ## 2
        else:
            manipulate_simple_ik(p, mobot, mani_location)       ## 3
            print(f"Generated Random Manipulation Location = {mani_location}")

        MANIP_COMPLETE = 1
        print("==== Manipulation complete =================")
        # ee_position, _, _ = get_robot_ee_pose(p, mobot.robotId)
        # print(f"End-Effector Position: {ee_position}")

    # GRASPING ATTACH ###########################################################################################
    elif GRASP and NAV_COMPLETE and MANIP_COMPLETE and not(GRASP_COMPLETE):
        if TASK_NUM == 1:
            grasp_attach(p, mobot, mug1_id) # 1
        elif TASK_NUM == 2:
            grasp_attach(p, mobot) # 2
        else:
            grasp_attach(p, mobot, grasp_id)
        print("==== Grasp Attach complete =================")
        GRASP_COMPLETE = 1

    # REMANING OPERATIONS #######################################################################################
    elif GRASP_COMPLETE and not(END):
        if TASK_NUM == 1:
            # Manipoulation 2
            # manipulate_simple_ik(p, mobot, [current_pos[0]-0.05, current_pos[1], 0.85]) # current_pos[2] + 0.2])                                    ## 1
            # Alternate method of manipulation 2
            arm_control(mobot, p , stretch=-1)
            time.sleep(2.7)
            arm_control(mobot, p, stretch=0, up=-1)
            time.sleep(1.2)
            arm_control(mobot, p, up=0)

            # Detach gripper
            gripper_control(mobot, p, cmd=0)
            grasp_detach(p, mobot)

            current_pos, _, _ = get_robot_base_pose(p, mobot.robotId)

            smoothed_waypoints2 = generate_waypoints(p, mobot, start_pos=current_pos, target_pos=bed_side_pos)      ### 1
            follow_waypoints(p, mobot, smoothed_waypoints2)

            # current_pos, _, _ = get_robot_base_pose(p, mobot.robotId)
            mug_position = get_mug_pose(p, mug_id=mug1_id)

            # Manipulation 3
            # manipulate_simple_ik(p, mobot, [mug_position[0], mug_position[1], mug_position[2] + 0.1])
            # Alternate method of manipulation 3
            arm_control(mobot, p, up=-1)
            time.sleep(2)
            arm_control(mobot, p, up=0)

            grasp_attach(p, mobot, mug1_id)

            time.sleep(1)

            # Manipulation 4
            # manipulate_simple_ik(p, mobot, [bed_side_pos[0], bed_side_pos[1] + 0.8, bed_side_pos[2] + 0.5])
            # Alternate method of manipulation 4
            arm_control(mobot, p , up=1)
            time.sleep(2.5)
            arm_control(mobot, p, stretch=1, up=0)
            time.sleep(4)
            arm_control(mobot, p, stretch=0)
            time.sleep(1)

        elif TASK_NUM == 2:
            current_pos, _, _ = get_robot_base_pose(p, mobot.robotId)
            follow_waypoints(p, mobot, [(current_pos[0]-0.23, current_pos[1])])            ## 2
        else:
            # Task 3
            pass

        # detach gripper
        gripper_control(mobot, p, cmd=0)
        grasp_detach(p, mobot)

        print("==== Navigation to GOAL complete =================")
        base_control(mobot, p, forward=0, turn=0)
        END = 1

    # COMPLETE TASK #########################################################################################
    else:
        print("=== TASK DONE =================")
        break

########################################################################################################
# Stop simulation
while True:
    p.stepSimulation()
    time.sleep(1 / 240.0)

########################################################################################################
