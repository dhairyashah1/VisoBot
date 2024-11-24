import numpy as np
import pybullet as p
import time
from stretch import init_scene
from utils.tools import get_robot_ee_pose, get_robot_base_pose, motion_planning_test
from stretch import base_control, arm_control

def compute_ik(p, robot_id, ee_link, target_position):
    """
    Use PyBullet's inverse kinematics solver to compute joint angles for a given target position.
    """
    joint_angles = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link,
        targetPosition=target_position
    )
    join_angles_indices = [5,
                           6, 7, 8, 9,
                           10, 11]
    # Filter the angles for the desired indices
    filtered_joint_angles = [joint_angles[i] for i in join_angles_indices]

    return filtered_joint_angles

def is_collision_free(p, robot_id, table_id):
    """
    Check if the current robot configuration is collision-free.
    """
    current_pos = get_robot_ee_pose(p, robot_id)[0]
    print('current pos: ', current_pos)
    contacts = p.getClosestPoints(bodyA=robot_id, bodyB=table_id, distance=0.02)
    print('is collision free: ', len(contacts) == 0)
    return len(contacts) == 0

def informed_rrt_ik(
    p, robot_id, table_id, joint_indices, ee_link, target_position, joint_limits, max_iterations=500, step_size=0.1
):
    """
    Informed RRT algorithm with inverse kinematics for planning a path in joint space.
    """
    # Get the current joint states
    current_joint_angles = [p.getJointState(robot_id, i)[0] for i in joint_indices]
    tree = [current_joint_angles]
    parent_map = {0: None}
    goal_node = None

    def sample_random_configuration():
        """
        Sample random joint angles within joint limits or bias toward the goal using IK.
        """
        if np.random.rand() < 0.2:  # 20% goal bias
            return compute_ik(p, robot_id, ee_link, target_position)
        return [np.random.uniform(low, high) for low, high in joint_limits]

    def nearest_configuration(sample):
        """
        Find the nearest joint configuration in the tree to the sampled configuration.
        """
        distances = [np.linalg.norm(np.array(node) - np.array(sample)) for node in tree]
        return np.argmin(distances)

    def interpolate_configuration(start, end, step_size):
        """
        Interpolate between two joint configurations with a step size.
        """
        direction = np.array(end) - np.array(start)
        distance = np.linalg.norm(direction)
        if distance < step_size:
            return end
        return start + direction / distance * step_size

    for i in range(max_iterations):
        # Sample a random configuration
        sample = sample_random_configuration()
        nearest_idx = nearest_configuration(sample)
        nearest_node = tree[nearest_idx]
        
        # Attempt to connect to the sampled configuration
        new_node = interpolate_configuration(nearest_node, sample, step_size)
        # Save the current state
        state_id = p.saveState()

        for idx, angle in zip(joint_indices, new_node):
            p.resetJointState(robot_id, idx, angle)
        
        if is_collision_free(p, robot_id, table_id):
            tree.append(new_node)
            parent_map[len(tree) - 1] = nearest_idx

            # Check if the end-effector is close to the target
            ee_position, _, _ = get_robot_ee_pose(p, robot_id)
            if np.linalg.norm(np.array(ee_position) - np.array(target_position)) < step_size:
                goal_node = len(tree) - 1
                break
        # Restore the saved state
        p.restoreState(state_id)
        p.removeState(state_id)

    if goal_node is None:
        print("Failed to find a path within the given iterations.")
        return None

    # Reconstruct the path from the tree
    path = []
    current = goal_node
    while current is not None:
        path.append(tree[current])
        current = parent_map[current]
    path.reverse()

    print("Path found:", path)
    return path

def execute_path(p, robot_id, path, joint_indices):
    """
    Execute the planned path by setting joint states step-by-step.
    """
    for joint_angles in path:
        for idx, angle in zip(joint_indices, joint_angles):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle
            )
        p.stepSimulation()
        time.sleep(0.05)

def get_current_arm_orientation():
        """Get the current arm orientation (yaw) from the end-effector."""
        _, _, current_ee_orn = get_robot_ee_pose(p, mobot.robotId)
        current_yaw = current_ee_orn[2]  # Assuming yaw is the third element
        print(f"Current Yaw (rad): {current_yaw}, (deg): {np.degrees(current_yaw)}")
        return current_yaw

def get_target_arm_orientation(target_pos):
    """Calculate the desired arm orientation based on the target position."""
    target_xy = np.array(target_pos[:2])
    current_ee_pos, _, _ = get_robot_ee_pose(p, mobot.robotId)
    current_xy = np.array(current_ee_pos[:2])
    
    # Calculate the desired yaw angle for the arm
    direction = target_xy - current_xy
    desired_yaw = np.arctan2(direction[1], direction[0]) + 0.8854
    print(f"Target XY: {target_xy}, Current XY: {current_xy}, Direction: {direction}")
    print(f"Desired Yaw (rad): {desired_yaw}, (deg): {np.degrees(desired_yaw)}")
    
    return desired_yaw

def orient_base_to_match_arm_orientation(target_pos, speed=0.1):
    """Orient the base to match the desired arm orientation."""
    current_yaw = get_current_arm_orientation()
    desired_yaw = get_target_arm_orientation(target_pos)
    
    # Calculate the angle difference
    angle_diff = np.arctan2(np.sin(desired_yaw - current_yaw), 
                            np.cos(desired_yaw - current_yaw))
    
    max_attempts = 200
    attempts = 0
    
    while attempts < max_attempts:
        if abs(angle_diff) < 0.001:  # Orientation threshold
            base_control(mobot, p, forward=0, turn=0)
            print("Base oriented to match arm orientation")
            return True
        
        # Apply rotation control
        turn = np.sign(angle_diff) * 0.8
        base_control(mobot, p, forward=0, turn=turn)
        
        p.stepSimulation()
        time.sleep(0.01)
        
        # Update current yaw and angle difference
        current_yaw = get_current_arm_orientation()
        angle_diff = np.arctan2(np.sin(desired_yaw - current_yaw), 
                                np.cos(desired_yaw - current_yaw))
        print('angle diff: ', angle_diff)
        attempts += 1
    
    print("Failed to orient base within max attempts")
    return False

# Initialize PyBullet
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setGravity(0, 0, -9.81)

# Initialize robot
mobot, table_id = init_scene(p, mug_random=False)

# Example usage:
joint_indices = [8, 10, 11, 12, 13, 14, 16]  # Define the indices for arm joints
joint_limits = []  # Define joint limits for each joint
for joint_index in joint_indices:
    joint_info = p.getJointInfo(mobot.robotId, joint_index)
    joint_lower_limit = joint_info[8]
    joint_upper_limit = joint_info[9]
    joint_limits.append((joint_lower_limit, joint_upper_limit))
target_position = [0.26, -0.75, 0.9]  # Define the target position for the end-effector
ee_link = 18
orient_base_to_match_arm_orientation(target_position)
path = informed_rrt_ik(
    p, mobot.robotId, table_id, joint_indices, ee_link, target_position, joint_limits
)
if path:
    execute_path(p, mobot.robotId, path, joint_indices)
    motion_planning_test(p, mobot.robotId, target_position)
    time.sleep(2)

    input("Press Enter to exit...")
