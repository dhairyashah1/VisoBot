import numpy as np
import pybullet as p
import time
from stretch import init_scene
from utils.tools import get_robot_ee_pose, get_robot_base_pose, motion_planning_test
from stretch import base_control, arm_control

class EndEffectorController:
    def __init__(self, robot, robot_id, end_effector_index, joint_indices, joint_limits):
        self.robot = robot
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.joint_indices = joint_indices
        self.joint_limits = joint_limits

    def move_to_position(self, target_position, max_steps=500, max_velocity=0.1):
        """
        Moves the end effector to the specified target position using prismatic joints.
        
        :param target_position: The desired position of the end effector (x, y, z)
        :param max_steps: Maximum steps to perform the movement
        :param max_velocity: Maximum velocity for the joints
        """
        current_position = np.array(p.getLinkState(self.robot_id, self.end_effector_index)[0])
        target_position = np.array(target_position)

        self.orient_base_to_match_arm_orientation(target_position)

        for step in range(max_steps):
            print(f"Step {step + 1}/{max_steps}: Moving to target position {target_position}, given current position {current_position}")
            direction = target_position - current_position
            distance = np.linalg.norm(direction)
            if distance < 0.01:
                print("Target reached.")
                arm_control(self.robot, p, up=0, stretch=0, roll=0, yaw=0)
                base_control(self.robot, p, forward=0, turn=0)
                break

            direction /= distance  # Normalize the direction
            step_size = min(max_velocity, distance)
            print(f"Moving in direction {direction} with step size {step_size}")
            next_position = current_position + direction * step_size
            print(f"Next position: {next_position}")

            # Check for collisions and adjust the next position if necessary
            if self.check_collision(next_position):
                print("Collision detected. Adjusting position.")
                next_position = self.avoid_obstacle(current_position, next_position)

            # Use inverse kinematics to calculate the joint angles for the next position
            joint_angles = p.calculateInverseKinematics(
                self.robot_id, 
                self.end_effector_index, 
                next_position
            )

            joint_angle_indices = [2, 5, 6, 7, 8, 9, 10, 11]

            # Move each joint to the calculated angle
            for i in range(len(self.joint_indices)):
                joint_index = self.joint_indices[i]
                joint_angle_index = joint_angle_indices[i]
                target_angle = joint_angles[joint_angle_index]

                # Ensure the target angle is within the joint limits
                lower_limit, upper_limit = self.joint_limits[joint_index]
                if target_angle < lower_limit:
                    target_angle = lower_limit
                elif target_angle > upper_limit:
                    target_angle = upper_limit

                # Set higher force and velocity for the base turn joint
                force = 1000 if joint_index == 0 else 100 
                velocity = max_velocity * 2 if joint_index == 0 else max_velocity

                # Set joint motor control with velocity control
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=100,
                    maxVelocity=max_velocity
                )

            # Step the simulation to move the joints
            p.stepSimulation()

            # Update the current position
            current_position = np.array(p.getLinkState(self.robot_id, self.end_effector_index)[0])

    def check_collision(self, position):
        """
        Checks for collisions at the given position.
        
        :param position: The position to check for collisions (x, y, z)
        :return: True if a collision is detected, False otherwise
        """
        # Temporarily move the end effector to the position to check for collisions
        temp_joint_angles = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, position)
        joint_angle_indices = [2, 5, 6, 7, 8, 9, 10, 11]
        for i in range(len(self.joint_indices)):
            p.resetJointState(self.robot_id, self.joint_indices[i], temp_joint_angles[joint_angle_indices[i]])
        p.stepSimulation()
        base_control(self.robot, p, forward=0, turn=0)

        # Check for collisions
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=i)
            if len(contact_points) > 0:
                return True
        return False

    def avoid_obstacle(self, current_position, next_position):
        """
        Adjusts the next position to avoid obstacles.
        
        :param current_position: The current position of the end effector (x, y, z)
        :param next_position: The desired next position of the end effector (x, y, z)
        :return: The adjusted next position to avoid obstacles (x, y, z)
        """
        # Simple obstacle avoidance by moving in a perpendicular direction
        direction = next_position - current_position
        perpendicular_direction = np.array([-direction[0], direction[1], direction[2]])
        print(f"Perpendicular direction: {perpendicular_direction}")
        perpendicular_direction /= np.linalg.norm(perpendicular_direction)
        adjusted_position = current_position + perpendicular_direction * 0.1

        # Check if the adjusted position is collision-free
        if not self.check_collision(adjusted_position):
            return adjusted_position

        # If the adjusted position is still in collision, try the opposite direction
        adjusted_position = current_position - perpendicular_direction * 0.1
        if not self.check_collision(adjusted_position):
            return adjusted_position

        print("Unable to avoid obstacles. Returning the original next position.")
        # If both directions are in collision, return the original next position
        return next_position
    
    def get_current_arm_orientation(self):
        """Get the current arm orientation (yaw) from the end-effector."""
        _, _, current_ee_orn = get_robot_ee_pose(p, self.robot.robotId)
        current_yaw = current_ee_orn[2]  # Assuming yaw is the third element
        print(f"Current Yaw (rad): {current_yaw}, (deg): {np.degrees(current_yaw)}")
        return current_yaw

    def get_target_arm_orientation(self, target_pos):
        """Calculate the desired arm orientation based on the target position."""
        target_xy = np.array(target_pos[:2])
        current_ee_pos, _, _ = get_robot_ee_pose(p, self.robot.robotId)
        current_xy = np.array(current_ee_pos[:2])
        
        # Calculate the desired yaw angle for the arm
        direction = target_xy - current_xy
        desired_yaw = np.arctan2(direction[1], direction[0])
        print(f"Target XY: {target_xy}, Current XY: {current_xy}, Direction: {direction}")
        print(f"Desired Yaw (rad): {desired_yaw}, (deg): {np.degrees(desired_yaw)}")
        
        return desired_yaw

    def orient_base_to_match_arm_orientation(self, target_pos, speed=0.1):
        """Orient the base to match the desired arm orientation."""
        current_yaw = self.get_current_arm_orientation()
        desired_yaw = self.get_target_arm_orientation(target_pos)
        
        # Calculate the angle difference
        angle_diff = np.arctan2(np.sin(desired_yaw - current_yaw), 
                                np.cos(desired_yaw - current_yaw))
        
        max_attempts = 200
        attempts = 0
        
        while attempts < max_attempts:
            if abs(angle_diff) < 0.005:  # Orientation threshold
                base_control(self.robot, p, forward=0, turn=0)
                print("Base oriented to match arm orientation")
                return True
            
            # Apply rotation control
            turn = np.sign(angle_diff) * 0.8
            base_control(self.robot, p, forward=0, turn=turn)
            
            p.stepSimulation()
            time.sleep(0.01)
            
            # Update current yaw and angle difference
            current_yaw = self.get_current_arm_orientation()
            angle_diff = np.arctan2(np.sin(desired_yaw - current_yaw), 
                                    np.cos(desired_yaw - current_yaw))
            print('angle diff: ', angle_diff)
            attempts += 1
        
        print("Failed to orient base within max attempts")
        base_control(self.robot, p, forward=0, turn=0)
        return False

# Example usage:
def main():
    # Initialize PyBullet and load the robot
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    mobot, table_id = init_scene(p, mug_random=False)
    
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
        print(f"Joint Index: {joint_index}, Joint Name: {joint_name}, Joint Type: {joint_type}, "
            f"Lower Limit: {joint_lower_limit}, Upper Limit: {joint_upper_limit}")
        
        joint_limits[joint_index] = (joint_lower_limit, joint_upper_limit)


    # Create the EndEffectorController
    end_effector_index = 18  # Replace with the actual end effector index
    controller = EndEffectorController(mobot, mobot.robotId, end_effector_index, joint_indices, joint_limits)

    # Move the end effector to the target position
    target_position = [0.27, -0.71, 0.92]
    # target_position = [-1, -3.8, 0.92]
    time.sleep(10)
    # controller.orient_base_to_match_arm_orientation(target_position)
    controller.move_to_position(target_position, max_velocity=0.05)
    arm_control(mobot, p, up=0, stretch=0, roll=0, yaw=0)
    base_control(mobot, p, forward=0, turn=0)
    motion_planning_test(p, mobot.robotId, target_position)

    # Keep the simulation running for debugging
    while True:
        p.stepSimulation()
        time.sleep(1 / 240.0)

if __name__ == "__main__":
    main()