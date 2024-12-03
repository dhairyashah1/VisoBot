# VisoBot
Project for CS4278/CS5478 Intelligent Robots: Algorithms and Systems

# **Brief Instructions for `main.py`**

### 1. **Initialization**
- Connects to PyBullet and sets up gravity.
- Loads the robot, environment, and static grid map with obstacle inflation.
- Defines key task flags, navigation, and manipulation coordinates.

### 2. **Tasks**
- Change the task num flag to select which task you want the robot to perform.
- **Task 1 & 2**: Pickup (+ fetch) and place mugs at specific positions.
- **Task 3**: Random navigation and manipulation tasks using generated coordinates.

### 3. **Navigation**
- Generates a global path using A* and smooths it with cubic interpolation (`generate_waypoints`).
- Follows the smoothed waypoints using inverse kinematics (`follow_waypoints`).

### 4. **Manipulation**
- Uses simple IK to control the arm and move to specified positions (`manipulate_simple_ik`).
- Attaches or detaches objects with constraints for grasping (`grasp_attach`/`grasp_detach`).

### 5. **Execution**
- Selects the task and generates the corresponding waypoints.
- Continuously runs the simulation with periodic updates.

**Key Imported Modules:**
- `pybullet` for simulation.
- `scipy` would also be needed to be installed in the conda environment
