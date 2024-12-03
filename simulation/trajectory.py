# takes in a list of waypoints and smoothens the trajectory
import numpy as np
from scipy.interpolate import CubicSpline
def smooth_trajectory_cubic(waypoints, num_points=100):
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))

    cs_x = CubicSpline(t, waypoints[:, 0], bc_type='clamped')
    cs_y = CubicSpline(t, waypoints[:, 1], bc_type='clamped')

    t_fine = np.linspace(0, 1, num_points)

    smooth_path = np.column_stack((cs_x(t_fine), cs_y(t_fine)))
    return smooth_path