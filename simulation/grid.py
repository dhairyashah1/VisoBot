# Idea
'''
1. Create a static grid which uses the object ids to create inflation and keepout zones in the grid
    a. Information we have -- object_ids and can use getAABB to find the locations/sizes of these objects
    b. Start point of robot
    c. Global origin at the center of a 51 x 51 grid (i.e. global origin at point 25, 25), points in grid are spaced 20cm apart so each cell is 20cm by 20cm
    d. Have a function to take a point x,y and determine which cell index it falls into
2. Use the grid to determine waypoints to navigate to using A*
3. Smoothen the trajectory
4. Apply inverse kinematics to go to those waypoints.
'''

import numpy as np
import random
import pybullet as p
from utils.tools import *

class StaticGrid:
    def __init__(self, grid_size=(103, 103), cell_size=0.1, inflation_radius=0.1):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.inflation_radius = inflation_radius
        self.grid = np.zeros(grid_size, dtype=int)  # 0 for free, 1 for inflated/keepout
        self.origin = (grid_size[0] // 2, grid_size[1] // 2)  # Center of the grid

    def world_to_grid(self, x, y):
        grid_x = int(x / self.cell_size) + self.origin[0]
        grid_y = int(y / self.cell_size) + self.origin[1]

        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        world_x = (grid_x - self.origin[0]) * self.cell_size
        world_y = (grid_y - self.origin[1]) * self.cell_size

        return world_x, world_y

    def mark_obstacle(self, object_id):
        AABB = getAABB(object_id=object_id)
        AABB_min, AABB_max = AABB[0], AABB[1]

        # Convert AABB corners to grid indices
        grid_min = self.world_to_grid(AABB_min[0], AABB_min[1])
        grid_max = self.world_to_grid(AABB_max[0], AABB_max[1])

        #inflation cells
        icells = int(self.inflation_radius / self.cell_size)

        # Mark the grid cells within AABB as occupied + inflation zones of 20cm
        for i in range(grid_min[0] - icells , grid_max[0] + 1 + icells):
            for j in range(grid_min[1] - icells, grid_max[1] + 1 + icells):
                if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                    self.grid[i, j] = 1 # mark as occupied

    def mark_obstacle_without_inflation(self, object_id):
        AABB = getAABB(object_id=object_id)
        AABB_min, AABB_max = AABB[0], AABB[1]

        # Convert AABB corners to grid indices
        grid_min = self.world_to_grid(AABB_min[0], AABB_min[1])
        grid_max = self.world_to_grid(AABB_max[0], AABB_max[1])

        # Mark the grid cells within AABB as occupied
        for i in range(grid_min[0], grid_max[0] + 1):
            for j in range(grid_min[1], grid_max[1] + 1):
                if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                    self.grid[i, j] = 1 # mark as occupied

    def update_grid_with_objects(self, object_ids):
        for object_id in object_ids:
            self.mark_obstacle(object_id)

    def mark_custom_cells(self, cell_tuples, value):
        for (x, y) in cell_tuples:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.grid[x, y] = value  # Mark as custom value
    
    def mark_custom_cell(self, cell_tuple, value):
        x, y = cell_tuple
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.grid[x, y] = value  # Mark as custom value

    def print_grid(self):
        for row in self.grid:
            print(' '.join(map(str, row)))