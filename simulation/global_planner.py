import numpy as np
import heapq
import random

directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
              (1, 1), (-1, 1), (1, -1), (-1, -1)]

def astar(grid, start, end, path_type="random"):
    """
    A* pathfinding algorithm to find the shortest path on a grid.

    Args:
    - grid: A 2D numpy array representing the grid.
    - start: A tuple (x, y) representing the starting position.
    - end: A tuple (x, y) representing the ending position.

    Returns:
    - path: A list of tuples representing the shortest path from start to end.
    """

    if grid.grid[end[0], end[1]] == 1: # check if point is blocked
        nearest_free_points = find_nearest_free_points(grid, end)

        # check path to each of the nearest free points 
        viable_paths = []
        for free_point in nearest_free_points:
            path = run_astar(grid, start, free_point)
            if path:
                viable_paths.append((path, free_point))
        
        if viable_paths:
            if path_type == "random":
                closest_path, _ = get_random_path(viable_paths)
            else:
                closest_path, closest_point = get_closest_path(viable_paths, end)
            return closest_path
        else:
            print("No path found to any nearest free point!")
    else:
        return run_astar(grid, start, end) # if endpoint is free

# def run_astar(grid, start, end):
#     open_set = []
#     closed_set = set()

#     g_score = {start: 0}

#     f_score = {start: euclidean_distance(start, end)}

#     came_from = {}

#     heapq.heappush(open_set, (f_score[start], start))

#     while open_set:
#         current_f_score, current = heapq.heappop(open_set)

#         if current == end:
#             return reconstruct_path(came_from, current)
        
#         closed_set.add(current)

#         for direction in directions:
#             neighbour = (current[0] + direction[0], current[1] + direction[1])

#             if not is_valid_move(grid, neighbour, closed_set):
#                 continue

#             tentative_g_score = g_score[current] + 1 # cost 1 for each move

#             if neighbour not in g_score or tentative_g_score < g_score[neighbour]:
#                 came_from[neighbour] = current
#                 g_score[neighbour] = tentative_g_score
#                 f_score[neighbour] = tentative_g_score + euclidean_distance(neighbour, end)

#                 if neighbour not in closed_set:
#                     heapq.heappush(open_set, (f_score[neighbour], neighbour))
#     return None # no path found

def run_astar(grid, start, end):
    open_set = []
    closed_set = set()

    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, end)}
    came_from = {}

    heapq.heappush(open_set, (f_score[start], start))

    while open_set:
        current_f_score, current = heapq.heappop(open_set)

        print(f"Processing {current} with f_score: {current_f_score}")  # Debugging line

        if current == end:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for direction in directions:
            neighbour = (current[0] + direction[0], current[1] + direction[1])

            if not is_valid_move(grid, neighbour, closed_set):
                continue

            tentative_g_score = g_score[current] + 1  # cost 1 for each move

            if neighbour not in g_score or tentative_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = tentative_g_score + euclidean_distance(neighbour, end)

                if neighbour not in closed_set:
                    heapq.heappush(open_set, (f_score[neighbour], neighbour))
                    print(f"Adding to open_set: {neighbour} with f_score: {f_score[neighbour]}")  # Debugging line
    return None  # no path found

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_valid_move(grid, position, closed_set): # valid if the cell is within bounds, not in closed set or has an obstacle
    x, y = position
    if 0 <= x < grid.grid_size[0] and 0 <= y < grid.grid_size[1]:
        return grid.grid[x, y] == 0 and position not in closed_set
    return False

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)  # add the start point
    path.reverse()  # reverse to get path from start to end
    return path

def find_nearest_free_points(grid, start, radius=6):
    """
    Finds a list of nearest free points from the given start point.
    The radius controls how far we search around the start point.
    """
    free_points = []
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            x, y = start[0] + dx, start[1] + dy
            if 0 <= x < grid.grid_size[0] and 0 <= y < grid.grid_size[1]:
                if grid.grid[x, y] == 0:  # Free point
                    free_points.append((x, y))
    return free_points

def get_closest_path(valid_paths, end): # to the endpoint
    closest_path = None
    closest_point = None

    min_distance = float('inf')

    for path, point in valid_paths:
        distance = euclidean_distance(point, end)
        if distance < min_distance:
            min_distance = distance
            closest_path = path
            closest_point = point

    return closest_path, closest_point


def get_random_path(valid_paths): # to the endpoint

    random_path = random.choice(valid_paths)
    return random_path[0], random_path[1]