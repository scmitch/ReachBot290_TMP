import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from P1_astar import DetOccupancyGrid2D, AStar
from utils import generate_planning_problem

import time
import copy
from solve_subproblem import solve_subproblem
from bresenham import bresenham

width = 10
height = 10
obstacles = [((6,7),(8,8)),((2,2),(4,3)),((2,5),(4,7)),((6,3),(8,5))]
occupancy = DetOccupancyGrid2D(width, height, obstacles)

scale = 4
width_sc = 10*scale
height_sc = 10*scale
obstacles_sc = [((6*scale, 7*scale), (8*scale, 8*scale)),\
             ((2*scale, 2*scale), (4*scale, 3*scale)),\
             ((2*scale, 5*scale), (4*scale, 7*scale)),\
             ((6*scale, 3*scale), (8*scale, 5*scale))]
occupancy_sc = DetOccupancyGrid2D(width_sc, height_sc, obstacles_sc)

x_init = (1, 1) # originally (1,1)
x_goal = (9, 9) # orignally (9,9)

x_init_sc = (1*scale, 1*scale)
x_goal_sc = (9*scale, 9*scale)

start_time = time.clock()

astar = AStar((0, 0), (width_sc, height_sc), x_init_sc, x_goal_sc, occupancy_sc, buffer=2)
astar.solve()

if not astar.solve():
   print("No path found")
else:
    plt.rcParams['figure.figsize'] = [8, 8]
    astar.plot_path()
    astar.plot_tree()

#-------------------------------------------------------------------------------------------------------------
def sight_line_clear(sub_init, path_point, astar):
    '''
    Check if all points intersected by a line-of-sight from sub_init to path_point are clear
    
    Parameter: sight_points, a list of all grid squares that the line of interest passes through
    Return: True, if line-of-sight is clear (i.e. none of these grid squares are occupied by the inflated obstacles)
    '''
    sight_points = list(bresenham(sub_init[0], sub_init[1], path_point[0], path_point[1]))
    for point in sight_points:
        if not astar.occupancy.is_free(point, buffer=2):
            return False
    return True

# Copy the path, b/c we will be continously removing points from "path_to_go" as we find subgoals further and further along
path_to_go = copy.copy(astar.path)

# Truncate .0 decimals to int's for Bresenham algorithm
for idx, point in enumerate(path_to_go):
    path_to_go[idx] = (int(point[0]), int(point[1]))

subgoals = []
while len(path_to_go) > 0:
    # The path is shortened so that the init for each new subproblem is now at the beginning of the path-to-go
    sub_init = path_to_go[0]

    # Test each point along the path-to-go for a clear line-of-sight. When the line-of-sight is broken, store subgoal
    for idx, path_point in enumerate(path_to_go):
        if not sight_line_clear(sub_init, path_point, astar):
            subgoals.append(path_to_go[idx-1]) # Save the last point with a clear line-of-sight as a subgoal
            path_to_go = path_to_go[idx-1:] # Shorten path to now begin at this subgoal, to consider just the remaining path-to-go
            break
        elif path_point == path_to_go[-1]:
            # When we reach the final point in path_to_go, we have reached the final goal and we're done
            subgoals.append(path_to_go[-1])
            path_to_go=[]

print "Subgoals List:"
print subgoals

plt.rcParams['figure.figsize'] = [6, 6]
astar.plot_path()
astar.plot_tree()
for goal in subgoals:
    plt.plot(goal[0], goal[1], color='black', marker='o', fillstyle='full', markersize=7, zorder=10)
    
#-------------------------------------------------------------------------------------------------------------

def state_from_body(body):
    foot1 = (body[0] - 1, body[1] + 1)
    foot2 = (body[0] + 1, body[1] + 1)
    foot3 = (body[0] - 1, body[1] - 1)
    foot4 = (body[0] + 1, body[1] - 1)
    state = [foot1, foot2, foot3, foot4, body]
    return state

'''
# To run simpler problems, you can use a format like the one below, and pass in astar=None
x_map = [0, 10]
y_map = [0, 10]
state_init = state_from_body((1,1))
state_goal = state_from_body((9,9))
goals_list = [state_goal]
goals_remaining = copy.copy(goals_list)
'''

x_map = [0, width_sc]
y_map = [0, height_sc]
operators = ['move_foot1', 'move_foot2', 'move_foot3', 'move_foot4', 'move_body']

goals_list = []
for body_goal in subgoals:
    goals_list.append(state_from_body(body_goal))
goals_remaining = copy.copy(goals_list)
state_init = state_from_body(x_init_sc)

fig_num = 0
for state_goal in goals_list:
    solved, node, fig_num = solve_subproblem(x_map, y_map, state_init, state_goal, operators, fig_num, goals_remaining, astar=astar)
    goals_remaining.pop(0)
    state_init = state_goal
    print("Solved?", "True" if solved else "False")

print("Full Runtime:", time.clock() - start_time)
