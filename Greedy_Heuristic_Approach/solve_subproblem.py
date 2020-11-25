import matplotlib.pyplot as plt
import numpy as np
import time # time.perf_counter() is for Python3; use time.clock() when running Python2
import copy

from ReachBot_class import ReachBot
from sklearn.neighbors import NearestNeighbors
from subproblem_updates import update_foot, update_body, plot_state


def order_operators(RB, operators):
    '''
    Parameter: operators - unordered list of operators, each being a string, defined as passed into solve_subproblem()
    Parameter: RB - 
    Return: operators_ordered - list of operators strings reordered as nearest-to-farthest by their distance to the goal
                                (the body will always be placed halfway through the front feet & the back feet)
    '''
    
    distances_to_goal = [np.linalg.norm(np.subtract(RB.state[i], RB.goal[-1])) for i in range(RB.num_feet)]
    sort_index = np.argsort(distances_to_goal)

    operators_ordered = []
    for i in range(RB.num_feet):
        idx = sort_index[i]
        operators_ordered.append('move_foot'+str(idx+1))

    operators_ordered.insert(RB.num_feet//2, 'move_body') # Insert body halfway through feet, inserted as (index, element)
    return operators_ordered


def solve_subproblem(x_map, y_map, state_init, state_goal, operators, fig_num, goals_remaining, astar=None):
    '''
    This function finds the best gait for the subproblem based on the initial state and the goal state, and then cycles
    through this order of operators moving the feet and the body in turn until the current state reaches the goal state.
    
    If a full cycle of the gait goes by without resulting in any updates to the state, then the state is declared "stuck". 
    The robot will never update from this position unless the feet are unstuck with some recovery routine, so the subproblem
    ends by return is_solved as False.
    '''
    
    t_start = time.clock() #time.perf_counter()

    # List all possible grid points on the map
    state_positions = [(x,y) for x in range(x_map[0], x_map[1]) for y in range(y_map[0], y_map[1])]
    
    # Create instance of ReachBot
    RB = ReachBot(state_init, state_goal, num_feet=4, max_foot_length=5)
    
    # Plot subgoal states appearing on the plot one-by-one, but only on the first subproblem
    if fig_num==0:
        for g in range(len(goals_remaining)):
            temp_goals_remaining = goals_remaining[:g]
            temp_RB = ReachBot(state_init, state_goal, num_feet=4, max_foot_length=5)
            plot_state(temp_RB, x_map, y_map, fig_num, temp_goals_remaining, astar)
            fig_num += 1

    plot_state(RB, x_map, y_map, fig_num, goals_remaining, astar)
    
    # Find optimal operator gait
    operators_ordered = order_operators(RB, operators)
    
    # Perform one cycle of updates to the feet & body at a time, until the subproblem is solved (current state reaches the goal)
    while not RB.is_solved:
        # Track previous state for stuck positions
        RB_prev = copy.deepcopy(RB)
        
        # Perform one update for each operator in the gait
        for j in range(RB.num_feet+1):
            op = operators_ordered[j]
            if op=='move_body':
                RB = update_body(RB, state_positions, astar)
            else:
                foot = int(op[-1])
                RB = update_foot(RB, state_positions, foot, j, astar)

            plot_state(RB, x_map, y_map, fig_num, goals_remaining, astar)
            fig_num += 1
        
        # Compute solved() to see if configuration after the new gait cycle has reached the sub-goal
        RB.solved()
        
        # Check if we got stuck, i.e. no updates on a full round of the gait actions
        states_matching = [RB.state[i] == RB_prev.state[i] for i in range(len(RB.state))]
        if all(states_matching):
            RB.is_solved = False
            break
            
    t_stop = time.clock() #time.perf_counter()
    print(round(t_stop-t_start, 2), "seconds")
    
    return RB.is_solved, RB, fig_num