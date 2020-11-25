import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import LineString


def feet_intersecting(body1, foot1, body2, foot2):
    '''
    This function is called from update_foot().
    Used to check if a new foot position violates this intersection constraint:
    Checks if the legs of foot1 and foot2 are intersecting each other when connected to their respective points on the body.
    
    Returns True/False
    (Uses "shapely.geometry" package)
    '''
    
    line = LineString([body1, foot1])
    other = LineString([body2, foot2])
    return line.intersects(other) # True/False


def body_causing_feet_intersection(body, foot1, foot2, foot3, foot4):
    '''
    This function is called from update_body(). 
    Used to check if moving the body to this new position will cause any two of the legs to intersect each other.
    Note: Not generalized to account for different numbers of feet.
        
    Parameters: Fixed positions for each of the feet; new candidate position for the body
    Returns: True/False
    '''    
    
    # Define the corners of the body in order to connect each foot to its corresponding shoulder joint
    corner_offset = [[0,1], [1,1], [0,0], [1,0]]
    corners = [body + corner_offset[j] for j in range(4)]
    
    # Define the four legs of the robot with line segments (using shapely.geometry package)
    foot1_line = LineString([corners[0], foot1])
    foot2_line = LineString([corners[1], foot2])
    foot3_line = LineString([corners[2], foot3])
    foot4_line = LineString([corners[3], foot4])
    
    # Check for intersection in all foot combinations
    intersect_12 = foot1_line.intersects(foot2_line)
    intersect_13 = foot1_line.intersects(foot3_line)
    intersect_14 = foot1_line.intersects(foot4_line)
    intersect_23 = foot2_line.intersects(foot3_line)
    intersect_24 = foot2_line.intersects(foot4_line)
    intersect_34 = foot3_line.intersects(foot4_line)
    
    return (intersect_12 or intersect_13 or intersect_14 or intersect_23 or intersect_24 or intersect_34)    
    

def body_intersecting(body, foot_idx, foot_pos):
    '''
    This function is called from both update_foot() and update_body().
    Check if the position of a given foot intersects the grid square defining the body.
    
    Strategy: Define line segments that form a plus shape in the middle of the body (named edge1 & edge2).
              Any foot that passes through the body must pass through at least one of these two line segments.
    
    Parameters: Body position, index of foot, position of foot
    Returns: True/False
    '''
    
    # Define the four corners of the grid square representing the body state
    corner_offset = [[0,1], [1,1], [0,0], [1,0]] # Updated when updated plotting function
    corners = [body + corner_offset[j] for j in range(4)]
    
    # Define two edges forming a plus in the middle of the grid square defining the body
    foot_segment = LineString([corners[foot_idx], foot_pos])
    edge1 = LineString([body+[0, 0.5], body+[1, 0.5]])
    edge2 = LineString([body+[0.5, 0], body+[0.5, 1]])
    
    return (foot_segment.intersects(edge1) or foot_segment.intersects(edge2))


def check_tension(RB, idx, candidate_position):
    '''
    This function is called from both update_foot() and update_body().
    Used as an approximation to check if the body is under tension by all the feet.
    Caution: This method does in fact have edge cases where the body is inside the bounding box but not under tension.
    
    Strategy: Create a bounding box defined by the by min/max x positions and min/max y positions of the four feet.
              If the body is within this bounding box, then usually it will be under tension.
    
    Parameters: RB - instance of ReachBot
                idx - index {0,1,2,3,4} in the state representation of the part of the robot (foot or body) that we want to update
                candidate_position - the new position of that part of the robot that we want to move & check for tension constraint
    
    Returns: True if body is "approximately" under tension (False if tension constraint is broken)
    '''
    
    proposed_config = copy.copy(RB.state) # Get current position
    proposed_config[idx] = candidate_position # Update to the proposed configuration
    x_positions = [proposed_config[j][0] for j in range(RB.num_feet)] # x positions of all feet
    y_positions = [proposed_config[j][1] for j in range(RB.num_feet)] # y positions of all feet
    
    # Check if body is within the bounding box
    body_pos = proposed_config[-1]
    return (min(x_positions) < body_pos[0] < max(x_positions)) and (min(y_positions) < body_pos[1] < max(y_positions))

    
def body_reach_space_from_foot(i, RB, state_positions):
    '''
    This function is called from update_body().
    When the foot is at a fixed location, this will return all possible positions within reach for the body to be located
    
    Parameters: RB - instance of ReachBot
                i - index of the current foot {1,2,3,4}, defined as passed into update_foot()
                state_positions - list of all possible grid positions on the map, used to find all states within the max-radius
                
    Returns: list of grid locations in which the body would be within the max-radius from this foot                
    '''
    
    neigh = NearestNeighbors(radius=RB.max_foot_length)
    neigh.fit(state_positions)
    foot_pos = RB.state[i-1]
    body_reach_space_indices = neigh.radius_neighbors([foot_pos], return_distance=False)
    body_reach_space = [state_positions[index] for index in body_reach_space_indices[0]]
    return body_reach_space


def update_foot(RB, state_positions, i, order_idx, astar=None):
    '''
    This function moves Foot(i) to the position as close to the goal as possible, while remaining within all constraints.
    
    Strategy: Find initial reach space within max_foot_length, then remove all illegal configurations that break one of our constraints.
              From the remaining legal positions in the reach space, choose the one closest to the goal.
              
              Note: When removing illegal states from the list, we cannot use "for state in reach_space..." due to indexing issues
                    that arise when modifying the length of the reach_space list once already in the loop
                    (Also, without updating the current index you would skip over any state after one gets removed, and not check 
                    it for constraints)
    
    Parameters: RB - instance of ReachBot
                i - index of the current foot {1,2,3,4}, as pulled from the string name of the operator
                state_positions - list of all possible grid positions on the map, used to find all states within the max-radius
                order_idx - the index of this foot in the operators_ordered list, i.e. its turn in the movement cycle
                            used to determine if the foot is a "back foot", to make sure that it does not out-step the body
                astar - optional argument used to check an occupany grid, when we are working with obstacles
    
    Returns: ReachBot object with Foot(i) updated to its new position
    '''
    
    # Define the position of the body and all of its corners, used for connecting each foot to the body
    body_pos = RB.state[-1]
    corner_offset = [[0,1], [1,1], [0,0], [1,0]]
    corners = [np.array(body_pos) + corner_offset[c] for c in range(RB.num_feet)]
    
    # Find initial reach space - list of grid squares within the radius of max_foot_length
    neigh = NearestNeighbors(radius=RB.max_foot_length)
    neigh.fit(state_positions)
    reach_space_indices = neigh.radius_neighbors([body_pos], return_distance=False)
    reach_space = [state_positions[index] for index in reach_space_indices[0]]
    
    # Constraint: Check that the current foot is not occupying the same grid square as another foot nor the body
    occupied_coords = copy.copy(RB.state)
    del occupied_coords[i-1] # Remove current foot from illegal positions (staying put is an option)
    m = len(reach_space)
    k = 0
    while k < m:    
        candidate_state = reach_space[k]        
        if candidate_state in occupied_coords:                
            reach_space.remove(candidate_state)
            k -= 1 # Decrement total length of reach_space list
            m -= 1 # If we remove a state, all the next states will shift backward, so we need to check the current index again
        k += 1
    
    # Constraint: Check that the current foot is not intersecting other feet
    other_feet = np.delete(np.arange(RB.num_feet), i-1) # All feet except the current foot of interest
    
    for j in other_feet:
        # Set up line connecting the center of another foot to its corresponding body corner (as shown on plot)
        body1 = corners[j]
        foot1 = np.array(RB.state[j]) + [0.5, 0.5]
        
        m = len(reach_space)
        k = 0
        while k < m:
            # Set up line connecting the center of candidate foot position to the current corresponding body corner
            candidate_state = reach_space[k]
            body2 = corners[i-1]
            foot2 = np.array(candidate_state) + [0.5, 0.5]
            
            if feet_intersecting(body1, foot1, body2, foot2):
                reach_space.remove(candidate_state)
                m -= 1
                k -= 1
            k += 1
            
    # Constraint: Check that foot does not intersect the body, using the center of the foot's current grid square
    m = len(reach_space)
    k = 0
    while k < m:
        candidate_state = reach_space[k]
        foot_pos = np.array(candidate_state) + [0.5, 0.5]
        if body_intersecting(np.array(body_pos), i-1, foot_pos):
            reach_space.remove(candidate_state)
            k -= 1
            m -= 1
        k += 1

    # Constraint: Check that body is under proper tension
    # Method 1: Check that body is positioned inside min/max box defined by feet positions
    m = len(reach_space)
    k = 0
    while k < m:
        candidate_state = reach_space[k]        
        if not check_tension(RB, i-1, candidate_state):
            reach_space.remove(candidate_state)
            k -= 1
            m -= 1
        k += 1
        
    # Constraint: Check that body is under proper tension
    # Method 2: Make sure the back 2 feet are not outstepping (nearer to the goal than) the body or front 2 feet
    body_to_goal = np.linalg.norm(np.subtract(RB.state[-1], RB.goal[-1]))
    if (order_idx==3) or (order_idx==4): # For 4 feet, order index is {0,1,2,3,4}
        m = len(reach_space)
        k = 0
        while k < m:
            candidate_state = reach_space[k]
            foot_pos = np.array(candidate_state) + [0.5, 0.5]
            dist_to_goal = np.linalg.norm(np.subtract(foot_pos, RB.goal[-1]))
            if dist_to_goal <= body_to_goal:
                reach_space.remove(candidate_state)
                k -= 1
                m -= 1
            k += 1    
    
    # Constraint: Check that foot is not placed inside an obstacle, using the center of the foot's current grid square
    if astar:
        m = len(reach_space)
        k = 0
        while k < m:
            candidate_state = reach_space[k]
            foot_pos = np.array(candidate_state) + [0.5, 0.5]
            if not astar.occupancy.is_free(foot_pos, buffer=0):
                reach_space.remove(candidate_state)
                k -= 1
                m -= 1
            k += 1
    
    # Check if there are any legal states available; if not, don't move
    if len(reach_space) == 0:
        return RB
    
    # Otherwise, move the foot as close as possible to the goal, from the remaining legal positions in its reach space
    else:
        foot_goal = RB.goal[i-1]
        distances = [np.linalg.norm(np.subtract(foot_goal,state), ord=1) for state in reach_space]
        p = np.argmin(distances)
        closest_position = reach_space[p]
        RB.move_foot(i-1, closest_position)
        return RB




def update_body(RB, state_positions, astar=None):
    '''
    This function moves the robot's body to the position as close to the goal as possible, while remaining within all constraints.
    
    Strategy: Find initial reach space within max_foot_length, then remove all illegal configurations that break one of our constraints.
              From the remaining legal positions in the reach space, choose the one closest to the goal
              
    Parameters: state_positions - list of all possible grid positions on the map, used to find all states within the max-radius
                astar - optional argument used to check an occupany grid, when we are working with obstacles
    
    Returns: ReachBot object with the body updated to its new position
    '''
    
    # Find the list of grid squares that are within the max_foot_length from each foot
    foot1_reach_space = body_reach_space_from_foot(1, RB, state_positions)
    foot2_reach_space = body_reach_space_from_foot(2, RB, state_positions)
    foot3_reach_space = body_reach_space_from_foot(3, RB, state_positions)
    foot4_reach_space = body_reach_space_from_foot(4, RB, state_positions)
    
    # Create initial reach space for body from the intersection of the four feet reach spaces
    body_reach_space = []
    for state in state_positions:
        if (state in foot1_reach_space) and (state in foot2_reach_space) and (state in foot3_reach_space) and (state in foot4_reach_space):
            body_reach_space.append(state)
    
    
    # Constraint: Check that body is not occupying the same grid square as a foot
    occupied_coords = copy.copy(RB.state)
    del occupied_coords[-1] # Remove current body position from illegal positions (staying put is an option)
    m = len(body_reach_space)
    k = 0
    while k < m:
        candidate_body_state = body_reach_space[k]        
        if candidate_body_state in occupied_coords:                
            body_reach_space.remove(candidate_body_state)
            k -= 1
            m -= 1
        k += 1
    
    
    # Constraint: Check that body does not cause any feet to intersect each other, using the center of the foot's current grid square
    m = len(body_reach_space)
    k = 0
    foot1 = np.array(RB.state[0]) + [0.5, 0.5]
    foot2 = np.array(RB.state[1]) + [0.5, 0.5]
    foot3 = np.array(RB.state[2]) + [0.5, 0.5]
    foot4 = np.array(RB.state[3]) + [0.5, 0.5]
    while k < m:
        candidate_body_state = body_reach_space[k]        
        if body_causing_feet_intersection(np.array(candidate_body_state), foot1, foot2, foot3, foot4):
            body_reach_space.remove(candidate_body_state)
            k -= 1
            m -= 1
        k += 1
    
    
    # Constraint: Check that body does not cause any foot to intersect it, using the center of the foot's current grid square
    # Test feet {1,2,3,4}
    for i in range(1, RB.num_feet+1):
        foot_pos = np.array(RB.state[i-1]) + [0.5, 0.5]
        
        k = 0
        m = len(body_reach_space)
        while k < m:
            candidate_body_state = body_reach_space[k]
            if body_intersecting(np.array(candidate_body_state), i-1, foot_pos):
                # Remove unreachable state that would otherwise cross another leg
                body_reach_space.remove(candidate_body_state)
                k -= 1
                m -= 1
            k += 1
    
    
    # Constraint: Check that body is under proper tension 
    # Method 1: Check that body is positioned inside min/max box defined by feet positions
    m = len(body_reach_space)
    k = 0
    while k < m:
        candidate_state = body_reach_space[k]        
        if not check_tension(RB, -1, candidate_state):
            body_reach_space.remove(candidate_state)
            k -= 1
            m -= 1
        k += 1
    
    # Constraint: Check that body is not placed inside an obstacle, using the center of the body's current grid square
    if astar:
        m = len(body_reach_space)
        k = 0
        while k < m:
            candidate_state = body_reach_space[k]
            body_pos = np.array(candidate_state) + [0.5, 0.5]
            if not astar.occupancy.is_free(body_pos, buffer=0):
                body_reach_space.remove(candidate_state)
                k -= 1
                m -= 1
            k += 1
    
    
    # Check if there are any legal states available; if not, don't move
    if len(body_reach_space) == 0:
        return leaf_node
    
    # Otherwise, move the body as close as possible to the goal, from the remaining legal positions in its reach space
    else:
        body_goal = RB.goal[-1]
        distances = [np.linalg.norm(np.subtract(body_goal,state), ord=1) for state in body_reach_space]
        p = np.argmin(distances)
        closest_position = body_reach_space[p]
        RB.move_body(closest_position)
        return RB


def plot_state(RB, x_map, y_map, fig_num, goals_remaining, astar=None):
    '''
    This function plots the current state of ReachBot as defined by RB, displays it, and saves it to a file.
    
    Each foot is plotted by coloring in the grid square between its state position (x,y) and the point (x+1, y+1).
    Legs of the robot are plotted as line segements connecting the center of the foot's grid square to the corresponding corner
        of the robot's body.
    For aesthetics I also plotted "swivel points" on either end fo the leg segment. These are commented out for big grids since they
        would otherwise be too big compared to the feet themselves. But uncomment for smaller grids (e.g. 10x10)
    
    Parameters: x_map, y_map - each is a tuple [low,high] defining the boundaries of the grid world
                fig_num - the figure number is appended to the saved file name so that the animation script 
                          will read them in the right order
                goals_remaining - a list of the remaining subgoals, plotted in a light gray to visualize the destination(s)
                astar - optional; if included, ReachBot's state will plot on top of the A* path & occupancy grid
    
    Returns: nothing (but saves plot to a PNG)
    '''

    fig, ax = plt.subplots()
    
    if astar:
        astar.plot_path(ax=ax, fig_num=fig_num)
    
    width = 1
    height = 1
    
    colors = ["blue", "red", "green", "yellow"]
    corner_offset = [[0,1], [1,1], [0,0], [1,0]]
    
    # Draw gray square for the body
    body_pos = RB.state[-1]
    xy = (body_pos[0], body_pos[1])
    ax.add_patch(
        plt.Rectangle(xy, height, width, color = 'black', alpha = 0.5)
        )
    
    # Draw colored square for each foot
    for i in range(4):
        # Shade in foot location
        foot_pos = RB.state[i]
        xy = (foot_pos[0], foot_pos[1])
        ax.add_patch(
            plt.Rectangle(xy, height, width, color = colors[i])
            )
        
        # Draw line connecting center of the foot square to the corresponding body corner
        offset = corner_offset[i]
        plt.plot([body_pos[0]+offset[0], foot_pos[0]+0.5], [body_pos[1]+offset[1], foot_pos[1]+0.5], color='black', linewidth=1)
        
        # Put swivel dots on feet & body corners
        #plt.plot(foot_pos[0]+0.5, foot_pos[1]+0.5, color='black', marker='o', fillstyle='full', markersize=6)
        #plt.plot(body_pos[0]+offset[0], body_pos[1]+offset[1], color='black', marker='o', fillstyle='full', markersize=6)
    
    # Draw all remaining goal positions
    for goal_state in goals_remaining:
        # Draw body goal
        body_pos = goal_state[-1]        
        xy = (body_pos[0], body_pos[1])
        ax.add_patch(
                plt.Rectangle(xy, height, width, color = 'black', alpha = 0.08)
            )

        # Draw each foot goal
        for i in range(4):
            # Shade in foot location
            foot_pos = goal_state[i]
            xy = (foot_pos[0], foot_pos[1])
            ax.add_patch(
                plt.Rectangle(xy, height, width, color='black', alpha=0.08, zorder=10)
                )

            # Draw line connecting foot to body corner
            offset = corner_offset[i]
            plt.plot([body_pos[0]+offset[0], foot_pos[0]+0.5], 
                     [body_pos[1]+offset[1], foot_pos[1]+0.5], 
                     color='black', alpha=0.2, linewidth=1)

            # Put swivel dots on feet & body corners
            #plt.plot(foot_pos[0]+0.5, foot_pos[1]+0.5, color='black', marker='o', alpha=0.2, fillstyle='full', markersize=6)
            #plt.plot(body_pos[0]+offset[0], body_pos[1]+offset[1], color='black', alpha=0.2, marker='o', fillstyle='full', markersize=6)
        
    
    # Swap x & y axes in order to line up with how the grid is indexed with matrix coordinates
    ax.set_xlim(x_map[0], x_map[1])
    ax.set_ylim(y_map[0], y_map[1]) # makes x start counting from the top
    ax.set_xticks(np.arange(x_map[0], x_map[1], step=2))
    ax.set_yticks(np.arange(y_map[0], y_map[1], step=2))
    
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('ReachBot Task Planning Solution')
    
    plt.show()
    plt.savefig("Snapshots/ReachBot_Astar_Plot"+str(fig_num)+".png")