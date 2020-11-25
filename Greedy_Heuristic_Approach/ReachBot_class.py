import copy
import sys
import os
import numpy as np
import pdb

class ReachBot:
    def __init__(self, init_state, goal_state, num_feet, max_foot_length):
        self.state = init_state
        self.goal = goal_state
        self.num_feet = num_feet
        self.max_foot_length = max_foot_length
        self.is_solved = False
    
    def move_foot(self, foot, pos):
        self.state[foot] = pos
        
    def move_body(self, pos):
        self.state[-1] = pos

    def solved(self):
        states_matching = [self.state[i] == self.goal[i] for i in range(len(self.state))]
        is_solved = all(states_matching)
        self.is_solved = is_solved
        return is_solved