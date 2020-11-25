import copy
import sys
import os
import numpy as np
import pdb

class STRIPS:
    def __init__(self, x_range, y_range):
        foot1 = (0,0)
        foot2 = (0,1)
        foot3 = (1,0)
        foot4 = (1,1)
        self.state = [foot1, foot2, foot3, foot4]
        
        foot1 = (x_range - 2, y_range - 2)
        foot2 = (x_range - 2, y_range - 1)
        foot3 = (x_range - 1, y_range - 2)
        foot4 = (x_range - 1, y_range - 1)
        self.goal = [foot1, foot2, foot3, foot4]
    
    def move_foot(self, foot, pos):
        self.state[foot] = pos

    def solved(self):
        foot_matching = [self.state[i] == self.goal[i] for i in range(len(self.state))]
        return all(foot_matching)
    
class Node:
    def __init__(self, strip, parent=None, operator=None, operator_arg=None):
        self.strip = copy.copy(strip)
        self.parent = parent
        self.operator = operator
        self.operator_arg = operator_arg
        self.depth = 0
        if parent:
            self.depth = parent.depth+1
        self.solved = strip.solved() 

    def act(self, operator, pos):
        strip = copy.deepcopy(self.strip)
        if operator == 'move_foot1':
            strip.move_foot(0,pos)
        elif operator == 'move_foot2':
            strip.move_foot(1,pos)
        elif operator == 'move_foot3':
            strip.move_foot(2,pos)
        elif operator == 'move_foot4':
            strip.move_foot(3,pos)
        new_node = Node(strip, parent=self, operator=operator, operator_arg=pos)
        return new_node

def get_plan(leaf_node):
    plan = []
    if not leaf_node.solved:
        return plan 
    while leaf_node.parent:
        plan.append([leaf_node.operator, leaf_node.operator_arg])
        leaf_node = leaf_node.parent
    plan.reverse()
    return plan

class Queue:
    def __init__(self):
        self.list = []

    def queue(self, data):
        self.list.insert(0, data)

    def dequeue(self):
        if len(self.list) > 0:
            return self.list.pop()
        return None

    def is_empty(self):
        return True if len(self.list) == 0 else False
