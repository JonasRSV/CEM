import tensorflow as tf
import numpy as np

class cross_entropy():

    def __init__(self, state_space, action_space):
        self.state_space  = state_space
        self.action_space = action_space


    def __call__(self, actor, state):
        return 0
    
    def __len__(self):
        return 2


