"""
    This script contains model definitions used by the project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class A2C_policy(nn.Module):
    '''
    Policy neural network
    '''
    def __init__(self, input_shape, n_actions):
        super(A2C_policy, self).__init__() 
        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU())

        self.mean_l = nn.Linear(32, n_actions[0])
        self.mean_l.weight.data.mul_(0.1)

        self.var_l = nn.Linear(32, n_actions[0])
        self.var_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(n_actions[0]))

    def forward(self, x):
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))


class A2C_value(nn.Module):
    '''
    Actor neural network
    '''
    def __init__(self, input_shape):
        super(A2C_value, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.lp(x.float())
