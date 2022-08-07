import numpy
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class GaussianPolicy(nn.Module):

    def __init__(self,input_shape,output_shape,hidden_dim,std = None):

        super().__init__()

        self.layer1 = nn.Linear(input_shape,hidden_dim)
        self.outputLayer1 = nn.Linear(hidden_dim,output_shape)
        self.outputLayer2 = None


        if std is None:
            self.outputLayer2 = nn.Linear(hidden_dim,output_shape)
        else:
            self.std = std


    def forward(self, input):

        mean = self.layer1(input)
        mean = F.relu(mean)
        mean = self.outputLayer1(mean)

        if self.std is None:
            std = self.layer1(input)
            std = self.outputLayer2(std)
        else:
            std = self.std

        distribution = Normal(mean, std)
        action = distribution.sample()

        log_prob = distribution.log_prob(action)
        log_prob = log_prob.sum()

        action = torch.tanh(action)
        action = action.numpy()

        std = torch.tensor([std])
        std = torch.clamp(std, min=0, max=1)

        return mean, std, action.squeeze(), log_prob
    
    def set_std(self, std):
        self.std=std

class ValueNetwork(nn.Module):

    def __init__(self,input_shape,output_shape, hidden_dim):

        super().__init__()

        self.layer = nn.Linear(input_shape, hidden_dim)
        self.output = nn.Linear(hidden_dim,1)

    def forward(self, input):

        v = self.layer(input)
        v = F.relu(v)
        v = self.output(v)
        return v




class GaussianPolicy2(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_dim1,hidden_dim2, std=None):

        super().__init__()

        self.layer1 = nn.Linear(input_shape, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.outputLayer1 = nn.Linear(hidden_dim2, output_shape)
        self.outputLayer2 = None


        if std is None:

            self.outputLayer2 = nn.Linear(hidden_dim2, 1)
        else:
            self.std = std

    def forward(self, input):

        mean = self.layer1(input)
        mean = F.relu(mean)
        mean = self.layer2(mean)
        mean - F.relu(mean)
        mean = self.outputLayer1(mean)

        if self.std is None:
            std = self.layer1(input)
            std = self.outputLayer2(std)
        else:
            std = self.std

        distribution = Normal(mean, std)
        action = distribution.sample()

        log_prob = distribution.log_prob(action)
        log_prob = log_prob.sum()

        action = torch.tanh(action)
        action = action.numpy()

        std = torch.tensor([std])
        std = torch.clamp(std, min=0, max=1)

        return mean, std, action[0], log_prob

    def set_std(self, std):
        self.std = std
