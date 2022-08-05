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
        self.outputLayer1 = nn.Linear(hidden_dim,1)
        self.outputLayer2 = None

        if not std:
            self.outputLayer2 = nn.Linear(hidden_dim,1)
        else:
            self.std = std


    def forward(self, input):

        mean = self.layer1(input)
        mean = F.relu(mean)
        mean = self.outputLayer1(mean)

        if not self.std:
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

        return mean, std, action[0]*3, log_prob






















