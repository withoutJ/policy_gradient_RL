from policy import GaussianPolicy
import torch
from torch.optim import Adam

class Actor():

    def __init__(self,observation_space,action_space,hidden_dim,std = None, lr=0.0001) -> None:

        self.observation_space= observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        if std:
            self.std = std
        self.policy = GaussianPolicy(observation_space,action_space,hidden_dim,std)
        self.lr = lr
        self.optimizer = Adam(self.policy.parameters(),lr=lr)

    def select_action(self, state):
        _,_, action, log_prob = self.policy(state)
        return action, log_prob

    def train(self, episode, gamma):
        losses = []

        for k in range(len(episode)):
            discount = 1
            G = 0
            for step in episode[k:]:
                G+=step["reward"]*discount
                discount*=gamma
            losses.append(-G*step["log_prob"])
            
        loss = torch.stack(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        
            
            