from policy import GaussianPolicy
import torch
from torch.optim import Adam

class Actor():

    def __init__(self,observation_space,action_space,hidden_dim,std = 0,stdNetwork = False, lr=0.001) -> None:

        self.observation_space= observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.std = std
        self.stdNetwork = stdNetwork
        self.policy = GaussianPolicy(observation_space,action_space,hidden_dim,std,stdNetwork)
        self.lr = lr
        self.optimizer = Adam(self.policy.parameters(),lr=lr)

    def select_action(self, state):
        _,_, action, log_prob = self.policy(state)
        return action, log_prob

    def train(self, episode, gamma):
        losses = []

        for k in len(episode):
            discount = 1
            G = 0
            for step in episode[k:]:
                G+=step["reward"]*discount
                discount*=gamma
            losses.append(-G*log_prob)
            
        loss = torch.stack(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        
            
            