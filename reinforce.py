from policy import GaussianPolicy

class Actor():

    def __init__(self,observation_space,action_space,hidden_dim,std = 0,stdNetwork = False) -> None:

        self.observation_space= observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.std = std
        self.stdNetwork = stdNetwork
        self.policy = GaussianPolicy(observation_space,action_space,hidden_dim,std,stdNetwork)


    def select_action(self, state):
        _,_, action, log_prob = self.policy(state)
        return action, log_prob

    def train(self, episode):
        pass