from policy import GaussianPolicy
from policy import ValueNetwork
import torch
from torch.optim import Adam
class ActorCritic():

    def __init__(self, observation_space, action_space, hidden_dim, std=None, gamma = 0.99, lr_critic=5e-5,lr_actor = 5e-4, actor_path=None,
                 critic_path=None) -> None:

        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.I = 1
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor

        self.std = std

        self.actor = GaussianPolicy(observation_space, action_space, hidden_dim, std)
        self.critic = ValueNetwork(observation_space, 1, hidden_dim)

        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))

        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        _, _, action, log_prob = self.actor(state)
        state = state.detach()
        return action, log_prob

    def set_iF(self):
        self.I = 1

    def train(self, previous_observation,observation,action,log_prob,reward,done):

        previous_observation = torch.from_numpy(previous_observation).float().unsqueeze(0)
        vs_curr = self.critic(previous_observation)

        if not done:
            observation = torch.from_numpy(observation).float().unsqueeze(0)
            vs_prime = self.critic(observation)
        else:
            vs_prime= torch.tensor([0]).float().unsqueeze(0)


        critic_loss = torch.nn.functional.mse_loss(reward + self.gamma*vs_prime, vs_curr)
        delta = reward + self.gamma*vs_prime - vs_curr
        actor_loss = -log_prob*delta.detach()
        #actor_loss = -log_prob * delta
        actor_loss *= self.I


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.I *= self.gamma