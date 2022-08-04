import gym 
from policy import GaussianPolicy
import torch

def simulate(env, actor,render = False):

    episode = []
    score = 0
    previous_observation = env.reset()
    done = False

    while not done:
        action, log_prob = actor.select_action(torch.tensor(previous_observation,dtype = torch.float))
        observation, reward, done, info = env.step([action])
        if render:
            env.render()
        episode.append({"previous_observation":previous_observation,"action":action, "observation":observation, "reward":reward, "log_prob":log_prob})
        previous_observation = observation
        score+=reward
    return episode, score