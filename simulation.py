import gym 
from policy import GaussianPolicy
import torch
import numpy as np

def simulate(env, agent,render = False, train = False):

    episode = []
    score = 0
    previous_observation = env.reset()
    done = False
    agent.set_iF()

    while not done:

        action, log_prob = agent.select_action(np.array(previous_observation))
        observation, reward, done, info = env.step(action)
        if render:
            env.render()

        if train:
            agent.train(previous_observation, observation, action, log_prob, reward, done)
        
        episode.append({"previous_observation":previous_observation,"action":action, "observation":observation, "reward":reward, "log_prob":log_prob})
        agent.buffer.store(previous_observation, action, observation, reward,log_prob, done)
        previous_observation = observation
        score += reward

    return episode, score

