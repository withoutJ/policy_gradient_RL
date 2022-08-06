import gym 
from policy import GaussianPolicy
import torch
import numpy as np

def simulate(env, actor,render = False):

    episode = []
    score = 0
    previous_observation = env.reset()
    done = False

    while not done:

        action, log_prob = actor.select_action(np.array(previous_observation))
        observation, reward, done, info = env.step(action)
        if render:
            env.render()

        episode.append({"previous_observation":previous_observation,"action":action, "observation":observation, "reward":reward, "log_prob":log_prob})
        previous_observation = observation
        score+=1

    return episode, score

def simulateAC(env,actor,gamma,lmbd,render = False):

    episode = []
    score = 0
    previous_observation = env.reset()
    done = False

    while not done:
        action, log_prob = actor.select_action(np.array(previous_observation))
        observation, reward, done, info = env.step(action)

        if render:
            env.render()

        actor.train(previous_observation,observation,action,gamma,lmbd)
