import gym
from reinforce import Actor
from simulation import simulate
import torch
import pandas as pd
import numpy as np
import os

def render(params):

    env = gym.make(params["env"])
    dir = f"results/{params['env']}/reinforce/std={params['std']}_hiddenDim={params['hidden_dim']}"
    actor = Actor(env.observation_space.shape[0],env.action_space,params["hidden_dim"],std = params["std"], model_path=os.path.join(dir,"best_model.pth"))
    mean = 0
    std = 0
    total_score = 0
    rewards = []

    for ep in range(params["num_episodes"]):
        episode, episode_score = simulate(env, actor, render=False)
        total_score+=episode_score
        rewards.append(episode_score)


    mean = total_score/params["num_episodes"]
    rewards = np.array(rewards)
    std = np.sqrt(np.sum(np.square(rewards - mean))/(params["num_episodes"]-1))
    print(f"Mean: {mean}, Std: {std}")

    episode, episode_score= simulate(env,actor,render=True)


if __name__=="__main__":
    params = {"num_episodes": 1000, "std": 0.3, "gamma": 0.99, "hidden_dim": 512,"env":"InvertedDoublePendulum-v4"}
    render(params)

    