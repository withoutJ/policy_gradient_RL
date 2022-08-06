import gym
from reinforce import Actor
from simulation import simulate
import torch
import pandas as pd
import numpy as np
import os

def render(params):

    env = gym.make(params["env"])
    dir = f"results/{params['env']}/reinforce/one_layer/std={params['std_file']}"
    actor = Actor(env.observation_space.shape[0],env.action_space,params["hidden_dim1"],params["hidden_dim2"],std = params['std'], model_path=os.path.join(dir,"best_model.pth"))
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

    print(episode_score)


if __name__=="__main__":
    params = {

        "num_episodes": 10,
        "std":0.001,
        "std_file": 0.25,
        "std_change": True,
        "gamma": 0.99,
        "hidden_dim1": 512,
        "hidden_dim2": None,
        "env": "InvertedDoublePendulum-v4",
        "lr": 1e-5
    }
    render(params)

    