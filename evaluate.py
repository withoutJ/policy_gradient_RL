import gym
from reinforce import Actor
from actorCritic import ActorCritic
from simulation import simulate
import torch
import pandas as pd
import numpy as np
import os

def render(params):

    env = gym.make(params["env"])
    dir = f"results/{params['env']}/{params['method']}/{params['exp']}"
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],params["hidden_dim1"],std = params['std'], actor_path=os.path.join(dir,"actor_model.pth"), critic_path = os.path.join(dir,"critic_model.pth"))
    mean = 0
    std = 0
    total_score = 0
    rewards = []

    for ep in range(params["num_episodes"]):
        episode, episode_score = simulate(env, agent, render=False, train = False)
        total_score+=episode_score
        rewards.append(episode_score)


    mean = total_score/params["num_episodes"]
    rewards = np.array(rewards)
    std = np.sqrt(np.sum(np.square(rewards - mean))/(params["num_episodes"]-1))
    print(f"Mean: {mean}, Std: {std}")

    episode, episode_score= simulate(env,agent,render=True)

    print(episode_score)


if __name__=="__main__":
    torch.manual_seed(0)
    params = {

        "num_episodes": 10,
        "std":0.001,
        "hidden_dim1": 256,
        "hidden_dim2": None,
        "method":"actor_critic",
        "exp":"experiment_4",
        "env": "Swimmer-v4"
    }
    render(params)

    