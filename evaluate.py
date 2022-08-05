import gym
from reinforce import Actor
from simulation import simulate
import torch
import pandas as pd
def render(params):

    env = gym.make(params["env"])

    actor = Actor(env.observation_space.shape[0],env.action_space,params["hidden_dim"],std = params["std"], model_path="results/inverted_double_pendulum/reinforce/std=0.2/best_model.pth")

    episode, episode_score= simulate(env,actor,render=True)
    print(episode_score)

if __name__=="__main__":
    params = {"num_episodes": 3000, "std": 0.00000000001, "gamma": 0.99, "hidden_dim": 512,"env":"InvertedDoublePendulum-v4"}
    render(params)

    