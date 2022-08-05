import gym
import torch
from policy import GaussianPolicy
from reinforce import Actor
from simulation import simulate
import csv
import pandas as pd

def main():

    params = {"num_episodes": 8000, "std": 0.5, "gamma": 0.99, "hidden_dim":512,"env":"InvertedDoublePendulum-v4"}


    env = gym.make(params["env"])

    actor = Actor(env.observation_space.shape[0],env.action_space,params["hidden_dim"],std = params['std'])
    episode_scores = []
    episode_idx = 0

    max_score = 0

    while episode_idx < params["num_episodes"]:
        episode, episode_score = simulate(env, actor)
        episode_scores.append(episode_score)

        actor.train(episode, params["gamma"])

        if episode_score>max_score:
            torch.save(actor.policy.state_dict(),"results/inverted_double_pendulum/reinforce/std=0.2/best_model.pth")

        print(f"Episode {episode_idx}: {episode_score}")

        episode_idx+=1
    df = pd.DataFrame(episode_scores, columns=['ep_score'])
    df.to_csv('results/inverted_double_pendulum/reinforce/std=0.2/best_model.csv')


if __name__ =="__main__":
    main()
