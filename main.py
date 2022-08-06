import gym
import torch
from policy import GaussianPolicy
from reinforce import Actor
from simulation import simulate
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():

    params = {

        "num_episodes": 50000,
        "std": 0.25,
        "std_change": True,
        "gamma": 0.99,
        "hidden_dim1" : 800,
        "hidden_dim2" : None,
        "env":"InvertedDoublePendulum-v4",
        "lr": 5e-5,
        "method":"actor_critic"
    }



    dir = f"results/{params['env']}/"+params["method"]
    print("**************************************************************************************")
    print("Selected directory: ", dir)
    print("**************************************************************************************")

    if not os.path.isdir(dir):
        os.makedirs(dir)

    num_experiments = len(os.listdir(dir))
    dir = os.path.join(dir,"experiment_"+str(num_experiments))

    env = gym.make(params["env"])

    actor = Actor(env.observation_space.shape[0], env.action_space, params["hidden_dim1"],params["hidden_dim2"], std = params['std'])

    episode_scores = []
    episode_idx = 0
    max_score = 0

    while episode_idx < params["num_episodes"]:

        #linear std update
        if episode_idx%1000 == 0:
            actor.policy.set_std(actor.policy.std-0.003)

        episode, episode_score = simulate(env, actor)
        episode_scores.append(episode_score)

        actor.train(episode, params["gamma"])

        if episode_score >= max_score:
            torch.save(actor.policy.state_dict(), os.path.join(dir, "best_model.pth"))

        if episode_idx % 1000 ==0 or episode_idx ==  params["num_episodes"]-1:
            print(f"Episode {episode_idx}: {episode_score}")

        episode_idx += 1

    readmeFile = os.path.join(dir,"read.txt")

    with open(readmeFile, 'w') as f:

        f.write('Model with parameters:\n')
        f.write(f"Num of episodes: {params['num_episodes']}\n")
        f.write(f"Initial std value: {params['std']}\n")
        f.write(f"Std changed: {params['std_change']}\n")
        f.write(f"Hidden dim: {params['hidden_dim1']}\n")
        if params['hidden_dim2'] is not None:
            f.write(f"Hidden dim: {params['hidden_dim2']}\n")
        f.write(f"Learning rate: {params['lr']}\n")

    df = pd.DataFrame(episode_scores, columns=['ep_score'])
    #df.to_csv(os.path.join(dir,"best_model.csv"))

    plt.plot(df['ep_score'])

    plt.savefig(os.path.join(dir,"training.png"))
    plt.show()

    print("**************************************************************************************")
    print("Uploaded into directory: ", dir)
    print("**************************************************************************************")


if __name__ =="__main__":
    torch.manual_seed(0)
    main()
