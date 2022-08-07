import gym
import torch
from policy import GaussianPolicy
from reinforce import Actor
from simulation import simulate
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from actorCritic import ActorCritic


def main():

    params = {

        "num_episodes": 5000,
        "std": 5,
        "std_change": 0.08,
        "gamma": 0.99,
        "hidden_dim1" : 256,
        "hidden_dim2" : None,
        "env":"Swimmer-v4",
        "lr": 1e-4,
        "lr_critic": 1e-3,
        "method":"actor_critic",
    }



    dir = f"results/{params['env']}/"+params["method"]
    print("**************************************************************************************")
    print("Selected directory: ", dir)
    print("**************************************************************************************")

    num_experiments = len(os.listdir(dir)) if os.path.isdir(dir) else 0
    dir = os.path.join(dir,"experiment_"+str(num_experiments+1))

    os.makedirs(dir)

    env = gym.make(params["env"], ctrl_cost_weight = 1)

    if params['method'] == 'reinforce':
        agent = Actor(env.observation_space.shape[0], env.action_space.shape[0], params["hidden_dim1"],params["hidden_dim2"], std = params['std'])
    elif params['method'] == 'actor_critic':
        agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],  params["hidden_dim1"], std=params['std'], gamma = params["gamma"], lr_critic=params['lr_critic'],lr_actor = params['lr'])
    episode_scores = []
    episode_idx = 0
    max_score = 0
    hits = 0
    mean = 0
    std = 0
    total_score = 0
    rewards = []

    while episode_idx < params["num_episodes"]:

        #linear std update
        if episode_idx % 1000 == 0:
            mean = total_score/1000
            rewards = np.array(rewards)
            std = np.sqrt(np.sum(np.square(rewards - mean))/(999))
            print(f"Mean: {mean} standard deviation: {std} after {episode_idx} number of episodes")
            mean = 0
            std  = 0
            total_score = 0
            rewards = []
            agent.actor.set_std(agent.actor.std-params['std_change'])

        episode, episode_score = simulate(env, agent,render = False, train = True)
        episode_scores.append(episode_score)

        total_score+=episode_score
        rewards.append(episode_score)

        #actor.train(episode, params["gamma"])
        if episode_score == 1000 :
            hits += 1
            if hits == 10:
                for g in agent.actor_optimizer.param_groups:
                    g['lr']*=0.95
                for g in agent.critic_optimizer.param_groups:
                    g['lr']*=0.95


        if episode_score >=  max_score:
            max_score = episode_score
            if params['method'] == 'reinforce':
                torch.save(agent.policy.state_dict(), os.path.join(dir, "best_model.pth"))
            elif params['method'] == 'actor_critic':
                torch.save(agent.actor.state_dict(), os.path.join(dir, "actor_model.pth"))
                torch.save(agent.critic.state_dict(), os.path.join(dir, "critic_model.pth"))

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
        f.write(f"Actor learning rate: {params['lr']}\n")
        f.write(f"Critic learning rate : {params['lr_critic']}\n")

    df = pd.DataFrame(episode_scores, columns=['ep_score'])
    moving_average = df['ep_score'].rolling(500).mean()
    df.to_csv(os.path.join(dir,"best_model.csv"))

    plt.plot(df['ep_score'],label = 'Scores')
    plt.plot(moving_average, label = 'Moving average')
    plt.legend()
    plt.title('Training scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.savefig(os.path.join(dir,"training.png"))
    plt.show()
    print('Hits: ', hits)
    print("**************************************************************************************")
    print("Uploaded into directory: ", dir)
    print("**************************************************************************************")


if __name__ =="__main__":
    torch.manual_seed(0)
    main()

