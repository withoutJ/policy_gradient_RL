import gym
import torch
from policy import GaussianPolicy
from reinforce import Actor
from simulation import simulate

def main():
    params = {"num_episodes": 10000, "std": 0, "gamma": 0.99}


    env = gym.make('InvertedPendulum-v4')
    obs_temp = env.reset()
    actor = Actor(obs_temp.shape[0],1,100,std = 0.00005)
    episode_scores = []
    episode_idx = 0

    max_score = 0

    while episode_idx < params["num_episodes"]:
        if episode_idx%1000 == 0:
            episode, episode_score = simulate(env, actor,True)
        else:
            episode, episode_score = simulate(env, actor)
        episode_scores.append(episode_score)

        actor.train(episode, params["gamma"])

        if episode_score>max_score:
            torch.save(actor.policy.state_dict(),"best_model.pth")

        print(f"Episode {episode_idx}: {episode_score}")

        episode_idx+=1

    
if __name__ =="__main__":
    main()