import gym 
from policy import GaussianPolicy
from reinforce import Actor
from simulation import simulate
params = {"num_episodes": 100, "std": 0.5, "hideen_dim":64}

def main():


    actor = Actor()
    env = gym.make('InvertedPendulum-v4')


    episode_scores = []
    episode_idx = 0

    while episode_idx < params["num_episodes"]:
        episode, episode_score = simulate(env, actor)
        episode_idx+=1

    
if __name__ =="__main__":
    main()