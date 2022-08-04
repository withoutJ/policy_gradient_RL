import gym 
from policy import GaussianPolicy
from reinforce import Actor
from simulation import simulate

def main():
    params = {"num_episodes": 100, "std": 0.5, "gamma": 0.9}

    actor = Actor()
    env = gym.make('InvertedPendulum-v4')

    episode_scores = []
    episode_idx = 0

    while episode_idx < params["num_episodes"]:
        episode, episode_score = simulate(env, actor)
        episode_scores.append(episode_score)

        actor.train(episode)

        gamma = params["gamma"]

        for k in len(episode):
            
            for step in episode[k:]:
                


        episode_idx+=1

    
if __name__ =="__main__":
    main()