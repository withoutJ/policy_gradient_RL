import gym
from reinforce import Actor
from simulation import simulate

def render(params):
    env = gym.make('InvertedPendulum-v4')

    actor = Actor(env.observation_space.shape[0],env.action_space,params["hidden_dim"],std = params["std"], model_path="best_model.pth")

    episode, episode_score= simulate(env,actor,render=True)

if __name__=="__main__":
    render()
    