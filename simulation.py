import gym 
from policy import GaussianPolicy 

def simulate(env, actor):

    episode = []
    score = 0
    previous_observation = env.reset()
    done = False

    while not done:
        action, log_prob = actor.select_action(previous_observation)
        observation, reward, done, info = env.step(action)
        episode.append([previous_observation, action, observation, reward, log_prob])
        previous_observation = observation
        score+=reward
    return episode, score