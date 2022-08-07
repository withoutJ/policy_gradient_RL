import random

class Buffer():
    def __init__(self, maxlen=100000):
        self.memory = []
        self.maxlen = maxlen 

    def store(self,previous_observation, action, observation, reward, log_prob):
        self.memory.append({"previous_observation":previous_observation,"action":action, "observation":observation, "reward":reward, "log_prob":log_prob})
        if len(self.memory)>self.maxlen:
            self.memory = self.memory[1:]
            
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)     

    def __len__(self):
        return len(self.memory)