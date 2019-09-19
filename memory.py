'''
A file for the memory of the reinforcement learning agent
'''

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_obs, new_action, new_reward):
        self.observations.append(new_obs)
        self.actions.append(new_action)
        self.rewards.append(new_reward)
