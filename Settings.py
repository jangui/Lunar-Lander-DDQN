
class Settings:
    def __init__(self, env):
        self.num_actions = env.action_space.n
        self.render = True
        self.episodes = 1
        self.observation_shape = env.observation_space.shape
        self.batch_size = 64
        self.replay_mem_size = 5000
        self.min_replay_len = 1000
        self.discount = 0.99

    def load_settings(json):
        #TODO call load settings from init to load settings from json
        pass
