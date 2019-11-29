
class Settings:
    def __init__(self, env):
        self.num_actions = env.action_space.n
        self.render = True
        self.episodes = 25000
        self.observation_shape = env.observation_space.shape
        self.batch_size = 64
        self.replay_mem_size = 5000
        self.min_replay_len = 1000
        self.epsilon = 0.5
        self.epsilon_decay = self.epsilon / (self.episodes // 2)
        self.render_period = self.episodes // 100
        self.discount = 0.9
        self.update_pred_model_period = 5
        self.stats_period = self.episodes // 1000
        self.save_period = self.episodes // 100
        self.model_name = "64-16"


    def load_settings(json):
        #TODO call load settings from init to load settings from json
        pass
