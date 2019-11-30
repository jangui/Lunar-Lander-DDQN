class Settings:
    def __init__(self, env):
        ###env settings
        self.num_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.render = False

        ###training settings
        self.episodes = 500

        self.replay_mem_size = 30000
        self.min_replay_len = 1000
        self.batch_size = 64

        self.update_pred_model_period = 5

        self.epsilon = 0.5
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.001

        self.discount = 0.99

        ###stats settigns
        self.render_period = 100
        self.stats_period = 50

        ###save settings
        self.model_name = "elon1.3"
        #self.save_period = self.episodes
        self.save_period = 500
        self.save_max = 110
        self.save_min = -50
        self.save_avg = 0
        self.save_loc = f"./training_models/{self.model_name}/"
        self.auto_save_loc = f"{self.save_loc}autosave/"

    def load_settings(settings_json):
        #TODO call load settings from init to load settings from json
        pass
