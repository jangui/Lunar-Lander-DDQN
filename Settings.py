
class Settings:
    def __init__(self, env):
        ###env settings
        self.num_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape
        #self.render = True

        ###training settings
        #self.episodes = 25000

        self.replay_mem_size = 5000
        self.min_replay_len = 1000
        self.batch_size = 64

        self.update_pred_model_period = 5

        #self.epsilon = 0.5
        #self.epsilon_decay = self.epsilon / (self.episodes // 2)
        self.epsilon = 1
        self.epsilon_decay = 0.99975
        self.min_epsilon = 0.001

        self.discount = 0.99

        ###stats settigns
        #self.render_period = self.episodes // 100
        #self.stats_period = self.episodes // 1000

        ###save settings
        self.model_name = "16-16"
        #self.save_period = self.episodes // 100
        self.save_max = 100
        self.save_min = -200
        self.save_avg = 0

        #debug settings
        self.episodes = 25
        self.render_period = 20
        self.stats_period = 5
        self.save_period = 24
        self.render = False


    def load_settings(json):
        #TODO call load settings from init to load settings from json
        pass
