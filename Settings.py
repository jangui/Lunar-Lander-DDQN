class Settings:
    def __init__(self):
        ###training settings
        self.render = False
        self.episodes = 20000

        self.replay_mem_size = 50000
        self.min_replay_len = 1000
        self.batch_size = 64

        self.update_pred_model_period = 5

        self.epsilon = 1
        self.epsilon_decay = 0.99975
        #self.epsilon_decay = 995
        self.min_epsilon = 0.01

        self.discount = 0.99

        self.early_stop_count = 10
        self.early_stop_margin = 170
        self.success_margin = 200

        ###stats settigns
        self.render_period = 100
        self.stats_period = 50
        self.rolling_avg_min = 100

        ###save settings
        self.model_name = "elon3.1"
        self.save_period = 200
        self.save_max = 250
        self.save_min = -150
        self.save_avg = 100
        self.save_loc = f"./training_models/{self.model_name}/models/"
        self.autosave_loc = f"{self.save_loc}autosave/"

