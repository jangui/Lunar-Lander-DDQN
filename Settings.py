class Settings:
    def __init__(self):
        ###training settings
        self.render = True
        self.episodes = 20000

        self.replay_mem_size = 50000
        self.min_replay_len = 1000
        self.batch_size = 64

        self.update_pred_model_period = 5

        self.epsilon = 1
        self.epsilon_decay = 0.99975
        self.min_epsilon = 0.01

        self.discount = 0.99

        self.success_margin = 200

        ###stats settigns
        self.render_period = 100
        self.stats_period = 50
        self.rolling_avg_min = 25

        ###save settings
        self.model_name = "Apollo1.0"
        self.save_period = 200
        self.save_max = 250
        self.save_min = 0
        self.save_avg = 100
        self.save_loc = f"./training_models/{self.model_name}/models/"
        self.autosave_loc = f"{self.save_loc}autosave/"

