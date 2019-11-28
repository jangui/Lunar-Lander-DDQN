
class Settings:
    def __init__(self, env):
        self.num_actions = env.action_space.n
        self.render = True
        self.episodes = 1
        self.observation_shape = env.observation_space.shape

