import os
import numpy as np
from time import time
from Stats import calc_aggr_stats

def make_save_folders(s):
    if not os.path.exists("./training_models"):
        os.mkdir("./training_models")
    if not os.path.exists(f"./training_models/{s.model_name}"):
        os.mkdir(f"./training_models/{s.model_name}")
    if not os.path.exists(f"./training_models/{s.model_name}/models"):
        os.mkdir(f"./training_models/{s.model_name}/models")
    if not os.path.exists(f"./training_models/{s.model_name}/models/autosave"):
        os.mkdir(f"./training_models/{s.model_name}/models/autosave")

def save_model(episode_rewards, episode, agent, min_reward=None, max_reward=None, avg_reward=None):
    """
    save a model along with information of the max
    min and avg of the models in the last aggragated
    episodes
    """
    if min_reward == None:
        min_reward = int(round(np.min(episode_rewards[-agent.s.stats_period:]), 0))
    if max_reward == None:
        max_reward = int(round(np.max(episode_rewards[-agent.s.stats_period:]), 0))
    if avg_reward == None:
        avg_reward = int(round(np.mean(episode_rewards[-agent.s.stats_period:]), 0))

    save_name = f"{agent.s.model_name}_{episode}episode_{max_reward}max_"
    save_name += f"{min_reward}min_{avg_reward}avg_{int(time())}"
    save_loc = f"{agent.s.save_loc}{save_name}.model"
    agent.model.save(save_loc)

def autosave(episode, agent):
    """
    autosave periodically
    period defined in settings
    """
    if episode % agent.s.save_period == 0:
        save_name = f"{agent.s.model_name}_{episode}episode_{int(time())}"
        save_loc = f"{agent.s.autosave_loc}{save_name}.model"
        agent.model.save(save_loc)

def save_good_model(episode_rewards, episode, agent):
    """
    save a model if in the last aggregate episodes
    it has a good max, min or avg
    """
    min_reward = int(round(np.min(episode_rewards[-agent.s.stats_period:]), 0))
    max_reward = int(round(np.max(episode_rewards[-agent.s.stats_period:]), 0))
    avg_reward = int(round(np.mean(episode_rewards[-agent.s.stats_period:]), 0))

    if max_reward > agent.s.save_max:
        save_model(episode_rewards, episode, agent, min_reward, max_reward, avg_reward)
    elif min_reward > agent.s.save_min:
        save_model(episode_rewards, episode, agent, min_reward, max_reward, avg_reward)
    elif avg_reward > agent.s.save_avg:
        save_model(episode_rewards, episode, agent, min_reward, max_reward, avg_reward)
