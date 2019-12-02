#!/usr/bin/env python3
import gym
import random
from Settings import Settings
from Agent import *
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

model_name = "elon3.1_12350episode_311max_214min_273avg_1575262915.model"
model_path = "./training_models/elon3.1/models/"
model_path += model_name
showcases = 10

def main():
    env = gym.make("LunarLander-v2")
    s = Settings()

    num_actions = env.action_space.n
    observation_space = env.observation_space.shape

    agent = Agent(num_actions, observation_space, s, model_path)

    for i in range(showcases):
        done = False
        state = env.reset()
        while not done:
            action = agent.get_action(state)

            new_state, reward, done, extra_info = env.step(action)

            env.render()
            state = new_state

    env.close()
    return

if __name__ == "__main__":
    main()
