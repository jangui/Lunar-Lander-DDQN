#!/usr/bin/env python3
import gym
import random
from Settings import Settings
from Agent import *
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

model_name = "16-16_165max_-878min_-323avg_1575033091.model"
model_path = "./models/"
#model_path += "/autosave/"
model_path += model_name
showcases = 10

def main():
    env = gym.make("LunarLander-v2")
    s = Settings(env)
    agent = Agent(s, model_path)

    for i in range(showcases):
        done = False
        state = env.reset()
        while not done:
            if random.randint(0,1):
                action = agent.get_action(state)
            else:
                action = 0

            new_state, reward, done, extra_info = env.step(action)

            env.render()
            state = new_state

    env.close()
    return

if __name__ == "__main__":
    main()