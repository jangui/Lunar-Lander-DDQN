#!/usr/bin/env python3
import gym
import random
from Settings import Settings
from Agent import *
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

model_name = "example.model"

model_path = "./models/"
#model_path += "/autosave/"
model_path += model_name

def main():
    env = gym.make("LunarLander-v2")
    s = Settings(env)
    agent = Agent(s, model_path)

    done = False
    state = env.reset()

    while not done:
        action = agent.get_action(state)

        new_state, reward, done, extra_info = env.step(action)

        env.render()

        env_info = (state, action, new_state, reward, done)
        agent.train(env_info)

        state = new_state

    env.close()
    return

if __name__ == "__main__":
    main()
