#!/usr/bin/env python3
import gym
import random
from Agent import Agent
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

model_name = "spacex.model"
model_path = f"./trained_models/{model_name}"

episodes = 10

def main():
    env = gym.make("LunarLander-v2")

    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    agent = Agent(num_actions, input_shape, model_path=model_path)

    for i in range(episodes):
        done = False
        state = env.reset()

        # game loop
        while not done:
            # get action from agent
            action = agent.get_action(state)

            # preform action and get info from env
            new_state, reward, done, extra_info = env.step(action)

            # render and update state
            env.render()
            state = new_state

    env.close()
    return

if __name__ == "__main__":
    main()
