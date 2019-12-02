#!/usr/bin/env python3
#Jaime Danguillecourt
import gym
import random
from tqdm import tqdm #progress bar
import time
import numpy as np

from Settings import Settings
from Agent import Agent
from Stats import *
from Save import *

"""
From OpenAI Gym:

Landing pad is always at coordinates (0,0).
Coordinates are the first two numbers in state vector.
Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
If lander moves away from landing pad it loses reward back.
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points.
Landing outside landing pad is possible.
Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
"""

def main():
    env = gym.make("LunarLander-v2")
    s = Settings()

    num_actions = env.action_space.n
    observation_space = env.observation_space.shape

    episode_rewards = []
    rewards_rolling_avg = []
    aggr_stats_lst = []

    model_path = "./training_models/elon3.0/models/"
    model_name = "elon3.0_8650episode_330max_-228min_218avg_1575153558.model"
    model_path += model_name
    #model_path = None
    agent = Agent(num_actions, observation_space, s, model_path)

    #make folders for saving training models
    make_save_folders(s)

    try:
        #loop for desired number of attempts at Lunar Lander
        for episode in tqdm(range(1, s.episodes+1), ascii=True, unit='episode'):
            episode_reward = 0
            done = False
            state = env.reset()

            #each loop is an attempt at lunar lander
            while not done:
                if s.epsilon > random.random():
                    #preform random action
                    #while epsilon is high more random actions will be taken
                    action = random.randint(0, num_actions-1)
                else:
                    #preform action based off network prediction
                    #as episilon decays this will be the usual option
                    action = agent.get_action(state)

                #take action and get data back from env
                new_state, reward, done, extra_info = env.step(action)

                """
                #learn to fly first (don't penalize for flying up)
                if action == 2:
                    reward += 0.3
                """

                #train agent
                env_info = (state, action, new_state, reward, done)
                agent.train(env_info)

                #render
                if s.render and (episode % s.render_period == 0):
                    env.render()

                state = new_state
                episode_reward += reward

            ######STATS AND SAVING######
            #add reward of each episode to list to track progress
            episode_rewards.append(episode_reward)

            #store rolling avg
            #only calc after some min amount of episodes
            if episode > s.rolling_avg_min:
                rewards_rolling_avg.append(np.mean(episode_rewards))
                #if rolling avg above whats considered success, stop training
                if rewards_rolling_avg[-1] > s.success_margin:
                    save_model(episode_rewards, episode, agent)
                    break

            #calc & save: max, min, & avg stats of aggrated episodes
            #size of aggregation is period of collecting stats
            if episode % s.stats_period == 0:
                aggr_stats = calc_aggr_stats(episode, episode_rewards, s, display=True)
                aggr_stats_lst.append(aggr_stats)

                #save model if has good aggr stats
                save_good_model(episode_rewards, episode,  agent)

            #save model periodically just in case
            #save period defined in settings
            autosave(episode, agent)
            ###########################

            #decay epsilon
            if s.epsilon > s.min_epsilon:
                s.epsilon *= s.epsilon_decay
                s.epsilon = max(s.epsilon, s.min_epsilon)


    #if we interrupt training lets still get plots for progress so far
    except KeyboardInterrupt:
        pass

    plot_results(episode_rewards, rewards_rolling_avg, aggr_stats_lst, s)
    env.close()
    return

if __name__ == "__main__":
    main()
