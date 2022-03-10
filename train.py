#!/usr/bin/env python3
import os
import gym
import random
from tqdm import tqdm #progress bar
import time
import numpy as np

from Agent import Agent
from stats import plot_results, calc_aggr_stats

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

    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    episode_rewards = []
    rewards_rolling_avg = []
    aggr_stats_lst = []

    model_name = "test"

    #model_path = f"./trained_models/{model_name}"
    model_path = None

    agent = Agent(num_actions, input_shape, model_path)

    #make folders for saving training models
    if not os.path.exists("./training_models"):
        os.mkdir("./training_models")
    if not os.path.exists(f"./training_models/{model_name}"):
        os.mkdir(f"./training_models/{model_name}")

    try:
        # loop for desired number of attempts at Lunar Lander
        for episode in tqdm(range(1, agent.episodes+1), ascii=True, unit='episode'):
            episode_reward = 0
            done = False
            state = env.reset()

            ## game loop ##
            while not done:
                # eplison-greedy exploration
                if agent.epsilon > random.random():
                    action = random.randint(0, num_actions-1)
                else:
                    action = agent.get_action(state)

                # take action and get data back from env
                new_state, reward, done, extra_info = env.step(action)

                """
                #learn to fly first (don't penalize using main engine)
                if action == 2:
                    reward += 0.3
                """

                # train agent
                env_info = (state, action, new_state, reward, done)
                agent.train(env_info)

                # render
                if agent.render and (episode % agent.render_period == 0):
                    env.render()

                state = new_state
                episode_reward += reward
            ## end of game loop ##

            ### stats and saving ###
            # save total reward for this episode
            episode_rewards.append(episode_reward)

            # store rolling avg after a minimum of episodes have passed
            if episode > agent.rolling_avg_min:
                rewards_rolling_avg.append(np.mean(episode_rewards))

                # stop training if rolling average above success margin
                if rewards_rolling_avg[-1] > agent.success_margin:
                    agent.save(episode)
                    break

            # calculate stats on a batch episodes
            if episode % agent.checkpoint_period == 0:
                aggr_stats = calc_aggr_stats(episode, episode_rewards, agent.checkpoint_period, display=True)
                aggr_stats_lst.append(aggr_stats)

                # save model if aggregation outpreformed the save thresholds
                max_reward, avg_reward, min_reward = aggr_stats
                if max_reward > agent.save_thresholds['best']:
                    agent.save_with_stats(episode, aggr_stats)
                elif avg_reward > agent.save_thresholds['avg']:
                    agent.save_with_stats(episode, aggr_stats)
                elif min_reward > agent.save_thresholds['worst']:
                    agent.save_with_stats(episode, aggr_stats)

            # auto save
            if episode % agent.autosave_period == 0:
                agent.save(episode)

            ### training updates ###
            #decay epsilon
            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon = max(agent.epsilon, agent.min_epsilon)

    except KeyboardInterrupt:
        # keyboard interrupts will stop training but not halt the entire program
        # this will allow us to plot results for training we have done so far
        pass

    # plot results for training
    plot_results(episode_rewards, rewards_rolling_avg, agent.rolling_avg_min, agent.success_margin, aggr_stats_lst)

    # clean up
    env.close()

if __name__ == "__main__":
    main()
