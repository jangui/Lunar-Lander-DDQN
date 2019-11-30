#!/usr/bin/env python3
#Jaime Danguillecourt
import gym
import random
from Settings import Settings
from Agent import Agent
from tqdm import tqdm #progress bar
import time
import matplotlib.pyplot as plt
import numpy as np
import os

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

def handle_stats_and_save(agent, episode, episode_rewards, s):
    min_reward = int(round(np.min(episode_rewards[-s.stats_period:]), 0))
    max_reward = int(round(np.max(episode_rewards[-s.stats_period:]), 0))
    avg_reward = int(round(np.sum(episode_rewards[-s.stats_period:])/s.stats_period, 0))
    print(f"\nEpisode: {episode}")
    print(f"Min Reward: {min_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Avg Reward: {avg_reward}")
    print()

    save_name = f"{s.model_name}_{episode}episode_{max_reward}max_"
    save_name += f"{min_reward}min_{avg_reward}avg_{int(time.time())}"

    #save good model based off max preforming agent
    save_loc = f"{s.save_loc}{save_name}.model"
    if max_reward > s.save_max:
        agent.model.save(save_loc)
    #save good model based off min preforming agent
    elif min_reward > s.save_min:
        agent.model.save(save_loc)
    #save good model based off avg of agents
    elif avg_reward > s.save_avg:
        agent.model.save(save_loc)

    return (min_reward, max_reward, avg_reward)

def plot_results(episode_rewards, final_stats):
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.show()

    min_models, max_models, avg_models = [], [], []
    for min_r, max_r, avg_r in final_stats:
        min_models.append(min_r)
        max_models.append(max_r)
        avg_models.append(avg_r)

    plt.plot(min_models)
    plt.plot(max_models)
    plt.plot(avg_models)
    plt.xlabel("Aggregated Episodes")
    plt.ylabel("Reward")
    plt.legend(['min','max','avg',], loc='upper left')
    plt.title("Reward vs Aggregate Episodes")
    plt.show()

def main():
    env = gym.make("LunarLander-v2")
    s = Settings(env)
    episode_rewards = []
    final_stats = []

    model_name = "256-128-elon1.2-_500episode_-7max_-771min_-226avg_1575081525.model"
    model_path = "./training_models/elon1.2/models/autosave/"
    model_path += model_name
    #model_path = None
    agent = Agent(s, model_path)

    #for saving_training_models
    if not os.path.exists("./training_models"):
        os.mkdir("./training_models")
    if not os.path.exists(f"./training_models/{s.model_name}"):
        os.mkdir(f"./training_models/{s.model_name}")
    if not os.path.exists(f"./training_models/{s.model_name}/autosave"):
        os.mkdir(f"./training_models/{s.model_name}/autosave")


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
                    action = random.randint(0, s.num_actions-1)
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

                env_info = (state, action, new_state, reward, done)
                #train model
                agent.train(env_info)

                if s.render and (episode % s.render_period == 0):
                    env.render()

                state = new_state
                episode_reward += reward

            #print some stats ever so often
            episode_rewards.append(episode_reward)
            if episode % s.stats_period == 0:
                reward_stats = handle_stats_and_save(agent, episode, episode_rewards, s)
                final_stats.append(reward_stats)

            #save model periodically just in case
            if episode % s.save_period == 0:
                min_reward = int(round(np.min(episode_rewards[-s.stats_period:]), 0))
                max_reward = int(round(np.max(episode_rewards[-s.stats_period:]), 0))
                avg_reward = int(round(np.sum(episode_rewards[-s.stats_period:])/s.stats_period, 0))

                save_name = f"{s.model_name}_{episode}episode_{max_reward}max_"
                save_name += f"{min_reward}min_{avg_reward}avg_{int(time.time())}"
                save_loc = f"{s.auto_save_loc}{save_name}.model"
                agent.model.save(save_loc)

            #decay epsilon
            if s.epsilon > s.min_epsilon:
                s.epsilon *= s.epsilon_decay
                s.epsilon = max(s.epsilon, s.min_epsilon)

    #if we interrupt training lets still get plots for progress so far
    except KeyboardInterrupt:
        pass

    plot_results(episode_rewards, final_stats)
    env.close()
    return

if __name__ == "__main__":
    main()
