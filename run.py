#!/usr/bin/env python3
#Jaime Danguillecourt
import gym
import random
from Settings import Settings
from Agent import *
from tqdm import tqdm
import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

"""
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
def handle_stats_and_save(episode, episode_rewards, s):
    min_reward = round(np.min(episode_rewards[-s.stats_period:]), 1)
    max_reward = round(np.max(episode_rewards[-s.stats_period:]), 1)
    avg_reward = round(np.sum(episode_rewards[-s.stats_period:])/s.stats_period, 1)
    print(f"Episode: {episode}")
    print(f"Min Reward: {min_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Avg Reward: {avg_reward}")
    print()

    #save good model
    if max_reward > 0:
        save_name = f"{s.model_name}_{max_reward}max_{min_reward}min_{avg_reward}avg_{int(time.time())}"
        agent.model.save(f"models/{save_name}.model")

    #save good model
    elif min_reward > -100:
        save_name = f"{s.model_name}_{max_reward}max_{min_reward}min_{avg_reward}avg_{int(time.time())}"
        agent.model.save(f"models/{save_name}.model")

    return min_reward, max_reward, avg_reward


def main():
    env = gym.make("LunarLander-v2")
    s = Settings(env)
    agent = Agent(s)
    episode_rewards = []

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
            min_reward, max_reward, avg_reward = handle_stats_and_save(episode, episode_rewards, s)

        #save model periodically just in case
        if episode % s.save_period == 0:
            save_name = f"{s.model_name}_{episode}_{max_reward}max_{min_reward}min_{avg_reward}avg"
            agent.model.save(f"models/autosave/{save_name}.model")



    env.close()
    return

if __name__ == "__main__":
    main()
