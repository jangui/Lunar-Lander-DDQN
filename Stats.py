import numpy as np
import matplotlib.pyplot as plt

def calc_aggr_stats(episode, episode_rewards, s, display=True):
    min_reward_aggr = int(round(np.min(episode_rewards[-s.stats_period:]), 0))
    max_reward_aggr = int(round(np.max(episode_rewards[-s.stats_period:]), 0))
    avg_reward_aggr = int(round(np.mean(episode_rewards[-s.stats_period:]), 0))

    if display==True:
        print(f"\nAggregate Results [Episode {episode}]")
        print(f"Min Reward: {min_reward_aggr}")
        print(f"Max Reward: {max_reward_aggr}")
        print(f"Avg Reward: {avg_reward_aggr}")
        print()
    return (min_reward_aggr, max_reward_aggr, avg_reward_aggr)

def plot_results(episode_rewards, rewards_rolling_avg, aggr_stats_lst, s):
    plt.plot(episode_rewards)

    #plot rolling avg with correct axis
    ravg_x_vals = [i for i in range(s.rolling_avg_min, len(episode_rewards))]
    plt.plot(ravg_x_vals, rewards_rolling_avg, color='red', linewidth=4.0)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend(['Reward','Rolling Average'], loc='lower left')
    plt.show()

    aggr_min_models, aggr_max_models, aggr_avg_models = [], [], []
    for min_r, max_r, avg_r in aggr_stats_lst:
        aggr_min_models.append(min_r)
        aggr_max_models.append(max_r)
        aggr_avg_models.append(avg_r)

    plt.plot(aggr_min_models)
    plt.plot(aggr_max_models)
    plt.plot(aggr_avg_models)
    plt.xlabel("Aggregated Episodes")
    plt.ylabel("Reward")
    plt.legend(['min','max','avg',], loc='lower left')
    plt.title("Reward vs Aggregate Episodes")
    plt.show()
