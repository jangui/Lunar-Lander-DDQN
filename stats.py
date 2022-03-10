import numpy as np
import matplotlib.pyplot as plt

def calc_aggr_stats(episode, episode_rewards, aggregation_size, display=True):
    max_reward = int(round(np.max(episode_rewards[-aggregation_size:]), 0))
    avg_reward = int(round(np.mean(episode_rewards[-aggregation_size:]), 0))
    min_reward = int(round(np.min(episode_rewards[-aggregation_size:]), 0))

    if display==True:
        print(f"\nAggregate Results [Episode {episode}]")
        print(f"Max Reward: {max_reward}")
        print(f"Avg Reward: {avg_reward}")
        print(f"Min Reward: {min_reward}")
        rolling_avg = int(round(np.mean(episode_rewards), 0))
        print(f"\nRolling Average: {rolling_avg}")
        print()
    return (max_reward, avg_reward, min_reward)

def plot_results(episode_rewards, rewards_rolling_avg, rolling_avg_min, success_margin, aggr_stats_lst):
    # plot episode rewards
    plt.plot(episode_rewards)

    # plot rolling avg
    ravg_x_vals = [i for i in range(rolling_avg_min, len(episode_rewards))]
    plt.plot(ravg_x_vals, rewards_rolling_avg, color='red', linewidth=4.0)

    # plot success margin
    success_margin = [success_margin for i in range(len(episode_rewards))]
    plt.plot(success_margin, color='black', linewidth=0.5)

    # set plot info
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend(['Reward','Rolling Average', 'Success Margin'], loc='lower left')

    # show plot
    plt.show()

    # get aggregate stats
    aggr_min_models, aggr_max_models, aggr_avg_models = [], [], []
    for min_r, max_r, avg_r in aggr_stats_lst:
        aggr_min_models.append(min_r)
        aggr_max_models.append(max_r)
        aggr_avg_models.append(avg_r)

    # plot success margin
    success_margin = [success_margin for i in range(len(aggr_max_models))]
    plt.plot(success_margin, color='black', linewidth=0.5)

    # plot aggregate stats
    plt.plot(aggr_min_models)
    plt.plot(aggr_max_models)
    plt.plot(aggr_avg_models)

    # set plot info
    plt.xlabel("Aggregated Episodes")
    plt.ylabel("Reward")
    plt.legend(['min','max','avg','Success Margin'], loc='lower left')
    plt.title("Reward vs Aggregate Episodes")

    # show plot
    plt.show()
