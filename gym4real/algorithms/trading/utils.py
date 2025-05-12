import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def evaluate_agent_with_baselines(models, params, plot_folder, scaler, prefix, seeds):
    # model = PPO("MlpPolicy", env, verbose=1, batch_size=115, policy_kwargs=dict(net_arch= dict(pi=[256, 256], vf=[256, 256])), gamma=0.99, n_steps=598*5, seed=seed, tensorboard_log=f"./ppo_trading_tensorboard/seed_{seed}")
    rewards_agents = []
    action_agents = []
    for model in models:
        env_agent = gym.make("gym4real/TradingEnv-v0",
                             **{'settings': params, 'scaler': scaler})

        rewards_agent_seed = []
        action_episodes = []
        rewards_agent = []
        for _ in range(env_agent.unwrapped.get_trading_day_num()):
            done = False
            action_episode = []
            obs, _ = env_agent.reset()

            while not done:
                action, _ = model.predict(observation=np.array(obs, dtype=np.float32))
                action_episode.append(action)
                next_obs, reward, terminated, truncated, _ = env_agent.step(action)
                rewards_agent.append(reward)
                obs = next_obs
                done = terminated or truncated

            action_episodes.append(action_episode)

        rewards_agents.append(rewards_agent)
        action_agents.append(action_episodes)

    rewards_agents = np.asarray(rewards_agents)
    env_bnh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_bnh = []
    for _ in range(env_bnh.unwrapped.get_trading_day_num()):
        done = False
        env_bnh.reset()
        print(env_bnh.unwrapped._day)
        while not done:
            next_obs, reward, terminated, truncated, _ = env_bnh.step(2)
            rewards_bnh.append(reward)
            done = terminated or truncated

    env_snh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_snh = []
    for _ in range(env_snh.unwrapped.get_trading_day_num()):
        done = False
        env_snh.reset()
        print(env_snh.unwrapped._day)
        while not done:
            next_obs, reward, terminated, truncated, _ = env_snh.step(0)
            rewards_snh.append(reward)
            done = terminated or truncated


    rewards_bnh = np.asarray(rewards_bnh)
    rewards_snh = np.asarray(rewards_snh)



    plt.figure()
    plt.plot( (rewards_bnh.cumsum() / env_snh.unwrapped._capital) * 100, label="B&H")
    plt.plot((rewards_snh.cumsum() / env_snh.unwrapped._capital) * 100, label="S&H")
    #plt.plot(range(len(rewards_bnh)), (rewards_agent.cumsum() / env_agent.unwrapped._capital) * 100, label="Agent")
    mean_cumsum = np.mean((rewards_agents.cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
    std_cumsum = np.std((rewards_agents.cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
    x = np.arange(rewards_agents.shape[1])
    plt.plot(x, mean_cumsum, label="Agent", color="green")
    plt.fill_between(x, mean_cumsum - std_cumsum, mean_cumsum + std_cumsum, alpha=0.30,  color="green")
    plt.title(f"Performance on {prefix} Set")
    plt.xlabel("Time")
    plt.ylabel("P&L (%)")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, prefix+"_pnl"))

    for i in range(len(action_agents)):
        action_agent = pd.DataFrame(action_agents[i]).fillna(0).astype(int)
        plt.figure()
        cmap = sns.color_palette(['red', 'white', 'green'])
        plt.title(f"Action Heatmap")
        sns.heatmap(action_agent, cmap=cmap)
        tick_locs = np.arange(0, action_agent.shape[1], params['trading_close'] - params['trading_open'] )
        timestamps = pd.date_range(f'{params['trading_open']}:00', f'{params['trading_close'] - 1}:59', periods=120)
        tick_labels = [timestamps[i].strftime('%H:%M') for i in tick_locs]
        plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=45)
        plt.savefig(os.path.join(plot_folder, prefix+f"_action_distribution_seed_{seeds[i]}"))