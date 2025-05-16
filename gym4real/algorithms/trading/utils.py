import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.evaluation import evaluate_policy
from collections import OrderedDict
import datetime

sns.set_theme()
sns.set_style('whitegrid')
sns.set_context("paper")
plot_colors = sns.color_palette('colorblind')
sns.set(font_scale=1.2)

alg_color = OrderedDict({
    'random': plot_colors[1],
    'longest_first': plot_colors[2],
    'shortest_first': plot_colors[3],
    'q-learning': plot_colors[0],
    'sarsa': plot_colors[4],
    'dqn': plot_colors[5],
    'ppo': plot_colors[6],
    'fqi': plot_colors[7],
    'b&h': plot_colors[2],
    's&h': plot_colors[3]
})

alg_markers = OrderedDict({
    'random': '.',
    'longest_first': 'o',
    'shortest_first': 's',
    'q-learning': 's',
    'sarsa': 's',
})

alg_labels = {
    'random': 'Random',
    'longest_first': 'LF',
    'shortest_first': 'SF',
    'q-learning': 'Q-Learning',
    'sarsa': 'SARSA',
    'dqn': 'DQN',
    'ppo': 'PPO',
    'fqi': 'FQI'
}


def evaluate_agent_with_baselines(models, params, plot_folder, scaler, prefix, seeds, agent_name, show=False):
    # model = PPO("MlpPolicy", env, verbose=1, batch_size=115, policy_kwargs=dict(net_arch= dict(pi=[256, 256], vf=[256, 256])), gamma=0.99, n_steps=598*5, seed=seed, tensorboard_log=f"./ppo_trading_tensorboard/seed_{seed}")
    rewards_agents = []
    action_agents = []
    daily_return_agent = []
    for model in models:
        env_agent = gym.make("gym4real/TradingEnv-v0",
                             **{'settings': params, 'scaler': scaler})

        rewards_agent_seed = []
        action_episodes = []
        rewards_agent = []
        datetimes = []
        for _ in range(env_agent.unwrapped.get_trading_day_num()):
            done = False
            action_episode = []
            obs, _ = env_agent.reset()
            while not done:
                action, _ = model.predict(observation=np.array(obs, dtype=np.float32), deterministic=True)
                action_episode.append(action)
                next_obs, reward, terminated, truncated, info = env_agent.step(action)

                rewards_agent.append(reward)
                datetimes.append(info['datetime'])
                obs = next_obs
                done = terminated or truncated

            action_episodes.append(action_episode)

        rewards_agents.append(rewards_agent)
        action_agents.append(action_episodes)

    rewards_agents = np.asarray(rewards_agents)
    daily_return_agent = np.asarray(daily_return_agent)
    env_bnh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_bnh = []
    for _ in range(env_bnh.unwrapped.get_trading_day_num()):
        done = False
        env_bnh.reset()
        #print(env_bnh.unwrapped._day)
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
        #print(env_snh.unwrapped._day)
        while not done:
            next_obs, reward, terminated, truncated, _ = env_snh.step(0)
            rewards_snh.append(reward)
            done = terminated or truncated


    rewards_bnh = np.asarray(rewards_bnh)
    rewards_snh = np.asarray(rewards_snh)



    plt.figure()
    plt.plot(datetimes, (rewards_bnh.cumsum() / env_snh.unwrapped._capital) * 100, label="B&H", color=alg_color['b&h'])
    plt.plot(datetimes, (rewards_snh.cumsum() / env_snh.unwrapped._capital) * 100, label="S&H", color=alg_color['s&h'])
    mean_cumsum = np.mean((rewards_agents.cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
    std_cumsum = np.std((rewards_agents.cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
    plt.plot(datetimes, mean_cumsum, label=agent_name, color=alg_color[agent_name.lower()])
    plt.fill_between(datetimes, mean_cumsum - std_cumsum, mean_cumsum + std_cumsum, alpha=0.30,  color=alg_color[agent_name.lower()])
    plt.title(f"Performance on {prefix} Set")
    plt.xlabel("Time")
    plt.ylabel("P&L (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if show is False:
        plt.savefig(os.path.join(plot_folder, prefix+"_pnl.pdf"))
    else:
        plt.show()

    for i in range(len(action_agents)):
        action_agent = pd.DataFrame(action_agents[i]).fillna(0).astype(int)
        plt.figure()
        cmap = sns.color_palette(['red', 'white', 'green'])
        plt.title(f"Action Heatmap | seed = {seeds[i]}")
        sns.heatmap(action_agent, cmap=cmap)
        cbar = plt.gca().collections[0].colorbar
        tick_locs = np.arange(0, action_agent.shape[1], params['trading_close'] - params['trading_open'] )
        timestamps = pd.date_range(f'{params['trading_open']}:00', f'{params['trading_close'] - 1}:59', periods=120)
        tick_labels = [timestamps[i].strftime('%H:%M') for i in tick_locs]
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Short", "Flat", "Long"])
        plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=45)
        plt.xlabel("Time of the Day")
        plt.ylabel("Days")
        plt.tight_layout()
        if show is False:
            plt.savefig(os.path.join(plot_folder, prefix+f"_action_distribution_seed_{seeds[i]}.pdf"))
        else:
            plt.show()

def evaluate_multiple_agents_with_baselines(models, params, scaler, prefix, path):

    #models is a dictionary with key = name of the algorithm, value the trained algorithms

    rewards_agents = {}
    action_agents = {}
    daily_return_agents = {}
    for k in models.keys():
        rewards_agents[k] = []
        action_agents[k] = []
        daily_return_agents[k] = []

    for k in models.keys():
        daily_return_agent_m = []
        for model in models[k]:
            env_agent = gym.make("gym4real/TradingEnv-v0",
                                 **{'settings': params, 'scaler': scaler})

            action_episodes = []
            rewards_agent = []
            datetimes = []
            daily_return_agent = []
            for _ in range(env_agent.unwrapped.get_trading_day_num()):
                done = False
                action_episode = []
                obs, _ = env_agent.reset()
                cum_rew_ep = 0
                while not done:
                    action, _ = model.predict(observation=np.array(obs, dtype=np.float32), deterministic=True )
                    action_episode.append(action)
                    next_obs, reward, terminated, truncated, info = env_agent.step(action)
                    cum_rew_ep += reward
                    rewards_agent.append(reward)
                    datetimes.append(info['datetime'])
                    obs = next_obs
                    done = terminated or truncated
                daily_return_agent.append((cum_rew_ep / env_agent.unwrapped._capital)*100)
                action_episodes.append(action_episode)

            daily_return_agent_m.append(daily_return_agent)

            rewards_agents[k].append(rewards_agent)
            action_agents[k].append(action_episodes)

        rewards_agents[k] = np.asarray(rewards_agents[k])
        daily_return_agents[k] = np.asarray(daily_return_agent_m).mean(0)

    env_bnh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_bnh = []
    daily_return_long = []
    for _ in range(env_bnh.unwrapped.get_trading_day_num()):
        done = False
        env_bnh.reset()
        cum_rew_ep = 0
        while not done:
            next_obs, reward, terminated, truncated, _ = env_bnh.step(2)
            cum_rew_ep += reward
            rewards_bnh.append(reward)
            done = terminated or truncated
        daily_return_long.append((cum_rew_ep/ env_agent.unwrapped._capital)*100)

    daily_return_long = np.asarray(daily_return_long)

    env_snh = gym.make("gym4real/TradingEnv-v0",
                       **{'settings': params, 'scaler': scaler})

    rewards_snh = []
    daily_return_short = []
    for _ in range(env_snh.unwrapped.get_trading_day_num()):
        done = False
        env_snh.reset()
        cum_rew_ep = 0
        while not done:
            next_obs, reward, terminated, truncated, _ = env_snh.step(0)
            cum_rew_ep += reward
            rewards_snh.append(reward)
            done = terminated or truncated

        daily_return_short.append((cum_rew_ep/ env_agent.unwrapped._capital)*100)

    daily_return_short = np.asarray(daily_return_short)
    rewards_bnh = np.asarray(rewards_bnh)
    rewards_snh = np.asarray(rewards_snh)



    plt.figure()

    plt.plot(datetimes, (rewards_bnh.cumsum() / env_snh.unwrapped._capital) * 100, label="B&H", color=alg_color['b&h'])
    plt.plot(datetimes, (rewards_snh.cumsum() / env_snh.unwrapped._capital) * 100, label="S&H", color=alg_color['s&h'])

    for k in models.keys():
        mean_cumsum = np.mean((rewards_agents[k].cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
        std_cumsum = np.std((rewards_agents[k].cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
        plt.plot(datetimes, mean_cumsum, label=k, color=alg_color[k.lower()])
        plt.fill_between(datetimes, mean_cumsum - std_cumsum, mean_cumsum + std_cumsum, alpha=0.30,  color=alg_color[k.lower()])

    #plt.title(f"Performance on {prefix} Set")
    plt.xlabel("Time")
    plt.ylabel("P&L (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, f"ppo_dqn_{prefix.lower()}_trading.pdf"))

    daily_return_agents['B&H'] = daily_return_long
    daily_return_agents['S&H'] = daily_return_short
    plot_data = pd.DataFrame(daily_return_agents)
    plot_data = plot_data.melt(var_name='Strategy', value_name='Daily P&L')
    palette = {}
    for k in models.keys():
        palette[k] = alg_color[k.lower()]

    palette["B&H"] = alg_color['b&h']
    palette["S&H"] = alg_color['s&h']
    ### Boxplot
    plt.figure()
    sns.boxplot(x = "Strategy", y = "Daily P&L", data = plot_data, palette=palette)
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, f"ppo_dqn_{prefix.lower()}_boxplot_trading.pdf"))



class EvalCallbackSharpRatio(BaseCallback):
    def __init__(self, eval_env, callback_on_new_best=None, n_eval_episodes=5,
                 eval_freq=10000,
                 log_path=None, best_model_save_path=None, deterministic=True, render=False, verbose=1):
        super(EvalCallbackSharpRatio, self).__init__(verbose)
        self.eval_env = eval_env
        self.callback_on_new_best = callback_on_new_best
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.render = render

        self.best_sr = -np.inf

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate again with per-episode rewards
            episode_rewards, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
                warn=False,
                deterministic=self.deterministic
            )


            # Sum of rewards for all episodes
            sr = np.mean(episode_rewards) / (np.std(episode_rewards) + 1e-5)
            print(f"Current SR: {sr:.4f}, Best so far: {self.best_sr:.4f}")

            if sr > self.best_sr:
                self.best_sr = sr
                print(f"New best model! SR: {sr:.2f}")
                print(f"Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f} ")
                print("==================")

                if self.best_model_save_path is not None:
                    path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(path)

                if self.callback_on_new_best is not None:
                    return self.callback_on_new_best.on_step()

            # Log to tensorboard if applicable
            if self.logger:
                self.logger.record("eval/sr", sr)
                self.logger.record("eval/reward", f"{np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

                self.logger.dump(self.num_timesteps)

        return True
