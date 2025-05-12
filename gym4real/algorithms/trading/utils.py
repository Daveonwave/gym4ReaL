import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.evaluation import evaluate_policy
from collections import OrderedDict

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
    'b&h': plot_colors[8],
    's&h': plot_colors[9]
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



    plt.figure(figsize=(5, 3.5))
    plt.plot( (rewards_bnh.cumsum() / env_snh.unwrapped._capital) * 100, label="B&H", color=alg_color['b&h'])
    plt.plot((rewards_snh.cumsum() / env_snh.unwrapped._capital) * 100, label="S&H", color=alg_color['s&h'])
    #plt.plot(range(len(rewards_bnh)), (rewards_agent.cumsum() / env_agent.unwrapped._capital) * 100, label="Agent")
    mean_cumsum = np.mean((rewards_agents.cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
    std_cumsum = np.std((rewards_agents.cumsum(1) / env_agent.unwrapped._capital) * 100, axis=0)
    x = np.arange(rewards_agents.shape[1])
    plt.plot(x, mean_cumsum, label=agent_name, color=alg_color[agent_name.lower()])
    plt.fill_between(x, mean_cumsum - std_cumsum, mean_cumsum + std_cumsum, alpha=0.30,  color=alg_color[agent_name.lower()])
    plt.title(f"Performance on {prefix} Set")
    plt.xlabel("Time")
    plt.ylabel("P&L (%)")
    plt.legend()
    if show is False:
        plt.savefig(os.path.join(plot_folder, prefix+"_pnl"))
    else:
        plt.show()

    for i in range(len(action_agents)):
        action_agent = pd.DataFrame(action_agents[i]).fillna(0).astype(int)
        plt.figure(figsize=(5, 3.5))
        cmap = sns.color_palette(['red', 'white', 'green'])
        plt.title(f"Action Heatmap | seed = {seeds[i]}")
        sns.heatmap(action_agent, cmap=cmap)
        tick_locs = np.arange(0, action_agent.shape[1], params['trading_close'] - params['trading_open'] )
        timestamps = pd.date_range(f'{params['trading_open']}:00', f'{params['trading_close'] - 1}:59', periods=120)
        tick_labels = [timestamps[i].strftime('%H:%M') for i in tick_locs]
        plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=45)
        if show is False:
            plt.savefig(os.path.join(plot_folder, prefix+f"_action_distribution_seed_{seeds[i]}"))
        else:
            plt.show()


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
