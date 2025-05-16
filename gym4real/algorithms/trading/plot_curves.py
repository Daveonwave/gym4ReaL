import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns


def extract_scalar(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"[Warning] Tag '{tag}' not found in {logdir}. Available tags: {ea.Tags()['scalars']}")
        return pd.DataFrame()
    events = ea.Scalars(tag)
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=["step", tag])
    return df


algo_name = "DQN"
base_dir = "./logs/tensorboard/trading/dqn/dqn_trading/dqn_1"
rollout_merge = None
eval_merge = None
seeds = [32517, 84029, 10473, 67288, 91352, 47605]
j = 0
for dir in os.listdir(base_dir):

    logdir = os.path.join(base_dir, dir)

    rollout_df = extract_scalar(logdir, "rollout/ep_rew_mean")
    rollout_df = rollout_df.rename(columns={"rollout/ep_rew_mean": f'{seeds[j]}'})
    eval_df = extract_scalar(logdir, "eval/mean_reward")
    eval_df = eval_df.rename(columns={"eval/mean_reward": f'{seeds[j]}'})

    if rollout_merge is None:
        rollout_merge = rollout_df
    else:
        rollout_merge = pd.merge(rollout_merge, rollout_df, on="step", how="outer").sort_values("step")

    if eval_merge is None:
        eval_merge = eval_df
    else:
        eval_merge = pd.merge(eval_merge, eval_df, on="step", how="outer").sort_values("step")
    j += 1

plot_colors = sns.color_palette('colorblind')
plt.figure(figsize=(10, 5))
i = 0
for column in rollout_merge.columns[1:].sort_values():
    plt.plot(rollout_merge['step'], rollout_merge[column], label=column, color=plot_colors[i])
    i += 1
plt.xlabel("Steps")
plt.ylabel("Reward")
#plt.title(f"Training Reward {algo_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./plots/train_curves_trading_{algo_name.lower()}.pdf")

plt.figure(figsize=(10, 5))
i = 0
for column in rollout_merge.columns[1:].sort_values():
    plt.plot(eval_merge['step'], eval_merge[column], label=column, color=plot_colors[i])
    i += 1
plt.xlabel("Steps")
plt.ylabel("Reward")
#plt.title(f"Evaluation Reward {algo_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./plots/eval_curves_trading_{algo_name.lower()}.pdf")
