"""M4 的入口：lag 扫描实验 —— 异步能省多少、代价是什么。

    python3 run_async.py

实验设计（这里有个容易做错的地方）
--------------------------------
异步同时带来两件事：**收益**（生成和训练重叠 → 墙钟变短）和**代价**（数据变旧
→ 策略偏移）。如果一起测，你分不清"结果变差"是因为数据旧，还是因为线程调度。

所以要把它们**分开测**：

* `threaded=False`（确定性）→ 单线程模拟，**墙钟不变**，只改变 lag。
  这是"**代价**的纯净测量"：其他一切都固定，只有数据的新旧程度在变。
* `threaded=True` → 测**收益**（墙钟）。玩具任务太小，这个数字大概率是负的
  （见文件末尾的说明），真实 LLM 上才会转正。

本脚本跑前者：固定 seed、固定步数，只扫 `max_lag = 0,1,2,4,8`，
然后画"滞后 → 策略偏移 → 最终水平"的关系。
"""

from __future__ import annotations

import os
import tempfile

import torch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "dsh-mpl-cache"))

from config import PPOConfig
from metrics import MetricLogger
from toy import CountOnesTask, TinyActorCritic
from trainer import evaluate, make_eval_prompts


def run_one(task: CountOnesTask, base: PPOConfig, max_lag: int, eval_prompts):
    """跑一次训练，返回这一组的汇总指标。"""
    from async_runner import train_async

    cfg = base.replace(max_lag=max_lag, tag=f"async_lag{max_lag}", log_every=0)
    torch.manual_seed(cfg.seed)                    # 模型初始化固定 → 各组可比
    model = TinyActorCritic()
    _, logger = train_async(task, model, cfg, logger=MetricLogger(cfg.out_dir, cfg.tag),
                            eval_prompts=eval_prompts, eval_every=0)
    final = evaluate(task, model, cfg, eval_prompts)

    n = len(logger.records)
    half = logger.records[n // 2:]                 # 只看后半段（前半段还在爬坡）
    mean_lag = sum(r["lag"] for r in half) / len(half)
    mean_clip = sum(r["clip_frac"] for r in half) / len(half)
    mean_kl = sum(r["approx_kl"] for r in half) / len(half)
    mean_ratio = sum(r["mean_ratio"] for r in half) / len(half)
    return dict(max_lag=max_lag, mean_lag=mean_lag, clip_frac=mean_clip,
                approx_kl=mean_kl, mean_ratio=mean_ratio,
                eval_reward=final["eval_reward"], gen_len=final["eval_gen_len"],
                logger=logger)


def main() -> None:
    base = PPOConfig.preset("toy").replace(
        threaded=False,          # 确定性模式：隔离"代价"，墙钟不变
        total_iters=200,
        rollout_prompts=32,
        inner_epochs=4,
    )
    print(base.summary())
    print()

    task = CountOnesTask()
    eval_prompts = make_eval_prompts(task, n=128)

    rows = []
    for max_lag in (0, 1, 2, 4, 8):
        r = run_one(task, base, max_lag, eval_prompts)
        rows.append(r)
        print(f"  max_lag={max_lag} | 实测 lag {r['mean_lag']:.2f}"
              f" | clip_frac {r['clip_frac']:.4f}"
              f" | mean_ratio {r['mean_ratio']:.4f}"
              f" | approx_kl {r['approx_kl']:.4f}"
              f" | 评估准确率 {r['eval_reward']:.3f}")

    # ---------------- 画"代价曲线" ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    lags = [r["mean_lag"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(lags, [r["clip_frac"] for r in rows], "o-", color="tab:red",
            label="clip_frac（被裁 token 比例）")
    ax.set(title="滞后 → 策略偏移", xlabel="实测 lag（版本落后数）", ylabel="clip_frac")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(lags, [r["mean_ratio"] for r in rows], "o-", color="tab:blue")
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set(title="滞后 → mean_ratio（偏离 1 的程度）",
           xlabel="实测 lag", ylabel="mean_ratio")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(lags, [r["eval_reward"] for r in rows], "o-", color="tab:green")
    ax.set(title="滞后 → 最终水平", xlabel="实测 lag", ylabel="评估准确率")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    fig.suptitle(f"异步 PPO 的代价：lag 扫描（{base.total_iters} 步，确定性模式）")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    # 注意：out_dir 是相对路径时，产物会落到**当前工作目录**，不是脚本所在目录。
    # 用 __file__ 锚定，避免"在别的目录跑一下就找不到图了"。
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       base.out_dir, "async_lag_sweep.png")
    os.makedirs(base.out_dir, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\n图：{out}")
    print("""
读图提示
--------
* **lag 一涨，clip_frac 就涨** —— 这是异步代价的直接机制：数据由旧策略产生，
  ratio 起手就不是 1，于是裁剪项开始吃掉梯度（有效 batch 变小）。
* 只要最终水平没有明显下降，说明这个 lag 还在"能接受"的范围内；
  一旦 clip_frac 冲过 ~0.3，梯度就大半被丢掉了，训练会明显变慢甚至退化。
* **别忘了另一面**：确定性模式下墙钟时间并没有变短 —— 它只是把"代价"隔离出来
  测干净了。真正的收益要用 `threaded=True` 测（跑 `python3 run_async_threaded.py`）。
  玩具任务上这个收益大概率是**负的**（模型太小、GIL、线程调度比计算还贵），
  真实 LLM 上 rollout 占大头时才会转正。
""")


if __name__ == "__main__":
    main()
