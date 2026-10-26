"""M3 的入口：跑一次同步 PPO 训练并出图。

    python3 run_toy.py

预期（CPU 约 7 秒）：评估准确率 2.3% → ~100%。
"""

from __future__ import annotations

import torch

from config import PPOConfig
from metrics import MetricLogger
from toy import CountOnesTask, TinyActorCritic
from trainer import make_eval_prompts, train_sync


def main() -> None:
    cfg = PPOConfig.preset("toy")
    print(cfg.summary())
    print()

    torch.manual_seed(cfg.seed)          # 固定模型初始化，保证可复现
    task = CountOnesTask()
    model = TinyActorCritic()
    eval_prompts = make_eval_prompts(task, n=128)

    # 先看训练前
    from trainer import evaluate
    before = evaluate(task, model, cfg, eval_prompts)
    print(f"训练前：评估准确率 {before['eval_reward']:.3f}"
          f"，平均生成长度 {before['eval_gen_len']:.1f}\n")

    model, logger = train_sync(task, model, cfg,
                               logger=MetricLogger(cfg.out_dir, cfg.tag),
                               eval_prompts=eval_prompts, eval_every=25)

    after = evaluate(task, model, cfg, eval_prompts)
    print(f"\n训练后：评估准确率 {after['eval_reward']:.3f}"
          f"，平均生成长度 {after['eval_gen_len']:.1f}")
    print(f"（生成长度从 {before['eval_gen_len']:.1f} 降到 {after['eval_gen_len']:.1f}："
          f"模型学会了尽早输出 STOP —— 这就是逐步代价的作用）")
    print(f"\n{logger.summary(20)}")
    print(f"\n图：{logger.plot(title='PPO 同步训练 · 数 1 的个数')}")


if __name__ == "__main__":
    main()
