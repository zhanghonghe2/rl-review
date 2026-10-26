"""M4 的另一半：测**收益**（墙钟时间），而不是代价。

    python3 run_async_threaded.py

为什么必须单独测：`run_async.py` 用确定性模式把"代价"隔离干净了（墙钟不变），
这里反过来 —— 固定 lag 策略，只比"串行"和"真线程重叠"的墙钟时间。

**预期（玩具任务上）会是负收益。** 这不是 bug，是标度问题：
异步的收益 ∝ rollout 时间 / 训练时间。玩具模型的 rollout 只要几毫秒，
线程调度 + GIL 争用的成本比它省下来的还多。真实 LLM 上 rollout 占绝对大头
（自回归生成是逐 token 串行的），这个比值才会 >> 1，异步才有意义。
"""

from __future__ import annotations

import time

import torch

from config import PPOConfig
from toy import CountOnesTask, TinyActorCritic
from trainer import evaluate, make_eval_prompts, train


def run(task, cfg, threaded: bool, eval_prompts, model_seed: int):
    from async_runner import AsyncSource

    torch.manual_seed(model_seed)
    model = TinyActorCritic()
    source = AsyncSource(task, model, cfg)
    t0 = time.perf_counter()
    train(task, model, cfg, source, eval_every=0)
    wall = time.perf_counter() - t0
    return wall, source.produced, evaluate(task, model, cfg, eval_prompts)["eval_reward"]


def main() -> None:
    task = CountOnesTask()
    eval_prompts = make_eval_prompts(task, n=64)
    base = PPOConfig.preset("toy").replace(total_iters=200, log_every=0, rollout_prompts=32)

    results = {}
    for threaded in (False, True):
        cfg = base.replace(threaded=threaded, max_lag=2, tag=f"thr_{threaded}")
        wall, produced, acc = run(task, cfg, threaded, eval_prompts, base.seed)
        results[threaded] = (wall, produced, acc)
        print(f"  threaded={str(threaded):<5} | 墙钟 {wall:6.2f}s | 生产 {produced} 批"
              f" | 评估 {acc:.3f}")

    t_serial, n_serial, a_serial = results[False]
    t_thr, n_thr, a_thr = results[True]
    print(f"\n  墙钟比（串行/线程）= {t_serial / t_thr:.3f}"
          f"  （>1 才是真的变快；<1 说明线程开销比省下的还多）")
    print(f"  异步只多生产了 {n_thr - n_serial} 批 —— 生产者的领先幅度受 max_lag 限制")
    print(f"  最终水平：串行 {a_serial:.3f} / 线程 {a_thr:.3f}"
          f"（差 {abs(a_serial - a_thr):.3f}，这部分才是数据变旧带来的代价）")


if __name__ == "__main__":
    main()
