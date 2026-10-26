"""冒烟测试：只检查环境和算法的不变量，秒级跑完。

    python test_smoke.py
"""

from __future__ import annotations

import numpy as np

import bandits
import dodge


def test_bandit_env_and_policies() -> None:
    env = bandits.BernoulliBandit(k=5, seed=0)
    assert env.probs.shape == (5,)
    assert 0.0 <= env.best_prob <= 1.0

    exp = bandits.run_experiment(steps=50, runs=8, k=5, seed=0)
    assert set(exp["results"]) == {
        "ε-greedy (ε=0.1)", "UCB1 (c=2)", "Thompson 采样", "梯度 Bandit (α=0.1)"
    }
    for r in exp["results"].values():
        assert r["avg_reward"].shape == (50,)
        assert 0.0 <= r["avg_reward"].mean() <= 1.0
        # 遗憾只在期望意义下单调：单次实验的样本噪声可以让它小幅回落
        assert r["regret"][-1] >= 0.0
        assert r["regret"][-1] >= r["regret"][len(r["regret"]) // 2] - 1.0


def test_bandit_learns() -> None:
    """给足步数后，好策略应该明显好过随机瞎拉。"""
    exp = bandits.run_experiment(steps=400, runs=20, k=5, seed=3)
    best = max(r["avg_reward"][-100:].mean() for r in exp["results"].values())
    assert best > exp["bandit"].probs.mean() + 0.05


def test_dodge_env_invariants() -> None:
    env = dodge.DodgeTigerEnv(size=6, tiger_period=2)
    state = env.reset()
    assert state == (5, 0, 5, 5, 0)
    assert len(state) == 5 and env.n_actions == 4

    steps = 0
    done = False
    while not done:
        state, reward, done, info = env.step(0)  # 一路向上撞墙
        steps += 1
        assert steps <= env.max_steps
    assert info["outcome"] in {"win", "caught", "timeout"}
    assert env.tiger != env.agent or info["outcome"] == "caught"


def test_q_learning_beats_random() -> None:
    """小网格、少量回合也应该能学出个样子。"""
    env = dodge.DodgeTigerEnv(size=5, tiger_period=2, max_steps=50)
    Q, history = dodge.q_learning(env, episodes=1500, seed=1)
    greedy = dodge.evaluate(env, Q, episodes=100, seed=9)
    random_policy = dodge.evaluate(env, None, episodes=100, seed=9)
    assert greedy["win_rate"] > random_policy["win_rate"]
    assert greedy["win_rate"] > 0.5, f"学得不够好：{greedy}"
    assert len(history["returns"]) == 1500


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过：{name}")
    print("\n全部冒烟测试通过 ✅")
