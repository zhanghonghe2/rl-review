"""多臂老虎机（Multi-Armed Bandit）策略对照实验。

"老虎机"就是一个有 k 个摇臂的赌博机：每个臂的收益服从一个固定但未知的分布。
目标是在 n 次拉杆里尽量多拿奖励 —— 等价于最小化遗憾（regret）：

    regret(T) = T * p*  -  sum_{t=1..T} r_t

本模块提供环境 `BernoulliBandit` 与四种经典策略（ε-greedy / UCB1 / Thompson
采样 / 梯度 Bandit），并把它们放进同一组实验里比较。策略全部按"多次独立实验"
向量化，所以几百次 run × 上千步也只要一两秒。

用法：
    python bandits.py            # 跑默认实验并打印结果
    python main.py bandit        # 同上（统一入口）
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

# matplotlib 默认把缓存写到 ~/.matplotlib，沙箱里可能不可写，先指到临时目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "dsh-mpl-cache"))


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
class BernoulliBandit:
    """k 臂伯努利老虎机：拉臂 i 以概率 p_i 得到奖励 1，否则得到 0。

    `probs` 是环境私有的真值，智能体只能通过一次次拉杆去估计它。
    """

    def __init__(self, k: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.k = k
        self.probs = rng.random(k)
        self.best_arm = int(np.argmax(self.probs))
        self.best_prob = float(self.probs[self.best_arm])

    def sample(self, arms: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """给一批（并行的）run 各拉一次指定的臂，返回奖励向量。"""
        return (rng.random(arms.shape[0]) < self.probs[arms]).astype(np.float64)

    def __repr__(self) -> str:
        ps = " ".join(f"{p:.2f}" for p in self.probs)
        return (
            f"BernoulliBandit(k={self.k}, 最优臂=#{self.best_arm} "
            f"p*={self.best_prob:.3f}, p=[{ps}])"
        )


# --------------------------------------------------------------------------- #
# 策略（全部向量化：一次推进 runs 个独立实验）
# --------------------------------------------------------------------------- #
class Policy:
    """策略基类：维护每条 run 自己的估计，并选出这一步要拉的臂。"""

    name = "policy"

    def __init__(self, k: int, runs: int) -> None:
        self.k = k
        self.runs = runs
        self.counts = np.zeros((runs, k), dtype=np.float64)
        self.values = np.zeros((runs, k), dtype=np.float64)
        self.t = 0

    def select(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def update(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.runs)
        self.counts[rows, arms] += 1.0
        n = self.counts[rows, arms]
        # 增量平均：value += (r - value) / n
        self.values[rows, arms] += (rewards - self.values[rows, arms]) / n
        self.t += 1

    @staticmethod
    def _argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """argmax，但并列时随机选一个（避免永远偏向编号小的臂）。"""
        noise = rng.random(values.shape) * 1e-9
        return np.argmax(values + noise, axis=1)


class EpsilonGreedy(Policy):
    """ε-greedy：以 ε 的概率随机探索，否则拉当前估值最高的臂。"""

    def __init__(self, k: int, runs: int, epsilon: float = 0.1) -> None:
        super().__init__(k, runs)
        self.epsilon = epsilon
        self.name = f"ε-greedy (ε={epsilon:g})"

    def select(self, rng: np.random.Generator) -> np.ndarray:
        explore = rng.random(self.runs) < self.epsilon
        greedy = self._argmax_random_tie(self.values, rng)
        random_arm = rng.integers(0, self.k, size=self.runs)
        return np.where(explore, random_arm, greedy)


class UCB1(Policy):
    """UCB1：估值 + 置信上界 c*sqrt(ln t / n)，乐观面对不确定性。"""

    def __init__(self, k: int, runs: int, c: float = 2.0) -> None:
        super().__init__(k, runs)
        self.c = c
        self.name = f"UCB1 (c={c:g})"

    def select(self, rng: np.random.Generator) -> np.ndarray:
        untried = self.counts == 0
        bonus = np.zeros_like(self.values)
        tried = ~untried
        if tried.any():
            bonus[tried] = self.c * np.sqrt(
                np.log(max(self.t, 1) + 1.0) / self.counts[tried]
            )
        scores = self.values + bonus
        # 没拉过的臂优先（给它一个足够大的分数）
        scores[untried] = np.inf
        return self._argmax_random_tie(scores, rng)


class ThompsonSampling(Policy):
    """Thompson 采样：给每个臂维护 Beta(α, β) 后验，按后验采样值选臂。"""

    def __init__(self, k: int, runs: int) -> None:
        super().__init__(k, runs)
        self.alpha = np.ones((runs, k), dtype=np.float64)
        self.beta = np.ones((runs, k), dtype=np.float64)
        self.name = "Thompson 采样"

    def select(self, rng: np.random.Generator) -> np.ndarray:
        samples = rng.beta(self.alpha, self.beta)
        return self._argmax_random_tie(samples, rng)

    def update(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.runs)
        self.alpha[rows, arms] += rewards
        self.beta[rows, arms] += 1.0 - rewards
        super().update(arms, rewards)


class GradientBandit(Policy):
    """梯度 Bandit：维护偏好 H，用 softmax 当策略，按 (r - baseline) 做梯度上升。"""

    def __init__(self, k: int, runs: int, alpha: float = 0.1) -> None:
        super().__init__(k, runs)
        self.alpha = alpha
        self.preference = np.zeros((runs, k), dtype=np.float64)
        self.baseline = np.zeros(runs, dtype=np.float64)
        self.probs = np.full((runs, k), 1.0 / k)
        self.name = f"梯度 Bandit (α={alpha:g})"

    def select(self, rng: np.random.Generator) -> np.ndarray:
        z = self.preference - self.preference.max(axis=1, keepdims=True)
        exp = np.exp(z)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        # 按概率分布采样（每条 run 一个）
        return np.array(
            [rng.choice(self.k, p=self.probs[i]) for i in range(self.runs)]
        )

    def update(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.runs)
        advantage = rewards - self.baseline  # 用平均奖励当 baseline 降方差
        self.preference += self.alpha * advantage[:, None] * (
            (np.arange(self.k)[None, :] == arms[:, None]).astype(np.float64)
            - self.probs
        )
        self.baseline += (rewards - self.baseline) / (self.t + 1)
        super().update(arms, rewards)


# --------------------------------------------------------------------------- #
# 实验
# --------------------------------------------------------------------------- #
def run_experiment(
    steps: int = 1000,
    runs: int = 200,
    k: int = 10,
    seed: int = 0,
    policies: list[Policy] | None = None,
) -> dict:
    """跑一组策略，返回 {名字: {"avg_reward", "optimal_rate", "regret"}}。

    每条 run 用同一个老虎机参数（同样的 p 向量）但独立的随机流，保证公平比较。
    """
    bandit = BernoulliBandit(k=k, seed=seed)
    rng = np.random.default_rng(seed + 1)

    if policies is None:
        policies = [
            EpsilonGreedy(k, runs, epsilon=0.1),
            UCB1(k, runs, c=2.0),
            ThompsonSampling(k, runs),
            GradientBandit(k, runs, alpha=0.1),
        ]

    results: dict[str, dict[str, np.ndarray]] = {}
    for policy in policies:
        reward_curve = np.zeros((steps, runs))
        optimal_curve = np.zeros((steps, runs))
        for t in range(steps):
            arms = policy.select(rng)
            rewards = bandit.sample(arms, rng)
            policy.update(arms, rewards)
            reward_curve[t] = rewards
            optimal_curve[t] = arms == bandit.best_arm

        cum_reward = reward_curve.mean(axis=1).cumsum()
        results[policy.name] = {
            "avg_reward": reward_curve.mean(axis=1),
            "optimal_rate": optimal_curve.mean(axis=1),
            "regret": bandit.best_prob * np.arange(1, steps + 1) - cum_reward,
        }
    return {"bandit": bandit, "steps": steps, "runs": runs, "results": results}


def summarize(exp: dict) -> list[tuple[str, float, float, float]]:
    """把曲线压成一张表：平均奖励 / 最优臂比例 / 累积遗憾。"""
    rows = []
    for name, r in exp["results"].items():
        rows.append(
            (
                name,
                float(r["avg_reward"].mean()),
                float(r["optimal_rate"].mean()),
                float(r["regret"][-1]),
            )
        )
    return sorted(rows, key=lambda x: -x[1])


def plot(exp: dict, out_path: str = "bandit_results.png", window: int = 20) -> str:
    """画平均奖励与最优臂比例两条曲线，返回图片路径。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def smooth(x: np.ndarray) -> np.ndarray:
        if window <= 1:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    t = np.arange(1, exp["steps"] + 1)
    t_smooth = np.arange(window, exp["steps"] + 1)

    for name, r in exp["results"].items():
        axes[0].plot(t_smooth, smooth(r["avg_reward"]), label=name)
        axes[1].plot(t_smooth, smooth(r["optimal_rate"]), label=name)

    axes[0].axhline(exp["bandit"].best_prob, ls="--", c="gray", lw=1,
                    label=f"理论最优 p*={exp['bandit'].best_prob:.3f}")
    axes[0].set(title="平均奖励（滑动平均）", xlabel="步数 t", ylabel="奖励")
    axes[1].set(title="选中最优臂的比例", xlabel="步数 t", ylabel="比例")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{exp['results'].__len__()} 种老虎机策略对照"
        f"（{exp['bandit'].k} 臂，{exp['steps']} 步 × {exp['runs']} 次独立实验）"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多臂老虎机策略对照实验")
    parser.add_argument("--steps", type=int, default=1000, help="每次实验的拉杆次数")
    parser.add_argument("--runs", type=int, default=200, help="独立实验次数")
    parser.add_argument("--arms", type=int, default=10, help="摇臂数量")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true", help="不画图")
    args = parser.parse_args(argv)

    exp = run_experiment(steps=args.steps, runs=args.runs, k=args.arms, seed=args.seed)
    print(exp["bandit"])
    print(f"\n实验设置：{args.steps} 步 × {args.runs} 次独立实验\n")
    header = f"{'策略':<22}{'平均奖励':>10}{'选中最优臂':>12}{'累积遗憾':>12}"
    print(header)
    print("-" * len(header))
    for name, avg_r, opt, regret in summarize(exp):
        print(f"{name:<22}{avg_r:>10.3f}{opt:>11.1%}{regret:>12.1f}")
    print(f"\n随机瞎拉的期望奖励 = {exp['bandit'].probs.mean():.3f}，"
          f"理论最优 = {exp['bandit'].best_prob:.3f}")

    if not args.no_plot:
        path = plot(exp)
        print(f"图已保存：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
