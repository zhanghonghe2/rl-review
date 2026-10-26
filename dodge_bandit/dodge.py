"""躲避老虎（Dodge the Tiger）：网格追逃任务 + 表格 Q-learning。

规则
----
* 8×8 网格，智能体（A）从左下角出发，目标是右上角的奶酪（C）。
* 老虎（T）从右下角出发，**每 2 步**朝智能体走一格（曼哈顿距离的贪心追踪）。
* 被老虎抓到：奖励 -1，回合结束；吃到奶酪：奖励 +1，回合结束。
* 每走一步扣 0.02（时间成本），超过 max_steps 还没吃到就算超时。

为什么这个任务值得一学：老虎的移动节奏是固定的（隔一步一动），所以最优策略不是
"永远远离老虎"，而是**卡着老虎的节奏横向拉扯、然后在它转身的那一步冲刺**。
这类"利用环境周期性"的策略只能靠试错学出来 —— 正是强化学习要干的事。

状态 = (鼠行, 鼠列, 虎行, 虎列, 步数奇偶)，共 8*8*8*8*2 = 8192 个状态；
动作 = 上 / 下 / 左 / 右。表格 Q-learning 足够，秒级训练完。

用法：
    python dodge.py                 # 训练 + 评估 + 出图
    python dodge.py --render        # 额外打印一条贪心策略的 ASCII 轨迹
    python main.py dodge            # 统一入口
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "dsh-mpl-cache"))

# 动作：上、下、左、右（行号向下增大）
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["上", "下", "左", "右"]


class DodgeTigerEnv:
    """网格追逃环境。状态是元组，方便直接当 Q 表索引。"""

    def __init__(
        self,
        size: int = 8,
        max_steps: int = 80,
        step_cost: float = -0.02,
        tiger_period: int = 2,
        allow_diagonal: bool = False,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.tiger_period = tiger_period
        self.allow_diagonal = allow_diagonal
        self.n_actions = len(ACTIONS)
        self.start = (size - 1, 0)          # 左下角
        self.goal = (0, size - 1)           # 右上角奶酪
        self.tiger_start = (size - 1, size - 1)  # 右下角
        self.state_shape = (size, size, size, size, tiger_period)
        self.reset()

    # ---------------------------------------------------------------- 辅助 --
    def _clip(self, pos: tuple[int, int]) -> tuple[int, int]:
        r = min(max(pos[0], 0), self.size - 1)
        c = min(max(pos[1], 0), self.size - 1)
        return (r, c)

    def _state(self) -> tuple[int, int, int, int, int]:
        return (self.agent[0], self.agent[1], self.tiger[0], self.tiger[1],
                self.t % self.tiger_period)

    def _tiger_step(self) -> None:
        """老虎朝智能体走一格：优先缩小距离更大的那个轴。"""
        dr = np.sign(self.agent[0] - self.tiger[0])
        dc = np.sign(self.agent[1] - self.tiger[1])
        if abs(self.agent[0] - self.tiger[0]) >= abs(self.agent[1] - self.tiger[1]):
            if dr != 0:
                self.tiger = (self.tiger[0] + int(dr), self.tiger[1])
                return
        if dc != 0:
            self.tiger = (self.tiger[0], self.tiger[1] + int(dc))

    # ------------------------------------------------------------ Gym 风格 --
    def reset(self) -> tuple[int, int, int, int, int]:
        self.agent = self.start
        self.tiger = self.tiger_start
        self.t = 0
        self.done = False
        self.outcome = ""      # win / caught / timeout
        self.trajectory = [(self.agent, self.tiger)]
        return self._state()

    def step(self, action: int) -> tuple[tuple[int, int, int, int, int], float, bool, dict]:
        assert not self.done, "回合已结束，请先 reset()"
        dr, dc = ACTIONS[action]
        self.agent = self._clip((self.agent[0] + dr, self.agent[1] + dc))
        reward = self.step_cost

        # 主动撞上老虎
        if self.agent == self.tiger:
            self.outcome = "caught"
            self.done = True
            self.trajectory.append((self.agent, self.tiger))
            return self._state(), -1.0, True, {"outcome": self.outcome}

        # 吃到奶酪
        if self.agent == self.goal:
            self.outcome = "win"
            self.done = True
            self.trajectory.append((self.agent, self.tiger))
            return self._state(), 1.0, True, {"outcome": self.outcome}

        self.t += 1
        if self.t % self.tiger_period == 0:      # 老虎只在固定的节奏上动
            self._tiger_step()

        if self.tiger == self.agent:
            self.outcome = "caught"
            self.done = True
            self.trajectory.append((self.agent, self.tiger))
            return self._state(), -1.0, True, {"outcome": self.outcome}

        if self.t >= self.max_steps:
            self.outcome = "timeout"
            self.done = True
            self.trajectory.append((self.agent, self.tiger))
            return self._state(), reward, True, {"outcome": self.outcome}

        self.trajectory.append((self.agent, self.tiger))
        return self._state(), reward, False, {}


# --------------------------------------------------------------------------- #
# Q-learning
# --------------------------------------------------------------------------- #
def q_learning(
    env: DodgeTigerEnv,
    episodes: int = 4000,
    alpha: float = 0.15,
    gamma: float = 0.97,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 0,
    verbose_every: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """表格 Q-learning：off-policy，用 max 做自举。"""
    rng = np.random.default_rng(seed)
    Q = np.zeros(env.state_shape + (env.n_actions,), dtype=np.float64)
    returns = np.zeros(episodes)
    outcomes = np.empty(episodes, dtype=object)

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        # ε 线性退火：先大胆探索，再逐步收敛到贪心
        eps = eps_start + (eps_end - eps_start) * ep / max(episodes - 1, 1)
        while not done:
            if rng.random() < eps:
                action = int(rng.integers(env.n_actions))
            else:
                q = Q[state]
                action = int(np.argmax(q + rng.random(env.n_actions) * 1e-9))
            next_state, reward, done, info = env.step(action)
            target = reward if done else reward + gamma * float(np.max(Q[next_state]))
            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
            ep_return += reward
        returns[ep] = ep_return
        outcomes[ep] = info.get("outcome", "timeout")
        if verbose_every and (ep + 1) % verbose_every == 0:
            recent = outcomes[max(0, ep - verbose_every + 1): ep + 1]
            win = np.mean([o == "win" for o in recent])
            print(f"  回合 {ep + 1:>5} | 近 {verbose_every} 回合胜率 {win:6.1%} "
                  f"| 平均回报 {returns[max(0, ep - verbose_every + 1): ep + 1].mean():+.3f}")

    wins = np.array([o == "win" for o in outcomes], dtype=np.float64)
    caught = np.array([o == "caught" for o in outcomes], dtype=np.float64)
    return Q, {"returns": returns, "wins": wins, "caught": caught}


def greedy_values(Q: np.ndarray) -> np.ndarray:
    return Q.max(axis=-1)


# --------------------------------------------------------------------------- #
# 评估 / 渲染
# --------------------------------------------------------------------------- #
def evaluate(
    env: DodgeTigerEnv,
    Q: np.ndarray | None,
    episodes: int = 500,
    seed: int = 123,
) -> dict[str, float]:
    """跑若干回合：Q=None 表示随机策略，用来当基线。"""
    rng = np.random.default_rng(seed)
    stats = {"win": 0, "caught": 0, "timeout": 0}
    returns = []
    steps_to_win = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            if Q is None:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(Q[state]))
            state, reward, done, info = env.step(action)
            ep_return += reward
        stats[info["outcome"]] += 1
        returns.append(ep_return)
        if info["outcome"] == "win":
            steps_to_win.append(len(env.trajectory) - 1)
    return {
        "win_rate": stats["win"] / episodes,
        "caught_rate": stats["caught"] / episodes,
        "timeout_rate": stats["timeout"] / episodes,
        "avg_return": float(np.mean(returns)),
        "avg_steps_to_win": float(np.mean(steps_to_win)) if steps_to_win else float("nan"),
    }


def rollout(env: DodgeTigerEnv, Q: np.ndarray, max_frames: int = 100) -> list:
    """跑一条贪心轨迹，返回 [(agent, tiger, action_name), ...]。"""
    env.reset()
    frames = []
    done = False
    while not done and len(frames) < max_frames:
        state = env._state()
        action = int(np.argmax(Q[state]))
        state, _, done, info = env.step(action)
        frames.append((env.trajectory[-1][0], env.trajectory[-1][1], ACTION_NAMES[action]))
    return frames


def _cell(agent, tiger, goal, r, c) -> str:
    if (r, c) == agent:
        return "A"
    if (r, c) == tiger:
        return "T"
    if (r, c) == goal:
        return "C"
    return "."


def ascii_frame(env: DodgeTigerEnv, agent, tiger) -> str:
    n = env.size
    rows = []
    for r in range(n):
        rows.append(" ".join(_cell(agent, tiger, env.goal, r, c) for c in range(n)))
    return "\n".join(rows)


def render_episode(env: DodgeTigerEnv, frames: list, every: int = 4) -> str:
    """把一条轨迹里每隔 every 步的画面并排打印出来。"""
    out = []
    picks = list(range(0, max(len(frames) - 1, 1), every)) + [len(frames) - 1]
    for idx in picks:
        agent, tiger, action = frames[idx]
        out.append(f"[第 {idx + 1} 步] 鼠上一步动作：{action}")
        out.append(ascii_frame(env, agent, tiger))
        out.append("")
    out.append(f"结局：{env.outcome}，共 {len(frames)} 步；A=鼠  T=虎  C=奶酪")
    return "\n".join(out)


def plot(env: DodgeTigerEnv, history: dict, Q: np.ndarray,
         out_path: str = "dodge_results.png", window: int = 100) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def smooth(x: np.ndarray) -> np.ndarray:
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(smooth(history["returns"]), color="tab:blue", label="回合回报（滑动平均）")
    ax.set(xlabel="回合", ylabel="每回合累积回报", title="学习曲线")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(smooth(history["wins"]), color="tab:green", label="吃奶酪成功率")
    ax2.set_ylabel("成功率")
    ax2.set_ylim(-0.02, 1.02)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    # 右图：贪心策略跑出来的那条轨迹
    ax = axes[1]
    frames = rollout(env, Q)
    n = env.size
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    ax.set_title("贪心策略的一条轨迹")

    agent_path = np.array([f[0] for f in frames])
    tiger_path = np.array([f[1] for f in frames])
    ax.plot(agent_path[:, 1], agent_path[:, 0], "-o", ms=3, lw=1.4,
            color="tab:blue", label="鼠的路径")
    ax.plot(tiger_path[:, 1], tiger_path[:, 0], "--s", ms=3, lw=1.0,
            color="tab:red", alpha=0.8, label="虎的路径")
    ax.scatter([env.goal[1]], [env.goal[0]], marker="*", s=220,
               color="goldenrod", edgecolor="k", zorder=5, label="奶酪")
    ax.scatter([env.start[1]], [env.start[0]], marker="o", s=60,
               color="tab:blue", edgecolor="k", zorder=5)
    ax.scatter([env.tiger_start[1]], [env.tiger_start[0]], marker="s", s=60,
               color="tab:red", edgecolor="k", zorder=5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("躲避老虎：8×8 网格追逃（老虎每 2 步动一次）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="躲避老虎：表格 Q-learning 追逃任务")
    parser.add_argument("--size", type=int, default=8, help="网格边长")
    parser.add_argument("--episodes", type=int, default=4000, help="训练回合数")
    parser.add_argument("--tiger-period", type=int, default=2, help="老虎每几步动一次")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="打印贪心策略的 ASCII 轨迹")
    parser.add_argument("--no-plot", action="store_true", help="不画图")
    args = parser.parse_args(argv)

    env = DodgeTigerEnv(size=args.size, tiger_period=args.tiger_period)
    print(f"环境：{args.size}×{args.size} 网格，状态数 {int(np.prod(env.state_shape))}，"
          f"动作数 {env.n_actions}，老虎每 {args.tiger_period} 步动一次")
    print(f"训练 {args.episodes} 回合……")
    Q, history = q_learning(env, episodes=args.episodes, seed=args.seed,
                            verbose_every=max(args.episodes // 8, 0))

    rand = evaluate(env, None, episodes=args.eval_episodes, seed=args.seed + 7)
    greedy = evaluate(env, Q, episodes=args.eval_episodes, seed=args.seed + 7)

    print(f"\n评估（各 {args.eval_episodes} 回合，贪心策略 vs 随机瞎走）")
    header = f"{'策略':<12}{'吃到奶酪':>10}{'被抓':>10}{'超时':>10}{'平均回报':>12}{'平均步数':>10}"
    print(header)
    print("-" * len(header))
    for name, s in (("随机瞎走", rand), ("Q-learning", greedy)):
        print(f"{name:<12}{s['win_rate']:>10.1%}{s['caught_rate']:>10.1%}"
              f"{s['timeout_rate']:>10.1%}{s['avg_return']:>12.3f}"
              f"{s['avg_steps_to_win']:>10.1f}")
    print(f"\n训练后期（最后 500 回合）平均回报 = {history['returns'][-500:].mean():+.3f}，"
          f"成功率 = {history['wins'][-500:].mean():.1%}")

    if args.render:
        print("\n一条贪心策略轨迹：")
        frames = rollout(env, Q)
        print(render_episode(env, frames, every=max(len(frames) // 4, 1)))

    if not args.no_plot:
        print(f"图已保存：{plot(env, history, Q)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
