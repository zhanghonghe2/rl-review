"""训练指标的记录、落盘与画图。

为什么单独一个文件
----------------
因为 **PPO 的调试 90% 靠看曲线**：`clip_frac`、`mean_ratio`、`approx_kl` 这三个数
比 loss 本身有用得多。把它们随手 `print` 出来是不会有人整理的，所以从第一天起
就要有一个能出图的东西。

设计上刻意保持"哑"：只做记账和画图，不认识模型、不认识 PPO。
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass, field

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "dsh-mpl-cache"))


@dataclass
class MetricLogger:
    """把每步的标量记下来，最后出 CSV + 图。

    支持**稀疏 key**（比如只在第 100、200 步记录的 `eval_reward`）：
    每个 key 各自记自己的 (step, value) 序列，缺的地方不补 0 —— 补 0 会在图上
    画出一条掉到 0 的假线，是最常见的画图误导。

    用法::

        logger = MetricLogger(out_dir=cfg.out_dir, tag=cfg.tag)
        logger.add(step=0, reward=0.12, clip_frac=0.0)
        logger.add(step=10, reward=0.31, clip_frac=0.08, eval_reward=0.25)
        logger.save_csv(); logger.plot()
    """

    out_dir: str = "runs"
    tag: str = "default"
    window: int = 20                      # 滑动平均窗口（画图用，不改原始数据）
    records: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def add(self, step: int | None = None, **metrics: float) -> None:
        """记一条。step 省略时自动用"下一条的序号"。"""
        if step is None:
            step = len(self.records)
        rec = {"step": int(step)}
        for k, v in metrics.items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"指标 {k!r} 不是标量（type={type(v).__name__}）：MetricLogger 只收实数。\n"
                    f"        列表/字典字段（比如 ppo_update 返回的 per_epoch 轨迹）"
                    f"请单独收集，不要塞进日志。"
                ) from e
        self.records.append(rec)

    def keys(self) -> list[str]:
        ks: list[str] = []
        for r in self.records:
            for k in r:
                if k != "step" and k not in ks:
                    ks.append(k)
        return ks

    def series(self, key: str) -> tuple[list[int], list[float]]:
        """取出某个 key 的 (steps, values)，只含真正记录过它的那些步。"""
        steps = [r["step"] for r in self.records if key in r]
        vals = [r[key] for r in self.records if key in r]
        return steps, vals

    def recent(self, key: str, n: int = 10) -> float:
        """最近 n 条的平均，用来打印。"""
        _, v = self.series(key)
        if not v:
            return float("nan")
        return sum(v[-n:]) / len(v[-n:])

    def smooth(self, key: str, window: int | None = None) -> tuple[list[int], list[float]]:
        """滑动平均（只用于画图，不动 self.records）。"""
        w = window or self.window
        s, v = self.series(key)
        if w <= 1 or len(v) < 2:
            return s, v
        out, acc = [], []
        for i, val in enumerate(v):
            acc.append(val)
            if len(acc) > w:
                acc.pop(0)
            out.append(sum(acc) / len(acc))
        return s, out

    # ------------------------------------------------------------------ #
    def _path(self, ext: str) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, f"{self.tag}_metrics.{ext}")

    def save_csv(self, path: str | None = None) -> str:
        path = path or self._path("csv")
        keys = self.keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step"] + keys)
            for r in self.records:
                w.writerow([r["step"]] + [r.get(k, "") for k in keys])
        return path

    def save_json(self, path: str | None = None) -> str:
        import json
        path = path or self._path("json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=1)
        return path

    def plot(self, path: str | None = None, title: str | None = None) -> str:
        """四格图：奖励 / loss / clip_frac+mean_ratio / approx_kl+entropy。

        缺的 key 会自动跳过（只在有数据时才画那一格）。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        path = path or self._path("png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        # ---- (0,0) 奖励：训练奖励（平滑）+ 评估奖励（不平滑，点线更合适）
        ax = axes[0, 0]
        if self.series("reward")[1]:
            s, v = self.smooth("reward")
            ax.plot(s, v, color="tab:blue", label=f"训练奖励（{self.window} 步滑动平均）")
            s2, v2 = self.series("reward")
            ax.plot(s2, v2, color="tab:blue", alpha=0.18, lw=1)
        if self.series("eval_reward")[1]:
            s, v = self.series("eval_reward")
            ax.plot(s, v, "o--", color="tab:green", ms=4, label="评估奖励（贪心）")
        ax.set(title="奖励", xlabel="step", ylabel="平均奖励")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # ---- (0,1) loss
        ax = axes[0, 1]
        for key, color in (("loss", "tab:red"), ("policy_loss", "tab:orange"),
                           ("value_loss", "tab:purple")):
            if self.series(key)[1]:
                s, v = self.smooth(key)
                ax.plot(s, v, color=color, label=key)
        ax.set(title="损失", xlabel="step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # ---- (1,0) clip_frac 与 mean_ratio（双轴）
        ax = axes[1, 0]
        if self.series("clip_frac")[1]:
            s, v = self.smooth("clip_frac")
            ax.plot(s, v, color="tab:red", label="clip_frac（被裁比例）")
            ax.axhline(0.3, ls=":", color="tab:red", lw=1)
            ax.text(s[0], 0.3, " 0.3 警戒线", fontsize=7, color="tab:red", va="bottom")
            ax.set_ylabel("clip_frac", color="tab:red")
        ax.set(title="裁剪与策略偏移", xlabel="step")
        ax.grid(alpha=0.3)
        if self.series("mean_ratio")[1]:
            ax2 = ax.twinx()
            s, v = self.smooth("mean_ratio")
            ax2.plot(s, v, color="tab:blue", label="mean_ratio")
            ax2.axhline(1.0, ls="--", color="gray", lw=1)
            ax2.set_ylabel("mean_ratio", color="tab:blue")
            ax2.set_ylim(min(0.9, min(v) - 0.05), max(1.1, max(v) + 0.05))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = (ax2.get_legend_handles_labels() if self.series("mean_ratio")[1] else ([], []))
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

        # ---- (1,1) approx_kl 与 entropy
        ax = axes[1, 1]
        if self.series("approx_kl")[1]:
            s, v = self.smooth("approx_kl")
            ax.plot(s, v, color="tab:brown", label="approx_kl")
            ax.axhline(0.02, ls=":", color="tab:brown", lw=1)
            ax.text(s[0], 0.02, " 0.02：通常该降 lr", fontsize=7, color="tab:brown", va="bottom")
        ax.set(title="KL 与熵", xlabel="step")
        ax.grid(alpha=0.3)
        if self.series("entropy")[1]:
            ax2 = ax.twinx()
            s, v = self.smooth("entropy")
            ax2.plot(s, v, color="tab:cyan", label="entropy")
            ax2.set_ylabel("entropy", color="tab:cyan")
        elif self.series("gen_len")[1]:
            ax2 = ax.twinx()
            s, v = self.smooth("gen_len")
            ax2.plot(s, v, color="tab:cyan", label="平均生成长度")
            ax2.set_ylabel("生成长度", color="tab:cyan")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels() if "ax2" in dir() else ([], [])
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

        fig.suptitle(title or f"PPO 训练 [{self.tag}]")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def summary(self, last_n: int = 10) -> str:
        keys = self.keys()
        if not keys:
            return f"[{self.tag}] 还没有记录"
        lines = [f"[{self.tag}] 共 {len(self.records)} 条记录，最近 {last_n} 条均值："]
        for k in keys:
            lines.append(f"    {k:<14} = {self.recent(k, last_n):+.4f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 自我演示（这个文件不需要"实现"，但也要能单独跑）
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    import random

    random.seed(0)
    logger = MetricLogger(out_dir="runs", tag="demo", window=10)
    for step in range(120):
        # 假装是一次训练：奖励上升、clip_frac 上升、KL 上升、熵下降
        t = step / 119
        logger.add(
            step=step,
            reward=0.05 + 0.6 * t + random.gauss(0, 0.05),
            loss=2.5 - 1.8 * t + random.gauss(0, 0.05),
            policy_loss=1.5 - 1.1 * t + random.gauss(0, 0.05),
            value_loss=1.2 - 0.7 * t + random.gauss(0, 0.05),
            clip_frac=0.01 + 0.22 * t + random.gauss(0, 0.01),
            mean_ratio=1.0 - 0.06 * t + random.gauss(0, 0.005),
            approx_kl=0.001 + 0.02 * t + random.gauss(0, 0.0005),
            entropy=1.7 - 0.8 * t + random.gauss(0, 0.02),
            gen_len=4.0 + 3.0 * t,
        )
        # 稀疏 key：只在每 40 步记录一次
        if step % 40 == 0:
            logger.add(step=step, eval_reward=0.1 + 0.5 * t)

    print(logger.summary())
    print("\n  reward 的 (步骤, 值) 条数:", len(logger.series("reward")[0]))
    print("  eval_reward 的条数:", len(logger.series("eval_reward")[0]), "← 稀疏 key 各自记各自")
    print("  save_csv →", logger.save_csv())
    print("  plot     →", logger.plot(title="PPO 训练演示（合成数据）"))
