"""画出 PPO 裁剪目标的形状：A > 0 与 A < 0 时，目标函数随 r 怎么变。

    python3 plot_clip.py        # 生成 clip_objective.png

上排 = 目标函数本身（L、L_clip、min 三条曲线叠在一起）
下排 = 它的梯度 dObj/dr。**下排才是重点**：PPO 的裁剪不是把梯度变小，
       而是把某一段的梯度**恰好变成 0**，而且"哪一段被清零"取决于 A 的符号。

对照 `loss.py::_case_b()` 的四个手算数字：
    A=+1, r=3.0 → 目标 = 1.2（被裁）      A=+1, r=0.5 → 目标 = 0.5（没裁）
    A=-1, r=3.0 → 目标 = -3.0（没裁）     A=-1, r=0.5 → 目标 = -0.8（被裁）
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "dsh-mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 0.2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_objective.png")


def objective(r: np.ndarray, A: float, eps: float = EPS) -> tuple[np.ndarray, ...]:
    """返回 (L, L_clip, min(L, L_clip), dObj/dr)。"""
    L = r * A
    L_clip = np.clip(r, 1.0 - eps, 1.0 + eps) * A
    obj = np.minimum(L, L_clip)

    # 解析梯度：A>0 时 r>1+ε 被裁；A<0 时 r<1-ε 被裁（区间内梯度恒为 A）
    clipped = (r > 1.0 + eps) if A > 0 else (r < 1.0 - eps)
    grad = np.where(clipped, 0.0, A)
    return L, L_clip, obj, grad


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    r = np.linspace(0.2, 3.0, 2001)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))

    for col, (A, who) in enumerate([(+1.0, "A = +1（好动作）"), (-1.0, "A = −1（坏动作）")]):
        L, L_clip, obj, grad = objective(r, A)
        if A > 0:
            lo, hi, zone_txt = 1.0 + EPS, 3.0, f"r > {1 + EPS:.1f}：梯度 = 0\n（概率已抬够，别再贪）"
        else:
            lo, hi, zone_txt = 0.2, 1.0 - EPS, f"r < {1 - EPS:.1f}：梯度 = 0\n（概率已砍够，别再砍）"

        # ---------------- 上排：目标函数 ----------------
        ax = axes[0, col]
        ax.axvspan(lo, hi, color="tab:red", alpha=0.08)
        ax.axvspan(1 - EPS, 1 + EPS, color="tab:green", alpha=0.08)
        ax.axvline(1.0, color="k", lw=0.8, ls="-.")
        ax.plot(r, L, "--", color="tab:gray", lw=1.6, label="L = r·A（未裁剪）")
        ax.plot(r, L_clip, ":", color="tab:orange", lw=2.0, label="L_clip = clip(r)·A")
        ax.plot(r, obj, "-", color="tab:blue", lw=3.0, label="min(L, L_clip) ← 实际优化的目标")
        ax.text(1.0, ax.get_ylim()[0], " 行为策略\n r=1", fontsize=8, va="bottom", color="k")
        ax.text((lo + hi) / 2, (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2, zone_txt,
                ha="center", va="center", fontsize=9, color="tab:red",
                bbox=dict(fc="white", ec="tab:red", alpha=0.85, boxstyle="round,pad=0.3"))
        ax.set_title(f"目标函数　{who}", fontsize=11)
        ax.set_ylabel("目标值（越大越好）")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right" if A > 0 else "upper right")

        # ---------------- 下排：梯度 ----------------
        ax = axes[1, col]
        ax.axvspan(lo, hi, color="tab:red", alpha=0.08)
        ax.axvspan(1 - EPS, 1 + EPS, color="tab:green", alpha=0.08)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.plot(r, grad, "-", color="tab:purple", lw=2.6)
        ax.fill_between(r, 0, grad, color="tab:purple", alpha=0.18)
        ax.axvline(1.0, color="k", lw=0.8, ls="-.")
        # 文字放轴上方、断点注释放轴下方，避免两者重叠
        ax.text((lo + hi) / 2, 0.42, "梯度 = 0\n这一段完全学不到东西",
                ha="center", va="center", fontsize=9, color="tab:red")
        step_at = 1 + EPS if A > 0 else 1 - EPS
        ax.annotate(f"断点 r={step_at:.1f}", xy=(step_at, 0),
                    xytext=(step_at + 0.55, -0.62),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
        ax.set_title(f"梯度 dObj/dr　{who}", fontsize=11)
        ax.set_xlabel("r = π_new(a|s) / π_old(a|s)")
        ax.set_ylabel("梯度")
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.25)

    fig.suptitle(f"PPO 裁剪目标 ε={EPS}：不是「把步子变小」，而是「把走太远的样本踢出学习」",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=140)
    print(f"已保存：{OUT}")

    # 顺带把 loss.py::_case_b 的四个数字打印出来对账
    print("\n和 _case_b 的四个手算数字对账：")
    for A in (+1.0, -1.0):
        for rr in (3.0, 0.5):
            _, _, obj, grad = objective(np.array([rr]), A)
            print(f"  A={A:+.0f}  r={rr}:  目标 = {obj[0]:+.4f}   梯度 = {grad[0]:+.1f}")


if __name__ == "__main__":
    main()
