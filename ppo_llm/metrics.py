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


#: 各平台常见的中文字体，**按优先级**排。服务器（Linux）只有 Noto/文泉驿 之类 ✓
CJK_FONTS = [
    "Arial Unicode MS", "PingFang SC", "PingFang HK", "Heiti TC", "Songti SC",  # macOS
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",              # Linux
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback",         # Linux
    "Microsoft YaHei", "SimHei",                                               # Windows
]


def pick_cjk_font(candidates: list[str] | None = None) -> str | None:
    """返回第一个**系统里真的装了**的中文字体名；一个都没有就返回 None ✓

    ★ 关键是"一个都没有时也不崩" ✓ —— 服务器上就是这种情况，
      而它在 Mac 上复现不了 ✗ 所以 candidates 参数化，让它可测 ✓
    """
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return None
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in (candidates if candidates is not None else CJK_FONTS):
        if name in installed:
            return name
    return None


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

    @classmethod
    def load_csv(cls, path: str, out_dir: str | None = None,
                 tag: str | None = None) -> "MetricLogger":
        """把 `save_csv` 的产物读回来，得到一个可直接 `plot()` 的 logger ✓

        ★ 用途：**服务器跑、本地画** ✓
          服务器上没有中文字体 ✗（图里中文是方框），但 CSV 是完整的 ✓
          把 CSV 拉回 Mac，用本地字体重画，中文就正常了 ✓
        """
        import os as _os
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lg = cls(out_dir=out_dir or _os.path.dirname(path) or ".",
                 tag=tag or _os.path.basename(path).replace("_metrics.csv", ""))
        for r in rows:
            step = int(r["step"])
            vals = {}
            for k, v in r.items():
                if k == "step" or v in ("", None):
                    continue
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
            if vals:
                lg.add(step=step, **vals)
        return lg

    def plot(self, path: str | None = None, title: str | None = None) -> str:
        """四格图：奖励 / loss / clip_frac+mean_ratio / approx_kl+entropy。

        缺的 key 会自动跳过（只在有数据时才画那一格）。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        picked = pick_cjk_font()
        plt.rcParams["font.sans-serif"] = ([picked] if picked else []) + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        path = path or self._path("png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        # ---- (0,0) 奖励：训练奖励（平滑）+ 评估奖励（不平滑，点线更合适）
        ax = axes[0, 0]
        if self.series("reward")[1]:
            s, v = self.smooth("reward")
            ax.plot(s, v, color="tab:blue", label=f"train reward ({self.window}-step avg)")
            s2, v2 = self.series("reward")
            ax.plot(s2, v2, color="tab:blue", alpha=0.18, lw=1)
        if self.series("eval_reward")[1]:
            s, v = self.series("eval_reward")
            ax.plot(s, v, "o--", color="tab:green", ms=4, label="eval reward (greedy)")
        ax.set(title="Reward", xlabel="step", ylabel="mean reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # ---- (0,1) loss
        ax = axes[0, 1]
        for key, color in (("loss", "tab:red"), ("policy_loss", "tab:orange"),
                           ("value_loss", "tab:purple")):
            if self.series(key)[1]:
                s, v = self.smooth(key)
                ax.plot(s, v, color=color, label=key)
        ax.set(title="Loss", xlabel="step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # ---- (1,0) clip_frac 与 mean_ratio（双轴）
        ax = axes[1, 0]
        if self.series("clip_frac")[1]:
            s, v = self.smooth("clip_frac")
            ax.plot(s, v, color="tab:red", label="clip_frac (clipped frac)")
            ax.axhline(0.3, ls=":", color="tab:red", lw=1)
            ax.text(s[0], 0.3, " 0.3 threshold", fontsize=7, color="tab:red", va="bottom")
            ax.set_ylabel("clip_frac", color="tab:red")
        ax.set(title="Clipping & policy shift", xlabel="step")
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
            ax.text(s[0], 0.02, " 0.02: usually lower lr", fontsize=7, color="tab:brown", va="bottom")
        ax.set(title="KL & entropy", xlabel="step")
        ax.grid(alpha=0.3)
        if self.series("entropy")[1]:
            ax2 = ax.twinx()
            s, v = self.smooth("entropy")
            ax2.plot(s, v, color="tab:cyan", label="entropy")
            ax2.set_ylabel("entropy", color="tab:cyan")
        elif self.series("gen_len")[1]:
            ax2 = ax.twinx()
            s, v = self.smooth("gen_len")
            ax2.plot(s, v, color="tab:cyan", label="mean gen length")
            ax2.set_ylabel("gen length", color="tab.cyan")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels() if "ax2" in dir() else ([], [])
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

        fig.suptitle(title or f"PPO training [{self.tag}]")
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
def _case_ascii_labels() -> None:
    """★ 图上的文字必须全是 ASCII —— 这条**能在 Mac 上验证服务器的问题** ✓

    服务器上没有中文字体 ✗ → 只要图里有一个汉字，就会变成方框 + 刷一堆警告 ✗
    所以：把 pick_cjk_font 打桩成"什么都没有"（服务器的实际情况 ✓），
    画一张四格全满的图，断言 **0 条 Glyph missing 警告** ✓
    """
    import tempfile
    import warnings

    import matplotlib
    matplotlib.use("Agg")

    import metrics as _m
    lg = _m.MetricLogger(out_dir=tempfile.mkdtemp(), tag="ascii")
    for i in range(30):
        lg.add(step=i, reward=0.4 + 0.01 * i, loss=1.0 - 0.02 * i,
               clip_frac=0.02 + 0.001 * i, mean_ratio=1.0,
               approx_kl=0.01 + 0.0005 * i, entropy=0.4, gen_len=120.0)
    for i in (0, 10, 20):
        lg.add(step=i, eval_reward=0.40 + 0.001 * i)

    orig = _m.pick_cjk_font
    _m.pick_cjk_font = lambda *a, **k: None          # 假装服务器：没有中文字体 ✓
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lg.plot(path=lg._path("png"))
    finally:
        _m.pick_cjk_font = orig

    bad = [str(x.message) for x in w
           if "missing from font" in str(x.message) or "Glyph" in str(x.message)]
    if bad:
        raise AssertionError(
            f"图上还有非 ASCII 文字（{len(bad)} 条 Glyph 警告）✗\n"
            f"        例如：{bad[0][:100]}\n"
            f"        → 把 plot() 里漏掉的那个标签也改成英文")
    print("    无中文字体下画四格全满的图：**0 条 Glyph 警告** ✓")
    print("    → 服务器上不需要装任何字体 ✓")


def _case_roundtrip() -> None:
    """★ save_csv → load_csv 必须**逐值一致** ✓

    这条用例的价值：读回来的数据如果悄悄变了（浮点格式、空值、列错位），
    我们会在"本地重画"时看到一张**错的图** ✗ 而且完全不会报错 ✗
    """
    import tempfile
    a = MetricLogger(out_dir=tempfile.mkdtemp(), tag="rt")
    a.add(step=0, reward=0.5, kl=0.0)
    a.add(step=1, reward=0.75, loss=-0.1254321, approx_kl=0.03125)
    a.add(step=2, eval_reward=0.453125)          # 稀疏 key（只有部分行有 ✓）
    path = a.save_csv()
    b = MetricLogger.load_csv(path)
    assert len(a.records) == len(b.records), (len(a.records), len(b.records))
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a.keys():
        sa, sb = a.series(k), b.series(k)
        assert sa[0] == sb[0], f"{k} 的 step 不一致"
        for x, y in zip(sa[1], sb[1]):
            if x != y:
                raise AssertionError(f"{k} 的值不一致：{x} != {y}")
    print(f"    {len(a.records)} 条 / {len(a.keys())} 个指标，逐值一致 ✓")
    print(f"    稀疏 key 也保住了（eval_reward 只有 1 条）✓")


def _case_font() -> None:
    """★ 字体挑选必须在「系统里一个中文字体都没有」时也不崩 ✓

    这条用例的价值：**服务器上的情况在这台 Mac 上复现不了** ✗
    只能靠传假的候选列表来复现 ✓
    """
    assert pick_cjk_font(["绝对不存在的字体XYZ"]) is None, "找不到时应返回 None，而不是报错"
    got = pick_cjk_font()
    assert got is None or isinstance(got, str)
    print(f"    本机挑中: {got!r}")
    print(f"    候选 {len(CJK_FONTS)} 个，含 Linux 的 Noto / 文泉驿 ✓")
    print(f"    全都不存在时返回 None ✓ 不崩（服务器装 fonts-noto-cjk 即可）")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        # 本地重画：python3 metrics.py runs/try15b_metrics.csv
        src = _sys.argv[1]
        lg = MetricLogger.load_csv(src)
        print(f"读了 {src}：{len(lg.records)} 条记录，{len(lg.keys())} 个指标")
        print("字体：", pick_cjk_font())
        out = lg.plot(title=f"{src.split('/')[-1]}（本地重画）")
        print("重画完成 →", out)
        _sys.exit(0)

    print("中文字体选择（跨平台）：")
    _case_font()
    print("\n图上文字是否纯 ASCII（服务器无字体的那种情况）：")
    _case_ascii_labels()
    print("\nCSV 往返（供本地重画）：")
    _case_roundtrip()

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
