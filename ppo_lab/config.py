"""PPO 训练的全部超参，集中在一个 dataclass 里。

为什么单独开一个文件
------------------
1. **超参散落在函数签名里 = 无法做扫描实验**。你上一版 `PPO(loraConfigActor,
   loraConfigCritic, epoches=3, batch_size=4)` 只有 4 个参数能被改，其余
   （γ、λ、clip、内层轮数、KL 系数……）全硬编码在循环里。想试 γ=0.99 和 γ=1.0
   的差别，得改代码；现在只要 `cfg.replace(gamma=0.99)`。
2. **默认值本身就是一份文档**。每个字段下面的注释写的是"为什么是这个值"和
   "改了会发生什么"，这比 README 更不容易过期。
3. **顶层零依赖**。这个文件不 import torch —— 于是它可以在任何地方被 import，
   也不会把 MPS/CUDA 的初始化拖进只想读配置的脚本里。

用法
----
>>> cfg = PPOConfig.preset("toy")
>>> cfg = cfg.replace(lr=1e-4, total_iters=50)   # 做扫描实验
>>> print(cfg.summary())
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Literal

AdvNorm = Literal["batch", "none", "per_sequence"]
KLEstimator = Literal["k1", "k3"]


@dataclass
class PPOConfig:
    # ===================================================================== #
    # 1. 采样：一次迭代采多少、采多长、怎么采
    # ===================================================================== #
    rollout_prompts: int = 32
    """每次迭代采多少个 prompt（走一次 rollout 的样本数）。

    这个值决定了"一批数据有多新鲜"。太小 → 梯度噪声大；太大 → 每次更新等得久，
    而且同一批数据被反复利用的收益递减。玩具任务 32~64 起步。
    """

    max_new_tokens: int = 12
    """生成的最大 token 数（玩具任务）；LLM 场景常见 256~1024。

    注意：它同时是"截断上限"。如果大量序列都撞到这个上限，说明模型没学会主动
    停止（或者 max_new_tokens 给太小），这会让 dones 的语义变成"被截断"而不是
    "自然终止"——两者对 bootstrap 的处理是不一样的。看日志里的截断率。
    """

    temperature: float = 1.0
    """采样温度。**PPO 需要探索，不要设成 0。**

    temperature=0 → 贪心 → 同一 prompt 永远生成同一条回答 → 优势估计的方差为 0、
    但**信号也为 0**（没有对比就没有优劣）。这是新手最常犯的错之一。
    """

    top_p: float | None = None
    """nucleus 采样。None = 关闭。只有 LLM 场景需要，玩具任务留 None。"""

    # ===================================================================== #
    # 2. 优化：内层复用几次、学习率、裁剪
    # ===================================================================== #
    inner_epochs: int = 4
    """**同一批 rollout 数据被复用几轮 —— 这就是 PPO 区别于 A2C 的核心。**

    A2C：采一批 → 更新一次 → 扔掉。
    PPO：采一批 → 更新 K 次 → 扔掉。

    为什么要复用：rollout 是整条链路里最贵的一步（要跑完整的前向生成），
    只用一次太浪费。为什么不能无限复用：数据是 on-policy 的，
    策略偏离 behavior policy 越远，ratio 越离谱，裁剪项越失效。
    K=4 是常见起点；观察 clip_frac，如果第 1 轮就 >0.3，说明 K 大了或者 lr 大了。
    """

    num_minibatches: int = 1
    """把 rollout batch 切成几份做 minibatch。1 = 不分（玩具任务够用）。

    必须能整除 rollout_prompts（__post_init__ 会检查）。
    """

    lr: float = 3e-4
    """学习率。PPO 比 SFT 敏感得多：太大会让 ratio 剧烈偏离 1，
    裁剪项把所有样本都裁掉 → 梯度恒为 0 → 看起来在训练，其实没学。"""

    grad_clip: float = 1.0
    """梯度范数裁剪。**PPO 的标配，不要省。**

    ratio 的梯度里有 exp()，advantage 又有重尾，不加裁剪很容易一步跑飞。
    """

    total_iters: int = 300
    """总迭代数（每次迭代 = 一次 rollout + inner_epochs 轮更新）。"""

    # ===================================================================== #
    # 3. PPO 目标：折扣、GAE、裁剪、KL、熵
    # ===================================================================== #
    gamma: float = 1.0
    """折扣因子。**对"一轮生成"的任务应该是 1.0，不是 0.99。**

    算一下就知道为什么：γ=0.99、序列长 200 时，终局奖励传到第一个 token 只剩
    0.99^200 ≈ 0.13 —— 87% 的奖励被"时间"吃掉了。LLM 的单轮生成没有真实的
    时间价值（不存在"早一步答对更值钱"），所以折扣只会带来**长度偏置**：
    短回答天然占便宜。

    只有当真存在"越快越好"的语义（比如每步有真实成本、或回合长度无界）时，
    才该用 γ<1。__post_init__ 会在 γ<1 时给你一条提醒。
    """

    lam: float = 0.95
    """GAE 的 λ。0 = 只看一步 TD（低方差高偏差）；1 = 回报减基线（无偏高方差）。

    ⚠️ 一个陷阱：如果奖励**只在终局**且没有任何逐步代价，那么同一序列内每个
    token 的 return 是同一个常数，λ 就完全不起作用了（数学上退化）。
    所以你的玩具任务一定要有"逐步代价"（比如每步 -0.01），否则你测不出 GAE。
    """

    clip_eps: float = 0.2
    """策略 ratio 的裁剪半径，PPO 论文的默认值，一般不用动。

    它的作用是"只在策略移动不大的范围内做梯度"。ratio 落在 [0.8, 1.2] 之外时
    该样本梯度为 0 —— 这也是为什么 clip_frac 是最好的健康度指标。
    """

    value_clip_eps: float | None = 0.2
    """价值函数的裁剪半径；None = 不裁剪。

    价值网络也会一步跑飞（它拟合的目标 returns 本身在随策略变化），裁剪能防。
    想让实现更简单可以设成 None 先跑通，但要知道少了什么。
    """

    value_coef: float = 0.5
    """价值损失的权重。critic 学得准，advantage 的方差才小。"""

    entropy_coef: float = 0.01
    """熵奖励系数（鼓励探索）。策略熵快速下降 = 过早收敛到单一行为，
    这时该调大它。但注意：它和"temperature 采样"是两个不同层面的探索。"""

    kl_coef: float = 0.0
    """对 reference 模型的 KL 惩罚系数。0 = 关闭。

    玩具任务没有"不能忘掉预训练能力"的顾虑，可以关。
    LLM 场景通常要开（0.01~0.05），否则模型会为了刷奖励而把语言能力训崩。
    """

    kl_estimator: KLEstimator = "k3"
    """KL 的估计量。**默认 k3，不要用 k1。**

    k1 = d（d = logπ_ref - logπ_current）：无偏但方差大，而且**可以是负数**，
         很多人被负 KL 搞晕。
    k3 = exp(d) - d - 1：恒 ≥ 0，方差小得多。见 Schulman 的
         "Approximating KL Divergence" (2020)。
    """

    adv_norm: AdvNorm = "batch"
    """优势归一化方式。**几乎总是用 "batch"。**

    "batch"        在整个 batch 的所有有效 token 上做 (A-μ)/σ ✅ 正确
    "none"         不归一化（critic 本身就是 baseline）
    "per_sequence" ⚠️ 反面教材，故意保留：在**每一条序列内部**归一化。
                   当奖励只在终局时，序列内每个 token 的 return 几乎相同，
                   减均值会把"这条回答好不好"这个信息整体减掉 ——
                   剩下来的只是 critic 在序列内的形状。
                   你上一版代码就是这么写的。想亲眼看到它有多坏，
                   跑一下 tests/test_adv_norm.py 里的"加常数不变"测试。
    """

    target_kl: float | None = None
    """可选的 KL 早停阈值：某轮内层更新的 approx_kl 超过它，就提前结束这一批的
    剩余内层轮数。这是 PPO 论文里的实用技巧，能让训练稳很多。
    None = 关闭。建议 M2 之后再实现，先跑通主循环。
    """

    # ===================================================================== #
    # 4. 异步：让"生产数据"和"消费数据"解耦
    # ===================================================================== #
    async_mode: bool = False
    """False = 同步：采一批 → 训一批 → 采下一批。
    True  = 异步：生产者持续采，消费者持续训，两者重叠。"""

    max_lag: int = 1
    """允许消费的数据最多落后当前策略多少个版本（staleness 上限）。

    这是异步真正的**质量旋钮**，不是 queue_size：
    lag=0 相当于同步；lag 越大吞吐越高，但数据越"过时"，
    ratio 偏离 1 越远、clip_frac 越高、有效梯度越少。
    M4 的核心实验就是画出 clip_frac 随 lag 变化的曲线。
    """


    threaded: bool = True
    """True  = 真开线程重叠（用来看墙钟时间真的变短）。
    False = 单线程、完全确定性的"一步滞后"（**单元测试用这个**，
            才能把异步和同步逐步对齐比较）。
    """

    # ===================================================================== #
    # 5. 复现性与日志
    # ===================================================================== #
    seed: int = 0
    device: str = "cpu"
    """'cpu' / 'mps' / 'cuda'。想自动挑一个用 PPOConfig.auto_device()。"""

    log_every: int = 10
    save_every: int = 100
    out_dir: str = "runs"
    """图和日志的落地目录。M4 的验收就是要在这里看到 lag 曲线。"""

    tag: str = "default"
    """实验名，用来区分不同 run 的输出文件。"""

    # ===================================================================== #
    # 校验：让错误尽早暴露，而不是在训练到第 200 步时才发现
    # ===================================================================== #
    def __post_init__(self) -> None:
        # ---- 硬错误：这些值直接让算法失去意义 ----
        if self.rollout_prompts < 1:
            raise ValueError("rollout_prompts 必须 ≥ 1")
        if self.inner_epochs < 1:
            raise ValueError("inner_epochs 必须 ≥ 1（不然就不是 PPO 了）")
        if not 0.0 <= self.clip_eps < 1.0:
            raise ValueError("clip_eps 必须在 [0, 1) 内")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma 必须在 (0, 1] 内")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam 必须在 [0, 1] 内")
        if self.lr <= 0:
            raise ValueError("lr 必须 > 0")
        if self.num_minibatches < 1:
            raise ValueError("num_minibatches 必须 ≥ 1")
        if self.rollout_prompts % self.num_minibatches != 0:
            raise ValueError(
                f"rollout_prompts({self.rollout_prompts}) 必须能被 "
                f"num_minibatches({self.num_minibatches}) 整除"
            )
        if self.adv_norm not in ("batch", "none", "per_sequence"):
            raise ValueError(f"未知的 adv_norm：{self.adv_norm}")
        if self.kl_estimator not in ("k1", "k3"):
            raise ValueError(f"未知的 kl_estimator：{self.kl_estimator}")
        if self.kl_coef > 0 and self.kl_estimator == "k1":
            warnings.warn(
                "kl_coef>0 且用 k1 估计量：k1 可以为负，KL 惩罚会间歇性变成奖励。"
                "除非你明确知道为什么要 k1，否则换 k3。",
                stacklevel=2,
            )
        if self.max_lag < 0:
            raise ValueError("max_lag 必须 ≥ 0")

        # ---- 提醒（不是错误）：这些值合理但通常不是你想要的 ----
        if self.gamma < 1.0:
            warnings.warn(
                f"gamma={self.gamma} < 1：单轮生成任务里折扣会造成长度偏置"
                f"（长度 200 时终局奖励只剩 {self.gamma ** 200:.1%}）。"
                "确认你真的需要折扣，否则用 1.0。",
                stacklevel=2,
            )
        if self.temperature == 0.0:
            warnings.warn(
                "temperature=0：贪心采样会让同一 prompt 永远产生同一条回答，"
                "优势估计退化（没有对比就没有优劣）。PPO 需要探索。",
                stacklevel=2,
            )
        # 注意这里**故意不检查** `threaded 只在 async_mode 下有效`：
        # async_mode 默认 False，如果在这儿警告，那么每次构造默认配置都会刷一条
        # 警告 —— 训练人忽略警告，比不警告更糟。这个检查应该放在真正用到它的地方
        # （async_runner 里），那里才知道它是否有意义。
        # 这是"校验"该遵守的规矩：**警告只在该条件真的会影响行为时才发。**

    # ===================================================================== #
    # 便捷方法
    # ===================================================================== #
    def replace(self, **kwargs) -> "PPOConfig":
        """返回改了若干字段的新配置（会重新跑一遍校验）。做扫描实验用。"""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """打印成人能看的分组表格。"""
        groups = {
            "采样": ["rollout_prompts", "max_new_tokens", "temperature", "top_p"],
            "优化": ["inner_epochs", "num_minibatches", "lr", "grad_clip", "total_iters"],
            "PPO 目标": ["gamma", "lam", "clip_eps", "value_clip_eps", "value_coef",
                        "entropy_coef", "kl_coef", "kl_estimator", "adv_norm", "target_kl"],
            "异步": ["async_mode", "max_lag", "threaded"],
            "杂项": ["seed", "device", "log_every", "save_every", "out_dir", "tag"],
        }
        lines = [f"PPOConfig [{self.tag}]"]
        for group, names in groups.items():
            lines.append(f"  {group}")
            for n in names:
                lines.append(f"    {n:<18} = {getattr(self, n)!r}")
        return "\n".join(lines)

    # ===================================================================== #
    # 预设 + 设备探测
    # ===================================================================== #
    @staticmethod
    def auto_device() -> str:
        """自动挑设备。这里**延迟** import torch —— 保持本模块顶层零依赖。"""
        try:
            import torch
        except ImportError:  # pragma: no cover
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def preset(cls, name: str) -> "PPOConfig":
        """几套开箱即用的配置，省得每次手填。"""
        if name == "toy":
            # 玩具任务：秒级跑完，关掉 KL（没有"别忘掉预训练"的顾虑）。
            # device 强制 cpu：这个任务在 CPU 上就是毫秒级，用 MPS 一点不快，
            # 却会引入"模型在 A 设备、batch 在 B 设备"的对齐风险。
            # （真要用 MPS，得让 rollout 把 prompt 也搬到模型所在设备 —— 那是 M5 的事。）
            # lr=1e-3 是**实测**出来的：300 步约 7 秒，评估准确率从 2.3% → ~100%。
            # （用 3e-3 会更快学会但随后崩掉：熵塌到 0、生成长度失控、approx_kl>0.15。
            #   这正是要看 clip_frac / approx_kl 曲线的原因。）
            # target_kl=0.02 能让曲线更平滑，但那是 ppo_update 的可选特性，先留 None。
            return cls(tag="toy", device="cpu", total_iters=300, lr=1e-3,
                       rollout_prompts=32, max_new_tokens=12, kl_coef=0.0)
        if name == "toy_debug":
            # 冒烟测试：几十秒内验证代码能跑通、ratio 起手为 1
            # max_new_tokens 必须 ≥ 任务最大答案长度 + 1：
            # 数 1 的任务最长 prompt 8 个符号 → 答案最多 [m]*8 + [STOP] = 9 个 token。
            # 给 8 的话 k=8 的题永远不可能答对（预算不够输出 STOP），
            # 会变成"有些题怎么训都错"这种很难查的困惑。
            return cls(tag="toy_debug", device="cpu", total_iters=5,
                       rollout_prompts=8, max_new_tokens=12, inner_epochs=2,
                       kl_coef=0.0, log_every=1)
        if name == "toy_async":
            # 异步：lag 是要扫的自变量
            return cls.preset("toy").replace(
                tag="toy_async", async_mode=True, max_lag=2, threaded=False
            )
        if name == "llm":
            # 真实模型：开 KL（防止把语言能力训崩），关熵奖励（KL 已经在约束）
            return cls(
                tag="llm", device=cls.auto_device(), rollout_prompts=8,
                max_new_tokens=256, inner_epochs=2, lr=1e-5, num_minibatches=1,
                kl_coef=0.02, entropy_coef=0.0, total_iters=200, log_every=1,
            )
        raise ValueError(f"未知预设：{name}（可选：toy / toy_debug / toy_async / llm）")


if __name__ == "__main__":
    # 直接跑这个文件，把每套预设打出来看一眼
    for name in ("toy_debug", "toy", "toy_async", "llm"):
        print(PPOConfig.preset(name).summary())
        print()

    # 扫描实验的写法演示
    base = PPOConfig.preset("toy")
    for gamma in (1.0, 0.99):
        for lag in (0, 1, 4):
            cfg = base.replace(gamma=gamma, max_lag=lag, tag=f"g{gamma}_lag{lag}")
            print(f"{cfg.tag:<16} gamma={cfg.gamma} max_lag={cfg.max_lag}")
