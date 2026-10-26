"""异步 PPO：把"生产数据"和"消费数据"解耦（M4）。

为什么异步不是"更快的同步"
--------------------------
异步**唯一的本质区别**是：训练用的数据由**更旧的策略**产生（staleness，滞后）。

* 收益：生成（rollout）和训练重叠 → 墙钟时间下降、设备利用率上升
* 代价：`ratio = exp(logπ_new - logπ_old)` 起手不再等于 1 → 裁剪项开始吃掉梯度
        （有效 batch 变小）→ 同样的步数学到的东西更少

所以异步的质量旋钮是 **lag**（滞后了几个版本），而它**必须被显式度量**。
这个文件的目标就是让 lag 从一个隐含副作用，变成一个**可观测、可画曲线**的量。

两种模式
--------
| | `threaded=False`（确定性） | `threaded=True`（真重叠） |
|---|---|---|
| 实现 | 单线程模拟：先补队列再取最旧的 | 后台线程 + 有界队列 |
| 用途 | **单元测试 / lag 曲线实验**（可复现，lag 精确可控） | 看墙钟时间真的变短 |
| lag | 由 `max_lag` 精确决定 | 不可控，但可观测 |

权重同步
--------
生产者必须用"某一版本的权重"生成，不能和训练器读同一份参数（会撕裂）。
这里用最朴素的办法：**生产者持有自己的模型副本**，每次生成前在锁内
`load_state_dict` 拷一份当前权重。真实系统里这一步是 NCCL broadcast 或
独立进程 + 参数服务器 —— 机制不同，**版本号语义完全相同**。

关于 `assert_ratio_starts_at_one`
--------------------------------
那个断言只在**同步**下成立（数据必须由当前策略产生）。异步下 lag ≥ 1 时
它**应该**失败 —— 这正是异步的本质。所以 `ppo_update` 需要接受一个开关
`assert_fresh`（默认 True；异步传 False）。
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque

import torch

from batch import RolloutBatch
from config import PPOConfig
from metrics import MetricLogger
from toy import CountOnesTask, TinyActorCritic, rollout
from trainer import RolloutSource, check_source     # 接口定义在**消费者**那一侧（trainer）


# =========================================================================== #
# 1. 数据来源：同步 / 异步两种实现（这就是全部区别）
# =========================================================================== #
# `RolloutSource`（接口声明）和 `check_source`（运行时契约）都定义在 trainer.py ——
# 因为**用它的是训练循环**，接口就该和消费者待在一起。这里 import 进来只是为了
# 让你在一个文件里就能看全"数据源"这一族。换句话说：见 trainer.py。
class SyncSource:
    """同步：每轮用**当前**权重现场采一批。lag 恒为 0。"""

    def __init__(self, task: CountOnesTask, model: TinyActorCritic, cfg: PPOConfig) -> None:
        self.task, self.model, self.cfg = task, model, cfg
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.version = 0
        self.produced = 0

    def get(self) -> tuple[RolloutBatch, int]:
        b = rollout(self.task, self.model, self.cfg, generator=self.gen)
        b.version = self.version
        self.produced += 1
        return b, 0

    def advance(self) -> None:
        self.version += 1

    def close(self) -> None:
        pass


class AsyncSource:
    """异步：生产者**跑在消费者前面**，消费者拿到的是不同版本的策略产生的数据。

    参数语义（两个都是队列相关的旋钮，别混）
    ---------------------------------------
    * `max_lag`  —— **队列深度上限**：生产者最多领先消费者几个批次。
                   它就是 lag 的来源（队列越深 → 数据越旧）。
    队列容量 = `max_lag + 1`（**两种模式一致**）：生产者最多领先 max_lag 个批次，
    队列满就等（背压）。所以 `max_lag` 既是"允许的滞后"，也是"队列深度"。

    三份副本的分工（**这是并发正确性的全部**）
    ----------------------------------------
    | 副本 | 谁写 | 谁读 | 作用 |
    |---|---|---|---|
    | `model` | **只有训练器（主线程）** | 只有训练器 | 被 `optimizer.step()` 原地更新 |
    | `published` | 训练器，**在锁内** | 生产者，**在锁内** | (权重, 版本号) 的原子快照 |
    | `worker` | **只有生产者线程** | 只有生产者线程 | 本次 rollout 用的稳定副本 |

    为什么必须这样：

    * **生产者绝不读 `model`** —— `optimizer.step()` 是逐个张量改参数的，
      边读边改会拿到"一半新一半旧"的**撕裂权重**（这份权重在任何时刻都不存在）。
    * **`published` 在锁里同时更新权重和版本号** —— 否则生产者可能读到
      "新权重 + 旧编号"，于是这批数据的 logprob 不对应任何真实版本，
      `lag` 度量的东西就不存在了。
    * **`worker` 只归生产者** —— 发布新权重不会打断一次正在进行的 rollout。

    真实系统里 `published` 这一步不是 `deepcopy`，而是 **NCCL broadcast 到推理引擎
    自己的 buffer**（vLLM 的 `update_weights`）—— 机制不同，**"发布必须原子"这个
    约束完全相同**。这里用三份副本，只是为了让玩具能看清结构。

    字段
    ----
    `version`           : 当前训练到的版本号（`advance()` 每次 +1）
    `published_version` : `published` 里那份权重的版本号
    `worker_version`    : `worker` 里那份权重的版本号（= 批次被标注的版本）
    `lock`              : 保护 published 的读写 + 版本号
    `queue`             : deque，待训练的批次，每个带自己的 `.version`
    """

    def __init__(self, task: CountOnesTask, model: TinyActorCritic, cfg: PPOConfig) -> None:
        self.task, self.model, self.cfg = task, model, cfg
        self.max_lag = cfg.max_lag          # 唯一的滞后旋钮：队列深度上限
        self.threaded = cfg.threaded
        self.lock = threading.Lock()
        self.queue: deque[RolloutBatch] = deque()
        # 生产者用自己的模型副本 + 自己的随机源（torch.Generator 不是线程安全的）
        # 三份副本，见类 docstring 的分工表
        self.worker = copy.deepcopy(model)          # 生产者独占
        self.published = copy.deepcopy(model)       # 训练器在锁内发布，生产者在锁内读
        self.published_version = 0
        self.worker_version = 0
        self.gen = torch.Generator().manual_seed(cfg.seed)   # 和 SyncSource 一致，便于等价性测试
        self.version = 0
        self.produced = 0
        self.dropped = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # ★ 留给你实现的两个方法（异步状态机就在这里）
    # ------------------------------------------------------------------ #
    def _produce_once(self) -> None:
        """生成一批，带上"生成时的版本号"，放进队列。

        ⚠️ **谁调用它，取决于模式**（这里是最容易死锁的地方）：
          * `threaded=False`：由 `get()` 调用（单线程，生产-消费交替）
          * `threaded=True` ：**只由生产者线程 `_loop()` 调用**。
            `get()` 绝不能也来生产 —— 两个生产者 + 一个有界队列 = 死锁：
            它们会互相等对方腾出空间，而唯一能腾空间的消费者正卡在 `_produce_once` 里。

        一句话：**有界队列 + 阻塞生产者 ⇒ 消费者只能消费。**

        要做三件事：

        1. **拿到一致的权重快照**
           * `threaded=False`：单线程，直接用 `self.model` 现场采（不会有撕裂）
           * `threaded=True`：必须在 `self.lock` 里把 `self.model` 的权重拷进
             `self.worker`：`self.worker.load_state_dict(self.model.state_dict())`，
             然后**用 `self.worker`** 生成。版本号也要在同一个锁里读出来 ——
             否则你会拿到"半个版本的权重 + 另一个版本的编号"。
        2. **生成**：`rollout(self.task, <哪个模型>, self.cfg, generator=self.gen)`
           （注意生成用的 max_new_tokens / temperature 都来自 cfg）
        3. **打标 + 入队**：`b.version = <快照时的版本号>`，然后
           * 确定性模式：直接 `self.queue.append(b)`
           * 线程模式：入队即可（队列上限由 `_loop()` 把关，你不用管）

        另外把 `self.produced += 1`。

        **不要在这里做背压**（"队列满了就等"）—— 队列上限由 `_loop()` 负责。
        为什么：这个方法在确定性模式下是 `get()` 调的。如果它内部会阻塞，
        那么在线程模式下就会出现"消费者卡在 `_produce_once` 里等空间，而唯一能
        腾空间的就是它自己"的**死锁**。（我真踩过，挂了 10 分钟才反应过来。）
        让**唯一的生产者**（`_loop`）独占阻塞权，就不会有这个问题。

        为什么需要快照而不是直接用 self.model：
            训练线程随时在 `optimizer.step()` 改参数。边读边改会拿到**撕裂的权重**
            （一半新一半旧），生成的 logprob 就没有对应的"版本"可言了。
        """

        if self.threaded:
            # 只读 self.published（训练器在锁内发布的快照），**绝不读 self.model**。
            # 发布版本比手里这份新 → 在锁内换一次；换完在**锁外**用自己的副本 rollout。
            with self.lock:
                if self.published_version != self.worker_version:
                    self.worker.load_state_dict(self.published.state_dict())
                    self.worker_version = self.published_version
                ver = self.worker_version      # ← 标的是"手里这份权重"的版本，不是训练器的版本
            b = rollout(self.task, self.worker, self.cfg, self.gen)
        else:
            ver = self.version
            b = rollout(self.task, self.model, self.cfg, self.gen)
        b.version = ver
        self.produced += 1
        self.queue.append(b)



        # b = rollout(self.task, self.model, self.cfg, self.gen)
        # b.version = self.version
        # self.produced += 1
        # self.queue.append(b)
        

    def get(self) -> tuple[RolloutBatch, int]:
        """取一批来训练，返回 `(batch, lag)`。

        要做的：

        1. **保证队列里有货**，但方式按模式不同：
           * `threaded=False`：`while len(self.queue) < self.max_lag + 1: self._produce_once()`
             —— 单线程模拟"生产者跑在前面"，这就是 lag 的来源。
           * `threaded=True`：**只等**，不要自己生产：
             `if not self._wait_for_item(): raise RuntimeError("等生产者超时（死锁？）")`
        2. `b = self.queue.popleft()`（取**最旧**的 —— 先进先出）
        3. `lag = self.version - b.version`（当前版本 - 生成版本）
        4. **打上 stale 标记**：`b.meta["stale"] = lag > 0`
           —— `ppo_update` 会用它来跳过 `assert_ratio_starts_at_one`
           （那个断言只在"数据由当前策略产生"时成立）
        5. 返回 `(b, lag)`

        如果取到的批次 lag 超过 `self.max_lag`（理论上不该发生，除非
        threaded 模式下生产者跑飞），就 `self.dropped += 1` 并**继续取下一批** ——
        别用太旧的数据训练。
        """

        if self.threaded:
            if not self._wait_for_item():
                raise RuntimeError("等生产者超时（死锁？）")
        else:
            while len(self.queue) < self.max_lag + 1:
                self._produce_once()
                    
        b = self.queue.popleft()
        lag = self.version - b.version
        b.meta["stale"] = lag > 0
        return b, lag


    # ------------------------------------------------------------------ #
    # 以下是管道（已实现）
    # ------------------------------------------------------------------ #
    def _wait_for_item(self, timeout: float = 5.0) -> bool:
        """（管道）等待队列里出现至少一个批次；超时返回 False。

        为什么必须带超时：并发代码出问题时**默认行为是挂死**，而挂死比报错
        难查一百倍。给它一个上限，死锁就会变成一句明确的话。
        """
        t0 = time.perf_counter()
        while not self.queue:
            if self._stop.is_set():
                return False
            if time.perf_counter() - t0 > timeout:
                return False
            self._stop.wait(0.001)
        return True

    def advance(self) -> None:
        """训练完一批后调用：版本号 +1；threaded 模式下**在同一个锁里发布新权重**。

        发布 = 「把 `model` 的权拷进 `published`」+「版本号 +1」，两件事必须原子。
        生产者就是靠 (权重, 版本号) 这一对来标注数据的；只要它可能读到
        "新权重 + 旧编号"，这批数据的 logprob 就不对应任何真实版本。

        ⚠️ 这一步在训练器的**关键路径**上（拷贝耗时算在主线程头上）。
        真实系统里这个代价由 NCCL broadcast 承担 —— 不是免费的，只是换了个地方付。
        """
        if self.threaded:
            with self.lock:
                self.published.load_state_dict(self.model.state_dict())
                self.version += 1
                self.published_version = self.version
        else:
            self.version += 1

    def _loop(self) -> None:
        """生产者线程主体。"""
        while not self._stop.is_set():
            if len(self.queue) >= self.max_lag + 1:
                self._stop.wait(0.001)          # 背压：队列满了就等
                continue
            self._produce_once()

    def start(self) -> None:
        if self.threaded and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# =========================================================================== #
# 2. 异步训练入口（就是 train() + AsyncSource）
# =========================================================================== #
def train_async(
    task: CountOnesTask,
    model: TinyActorCritic,
    cfg: PPOConfig,
    logger: MetricLogger | None = None,
    eval_prompts=None,
    eval_every: int | None = None,
):
    """异步 PPO：**和 `train_sync` 是同一个 `train()`**，只是换了个 source。

    这句话不是修辞 —— 你可以 diff 这两个函数，它们的主体完全一样。
    """
    from trainer import make_eval_prompts, train          # 延迟 import，避免循环依赖

    source = AsyncSource(task, model, cfg)
    return train(task, model, cfg, source, logger, eval_prompts, eval_every)


# =========================================================================== #
# 3. 测试用例
# =========================================================================== #
def _cfg(**kw) -> PPOConfig:
    base = dict(tag="async_test", device="cpu", rollout_prompts=8, max_new_tokens=12,
                inner_epochs=1, lr=1e-2, grad_clip=1.0, total_iters=4, log_every=0)
    base.update(kw)
    return PPOConfig(**base)


def _run(task, cfg, model_seed: int):
    """跑一次训练，返回奖励序列（可复现）。"""
    from trainer import train
    torch.manual_seed(model_seed)
    model = TinyActorCritic()
    src = AsyncSource(task, model, cfg)
    _, logger = train(task, model, cfg, src, eval_every=0)
    return [round(r["reward"], 6) for r in logger.records], logger


def _case_a() -> None:
    """★ 等价性：**lag=0 的异步必须和同步逐步完全一致**。

    这是最关键的测试：如果 lag=0 时两者不同，说明"异步"引入了和数据无关的差异
    （比如随机源被多消费了、权重快照时机不对），那后面测出来的"lag 的代价"就是假的。

    必须同时固定：模型初始化 seed、生成用的随机源 seed、以及生产的批次数。
    """
    from trainer import train_sync
    task = CountOnesTask()
    cfg = _cfg(max_lag=0, threaded=False)

    torch.manual_seed(cfg.seed)
    m1 = TinyActorCritic()
    _, lg_sync = train_sync(task, m1, cfg, eval_every=0)
    r_sync = [round(r["reward"], 6) for r in lg_sync.records]

    r_async, _ = _run(task, cfg, cfg.seed)

    if r_sync != r_async:
        raise AssertionError(
            f"max_lag=0 时异步和同步结果不同：\n        同步 {r_sync}\n        异步 {r_async}\n"
            f"        → 固定 seed + lag=0 时，两条路径必须**完全一致**。"
            f"检查：随机源是否被多消费、生产的批次数是否和消费的一致"
        )
    print(f"    lag=0 的异步 == 同步：{r_sync}  ✅")


def _case_b() -> None:
    """版本号与 lag：max_lag=k 时，稳态 lag 应该正好是 k，而且版本号单调。"""
    task = CountOnesTask()
    for k in (0, 1, 2, 4):
        cfg = _cfg(max_lag=k, threaded=False, total_iters=8)
        _, lg = _run(task, cfg, cfg.seed)
        lags = [r["lag"] for r in lg.records]
        vers = [r["version"] for r in lg.records]
        # lag 是**爬坡**上来的：预热时队列里全是 v0 的批次，要 k+1 步才被消费完，
        # 所以前 k+1 个 lag 是 0,1,...,k，之后才恒定在 k。
        # （我第一版从第 2 步就开始查稳态，max_lag=4 时直接误报。）
        steady = lags[k + 1:]
        if not steady:
            raise AssertionError(f"total_iters 太小，看不到 max_lag={k} 的稳态")
        if not (max(lags) == k and all(x == k for x in steady)):
            raise AssertionError(
                f"max_lag={k} 的 lag 序列 {[int(x) for x in lags]} 不符合预期："
                f"\n        应该是前 {k + 1} 步爬坡 0..{k}，之后恒为 {k}"
            )
        if vers != sorted(vers):
            raise AssertionError(f"version 不单调：{vers}")
        print(f"    max_lag={k} → lag 序列 {[int(x) for x in lags]}，version {[int(x) for x in vers]}  ✅")

    # 队列深度必须被 max_lag 限制住（否则生产会无限跑在前面）
    cfg = _cfg(max_lag=3, threaded=False, total_iters=6)
    torch.manual_seed(cfg.seed)
    src = AsyncSource(task, TinyActorCritic(), cfg)
    peak = 0
    for _ in range(cfg.total_iters):
        src.get()
        peak = max(peak, len(src.queue))
        src.advance()
    if peak > cfg.max_lag + 1:
        raise AssertionError(f"队列峰值 {peak} 超过了 max_lag+1={cfg.max_lag + 1}（背压失效）")
    print(f"    队列峰值 {peak} ≤ max_lag+1={cfg.max_lag + 1}（背压生效）  ✅")


def _case_c() -> None:
    """★ 异步的本质：**lag ≥ 1 时，ratio 起手必然不是 1**。

    这就是"异步的代价"的全部来源 —— 数据由旧策略产生，所以
    `logπ_new - logπ_old` 天然不为 0，裁剪项开始吃掉梯度。

    反过来，lag=0 的批次上那个断言必须通过（说明数据是新鲜的）。
    """
    from loss import assert_ratio_starts_at_one
    from batch import token_logprobs
    from trainer import ppo_update

    task = CountOnesTask()
    cfg = _cfg(max_lag=3, threaded=False, total_iters=6)
    torch.manual_seed(cfg.seed)
    model = TinyActorCritic()
    src = AsyncSource(task, model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def recompute_ratio(b):
        tm = b.meta["prompt_mask"] | b.mask
        with torch.no_grad():
            logits, _, _ = model(b.input_ids, tm)
            lp = token_logprobs(logits, b.input_ids)
        return lp

    # 先训练一步，让权重动起来
    b0, lag0 = src.get()
    src.advance()
    ppo_update(model, opt, b0, cfg, assert_fresh=(lag0 == 0))

    # 现在取到的批次是"更早版本"产生的
    b1, lag1 = src.get()
    if lag1 < 1:
        raise AssertionError(f"期望拿到 lag ≥ 1 的批次，实际 lag={lag1}")
    lp1 = recompute_ratio(b1)
    try:
        assert_ratio_starts_at_one(lp1, b1.logprobs, b1.mask)
        raise AssertionError(
            f"lag={lag1} 的批次居然通过了 ratio≡1 的断言 —— 说明它其实不是旧数据，"
            f"异步没有真的产生滞后"
        )
    except AssertionError as e:
        if "ratio 起始偏差" not in str(e):
            raise
        print(f"    lag={lag1}：ratio 起手偏差非零（数据确实是旧的）  ✅")
    print(f"    meta['stale'] = {b1.meta.get('stale')}（应为 True）  ✅")
    if b1.meta.get("stale") is not True:
        raise AssertionError("lag>0 的批次必须打上 meta['stale']=True 标记")


def _case_d() -> None:
    """真并发 + **消费者不许自己生产**（防死锁 / 防重复生产的纪律）。

    上一版只检查「生产的比消费的多」，结果在第 1~3 步会**假通过** ——
    因为消费者自己也在生产，反而「生产得更多」。测试太弱，不是代码对。

    现在直接检查**谁在生产**：用 spy 记录每次 `_produce_once` 是在哪个线程调用的。

    消费者一旦也来生产，会付三个代价：
      1. rollout 白跑一倍（算力浪费）
      2. 队列上限失效（lag 不再受控）
      3. 如果 `_produce_once` 里含阻塞等待 → **直接死锁**（我真踩过）
    """
    import threading
    import time

    from trainer import train

    task = CountOnesTask()

    def run(threaded: bool):
        cfg = _cfg(max_lag=2, threaded=threaded, total_iters=6)
        torch.manual_seed(cfg.seed)
        model = TinyActorCritic()
        src = AsyncSource(task, model, cfg)

        callers: list[str] = []
        orig = src._produce_once

        def spy() -> None:                       # 记录是哪个线程在调它
            callers.append(threading.current_thread().name)
            orig()

        src._produce_once = spy
        t0 = time.perf_counter()
        train(task, model, cfg, src, eval_every=0)
        return time.perf_counter() - t0, src.produced, callers

    t_sync, n_sync, _ = run(False)
    t_thr, n_thr, callers = run(True)

    if n_thr <= n_sync:
        raise AssertionError(
            f"threaded=True 时生产的批次数（{n_thr}）没有多于串行（{n_sync}）"
            f"—— 生产者线程似乎没在跑"
        )
    by_main = [c for c in callers if c == "MainThread"]
    if by_main:
        raise AssertionError(
            f"线程模式下，消费者（MainThread）自己生产了 {len(by_main)} 批"
            f"—— 违反了「队列只由生产者线程写入」的纪律\n"
            f"        后果：rollout 白跑一倍、队列上限失效，"
            f"而且若 `_produce_once` 里含阻塞等待就会**死锁**\n"
            f"        → `get()` 在 threaded 模式下必须**只等待**：\n"
            f"          `if not self._wait_for_item(): raise RuntimeError(...)`"
        )
    print(f"    串行 {t_sync:.2f}s / 生产 {n_sync} 批；"
          f"线程 {t_thr:.2f}s / 生产 {n_thr} 批（预取 {n_thr - n_sync} 批）")
    print(f"    生产者的线程：{sorted(set(callers))}（不含 MainThread = 消费者没在产）  ✅")


def _case_e() -> None:
    """★ 权重发布必须**原子**：每一批数据都要对应一个**真实存在过**的版本。

    这是异步正确性的核心契约：

        `b.logprobs` 必须能由「`b.version` 这个版本的权重」复现出来。

    为什么它是要害：如果生产者读到 `optimizer.step()` 改到一半的**撕裂权重**，
    或者把新权目标成了旧版本号，这批数据就**不对应任何真实版本** ——
    而 `lag` 度量的正是"它落后几个版本"，前提是它确实属于某个版本。
    度量一个不存在的东西，是你最不想要的那种错误：曲线照画，结论全错。

    做法：每次 `advance()` 之后把**发布出来的权重**存进 `history[version]`；
    训练跑完后逐批复现 —— 用 `history[b.version]` 重算 logprob，必须和 `b.logprobs` 一致。

    （旧设计会在这里挂掉：它读的是 `self.model`，而 `model` 在锁外被 `ppo_update` 改。）
    """
    from batch import token_logprobs
    from trainer import train

    task = CountOnesTask()
    # max_lag 故意调大：队列不容易满，生产者才不会一直阻塞在"队列满"上，
    # 否则它根本没机会在竞态窗口里去采样。（窗口小 + 生产者被挡住 = 测不出来）
    cfg = _cfg(max_lag=8, threaded=True, total_iters=6)

    torch.manual_seed(cfg.seed)
    model = TinyActorCritic()
    src = AsyncSource(task, model, cfg)

    if not hasattr(src, "published"):
        raise AssertionError(
            "AsyncSource 没有 `published` 副本 —— 生产者和训练器还在共用 `model`。\n"
            "        → 生产者必须只读「训练器在锁内发布出去的快照」，绝不读 `self.model`"
        )

    # ---- E1（**确定性**，不靠时序运气）：生产者绝不碰 self.model ----
    #
    # `src.model` 换成代理：非主线程调 state_dict() 就记一笔。
    # 训练器自己 `advance()` 时也读 `src.model`（发布那一步），但它在主线程，放行。
    # 所以这个检查**只**抓「生产者在读 model」。
    import threading

    violations: list[str] = []

    class _ThreadGuard:
        def __init__(self, inner):
            self._inner = inner

        def state_dict(self, *a, **kw):
            if threading.current_thread() is not threading.main_thread():
                violations.append(threading.current_thread().name)
            return self._inner.state_dict(*a, **kw)     # 仍返回真权重，让训练跑完

    src.model = _ThreadGuard(model)

    def snap(m) -> dict:
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    history: dict[int, dict] = {0: snap(model)}      # version 0 = 初始权重
    batches: list = []
    orig_advance, orig_get = src.advance, src.get

    def advance_rec() -> None:
        orig_advance()
        history[src.version] = snap(src.published)   # 「这个版本被发布出来的权重」

    def get_rec():
        b, lag = orig_get()
        batches.append((b, lag))
        return b, lag

    src.advance, src.get = advance_rec, get_rec

    # 把竞态窗口**人为放大**：在 `ppo_update` 改完权重、`advance()` 更新版本号之间
    # 插一段延迟。这不是作弊 —— 竞态本来就存在，只是窗口只有几微秒，
    # 靠反复跑碰运气是测不出来的。**放大窗口是测试竞态的标准手法。**
    import time

    import trainer as trainer_mod

    real_update = trainer_mod.ppo_update

    def update_with_delay(*a, **kw):
        out = real_update(*a, **kw)
        time.sleep(0.02)          # ← 窗口：权重已新，版本号还旧
        return out

    trainer_mod.ppo_update = update_with_delay
    try:
        train(task, model, cfg, src, eval_every=0)
    finally:
        trainer_mod.ppo_update = real_update

    if violations:
        raise AssertionError(
            f"生产者线程读了 {len(violations)} 次 `self.model` —— 它应该只读 `self.published`\n"
            f"        `model` 正被 `optimizer.step()` 原地改，直接读会拿到**撕裂权重**\n"
            f"        （这份权重在任何时刻都不存在 → 这批数据的 logprob 不对应任何版本）"
        )
    print(f"    E1 生产者从未触碰 self.model（主线程发布了 {src.version} 个版本）  ✅")

    checker = TinyActorCritic()
    bad: list[str] = []
    for b, lag in batches:
        if b.version not in history:
            bad.append(f"version={b.version} 从未被发布过")
            continue
        checker.load_state_dict(history[b.version])
        tm = b.meta["prompt_mask"] | b.mask
        with torch.no_grad():
            logits, _, _ = checker(b.input_ids, tm)
            lp = token_logprobs(logits, b.input_ids)
        dev = float(((lp - b.logprobs).abs() * b.mask).max())
        if dev > 1e-5:
            bad.append(f"version={b.version} 的 logprob 复现不出来（偏差 {dev:.4f}）")

    if bad:
        raise AssertionError(
            "有批次不对应任何真实版本：\n        " + "\n        ".join(bad[:5]) +
            f"\n        （共 {len(batches)} 批，{len(bad)} 批有问题）\n"
            f"        → 说明「拷贝权重」和「版本号 +1」不是原子的，"
            f"或者生产者读到了正在被更新的 `model`"
        )
    lags = [int(lag) for _, lag in batches]
    print(f"    {len(batches)} 批全部能按自己标注的版本复现（lag 序列 {lags}）  ✅")
    print(f"    history 里的版本：{sorted(history)}（每个都是一个真实发布过的快照）  ✅")


if __name__ == "__main__":
    cases = [("用例 A：lag=0 的异步必须 == 同步（★ 最关键）", _case_a),
             ("用例 B：版本号与 lag（含背压）", _case_b),
             ("用例 C：异步的本质 —— lag≥1 时 ratio 起手不是 1", _case_c),
             ("用例 D：真并发（生产者跑在消费者前面）", _case_d),
             ("用例 E：权重发布必须原子（每批都对应真实版本）", _case_e)]
    passed = 0
    for title, fn in cases:
        print(f"\n{title}")
        try:
            fn()
            passed += 1
        except NotImplementedError as e:
            print(f"    ⏳ 还没实现：{e}")
        except AssertionError as e:
            print(f"    ❌ {e}")
        except Exception as e:
            print(f"    💥 未预期的异常 {type(e).__name__}: {e}")
    print(f"\n{'=' * 60}")
    if passed == len(cases):
        print(f"🎉 {len(cases)} 个用例全部通过 —— M4 完成，跑 python3 run_async.py 看代价曲线")
    else:
        print(f"进度：{passed}/{len(cases)} 通过（NotImplementedError 是正常的）")
