"""prm_reward.py —— 过程奖励（PRM）的**核心**：分段 → 判分 → 逐 token 奖励 ✓✓

## 独立于旧 PPO ✓✓（只 import，不改 ✓）

    复用：`data`（题库）/ `math_data`（答案判据）/ tokenizer
    不碰：`model.py` / `batch.py` / `update.py` / `main.py` / `sft.py` ✗✗
    → 旧的 GRPO 流程一行都不受影响 ✓

## 为什么 PPO 那侧**不需要改** ✗✓

`batch.py` 里 `rewards` 本来就是 **(B, T)** ✓，注释写着
"终局稀疏奖励就只在最后一个有效位置非零" ✓
→ ★ 我们只要把"段末位置"填上分数 ✓ → `update.ppo_update` 里
   `compute_gae(rewards=batch.rewards, ...)` **直接就能用** ✓✓

## 奖励定义（和 PRM_DESIGN.md 一致 ✓）

    第 j 段判对（1 分）→ `r[段 j 的最后一个 token] = 1`
    全题做对          → `r[最后一个有效 token] += 1`（完成奖励 ✓）
    其它位置           → 0

    ★ γ=1 时 GAE 的**回报** = 从该位置往后的和（后缀 ✓）
      → 段 j 的回报 = 1[本段对] + 之后的正确段数 + bonus ✓✓
      → 这天然就是"走到第几段"的信号 ✓；而判 0 之后全是 0 ✓ → 错段回报 0 ✓ 必被惩罚 ✓

## 判分器

    `Judge` 三种实现 ✓：
      · `MockJudge`      —— 按行是否含"WRONG"标 0/1 ✓（**只为自测** ✓ 不联网 ✓）
      · `CallableJudge`  —— 给别人包现成的判分函数 ✓
      · `ApiJudge`       —— 调大模型 API ✓ **带缓存** ✓（同一段文本只判一次 ✓✓）

## 自测（纯 CPU ✓ 不联网 ✓ 不需要模型 ✓）

    python3 prm_reward.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: 五段的标记 ✓（顺序即判分顺序 ✓）
HEADS = ("[Overview]", "[Plan]", "[Solution]", "[Check]", "[Answer]")


# =========================================================================== #
# 1. 分段 ✓（**字符位置** → 之后由调用方映射到 token ✓）
# =========================================================================== #
def find_sections(text: str) -> list[tuple[str, int, int]]:
    """返回 `[(head, start, end), ...]` ✓ `start/end` 是**该段内容**的字符区间 ✓

    规则：段的边界 = 下一个 head 出现处 ✓；缺段就跳过 ✓（不报错 ✓ 判分自然给 0 ✗）
    """
    hits = []
    for h in HEADS:
        i = text.find(h)
        if i >= 0:
            hits.append((i, h))
    hits.sort()
    out: list[tuple[str, int, int]] = []
    for k, (i, h) in enumerate(hits):
        s = i + len(h)
        e = hits[k + 1][0] if k + 1 < len(hits) else len(text)
        out.append((h, s, e))
    return out


def section_contents(text: str) -> list[tuple[str, str]]:
    """`[(head, 内容（已 strip）), ...]` ✓"""
    return [(h, text[s:e].strip()) for h, s, e in find_sections(text)]


# =========================================================================== #
# 2. ★★ 把"段的字符末尾"映射到"token 下标"（这一段最容易错 ✗✓）
# =========================================================================== #
def char_to_token_index(tokenizer, ids: list[int], char_pos: int,
                        prompt_len: int = 0) -> int:
    """`char_pos`（在**生成段文本**里的字符位置 ✓）→ 对应的 token 下标 ✓

    ★★★ 这里原来有个致命错误 ✗✓：
        我用"**逐 token 解码再拼起来**"来建字符→token 映射 ✓
        → 只要输出里有**非 ASCII**（`π` / `≤` / 全角符号 ✓）→
          单个 token 解码出 `�` ✗ → **拼起来 ≠ 整体解码** ✗ → 断言直接抛 ✓
        → 实测：**8/8 全部解码失败** ✗✗ → 段判对恒 0 ✓ → PRM 完全没信号 ✓✓

    ★★ 现在改成**二分** ✓✓：找最小的 t，使 `decode(ids[:t+1])` 的长度 > char_pos ✓
       · 用的是**同一条** `decode` 路径 ✓ → 非 ASCII 也**永远一致** ✓✓
       · 解码长度对 t **单调不减** ✓ → 二分合法 ✓✓
       · 复杂度 O(log T) ✓（每段只调用一次 ✓）
    """
    lo, hi = prompt_len, len(ids) - 1
    if hi < lo:
        return max(0, len(ids) - 1)

    def _len_at(t: int) -> int:
        # ★★ 必须**只解码生成段** ✗✓ —— `char_pos` 是生成段内的位置 ✓
        #   我第一版解码 `ids[:t+1]`（含 prompt ✗）→ 长度多了 prompt_len 个字符 ✗✗
        #   → 二分落点整体**偏移** ✓ → 自测里 5 段全错 ✓（自测立刻抓到了 ✓✓）
        return len(tokenizer.decode(ids[prompt_len:t + 1], skip_special_tokens=True))

    # ① 生成段为空 → 直接返回最后一个 ✓
    if _len_at(hi) <= char_pos:
        return hi
    # ② 二分找第一个"长度 > char_pos"的 t ✓
    while lo < hi:
        mid = (lo + hi) // 2
        if _len_at(mid) > char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


# =========================================================================== #
# 2b. ★★★ 按**自然段落**切块 ✓✓（给"不写 5 段"的模型用 ✓）
# =========================================================================== #
#: ★ 这些区间内部**不许切** ✗✓（LaTeX 公式/环境/代码块 ✓）
_PROTECT = (
    re.compile(r"\\\[.*?\\\]", re.S),          # \[ ... \]
    re.compile(r"\$\$.*?\$\$", re.S),              # $$ ... $$
    re.compile(r"\\\(.*?\\\)", re.S),          # \( ... \)
    re.compile(r"\\begin\{.*?\}.*?\\end\{.*?\}", re.S),  # \begin{}..\end{}
)


def _protected_mask(text: str) -> list[bool]:
    """标出**不许切**的字符 ✓（长度 = len(text) ✓）"""
    mask = [False] * len(text)
    for pat in _PROTECT:
        for m in pat.finditer(text):
            for i in range(m.start(), m.end()):
                if i < len(mask):
                    mask[i] = True
    return mask


def _snap(a: int, b: int, mask: list[bool]) -> tuple[int, int]:
    """★ 只把**终点往后推**到保护区之外 ✓✓ 且**最多推 300 字** ✗✓

    ✗ 第一版错在两处：
      ① 同时把起点往回挪 → 块 N 起点跳进块 N-1 ✓ → **重叠** ✗✗
      ② 没有距离上限 → 若保护区连成一片 ✓ → `b` 一路推到文本末尾 ✗✗ → **全塌成 1 块** ✓
    ✓ 现在：起点不动 ✓ · 终点最多推 300 字 ✓（推不到就**不推** ✓ 宁可切在公式旁 ✓）
    """
    n = len(mask)
    _limit = min(n, b + 300)
    while b < _limit and mask[b - 1]:
        b += 1
    return a, b


def block_spans(text: str, min_chars: int = 80) -> list[tuple[int, int]]:
    """把任意 CoT 文本切成若干块 ✓ 返回**字符区间** ✓

    ★ 为什么需要它 ✗✓：Qwen2.5-Math 写的是**它自己的 CoT**（带空行分段 ✓ 没有段名 ✗）
      → 若坚持 5 段 → 又要 SFT 教格式 ✗ → 又伤能力 ✗✗
      → ★ 按**自然段落**切 ✓ → 不需要任何格式 SFT ✓✓

    规则：
      ① 优先按**空行**切 ✓（它自己就是这么分段的 ✓）
      ② 太碎的块（< min_chars）**合并到前一块** ✓（避免"一句话一块"→ 判分太碎 ✓）
      ③ 没有任何空行 → **按句号均分** ✓（保底 ✓）
    """
    import re as _re
    parts = [m for m in _re.split(r"\n\s*\n", text)]
    # 算每块的绝对区间 ✓
    spans: list[tuple[int, int]] = []
    pos = 0
    for pt in parts:
        i = text.index(pt, pos) if pt else pos
        spans.append((i, i + len(pt)))
        pos = i + len(pt)
    # ② 合并太碎的 ✓
    merged: list[tuple[int, int]] = []
    # ★★★ 合并**必须有上限** ✗✓ —— 我第一版只判"当前块 < min_chars" ✓
    #   实测后果：5 段里有 4 段都很短 ✓ → **全部并进第 1 段** ✗✗ → 越并越大 ✓ → **最后 1 块** ✓
    #   → 现在：合完如果超过 `_MERGE_MAX` 字，就**不合并** ✓✓
    # ★★★ 正确的判据：**合并后仍然 < min_chars 才合并** ✓✓
    #   ✗ 我用过两个错规则：
    #     ① 只判"当前块 < min_chars" ✗ → 每块都短时**一路合光** ✓ → 1 块 ✗✗
    #     ② 加上限 250 ✗ → 但整段才 158 字 ✓ → **仍在限内** → 还是合光 ✗✗
    #   ✓ 现在：`已合并长度 + 当前块长度 < min_chars` ✓ → 块最多长到 80 字 ✓✓
    #     → 碎片会被并起来 ✓ 但**永远不会长成巨块** ✓✓
    for a, b in spans:
        if merged and ((merged[-1][1] - merged[-1][0]) + (b - a)) < min_chars:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    # ★★★ 切点不能在 **LaTeX 公式内部** ✗✓ —— 实测样本 2 的块 1 结尾是
    #   `\theta = \tan^{-1}\` ✓ → **半条公式** ✓ → 判分器必然判 0 ✗✗ → 假阴性 ✓
    _mask = _protected_mask(text)
    merged = [_snap(x, y, _mask) for x, y in merged]
    merged = [(a, b) for a, b in merged if text[a:b].strip()]
    # ★★ ④ 超长块**再按句切** ✓✓ —— 实测：一条无空行的长答案会被切成 **1 块** ✗
    #   → 那一段的 PRM 就等于 0/1（一处错整块 0）✗✗ → 过程信号全丢 ✓
    #   → 规则：单块 > 400 字就按句切，每约 2 句一块 ✓
    _MAX = 400
    split2: list[tuple[int, int]] = []
    for a0, b0 in merged:
        if (b0 - a0) <= _MAX:
            split2.append((a0, b0))
            continue
        sub = text[a0:b0]
        cuts = [0] + [m.end() for m in _re.finditer(r"[.!?](?=\s|$)", sub)] + [len(sub)]
        cuts = sorted(set(cuts))
        k = 0
        while k < len(cuts) - 1:
            # ★ 每块尽量接近 _MAX 但不超过它 ✓（至少 1 句 ✓）
            j = k + 1
            while j + 1 < len(cuts) and (cuts[j + 1] - cuts[k]) <= _MAX:
                j += 1
            _s0, _s1 = _snap(a0 + cuts[k], a0 + cuts[j], _mask)
            split2.append((_s0, _s1))
            k = j
        split2 = [(x, y) for x, y in split2 if text[x:y].strip()] or split2
    if split2:
        merged = split2

    # ③ 保底：没空行就按句子均分 ✓（每约 2 句一块 ✓）
    if len(merged) <= 1:
        sents = [m for m in _re.finditer(r"[.!?](?=\s|$)", text)]
        if len(sents) >= 4:
            cut = [0] + [m.end() for m in sents] + [len(text)]
            merged, step = [], 2
            for k in range(0, len(cut) - 1, step):
                merged.append(_snap(cut[k], cut[min(k + step, len(cut) - 1)], _mask))
            merged = [(a, b) for a, b in merged if text[a:b].strip()]
    return merged


def block_end_tokens(tokenizer, ids: list[int], text: str,
                     prompt_len: int, min_chars: int = 80
                     ) -> tuple[list[int], list[str]]:
    """按段落切块 ✓ → `(每块末尾的 token 下标, 每块内容)` ✓"""
    idxs: list[int] = []
    contents: list[str] = []
    for a, b in block_spans(text, min_chars):
        c = text[a:b].strip()
        if not c:
            continue
        e2 = a + len(text[a:b].rstrip())
        idxs.append(char_to_token_index(tokenizer, ids, e2 - 1, prompt_len))
        contents.append(c)
    return idxs, contents


def make_block_judge(api_judge):
    """把 `ApiJudge` 包成"判**推理片段**"的判分器 ✓✓

    ★ 原来的 `JUDGE_SYSTEM` 是给 5 个具名段写的 ✗ → 直接拿去判自由段落**不合适** ✗✓
      这里换一套更通用的标准 ✓：这一小步推理有没有错 ✗
    """
    def _f(problem: str, head: str, so_far: str, content: str) -> int:
        return int(api_judge(problem, "reasoning step", so_far, content))
    return _f


def section_end_tokens(tokenizer, ids: list[int], text: str,
                       prompt_len: int) -> tuple[list[int], list[tuple[str, str]]]:
    """返回 `(每段末尾的 token 下标, [(head, 内容), ...])` ✓

    ★ 段的"最后一个 token"取**内容区间的末尾** ✓ 而不是下一个 head 之前 ✓
      因为 head 本身不该被奖励 ✓（它只是标记 ✓）
    """
    secs = find_sections(text)
    idxs: list[int] = []
    contents: list[tuple[str, str]] = []
    for h, s, e in secs:
        c = text[s:e].strip()
        if not c:                                   # 空段 → 记 0 分 ✓ 位置仍给（不奖励 ✓）
            idxs.append(prompt_len)
            contents.append((h, ""))
            continue
        # 内容真正的结束位置（去空白之后 ✓）
        e2 = s + len(text[s:e].rstrip())
        idxs.append(char_to_token_index(tokenizer, ids, e2 - 1, prompt_len))
        contents.append((h, c))
    return idxs, contents


# =========================================================================== #
# 3. 判分器 ✓
# =========================================================================== #
class MockJudge:
    """★ 只为自测 ✓：内容里出现 `WRONG` 判 0 ✓ 否则判 1 ✓"""

    name = "mock"

    def __call__(self, problem: str, head: str, so_far: str, content: str) -> int:
        return 0 if "WRONG" in content else 1


class CallableJudge:
    """把任意 `f(problem, head, so_far, content) -> 0/1` 包成判分器 ✓"""

    def __init__(self, fn, name: str = "callable"):
        self._fn, self.name = fn, name

    def __call__(self, problem: str, head: str, so_far: str, content: str) -> int:
        return int(self._fn(problem, head, so_far, content))


JUDGE_SYSTEM = (
    "You are grading ONE section of a structured math solution.\n"
    "Each section has a different standard:\n"
    "- [Overview]: must correctly state the problem type / goal / key idea.\n"
    "- [Plan]: high-level steps in plain English. DO NOT require numbers here; "
    "a correct plan scores 1.\n"
    "- [Solution]: the actual math. Every computation must be correct.\n"
    "- [Check]: a valid verification of the answer.\n"
    "- [Answer]: must state the final answer.\n"
    "Answer with the single character 1 (this section is correct and appropriate) "
    "or 0 (it contains an error). Output ONLY 0 or 1, nothing else."
)

# ★★★ 给**自由 CoT 片段**用的标准 ✓✓（不是 5 个具名段 ✗）
JUDGE_SYSTEM_BLOCK = (
    "You are checking ONE step of a math solution being written by another model.\n"
    "The step is a paragraph of its reasoning. Judge ONLY whether this paragraph is "
    "correct and follows validly from the steps already judged correct.\n"
    "Be strict about arithmetic and logic, but do NOT penalize informal language, "
    "hedging, or the model admitting uncertainty.\n"
    "IMPORTANT: earlier blocks are marked with their score. If an earlier block scored 0 "
    "and this paragraph correctly identifies and fixes that mistake, score this paragraph 1. "
    "Acknowledging and correcting an error is good reasoning, not a defect.\n"    "IMPORTANT: if the text is garbled, repetitive, self-contradictory, or contains "
    "non-English characters or random words spliced into formulas (a sign of a broken "
    "generation), score 0 REGARDLESS of how plausible the structure looks.\n"
    "Answer with the single character 1 (this paragraph is correct) "
    "or 0 (it contains a real error). Output ONLY 0 or 1."
)

JUDGE_TMPL_BLOCK = """Problem:
{problem}

Steps already judged correct:
{so_far}

Next paragraph of the solution:
{content}

Is this paragraph correct? Answer ONLY 0 or 1."""

JUDGE_TMPL = """Problem:
{problem}

Already-verified steps:
{so_far}

Section to grade: {head}
{content}

Is this section correct per the standard for {head}? Answer ONLY 0 or 1."""


class ApiJudge:
    """★ 调大模型 API 判分 ✓✓ **带磁盘缓存** ✓（同一段文本只判一次 ✓）

    成本控制 ✓：
      · 缓存键 = (题, 段名, 前缀, 段内容) 的 md5 ✓ → 重跑/重采样都命中 ✓✓
      · `--judge-frac` 那种"只判一部分"的开关留给了调用方 ✓ 这里不做 ✗
    """

    def __init__(self, model: str = "deepseek-flash", cache_path: str | None = None,
                 temperature: float = 0.0, base_url: str = "https://api.deepseek.com",
                 retries: int = 3, mode: str = "section"):
        self.name = f"api:{model}"
        self.model, self.temperature = model, temperature
        self.base_url, self.retries = base_url, retries
        self.mode = mode                       # ★ "section"（5 段）或 "block"（自由段落 ✓）
        self.cache_path = cache_path or "runs/prm_judge_cache.jsonl"
        self._cache: dict[str, int] = {}
        self._client = None
        self.hits = self.calls = 0
        self.fail = 0
        self.n_zero = 0
        # ★★★ 真实单次延迟 ✓✓ —— 我曾用"32 路 2.1s"推断"每次 0.06s" ✗
        #   那是极小 prompt ✓ 而真实 ~1,400 token → 每次 **~12s** ✗
        self._pending: list[tuple[str, int]] = []   # ★ 待写盘的缓存条目 ✓
        # ★★★ 我的补丁把 `self.t_http = 0.0` **整行替换掉了** ✗✗
        #   → `__call__` 里 `self.t_http += ...` 直接 AttributeError ✓
        #   → **每一次判分都抛异常** ✓ → 重试 3 次后 RuntimeError ✓ → 整个 run 死 ✓
        #   ★ 教训：`.replace(a, b)` 必须把 a **原样包含在 b 里** ✓✓
        self.t_http = 0.0
        self.n_retry = 0                      # ★ 重试次数 ✓（区分"延迟高"vs"被限流" ✓✓）
        self.n_http = 0
        self.verbose = bool(os.environ.get("PRM_JUDGE_VERBOSE"))
        # ★★ 并发要加锁 ✓✓ —— 缓存 dict 和追加写都非线程安全 ✗
        import threading as _th
        self._lock = _th.Lock()
        self._pool = None
        if os.path.exists(self.cache_path):
            for line in open(self.cache_path, encoding="utf-8"):
                if line.strip():
                    try:
                        d = json.loads(line)
                        self._cache[d["k"]] = int(d["v"])
                    except Exception:                    # noqa: BLE001
                        pass
            print(f"  [judge] 缓存命中池 {len(self._cache)} 条 ✓（{self.cache_path}）")

    def _key(self, problem: str, head: str, so_far: str, content: str) -> str:
        raw = f"{self.model}\x00{self.mode}\x00{problem}\x00{head}\x00{so_far}\x00{content}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _client_or_make(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("没有 API key ✓ 请 export DEEPSEEK_API_KEY=...")
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def score_many(self, jobs: list[tuple[str, list[tuple[str, str]]]],
                   workers: int = 0) -> list[list[int]]:
        """★★★ **并发**判多条样本 ✓✓（每条内部仍串行 ✓）

        为什么这样切 ✗✓：
            · **同一题内不能并行** ✗ —— 块 k 需要块 1..k-1 的 `so_far` ✓（顺序依赖 ✓）
            · **不同样本完全独立** ✓ → **样本级并发** ✓✓
            · API 调用是 **I/O 等待** ✓ → 线程就够 ✓（不用进程 ✓✓）

        ★ 实测意义：32 条 × ~10 块 = **320 次串行** ✗ → 2.6s → **14 分钟/步** ✗✗
          16 路并发 → **~1 分钟/步** ✓✓
        """
        import concurrent.futures as _cf
        # ★★★ `workers=0` → **自动等于任务数** ✓✓
        #   = 每个样本一个任务 ✓ → **并发度 = batch** ✓（32 条 → 32 路 ✓）
        #   这正是"按 batch 并发"的含义 ✓ 不会多开也不会少开 ✓✓
        _w = max(1, int(workers) if workers and int(workers) > 0 else len(jobs))
        if self._pool is None:
            self._pool = _cf.ThreadPoolExecutor(max_workers=_w)
        else:
            self._pool._max_workers = _w        # ★ 换了 batch 就跟着换 ✓
        futs = [self._pool.submit(score_sections, self, q, c) for q, c in jobs]
        out = [f.result() for f in futs]
        self._flush()                     # ★ 收尾把不足 64 条的余量写掉 ✓✓
        return out

    def score_block_jobs(self, jobs: list[tuple[str, str, str]],
                         workers: int = 0) -> list[int]:
        """★★★ **块级全并行** ✓✓ —— 320 个块任务一次性发出 ✓

        wall = **1 次 API 往返** ✓（原来每条样本内部 10 块串行 ✗ → 10 轮 ✗）
        """
        import concurrent.futures as _cf
        _w = max(1, int(workers) if workers and int(workers) > 0 else len(jobs))
        # ★ 并发上限可调 ✓（被限流时降它 ✓ 而不是降 batch ✗）
        # ★★ 默认上限 **64** ✓✓ —— 320 个同时发几乎肯定被 429 ✗
        #   而 429 会走退避（1s/2s）→ **反而比串行更慢** ✗✓
        #   → 64 路：320/64 = 5 轮 × 延迟 ✓ 仍是串行(10 轮)的 2 倍  速 ✓
        #   → 若确认不限流，`PRM_JUDGE_CONCURRENCY=160` 再试 ✓✓
        _cap = int(os.environ.get("PRM_JUDGE_CONCURRENCY", "64") or 64)
        if _cap > 0:
            _w = min(_w, _cap)
        # ★★★ 并发数必须**可见** ✗✓ —— 它曾静默变成 1（0 被 max(1,·) 吃掉 ✗）
        #   → 320 次判分串行 ✓ → GPU 卡几分钟 ✓ 而日志只有 5 行 ✓ → 完全看不出 ✓✓
        if self._pool is None or self._pool._max_workers != _w:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
            self._pool = _cf.ThreadPoolExecutor(max_workers=_w)
            print(f"  [judge] ★ 块级并发 **{_w}** 路 · 任务 {len(jobs)} 个 ✓"
                  f"（320/{_w} = {len(jobs) // max(_w, 1) + 1} 轮）", flush=True)
        futs = [self._pool.submit(self, prob, "step", pre, cont)
                for prob, pre, cont in jobs]
        out = [f.result() for f in futs]
        self._flush()
        return out

    def _save(self, k: str, v: int) -> None:
        """★★ 缓存**批量写** ✓✓ —— 原来每次判分都 open/close 一次 ✗✗

        实测代价 ✗：32 条 × ~10 块 = **320 次** ✓
            · 每次 `makedirs` + `open` + `write` + `close` ✗
            · 全程持有全局锁 ✓ → **把 32 路并发压成串行** ✗✗
            · 而服务器是 **K8s pod 的网络盘** ✓ → 单次 0.3~0.5s ✓
            → 判分时间从 119s 涨到 **274s** ✗（且随缓存增长越来越慢 ✓）
        ✓ 现在：先进内存 ✓ 满 64 条或退出时**一次写出** ✓✓
        """
        with self._lock:
            self._pending.append((k, v))
            if len(self._pending) < 64:
                return
            batch, self._pending = self._pending, []
        self._flush(batch)

    def _flush(self, batch: list[tuple[str, int]] | None = None) -> None:
        """一次写出多条 ✓（只在**不持锁**的时候写盘 ✓✓）"""
        with self._lock:
            if batch is None:
                batch, self._pending = self._pending, []
            elif not batch:
                return
            if not batch:
                return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write("".join(json.dumps({"k": a, "v": b}, ensure_ascii=False) + "\n"
                                for a, b in batch))
        except OSError as e:                      # ★ 写盘失败不该影响训练 ✓
            print(f"  [judge] ⚠️ 缓存写盘失败（不影响奖励 ✓）：{e}", flush=True)

    def __call__(self, problem: str, head: str, so_far: str, content: str) -> int:
        k = self._key(problem, head, so_far, content)
        with self._lock:
            if k in self._cache:
                self.hits += 1
                return self._cache[k]
        if self.mode == "block":
            prompt = JUDGE_TMPL_BLOCK.format(problem=problem, so_far=so_far or "(none)",
                                             content=content)
            _sys = JUDGE_SYSTEM_BLOCK
        else:
            prompt = JUDGE_TMPL.format(problem=problem, so_far=so_far or "(none)",
                                       head=head, content=content)
            _sys = JUDGE_SYSTEM
        last = None
        for attempt in range(self.retries):
            try:
                _t1 = time.perf_counter()
                r = self._client_or_make().chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    # ★★★ 必须给足 ✗✗ —— `deepseek-flash` 是**推理模型** ✓
                    #   它先写 reasoning_content（思考 ✓）再写 content（答案 ✓）
                    #   `max_tokens=16` → 思考就被截 ✓ → **content = ''** ✗（实测 ✓）
                    #   → 于是"判分器坏了"被我读成"模型答错了" ✗✗ → 全批 0% ✓
                    max_tokens=512,
                    messages=[{"role": "system", "content": _sys},
                              {"role": "user", "content": prompt}],
                )
                with self._lock:
                    self.t_http += time.perf_counter() - _t1
                    self.n_http += 1
                self.calls += 1
                _msg = r.choices[0].message
                txt = (_msg.content or "").strip()
                # ★ 推理模型：content 可能空 ✓ → 答案在 reasoning_content 里 ✓
                if not txt:
                    txt = (getattr(_msg, "reasoning_content", "") or "").strip()
                _fin = getattr(r.choices[0], "finish_reason", None)
                # ★★★ 空/截断 → **不是"模型答错"** ✗✗ → 作废并重试 ✓✓
                #   （这条最关键：我原来把"判分器坏了"记成 0 分 ✗ → 全批 0% ✓✓）
                import re as _re0
                if not txt or _fin == "length" or not _re0.search(r"[01]", txt):
                    self.fail += 1
                    print(f"  [judge] ✗✗ 判分作废（空/{_fin!r}/无 0-1 ✗ "
                          f"回复={txt[:40]!r}）→ 重试 ✓ 不是模型答错 ✓", flush=True)
                    continue
                # ★ 解析放宽 ✓✓ —— 原来是 `startswith("1")` ✗
                #   模型若回 "The section is correct: 1" ✓ → 也被判 0 ✗✗
                #   现在：取**第一个出现的 0/1 数字** ✓
                import re as _re
                _m = _re.search(r"[01]", txt)
                v = int(_m.group(0)) if _m else 0
                # ★★★ 日志要**克制** ✗✓ —— 原来"每次判 0 都打" ✓
                #   而段判对率是 ~82% ✓ → **一次 320 次判分里 ~58 个 0** ✗
                #   → 45 步 = **2,600 行** ✓ 把真正的错误信息全冲掉了 ✗✗
                #   ✓ 现在：① 前 3 次 ✓ ② 采样计数 ✓ ③ **只有 verbose 才全打** ✓
                self.n_zero += 0 if v else 1
                if self.verbose:
                    print(f"  [judge] {head} → {v} ✓ {txt[:40]!r} ✓", flush=True)
                elif self.calls <= 3:
                    print(f"  [judge] 首判 {head} → {v} ✓ {txt[:40]!r} ✓", flush=True)
                elif v == 0 and self.n_zero <= 5:
                    # ★ 只打**前 5 个 0** ✓ 再多就是同一类问题 ✓
                    print(f"  [judge] 判 0（第 {self.n_zero} 个）{head} → "
                          f"{txt[:40]!r} ✓", flush=True)
                with self._lock:
                    self._cache[k] = v
                self._save(k, v)
                return v
            except Exception as e:                       # noqa: BLE001
                last = e
                # ★★★ 这一步可能吃掉**大量时间** ✗✓ —— 任何异常（429/超时 ✓）都
                #   退避 `1s / 2s / 4s` ✓ 再试 ✓ → 而**次数从来没被统计** ✗✗
                #   若 20% 的调用被限流 ✓ → 320 × 0.2 × 3s ≈ **190s** ✗ ← 可能就是 274s ✓
                with self._lock:
                    self.n_retry += 1
                time.sleep(min(20.0, 2 ** attempt))
        self.fail += 1
        print(f"  [judge] ✗✗ 判分**彻底失败** {type(last).__name__}: {str(last)[:100]}",
              flush=True)
        # ★ 抛出去 ✓ 让上层知道"这一批的判分不可信" ✗ 别静默记 0 ✓✓
        raise RuntimeError(f"judge failed for {head}: {last}")


# =========================================================================== #
# 4. ★★ 逐段判分 → 逐 token 奖励 ✓
# =========================================================================== #
def score_sections(judge, problem: str,
                   contents: list[tuple[str, str]],
                   stop_at_zero: bool = False, keep: int = 3) -> list[int]:
    """逐块判分 ✓✓ **默认不停判** ✗✓（这是关键改动 ✓）

    ## 为什么改成"不停判" ✗✓

    原来"遇 0 立刻停" ✓ → 实测代价很大 ✗：
        样本 2 的块 5 写了 `tan^{-1}(3/0)`（真错 ✓ 判 0 正确 ✓）
        → **块 6 写了"这个式子未定义，改用几何位置判断"** ✓✓ ← ★ **这是自我纠正** ✓
        → 但"停判"让块 6~9 全记 0 ✗✗ → **答案明明是对的 ✓ 却只拿到 4/9** ✓

    ★ CoT 最值钱的地方**恰恰是"发现错、改过来"** ✓✓
      → 把"纠正"判成"整段作废" ✗ → 等于惩罚了最有价值的轨迹 ✓

    ## 现在的做法 ✓

        · **继续判后面的块** ✓
        · `so_far` 里**带上每一块的分数** ✓✓ → 判分器能看到"前面错了" ✓
          → 于是它能判断"这一段是在纠正，还是又错了" ✓✓
        · `stop_at_zero=True` 仍保留 ✓（想省 API 时可以开 ✓）
    """
    scores: list[int] = []
    so_far = ""
    _hist: list[str] = []
    for head, content in contents:
        if stop_at_zero and scores and scores[-1] == 0:
            scores.append(0)                           # 省 API 模式 ✓
            continue
        if not content:
            scores.append(0)
            _hist.append(f"{head}\n(empty)")
            so_far = "\n\n".join(_hist[-keep:]) + "\n\n"
            continue
        s = int(judge(problem, head, so_far, content))
        scores.append(1 if s else 0)
        # ★★ **所有块都进 so_far，并标明分数** ✓✓
        #   判分器因此知道"前一步判了 0" ✓ → 能识别"这段是纠正" ✓✓
        _hist.append(f"{head}  [scored {1 if s else 0}]\n{content}")
        # ★★★ `so_far` 只留**最近 `keep` 块** ✗✓
        #   原来累积**全部**块 ✓ → 长答案的判分 prompt 能到 3~4,000 token ✗✗
        #   → 判分变慢变贵 ✓（而"这一步对不对"只需要紧邻上下文 ✓✓）
        so_far = "\n\n".join(_hist[-keep:]) + "\n\n"
    return scores


# =========================================================================== #
# 3b. ★★★ 块级**全并行**判分 ✓✓（前缀只依赖原文 ✓ 不依赖判决 ✓）
# =========================================================================== #
def block_jobs(problem: str, contents: list[tuple[str, str]],
               keep: int = 3) -> list[tuple[str, str, str]]:
    """把一条样本拆成 **每块一个独立任务** ✓✓ → 返回 `[(题, 前缀, 本块)]` ✓

    ★★★ 为什么这样可以全并行 ✗✓
        我原来把"前面各块**判了几分**"写进 prompt ✓ → 块 i 要等块 i-1 的判决 ✗✗
        → 于是每条样本内部**必须串行** ✓ → 32 条 × 10 块 = 10 轮 ✓ → 200s ✗
        ★ 但那个分数标记是**我自己的设计** ✗ → 去掉它，prompt 只由**原文**决定 ✓
        → ★★ A / AB / ABC / ABCD / ABCDE **五段同时发** ✓✓
             wall = **1 次往返** ✓（不是 10 次 ✗）
        ★ 取舍：判分器看不到"前一块判了 0" ✗ 但**看得到前一块的原文** ✓
          错在哪它自己能看出来 ✓ 而且**不受前序判决锚定** ✓（更独立 ✓）
    """
    out: list[tuple[str, str, str]] = []
    texts = [c for _, c in contents]
    for i, (_, c) in enumerate(contents):
        prefix = "\n\n".join(texts[max(0, i - keep):i])
        out.append((problem, prefix, c))
    return out


def score_block_jobs(judge, jobs: list[tuple[str, str, str]],
                     workers: int = 0) -> list[int]:
    """★ 判一批**块级**任务 ✓✓ 全部并发 ✓（不是逐条样本串行 ✓）"""
    if hasattr(judge, "score_block_jobs"):
        return judge.score_block_jobs(jobs, workers=workers)
    return [int(judge(prob, "step", pre, cont)) for prob, pre, cont in jobs]


def build_reward_matrix_paper(rows: list[dict], B: int, T: int,
                              done_idx: torch.Tensor,
                              beta: float = 0.15) -> tuple[torch.Tensor, dict]:
    """★★★ 论文式奖励 `r = σ(α·rm) + (rv − 1)` 的**逐 token 版** ✓✓

    | 信号 | 落在哪 | 作用 |
    |---|---|---|
    | `rv`（答案对错 ✓ 机械判 ✓）| **done 位置** ✓ | ★ **定生死** ✓（±1 ✓）|
    | `rm`（段判对 ✓ 可刷 ✗）| **各段末尾** ✓ | 只做**组内排序** ✓（总权重 ≤ 5β < 1 ✓）|

    ★★ 核心约束（这是上次失败的解药 ✗✓）：
        **5 段全对合计 5β = 0.75 < 1** ✓ → **段分永远翻不了正负** ✓✓
        → "段段漂亮但答错" = **−1 + 0.75 = −0.25** ✗ ✗（必须答对才可能拿正分 ✓）
        → 而"答对且全对" = **+1 + 0.75 = +1.75** ✓✓

    ★ 为什么这修掉了上次的坑 ✗：
        上次 `r = 段分` 是**唯一信号** ✗ → 模型只要"让判分器满意"就能拿满分 ✓✗
        现在段分**最多值 0.75** ✓ → 而答错直接 **−1** ✗ → **主信号是答对** ✓✓
    """
    r = torch.zeros(B, T)
    stats = {"segments": 0, "correct_segments": 0, "full": 0, "rv_pos": 0}
    for k, row in enumerate(rows):
        ok = bool(row.get("all_correct"))
        stats["full"] += 1 if ok else 0
        stats["rv_pos"] += 1 if ok else 0
        # ★★★ 段分要**按段数归一** ✗✓ —— 否则段落模式下段数无上限 ✗
        #   5 段：5×0.15 = 0.75 < 1 ✓ 安全 ✓
        #   段落：10 段 → 1.5 > 1 ✗✗ → "段全对但答错" = −1+1.5 = **+0.5** ✗
        #         → ★ **又会被刷** ✓（回到上次那个坑 ✗）
        #   → 每家每户除以自己的段数 ✓ → 段分总贡献**恒 ≤ beta** ✓✓ 与段数无关 ✓
        _n = max(1, len(row["ends"]))
        _b = float(beta) / _n
        for e, sc in zip(row["ends"], row["scores"]):
            stats["segments"] += 1
            if sc:
                stats["correct_segments"] += 1
                if 0 <= e < T:
                    r[k, e] += _b
        # ② rv：**主信号** ✓（±1 ✓ 落在 done ✓ 和 `dones` 对齐 ✓）
        d = int(done_idx[k])
        if 0 <= d < T:
            r[k, d] += 1.0 if ok else -1.0

    return r, stats


def build_reward_matrix(rows: list[dict], B: int, T: int,
                        done_idx: torch.Tensor,
                        bonus: float = 1.0) -> tuple[torch.Tensor, dict]:
    """★ 组装 `(B, T)` 奖励 ✓✓

    `rows[k]` 必须含：`ends`（段末 token 列表 ✓ 绝对下标）、`scores`（0/1 ✓）、
                     `all_correct`（bool ✓）
    `done_idx[k]` = 第 k 条轨迹**最后一个有效位置**（收 EOS 的位置 ✓ 或最后一个 token ✓）
                    放完成奖励用它 ✓ 和 `dones` 对齐 ✓
    """
    r = torch.zeros(B, T)
    stats = {"segments": 0, "correct_segments": 0, "full": 0}
    for k, row in enumerate(rows):
        ends, scores = row["ends"], row["scores"]
        for j, (e, s) in enumerate(zip(ends, scores)):
            if e < 0 or e >= T:
                continue                                # 越界 → 跳过 ✓（宁可少给 ✓）
            stats["segments"] += 1
            if s:
                r[k, e] += float(s)
                stats["correct_segments"] += 1
        if row.get("all_correct"):
            stats["full"] += 1
            d = int(done_idx[k])
            if 0 <= d < T:
                r[k, d] += float(bonus)                 # ★ 完成奖励 ✓
    return r, stats


# =========================================================================== #
# 5. 自测（纯 CPU ✓）
# =========================================================================== #
class _FakeTok:
    """按**字符**切 token 的最简 tokenizer ✓（ASCII 下和真 tokenizer 行为一致 ✓）"""

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(int(i)) for i in ids)

    def __call__(self, s: str, add_special_tokens: bool = False):
        return {"input_ids": [ord(c) for c in s]}


_TEXT = """[Overview]
A compound interest problem.
[Plan]
P1. Use the formula.
P2. Round the result.
[Solution]
S1. [calc] Apply the formula -> <<2000*(1+0.05)^18=4813.2384673>>
S2. [calc] Round -> <<4813.2384673=4813.24>>
[Check]
V1. It is positive -> <<4813.24 > 0>>
[Answer]
\\boxed{4813.24}"""


def _self_test() -> int:
    fails = 0

    def ck(c, m):
        nonlocal fails
        print(f"  {'✓' if c else '✗✗'} {m}")
        fails += 0 if c else 1

    print("① 分段 ✓")
    secs = section_contents(_TEXT)
    ck([h for h, _ in secs] == list(HEADS), f"五段齐全且有序 ✓（{len(secs)} 段）")
    ck(secs[0][1].startswith("A compound"), "第一段内容对 ✓")
    ck("\\boxed{4813.24}" in secs[-1][1], "最后一段含 boxed ✓")
    ck(section_contents("[Plan]\nP1. x\n") == [("[Plan]", "P1. x")], "缺段只返回有的 ✓")

    print("\n② ★ 字符→token 映射（最容易错的地方 ✗✓）")
    tok = _FakeTok()
    ids = [ord(c) for c in _TEXT]
    Lp = 10                                        # 假装 prompt 有 10 个 token ✓
    ends, contents = section_end_tokens(tok, [0] * Lp + ids, _TEXT, Lp)
    ck(len(ends) == 5, f"五段都有末位 ✓（{[e - Lp for e in ends]}）")
    ck(all(Lp <= e < Lp + len(ids) for e in ends), "末位都在生成段内 ✓")
    ck(ends == sorted(ends), "末位单调递增 ✓✓（段是按顺序的 ✓）")
    # ★ 逐段核对：段末 token 解出来的字符应该在**该段内容**里 ✓
    for (h, c), e in zip(contents, ends):
        ch = chr(ids[e - Lp])
        ck(ch == c[-1], f"{h} 末位解出 {ch!r} == 内容最后一字 {c[-1]!r} ✓✓")

    print("\n③ 判分：**默认不停判** ✓✓（能识别自我纠正 ✓）")
    calls = []

    def judge(prob, head, so_far, content):
        calls.append(head)
        return 1 if head in ("[Overview]", "[Plan]") else 0

    sc = score_sections(CallableJudge(judge), "prob", contents)
    ck(sc == [1, 1, 0, 0, 0], f"scores = {sc} ✓（判 0 的块之后**继续判** ✓）")
    ck(len(calls) == 5, f"调了 {len(calls)} 次 ✓（= 块数 ✓ 不再提前停 ✓✓）")

    print("\n③b 省 API 模式（stop_at_zero=True）✓")
    calls.clear()
    sc2 = score_sections(CallableJudge(judge), "prob", contents, stop_at_zero=True)
    ck(sc2 == [1, 1, 0, 0, 0], f"scores 相同 ✓（{sc2}）")
    ck(len(calls) == 3, f"只调了 {len(calls)} 次 ✓（判 0 后停止 ✓ 省 API ✓）")

    print("\n③c ★ so_far 里**带分数** ✓✓（判分器能看到前面错了 ✓）")
    seen = []

    def judge2(prob, head, so_far, content):
        seen.append(so_far)
        return 1 if len(seen) == 1 else 0

    score_sections(CallableJudge(judge2), "prob", contents)
    ck("scored 1" in seen[1], f"第 2 次的 so_far 含分数标记 ✓（{seen[1][:40]!r}）")

    print("\n④ 奖励矩阵：只落在**段末** ✓✓")
    B, T = 1, 200
    rows = [{"ends": [20, 40, 60, 80, 100], "scores": [1, 1, 0, 0, 0],
             "all_correct": False}]
    r, st = build_reward_matrix(rows, B, T, torch.tensor([150]), bonus=1.0)
    nz = torch.nonzero(r[0]).flatten().tolist()
    ck(nz == [20, 40], f"非零位置 = {nz} ✓（只有判对的段 ✓）")
    ck(st["correct_segments"] == 2 and st["full"] == 0, f"统计 {st} ✓")

    print("\n⑤ 完成奖励 ✓（全对时多一分，落在最后一个有效 token ✓）")
    rows = [{"ends": [20, 40, 60, 80, 100], "scores": [1, 1, 1, 1, 1],
             "all_correct": True}]
    r, st = build_reward_matrix(rows, 1, T, torch.tensor([150]), bonus=1.0)
    ck(abs(float(r[0].sum()) - 6.0) < 1e-6, f"总分 = {float(r[0].sum())} ✓（5 段 + 1 bonus ✓✓）")
    ck(float(r[0, 150]) == 1.0, "bonus 落在 done_idx ✓✓（和 `dones` 对齐 ✓）")
    ck(st["full"] == 1, "统计 full=1 ✓")

    print("\n⑥ ★★ GAE 的回报 = **后缀和** ✓✓（段 j = 本段 + 之后所有段 + bonus ✓）")
    from update import compute_gae
    mask = torch.zeros(1, T)
    mask[0, 10:160] = 1                             # 生成段 ✓
    values = torch.zeros(1, T)
    rewards = r.clone()
    # ★ `compute_gae` 里写的是 `~dones` ✗ → **必须是 bool** ✓（float 会 TypeError ✓ 实测踩了 ✓）
    dones = torch.zeros(1, T, dtype=torch.bool)
    dones[0, 150] = True
    adv, ret = compute_gae(rewards=rewards, values=values, next_values=torch.zeros(1, T),
                           dones=dones, mask=mask, gamma=1.0, lam=1.0)
    # ★★★ 我之前写的是"前缀和" ✗✗ —— **错了** ✓✓
    #   GAE 的 return 是 `G_t = r_t + γ·G_{t+1}` ✓ → **从 t 往后的和**（后缀 ✓）
    #   所以段 j 末尾的回报 = 1[本段对] + **它之后所有对段的个数** + bonus ✓
    #   → 这**恰好**是对的 ✓✓：回报单调反映"这条轨迹走到了第几段" ✓
    #     而**判 0 之后的段全是 0** ✓ → 错段的回报 = 0 ✓ → 优势为负 ✓✓（正确惩罚 ✓）
    ck(abs(float(ret[0, 20]) - 6.0) < 1e-5,
       f"ret[20] = {float(ret[0,20]):.3f} ✓（本段 1 + 后面 4 段 + bonus = **6** ✓）")
    ck(abs(float(ret[0, 40]) - 5.0) < 1e-5, f"ret[40] = {float(ret[0,40]):.3f} ✓（越来越小 ✓）")
    ck(abs(float(ret[0, 100]) - 2.0) < 1e-5,
       f"ret[100] = {float(ret[0,100]):.3f} ✓（最后一段 = 1 + bonus = **2** ✓）")
    # ★ 关键性质：**判 0 的段**回报必须是 0 ✓✓（否则错的段会被强化 ✗✗）
    rows_bad = [{"ends": [20, 40, 60, 80, 100], "scores": [1, 1, 0, 0, 0],
                 "all_correct": False}]
    rb, _ = build_reward_matrix(rows_bad, 1, T, torch.tensor([150]), bonus=1.0)
    _, retb = compute_gae(rewards=rb, values=torch.zeros(1, T),
                          next_values=torch.zeros(1, T), dones=dones.clone(),
                          mask=mask, gamma=1.0, lam=1.0)
    ck(abs(float(retb[0, 40]) - 1.0) < 1e-5, f"错段之前 ret[40] = {float(retb[0,40]):.3f} ✓")
    ck(abs(float(retb[0, 60])) < 1e-5,
       f"★ **判 0 的段 ret[60] = {float(retb[0,60]):.3f}** ✓✓（优势必为负 → 正确惩罚 ✓）")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓ —— 分段/映射/停判/奖励矩阵/GAE 回报都对 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
