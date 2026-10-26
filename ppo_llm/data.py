"""GSM8K 数据 + **规则奖励**（RLVR 的第一步）。

为什么单独一个文件、而且由我写好并测过
------------------------------------
因为 RLVR 的成败几乎全在这三件小事上，而它们**错了都不报错，只会让奖励悄悄失真**：

1. **答案抽取**：模型输出什么格式算"答对了"？
2. **数值归一化**：`18` vs `18.0` vs `1,234` vs `$18` 算不算同一个答案？
3. **奖励尺度**：0/1 还是别的？奖励全 0 时 PPO 一点梯度都没有。

这一份的每个函数都有自测（`python3 data.py`），用例是我从真实输出里挑出来的坑。

数据来源
--------
GSM8K 官方仓库的 **JSONL**（不是 HF 的 parquet）：
`raw.githubusercontent.com/openai/grade-school-math/.../test.jsonl`

这样**不需要 `datasets` / `pyarrow`**——本机两个都没装，而且 pip 装不上（沙箱）。
第一次运行会自动下载并缓存到 `data/`。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))          # 用 __file__ 锚定，
_CACHE = os.path.join(_HERE, "data")                        # 别让产物落到"当前目录"
_BASE_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data")


# =========================================================================== #
# 1. 数据
# =========================================================================== #
def ensure_gsm8k(split: str = "test") -> str:
    """确保 JSONL 在本地缓存里，返回路径（没有就下载）。"""
    assert split in ("train", "test"), split
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"gsm8k_{split}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    url = f"{_BASE_URL}/{split}.jsonl"
    print(f"下载 {url} ...")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return path


def load_gsm8k(split: str = "test") -> list[dict]:
    """返回 `[{"question":..., "gold":..., "solution":...}, ...]`。

    `gold` 是**归一化后的答案字符串**（用来和模型抽出来的答案比）。
    """
    out = []
    with open(ensure_gsm8k(split), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            raw = d["answer"].split("####")[-1].strip()
            out.append({"question": d["question"], "gold": raw, "solution": d["answer"]})
    return out


def fewshot_examples(train: list[dict], k: int = 4) -> list[dict]:
    """取前 k 条训练样本当 few-shot 示例。

    ⚠️ 必须从 **train** 里取，不能从 test 里取——否则基线被污染，
    你测出来的"学习"有一部分是抄答案抄来的。
    """
    return train[:k]


# =========================================================================== #
# 2. 答案抽取与归一化（奖励的核心）
# =========================================================================== #
# 优先找 `#### 数字`；找不到就退回"文本里最后一个数字"。
_FINAL_RE = re.compile(r"####\s*([^\n]*)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


#: 前多少条留给 SFT。RL 只用**后面**那一段 —— 两边严格不相交。
#: 改这个值会让已有的 adapter 和数据池不再对应，所以别随便动。
SFT_SIZE = 1500


def split_train(sft_size: int | None = None) -> tuple[list[dict], list[dict]]:
    """把 GSM8K train 切成**严格不相交**的两段：`(sft_pool, rl_pool)`。

    为什么要有这个函数，而不是在两个文件里各切一刀：

        切分必须是**一个事实**。各切一刀的话，`sft.py` 用 `train[:1600]`、
        `main.py` 用 `train[1:]`，重叠 1599 条 ✗ —— 而**它不会报错**，
        只会让 RL 的训练奖励虚高（那部分是记忆，不是泛化）✗

    few-shot 示例也要从 **sft_pool** 里取：如果示例同时是 RL 的训练样本，
    那道题的答案就**明写在 prompt 里** → 模型抄一下就有奖励 ✗
    """
    tr = load_gsm8k("train")
    n = SFT_SIZE if sft_size is None else sft_size
    if not 0 < n < len(tr):
        raise ValueError(f"sft_size={n} 不合法（train 共 {len(tr)} 条）")
    return tr[:n], tr[n:]


def load_test_file(path: str) -> list[dict]:
    """★ 读**任意** jsonl 当测试集 ✓✓ —— 统一成 `{question, gold}`

    为什么需要它 ✗：官方 test 是固定文件 ✓ → 但我们做了"带三段式模板"的测试集 ✓
    → 评估必须能**指定文件** ✓，否则新测试集永远用不上 ✓✓

    支持的字段组合（**兼容所有我们的产物** ✓）：
        {"question", "gold"}                        ← clean_staged 的输出 ✓
        {"question", "answer"}                      ← 官方 GSM8K ✓（gold 从 #### 抽 ✓）
        {"question", "solution"}                    ← 三段式 ✓（gold 从 #### 抽 ✓）
        {"question", "raw_question", "gold", ...}   ← clean_staged 全字段 ✓

    `question` **原样保留** ✓ —— 如果它带三段式模板 ✓ 就带着进 prompt ✓✓
    （和"训练时的输入"一致 ✓ 这正是我们要的 ✓）

    返回的每条都保证有：`question` ✓ `gold` ✓（其它字段原样留着 ✓）
    """
    import json as _json
    if not os.path.exists(path):
        raise FileNotFoundError(f"测试集不存在：{path} ✗")
    out: list[dict] = []
    bad: list[int] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            q = d.get("question")
            if not q:
                bad.append(i)
                continue
            g = d.get("gold")
            if g is None:
                # ★ 从 answer / solution 里的 `#### N` 抽 ✓（和官方 loader 一致 ✓）
                for k in ("answer", "solution"):
                    if isinstance(d.get(k), str) and "####" in d[k]:
                        g = d[k].split("####")[-1].strip()
                        break
            if g is None or g == "":
                bad.append(i)
                continue
            out.append({**d, "question": q, "gold": str(g), "_line": i})
    if bad:
        raise ValueError(
            f"{path} 有 {len(bad)} 条缺 question 或 gold/#### ✗（前几个行号：{bad[:5]}）\n"
            f"  需要字段：question ✓ + (gold 或 answer/solution 里含 #### N) ✓")
    if not out:
        raise ValueError(f"{path} 是空的 ✗")
    return out


def normalize_number(s: str | None) -> float | None:
    """把一段字符串里的**数字**抽出来并归一化成 float；抽不到返回 None。

    处理这些真实会遇到的写法：
        "18"      → 18.0
        "18.0"    → 18.0      （和 18 视为同一个答案）
        "1,234"   → 1234.0    （千分位）
        "$18"     → 18.0      （货币符号）
        "18 apples"→18.0      （单位）
        "-7"      → -7.0
        "18."     → 18.0      （句尾句号）
    """
    if s is None:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_answer(text: str) -> str | None:
    """从模型输出里抽最终答案（字符串形式），抽不到返回 None。

    顺序：
      1. 最后一个 `#### ...` 后面的内容
      2. 都没有 → 全文里最后一个数字（**宽松模式**）
    """
    hits = _FINAL_RE.findall(text)
    for h in reversed(hits):
        if normalize_number(h) is not None:
            return h.strip()
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def reward_fn_base(text: str, gold: str) -> bool:
    """**规则奖励**：答案数值相等给 1.0，否则 0.0（RLVR 的标准做法）。

    用"数值相等"而不是"字符串相等"：`18` 和 `18.0`、`1,234` 和 `1234`
    在数学上是一个答案，按字符串比会白白判错——这类假阴性会直接污染训练信号。
    """
    p, g = normalize_number(extract_answer(text)), normalize_number(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if abs(p - g) < 1e-6 else 0.0


# =========================================================================== #
# 3. Prompt 构造
# =========================================================================== #
# ★★★ 全局 system 开关 ✓✓
#   None  → 用默认的 GSM8K system（"必须写 #### <number>"）✓
#   ""    → **完全不要 system 消息** ✓✓ ← PRM 数据必须用这个 ✗
#   其它   → 直接用这个字符串 ✓
#   为什么需要 ✗：PRM 的题干已经在 `question` 里写清了 5 段格式 ✓
#   而默认 system 却在命令"输出 #### <number>" ✗ → **指令与监督信号冲突** ✗✗
#   → 模型会被 system 带向 `####` ✓ 而不是 `\boxed{}` + 5 段 ✗
# ★ 也支持**环境变量** ✓✓ —— 这样 `main.py` / `probe_passk.py` / `sample_rft.py`
#   都**不用各加一个 flag** ✓（它们都经 `model.rollout` → `build_messages` ✓）
#
#     PPO_SYSTEM=none   python3 main.py ...        ← PRM 数据用这个 ✓✓
#     PPO_SYSTEM=default python3 main.py ...       ← GSM8K 用这个（默认 ✓）
SYSTEM_OVERRIDE: str | None = None
_ENV_SYS = os.environ.get("PPO_SYSTEM", "").strip().lower()
if _ENV_SYS in ("none", "empty", "off", "no"):
    SYSTEM_OVERRIDE = ""


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, "
    "then write the final answer on a new line exactly as: #### <number>"
)


#: GSM8K 解答里的中间算式标注，如 <<48+24=72>>
_ANN = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
#: 只允许这些字符的算式才敢 eval（含变量/文字的判不了，跳过）
_SAFE = re.compile(r"[0-9+\-*/(). ]+")


def check_arith(expr: str, claimed: str) -> bool | None:
    """<<48+24=72>> 里的算式**真的**等于右边吗？判不了就返回 None。"""
    if not _SAFE.fullmatch(expr):
        return None
    try:
        got = eval(expr, {"__builtins__": {}}, {})      # noqa: S307  字符已白名单
    except Exception:
        return None
    try:
        want = float(claimed.replace(",", "").strip())
    except ValueError:
        return None
    return abs(got - want) < 1e-6


def process_stats(text: str) -> tuple[int, int]:
    """返回 (可判定的算式数, 其中算对的数)。"""
    ok = tot = 0
    for e, c in _ANN.findall(text):
        r = check_arith(e, c)
        if r is not None:
            tot += 1
            ok += int(r)
    return tot, ok


def answer_strict(text: str, gold: str) -> bool:
    """**必须显式写出 `#### N`** 且数值正确（宽松回退不算）。

    训练奖励用这个 ✓ 理由：宽松抽取会让"截断/没写完"的输出靠"最后出现的数字"
    白拿结果奖励 ✗ —— 实测就是它让"永不停止"的退化解拿到了分 ✓
    """
    v = parsed_strict(text) if "parsed_strict" in globals() else None
    if v is None:
        hits = re.findall(r"####\s*([^\n]*)", text)
        for h in reversed(hits):
            v = normalize_number(h)
            if v is not None:
                break
    g = normalize_number(gold)
    return v is not None and g is not None and abs(v - g) < 1e-6


def gated_reward(text: str, gold: str, process_coef: float, finished: bool) -> float:
    """带截断门的过程奖励：**没吐 EOS 就不给过程项**（结果项照常，但已是严格判据）。"""
    base, proc = reward_parts(text, gold, process_coef)
    return base + (proc if finished else 0.0)


#: ★ 宽松判据认的**显式答案标记** ✓✓ —— 只认这些，**不认"全文最后一个数字"** ✗
#   为什么必须这么限 ✗：`reward_fn_base` 那种"从整段抽数字"会让**截断/没写完**的输出
#   靠"最后出现的数字"白拿分 ✗（实测过：永不停止的退化解拿到了分 ✓）
#   → 这里只认**明确的答案标记** ✓ → 基座那种 `**Answer:** $18` 能过 ✓，而"写到一半被切断"过不了 ✓✓
_ANS_MARKERS = (
    (r"####\s*([^\n]*)", "####"),                                    # 我们的方言 ✓
    (r"\\boxed\{([^}]*)\}", "boxed"),                                # LaTeX ✓
    (r"\*\*\s*Answer\s*:?\s*\*\*\s*([^\n]*)", "**Answer:**"),   # ★ 基座用这个 ✓
    (r"(?im)^\s*\**\s*Answer\s*:?\s*\**\s*([^\n]*)", "Answer:"), # 行首 Answer: ✓
    (r"(?i)the\s+(?:final\s+)?answer\s+is\s*[:=]?\s*([^\n.]*)", "the answer is"),  # ✓
)


def parsed_lenient(text: str) -> float | None:
    """从**显式答案标记**后面抽数字 ✓✓（比 strict 宽松 ✓ 比 naive 安全 ✗）"""
    for pat, _name in _ANS_MARKERS:
        hits = re.findall(pat, text)
        if not hits:
            continue
        # ★ 取**最后一个**标记 ✓（模型可能先写草稿再给答案 ✓）
        for h in reversed(hits):
            v = normalize_number(h)
            if v is not None:
                return v
    return None


def answer_lenient(text: str, gold: str) -> bool:
    """★ 宽松判据 ✓ —— 认多种"最终答案"方言 ✓ 但**要求有显式标记** ✓

    适用场景 ✗：**RL 跑在基座/未按 `####` 格式训练的模型上** ✓
      （基座写 `**Answer:** $18` ✓ → strict 判 0 ✗ → RL 会先花几百步"改造方言" ✗✗）
    仍然防的退化 ✗：
      · 截断/没写完 → 没有标记 → **0** ✓
      · "写到一半正好有个数字" → 不算 ✓（必须跟在标记后 ✓）
    """
    v = parsed_lenient(text)
    if v is None:
        return False
    g = normalize_number(gold)
    return g is not None and abs(v - g) < 1e-6


def reward_parts(text: str, gold: str, process_coef: float = 0.0,
                mode: str = "strict") -> tuple[float, float]:
    """返回 `(结果项, 过程项)`，**分开**给调用方。

    为什么要拆开：`rollout` 需要知道"这条轨迹有没有写完"才能决定过程项给不给
    （见下面的截断门）✓ 合在一起就算不出来了。
    """
    # ★ `mode`：`strict`（必须 #### ✓ 我们自己的模型 ✓）· `lenient`（认多种方言 ✓ 基座/未训练 ✓）
    if mode == "math":
        # ★ MATH 类：答案是 **LaTeX**（\frac / \sqrt / 坐标对 ✗ 不是数字 ✓）
        #   → 用 sympy/math_verify 判等价 ✓（惰性 import ✓ math_data 不依赖本模块 ✓ 无循环 ✓）
        from math_data import answer_math
        _ok = answer_math(text, gold)
    elif mode == "lenient":
        _ok = answer_lenient(text, gold)
    else:
        _ok = answer_strict(text, gold)
    base = 1.0 if _ok else 0.0
    if process_coef == 0.0:
        return base, 0.0
    tot, ok = process_stats(text)
    return base, process_coef * (ok / tot if tot else 0.0)


def reward_fn(text: str, gold: str, process_coef: float = 0.0) -> float:
    """答对(0/1) + process_coef × 算对比例。

    ⚠️ 没有可判定算式时，过程项是 **0**（不是跳过）——
       这正是「输出变短」被惩罚的机制 ✓
    ⚠️ 但**截断的轨迹必须另外处理**：它照样能拿满过程项，于是"一直写下去"
       能刷分 → EOS 概率被压到 0 → 永不停止 ✗（实测发生了：EOS 率 98.4% → 0%）
       所以真正的入口是 `rollout` 里的截断门，不是这个函数 ✓
    """
    b, pr = reward_parts(text, gold, process_coef)
    return b + pr


def build_messages(question: str, examples: list[dict] | None = None,
                   system: str | None = None) -> list[dict]:
    """构造 chat 消息列表：system + (few-shot 多轮) + 当前问题。

    ★ 关于 few-shot 的地位变了 ✗（SFT 之前 vs 之后 ✓）：
      SFT 之前：0.5B zero-shot 几乎不输出 `#### N` ✗ → 奖励全 0 → 那时 few-shot 是在"买"格式 ✓
      SFT 之后：0-shot 的格式合规率已经 **0.844** ✓ → **不再需要 few-shot** ✓✓
        （而且 0-shot 的 prompt 只有 101~111 token ✓ vs 4-shot 的 675 ✓ → 快 6 倍 ✓✓）
      所以现在所有 run 都用 `--n-shot 0` ✓，few-shot 只是一条备用路径 ✓
    """
    _sys = SYSTEM_OVERRIDE if system is None else system
    msgs = [] if _sys == "" else [{"role": "system",
                                   "content": _sys if _sys else SYSTEM_PROMPT}]
    for ex in (examples or []):
        msgs.append({"role": "user", "content": ex["question"]})
        # 字段名是 `solution` ✓ —— loader 把原始 JSONL 里的 `answer` 规范化过了 ✓
        # （我曾以为这是 bug ✗ 因为我拿**原始文件**去比**加载后的对象** ✓
        #  而当时实测打印的就是「✅ 没崩」✓ —— 我无视了它 ✗ 自测立刻抓住了我 ✓）
        msgs.append({"role": "assistant", "content": ex["solution"]})
    # ★ 模板**写在数据里** ✓（`question` = 模板 + 裸题）→ 代码只原样传 ✓
    msgs.append({"role": "user", "content": question})
    return msgs


# =========================================================================== #
# 4. 自测（用例都来自真实会踩的坑）
# =========================================================================== #
def _case_split() -> None:
    """分割必须严格不相交，而且 few-shot 不能从 RL 池里取。"""
    sft_pool, rl_pool = split_train()
    q_sft = {e["question"] for e in sft_pool}
    q_rl = {e["question"] for e in rl_pool}
    overlap = q_sft & q_rl
    if overlap:
        raise AssertionError(
            f"SFT 池和 RL 池重叠了 {len(overlap)} 条 ✗ —— RL 会在 SFT 训过的题上拿奖励，"
            f"训练奖励虚高（记忆当泛化）")
    if len(sft_pool) + len(rl_pool) != len(load_gsm8k("train")):
        raise AssertionError("两段加起来不等于全集 —— 有数据被漏掉或重复")
    # 反证：老代码的切法必须被这条判据抓住
    old_sft = {e["question"] for e in load_gsm8k("train")[:1600]}
    old_rl = {e["question"] for e in load_gsm8k("train")[1:]}
    if not (old_sft & old_rl):
        raise AssertionError("这条判据抓不住老代码的泄漏 —— 说明它没有牙齿 ✗")
    print(f"    SFT 池 {len(sft_pool)} 条 / RL 池 {len(rl_pool)} 条，重叠 0 ✅"
          f"（老切法重叠 {len(old_sft & old_rl)} 条，会被抓住）")


if __name__ == "__main__":
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'✅' if ok else '❌'} {name}: got={got!r} want={want!r}")
        if not ok:
            fails.append(name)

    print("抽取与归一化：")
    check("标准", normalize_number("#### 18".split("####")[1]), 18.0)
    check("小数=整数", normalize_number("18.0"), 18.0)
    check("千分位", normalize_number("1,234"), 1234.0)
    check("货币", normalize_number("$18"), 18.0)
    check("带单位", normalize_number("18 apples"), 18.0)
    check("负数", normalize_number("-7"), -7.0)
    check("句尾句号", normalize_number("18."), 18.0)
    check("非数字", normalize_number("no number"), None)

    print("\n过程奖励：")
    check("完全正确+算式对", reward_fn("<<48+24=72>>\n#### 72", "72", 0.5), 1.5)
    check("答对但算式错", reward_fn("<<48+24=99>>\n#### 72", "72", 0.5), 1.0)
    check("答错算式也对", reward_fn("<<48+24=72>>\n#### 99", "72", 0.5), 0.5)
    check("半对", reward_fn("<<1+1=2>> <<2+2=5>>\n#### 3", "3", 0.5), 1.25)
    check("系数0 = 老行为", reward_fn("<<48+24=72>>\n#### 72", "72", 0.0), 1.0)
    _short = reward_fn("#### 72", "72", 0.5)
    _long = reward_fn("<<48+24=72>>\n#### 72", "72", 0.5)
    check("★防塌缩：带算式必须比不带高", _long > _short, True)
    check("无标注时过程项=0", _short, 1.0)
    check("判不了的算式跳过", reward_fn("<<x+y=3>>\n#### 3", "3", 0.5), 1.0)
    check("拆开后 base+process 等于原值",
          sum(reward_parts("<<48+24=72>>\n#### 72", "72", 0.5)),
          reward_fn("<<48+24=72>>\n#### 72", "72", 0.5))
    # ★ 截断门的牙齿：同一个输出，finished=True/False 必须不同
    _t = "<<48+24=72>>\n#### 72"
    check("写完 → 拿到过程项", gated_reward(_t, "72", 0.5, True), 1.5)
    check("截断 → 过程项清零", gated_reward(_t, "72", 0.5, False), 1.0)
    # ★ 严格判据的牙齿：没写 #### 的输出，结果项必须是 0
    check("没写 #### → 结果项 0（宽松回退不算）",
          reward_parts("the answer is 72", "72", 0.0)[0], 0.0)
    check("写了 #### → 结果项 1", reward_parts("#### 72", "72", 0.0)[0], 1.0)

    print("\nSFT / RL 数据切分（防泄漏）：")
    try:
        _case_split()
        print("  ✅ 分割不相交")
    except AssertionError as e:
        print(f"  ❌ {e}")
        fails.append("split")

    print("\nextract_answer：")
    check("标准", extract_answer("blah\n#### 18"), "18")
    check("多个取最后", extract_answer("#### 18\n后面又写 #### 20"), "20")
    check("无标记退回最后数字", extract_answer("the answer is 42"), "42")
    check("完全没数字", extract_answer("I cannot solve this"), None)

    print("\nreward_fn（0/1）：")
    check("完全正确", reward_fn("#### 18", "18"), 1.0)
    check("格式差但数值对", reward_fn("#### 18.0", "18"), 1.0)
    check("千分位对", reward_fn("#### 1,234", "1234"), 1.0)
    check("答错", reward_fn("#### 17", "18"), 0.0)
    check("没给答案", reward_fn("I don't know", "18"), 0.0)
    check("只是过程里出现过对的数（应判错）", reward_fn("18 - 3 - 4 = 11\n所以是 11", "18"), 0.0)

    print("\n数据：")
    tr = load_gsm8k("train")
    te = load_gsm8k("test")
    print(f"  ✅ train {len(tr)} 条 / test {len(te)} 条")
    assert len(tr) > 7000 and len(te) > 1300, "条数不对"
    assert all(normalize_number(d["gold"]) is not None for d in te[:200]), "gold 抽不出数字"
    msgs = build_messages("1+1=?", fewshot_examples(tr, 2))
    print(f"  ✅ build_messages: {len(msgs)} 条消息（2-shot → 1 system + 4 轮 + 1 问题）")
    assert len(msgs) == 6

    print(f"\n{'=' * 60}")
    print("🎉 data.py 全部自测通过" if not fails else f"❌ 失败 {len(fails)} 项：{fails}")
    # ★ 失败必须真的非 0 退出 ✗
    #   （此前只 print 不退出 → `python3 data.py` 的退出码恒为 0 ✓
    #     于是"只看退出码"的门禁**看不见任何失败** ✓✓ 我上一轮还写了句
    #     `if __name__ == "__main__": pass` 假装修好了 —— 那比不修更坏 ✗）
    if fails:
        raise SystemExit(1)

