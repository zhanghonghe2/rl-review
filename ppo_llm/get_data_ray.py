"""get_data_ray.py —— GSM8K 三段式数据生成（Ray 4 路并发 ✓ 断点续跑 ✓ 重试 ✓）

## 相比原来的 get_data.py 改了什么 ✗

| 原来 | 现在 |
|---|---|
| `ray` 导入了但**没用** ✗（全串行）| **4 路并发** ✓✓（`--workers 4`）|
| API key **硬编码在源码里** ✗✗ | 从环境变量读 ✓（`DEEPSEEK_API_KEY` ✓）|
| 输出 `open(..., "a")` **追加** ✗ → 重跑重复 ✓ | **断点续跑** ✓：按 `idx` 跳过已完成的 ✓✓ |
| 无重试 ✗ → 一次 429 就整轮挂 ✓ | **指数退避重试** ✓✓ |
| prompt 含 `✓ ✗ ★ ─ →` 等非 ASCII ✗ | 全部换成 ASCII ✓✓（`→` 只保留在**输出格式**里 ✗ hmm 见下 ✓）|
| 每条 print 整段 ✗ | 只打进度 ✓（`--verbose` 才打样例 ✓）|
| 记录里没有 `idx` ✗ | **写入 `idx`** ✓ → 续跑精确 ✓✓ |

## 用法

    # ★ 生成第 4000 条到结尾（默认行为 ✓）
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 get_data_ray.py --start 4000

    # 指定并发 / 输出文件 / 只跑几条试试
    python3 get_data_ray.py --start 4000 --workers 4 --out runs/staged_4k.jsonl --limit 5

    # 续跑（会自动跳过已完成的 ✓ 重复跑同一条命令即可 ✓）
    python3 get_data_ray.py --start 4000

## 注意

* **API key**：脚本只从环境变量读 ✓。你原来那份代码里写死过一个 key ✗✗ →
  建议**轮换掉它** ✓（它已随源码一起存在过 ✓）。
* **并发上界**：`--workers` 默认 4 ✓。再高要看你账号的 RPM/TPM 限额 ✓；
  遇到 429 会自动退避重试 ✓ 但太多限流反而更慢 ✓。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

# ── prompt：和原来**逐字一致**（只把非 ASCII 换成 ASCII ✓ 语义不变 ✓）──────────
#   ★ 注意：输出格式里那个 `->` 是我们**要求模型**写的 ✓
#     而解析器只认 <<expr=result>> ✓ → `->` 是 ASCII ✓ 完全没问题 ✓✓
SYSTEM = (
    "You are a meticulous elementary-school math teacher. "
    "Rewrite the solution of the problem below into a strict three-stage format. "
    "Write everything in English only."
)

PROMPT_TMPL = """[Problem]
{question}

---------------------------- FORMAT (must follow exactly) ----------------------------

Output exactly three sections, in this order, with the headers copied verbatim:

[Plan]
   Plain English prose explaining what to compute, what to compute first, what next.
   Do not write any arithmetic expression here (words like "twice" are fine;
   things like `48+24` are not).

[Calculation]
   One line per step. Each line is plain English describing the step, followed by
   exactly one calculation for that step, in this exact shape:

       <what this step does, in plain English> -> <<expression=result>>

   Rules for every line:
     - FIRST say what the step is doing (e.g. "halve the April amount to get May",
       "add the two months") -- words only, no arithmetic in this part
     - THEN give exactly one <<expression=result>> for that description
     - the description and the expression must match (if it says "add", use +)
     - the expression may contain only digits and + - * / ( ) . and spaces
     - it must not contain dollar signs, percent signs, commas, units, letters or
       variables (put any unit outside the <<>>, after it)
     - the right-hand side must be the exact value of that expression
       (for fractions, write a decimal with enough digits)
   The last line must produce the quantity the problem asks for.

[Answer]
   The last line must be exactly: #### number
   (a bare number -- no units, no commas, no explanation)

---------------------------- THREE IRON RULES ----------------------------

1) Inside <<>> keep only bare arithmetic; exactly ONE <<>> per line.
   BAD:  <<48+24=72 clips>>          GOOD: <<48+24=72>> clips
   BAD:  add them <<48+24=72>> and also <<72*2=144>>
   GOOD: add the two months -> <<48+24=72>>
2) The right-hand side must really equal the left. Recompute it yourself;
   one wrong step ruins the whole record.
3) The answer line must be exactly one line of the form #### N, where N is a bare number.

---------------------------- EXAMPLE 1 ----------------------------

[Problem]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips
in May. How many clips did Natalia sell altogether in April and May?

[Plan]
The number sold in May is half the number sold in April. So first compute the May
quantity, then add the two months together.

[Calculation]
Halve the April amount to get the May amount -> <<48/2=24>>
Add the two months together -> <<48+24=72>>

[Answer]
#### 72

---------------------------- EXAMPLE 2 (decimals) ----------------------------

[Problem]
A shirt costs 12.5 dollars. I buy 3 shirts and pay with a 50 dollar bill.
How much change do I get?

[Plan]
First compute the total price of three shirts, then subtract that total from the
amount paid.

[Calculation]
Multiply the price of one shirt by three -> <<12.5*3=37.5>>
Subtract the total price from the amount paid -> <<50-37.5=12.5>>

[Answer]
#### 12.5

---------------------------- SELF-CHECK (before you answer) ----------------------------

- All three headers present, in the order: [Plan], [Calculation], [Answer]
- [Plan] contains no arithmetic expression like a+b
- Every line of [Calculation] is: <plain-English step> -> <<...=...>>
- Each [Calculation] line has exactly one <<>>, and the words match the operator
- Every expression inside <<>> uses only 0-9 + - * / ( ) . and spaces
- Every right-hand side is actually correct (recompute it)
- The last line is exactly #### N (bare number, no units)
- The N in [Answer] matches the final quantity computed in [Calculation]
- Everything is in English and plain ASCII (no Chinese characters, no fancy symbols)

Output only these three sections. No extra commentary, no markdown code fences,
no restating the problem.
"""

#: 外层要求（写进 `question` 字段的那段 ✓ 保持和原来一致 ✓）
QUESTION_TMPL = """Answer the following math problem in a strict three-stage format.

[Problem]
{question}

Requirements:
[Plan]
Explain the solution steps in plain English (no arithmetic expressions).

[Calculation]
One line per step, shaped: <plain-English step> -> <<expression=result>> Say what the
step does first (no arithmetic in the words), then give exactly one <<>> for it. The
expression may contain only digits and + - * / ( ) . and spaces.

[Answer]
The final line must be `#### number`.

Write everything in English only.
"""


# ── 非 ASCII 清理 ✓（模型偶尔会吐 ✓ ✗ ★ 之类 ✓）──────────────────────────
_KEEP = {"\u2192": "->"}          # 万一模型还是用了箭头 ✓ 换成 ASCII ✓
_DROP = set("\u2713\u2714\u2717\u2718\u2605\u2606\u26a0\u2705\u274c")


def to_ascii(t: str) -> str:
    out = []
    for ch in t:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in _KEEP:
            out.append(_KEEP[ch])
        elif ch in _DROP:
            continue
        else:
            out.append("?")        # 无法识别的 → 打问号 ✓（后面会被筛掉 ✓）
    return "".join(out)


# ── 一条记录 ✓ ────────────────────────────────────────────────────
def gen_one(client, idx: int, question: str, model: str,
            temperature: float, max_retries: int = 6) -> dict | None:
    """调一次 API 生成一条 ✓ 失败返回 None ✓（Ray 那边还会再重试 ✓）"""
    prompt = PROMPT_TMPL.format(question=question)
    last = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = r.choices[0].message.content or ""
            return {
                "idx": idx,
                "question": QUESTION_TMPL.format(question=question),
                "ans": to_ascii(text),
                # ★ 顺便带上裸题和 gold ✓ 后面能直接查 ✓ 少一次匹配 ✓✓
                "raw_question": question,
            }
        except Exception as e:                              # noqa: BLE001
            last = e
            # ★ 429 / 5xx / 超时 → 指数退避 ✓✓
            wait = min(60.0, (2 ** attempt) + random.random())
            print(f"  [warn] idx={idx} 第 {attempt + 1} 次失败：{type(e).__name__}: "
                  f"{str(e)[:120]} → {wait:.1f}s 后重试", flush=True)
            time.sleep(wait)
    print(f"  [error] idx={idx} 放弃 ✗ 最后一次：{last}", flush=True)
    return None


# ── 主流程 ✓ ──────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default=None, dest="train_jsonl",
                    help="可选：直接读这个 jsonl 当题库（每条要有 question）。"
                         " 默认用 data.load_gsm8k(train) ✓ 不需要 parquet ✓")
    ap.add_argument("--out", default=None, help="默认 gsm8k_new_train_4k.jsonl")
    ap.add_argument("--start", type=int, default=4000, help="从第几条开始（含 ✓）")
    ap.add_argument("--end", type=int, default=None, help="到第几条为止（不含 ✓ 默认到结尾 ✓）")
    ap.add_argument("--limit", type=int, default=None, help="这次最多生成几条（试跑用 ✓）")
    ap.add_argument("--workers", type=int, default=4, help="并发路数（默认 4 ✓）")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--flush-every", type=int, default=20, help="每 N 条落盘一次 ✓")
    ap.add_argument("--verbose", action="store_true", help="打印每条结果（调试用 ✗ 会很长 ✓）")
    a = ap.parse_args()

    out = a.out or "gsm8k_new_train_4k.jsonl"

    # ★ API key 只从环境变量读 ✓✓
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("✗✗ 没有 API key ✓ 请先 export DEEPSEEK_API_KEY=sk-xxxx", file=sys.stderr)
        print("     （不要把它写回源码里 ✗ 原来的 get_data.py 就写死过一个 ✓ 建议轮换 ✓）",
              file=sys.stderr)
        return 2

    from openai import OpenAI
    import data as D

    # 用**我们自己的** loader ✓✓ —— 不依赖 parquet / datasets ✓
    if a.train_jsonl:
        rows = [json.loads(l) for l in open(a.train_jsonl, encoding="utf-8") if l.strip()]
    else:
        rows = D.load_gsm8k("train")
    questions = [r["question"] for r in rows]
    N = len(questions)
    # 防线：官方 train 是 7473 条 ✓；test 只有 1319 ✓ → 太少就说明拿错了 ✓
    if N < 5000 and not a.train_jsonl:
        raise SystemExit(f"只读到 {N} 条 → 可能拿成 test 了（train 应为 7473）")
    print(f"题库 {N} 条 ✓（来源：{a.train_jsonl or 'data.load_gsm8k(train)'}）")
    end = N if a.end is None else min(a.end, N)
    idxs = list(range(max(0, a.start), end))
    print(f"数据集 {N} 条 · 本次范围 [{a.start}, {end}) 共 {len(idxs)} 条 · "
          f"并发 {a.workers} 路 ✓")

    # ── 断点续跑 ✓：读已有输出里的 idx（旧记录没有 idx → 用 raw_question 兜底 ✓）──
    done: set[int] = set()
    done_q: set[str] = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d.get("idx"), int):
                    done.add(d["idx"])
                if d.get("raw_question"):
                    done_q.add(d["raw_question"])
        print(f"  已有输出 {out}：按 idx 判定已完成 {len(done)} 条 "
              f"· 按题目文本 {len(done_q)} 条 ✓")
    todo = [i for i in idxs if i not in done and questions[i] not in done_q]
    if a.limit is not None:
        todo = todo[: a.limit]
    print(f"  待生成 **{len(todo)}** 条 ✓（跳过已完成 {len(idxs) - len(todo)} 条 ✓）")
    if not todo:
        print("  没有要做的 ✓ 全部已完成 ✓")
        return 0

    # ── Ray ✓ ────────────────────────────────────────────────────
    import ray

    @ray.remote(num_cpus=0.25, max_retries=3)
    def task(i: int, q: str, okey: str, model: str, temp: float):
        # ★ client 在每个 worker 里**懒建一次** ✓（不能跨进程传 ✓）
        c = globals().get("_CLIENT")
        if c is None or globals().get("_CLIENT_KEY") != okey:
            c = OpenAI(api_key=okey, base_url="https://api.deepseek.com")
            globals()["_CLIENT"] = c
            globals()["_CLIENT_KEY"] = okey
        return gen_one(c, i, q, model, temp)

    if not ray.is_initialized():
        # ★ 只为并发用 Ray ✓ CPU 需求很小（任务是 I/O 等待 ✓）
        ray.init(num_cpus=max(1, a.workers), include_dashboard=False,
                 log_to_driver=False, ignore_reinit_error=True)

    futs = [task.remote(i, questions[i], key, a.model, a.temperature) for i in todo]
    total = len(futs)
    got = 0
    t0 = time.perf_counter()
    pend = futs
    buf: list[dict] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        with open(out, "a", encoding="utf-8") as f:      # ★ 逐条落盘 ✓ 崩溃也只丢几条 ✓
            for d in buf:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        buf = []

    try:
        while pend:
            ready, pend = ray.wait(pend, num_returns=min(a.flush_every, len(pend)),
                                   timeout=300)
            for f in ready:
                try:
                    d = ray.get(f)
                except Exception as e:                    # noqa: BLE001
                    print(f"  [error] task 失败：{type(e).__name__}: {e}", flush=True)
                    d = None
                got += 1
                if d is not None:
                    buf.append(d)
                    if a.verbose:
                        print(f"\n  ── idx={d['idx']} ──\n{d['ans'][:400]}")
                if got % a.flush_every == 0 or not pend:
                    flush()
                    el = time.perf_counter() - t0
                    print(f"  进度 {got}/{total} ✓（{el:.0f}s，"
                          f"{got / max(el, 1e-9) * 60:.1f} 条/分钟 ✓）", flush=True)
    finally:
        flush()
        if ray.is_initialized():
            ray.shutdown()

    print(f"\n完成 ✓ 用时 {time.perf_counter() - t0:.0f}s · 输出 {out} ✓")
    print(f"  继续跑同一条命令会自动跳过已完成的 ✓（按 idx ✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
