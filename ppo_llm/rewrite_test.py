"""rewrite_test.py —— 造**私有 GSM8K 测试集**（换人名/场景/数字 ✓ 保持同难度 ✓✓）

## 为什么要它 ✗

基座 3B 在官方 GSM8K test 上拿 **0.827** ✓，而且**会写 `#### N`** ✗✓ ——
而 GSM8K 的题目里**没有任何** `####` 的描述 ✓ → 说明**它见过 GSM8K 的解答** ✗✓（数据泄漏 ✓）
→ 所以 0.827 **不可信** ✗ → 必须用**模型没见过的题**来量 ✓✓

## 做法（三关校验 ✓ 只保留可验证的 ✓✓）

    ① **改写**：换人名 / 场景 / 物品 / **数字** ✓（数学结构不变 ✓）
       同时要求输出三段式 + `<<expr=result>>` + `#### N` ✓ → 这样**答案能机械验证** ✓✓
    ② **独立重解**：再调一次 ✓ 只让它给 `#### N` ✓
    ③ **三重校验**（任一不过就丢弃 ✗）：
         · 所有 `<<expr=result>>` **都成立** ✓（`data.check_arith` ✓）
         · `#### N` == 最后一条 `<<>>` 的结果 ✓
         · `#### N` == 独立重解的答案 ✓（**两次一致** ✓✓）
    → ★ 这样 gold **不是"我们信它"** ✗ 而是**"它三次自洽"** ✓✓

## 用法

    export DEEPSEEK_API_KEY=sk-xxxx
    cd /data/ppo

    # 生成 128 题（会多要一些 ✓ 因为有丢弃率 ✓）
    python3 rewrite_test.py --src data/gsm8k_test_0.jsonl --out data/gsm8k_private128.jsonl \
        --n 128 --workers 4

    # 只校验已有文件 ✓
    python3 rewrite_test.py --verify data/gsm8k_private128.jsonl

    # 自测（不需要网络 ✓）
    python3 rewrite_test.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D

SYSTEM = ("You are a meticulous math competition editor. You rewrite problems and "
          "verify arithmetic. Output only what is asked, in plain ASCII.")

REWRITE = """Rewrite the math word problem below into a **brand-new equivalent problem**.

Hard requirements:
- Keep the **same solution structure**: same number of reasoning steps, same operations
  in the same order (e.g. if the original halves then adds, yours must halve then add).
- Change ALL of these: every **person name**, the **scene/setting**, the **objects**,
  and **every number** (including counts, prices and durations).
- The new numbers must keep the problem solvable with the same plan and the answers must
  be "nice" (no weird decimals beyond 2 places).
- Do NOT copy the original wording; write it as a fresh problem.
- Write everything in English, plain ASCII only.

Then give the solution in this exact three-stage format:

[Problem]
<the rewritten problem, no answer inside>

[Plan]
<plain-English plan, words only, no arithmetic>

[Calculation]
One line per step, shaped:  <plain-English step> -> <<expression=result>>
Only digits and + - * / ( ) . and spaces inside <<>>.
The right-hand side must be exactly correct.

[Answer]
#### <the final number>

Original problem:
[Problem]
{question}

Original solution (for reference on the STRUCTURE only — do not copy its numbers):
{solution}
"""

RESOLVE = """Solve this problem and reply with ONLY the final number (no words, no units).

[Problem]
{question}
"""


def _ascii(t: str) -> str:
    out = []
    for ch in t:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in "\u2192":
            out.append("->")
        elif ch in "\u2713\u2714\u2717\u2718\u2605\u2606":
            continue
        else:
            out.append("?")
    return "".join(out)


def _stages(txt: str):
    H = ("[Problem]", "[Plan]", "[Calculation]", "[Answer]")
    idx = [txt.find(h) for h in H]
    if any(i < 0 for i in idx) or idx != sorted(idx):
        return None
    return (txt[idx[0] + len(H[0]):idx[1]].strip(),
            txt[idx[1] + len(H[1]):idx[2]].strip(),
            txt[idx[2] + len(H[2]):idx[3]].strip(),
            txt[idx[3] + len(H[3]):].strip())


def verify_one(rec: dict) -> tuple[bool, str]:
    """★ 三重校验 ✓ 返回 (能不能用, 原因)"""
    txt = rec.get("raw", "")
    parts = _stages(txt)
    if parts is None:
        return False, "缺四段标题"
    problem, plan, calc, ans = parts
    if not problem or not calc or not ans:
        return False, "某段为空"
    if any(ord(c) > 127 for c in txt):
        return False, "非 ASCII"
    if re.search(r"<<|>>", plan) or re.search(r"\d\s*[+\-*/]\s*\d", plan):
        return False, "思路里出现算式"
    # ① 所有 <<>> 都必须成立
    pairs = D._ANN.findall(calc)
    if len(pairs) < 2:
        return False, f"算式太少（{len(pairs)} 条）"
    tot, ok = D.process_stats(calc)
    if tot == 0 or ok != tot:
        return False, f"算式有错（{ok}/{tot}）"
    # ② #### == 最后一条 <<>> 的结果
    m = re.search(r"####\s*([0-9.\-]+)", ans)
    if not m:
        return False, "answer 段没有 #### N"
    final = D.normalize_number(m.group(1))
    last_calc = D.normalize_number(pairs[-1][1])
    if final is None or last_calc is None or abs(final - last_calc) > 1e-6:
        return False, "#### 与最后算式不一致"
    # ③ 独立重解一致
    g2 = D.normalize_number(str(rec.get("_resolve", "")))
    if g2 is None:
        return False, "重解没有数字"
    if abs(final - g2) > 1e-6:
        return False, f"重解不一致（{m.group(1)} vs {rec['_resolve']!r}）"
    # ④ 数字必须**换过**（否则等于没改写 ✗）
    orig_nums = set(re.findall(r"\d+(?:\.\d+)?", rec.get("orig_question", "")))
    new_nums = set(re.findall(r"\d+(?:\.\d+)?", problem))
    if orig_nums and new_nums == orig_nums:
        return False, "数字一个都没换"
    rec["question"] = problem
    rec["solution"] = txt[txt.find("[Plan]"):].strip()
    rec["gold"] = str(final).rstrip("0").rstrip(".") if final != int(final) else str(int(final))
    return True, ""


def _one(client, rec: dict, model: str, max_retries: int = 5) -> dict:
    q, sol = rec["orig_question"], rec.get("orig_solution", "")
    for a in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.8,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": REWRITE.format(question=q, solution=sol)}])
            raw = _ascii(r.choices[0].message.content or "")
            parts = _stages(raw)
            out = {**rec, "raw": raw}
            if parts is None:
                return out
            # 独立重解 ✓
            r2 = client.chat.completions.create(
                model=model, temperature=0.0,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": RESOLVE.format(question=parts[0])}])
            out["_resolve"] = _ascii(r2.choices[0].message.content or "").strip()
            return out
        except Exception as e:                              # noqa: BLE001
            wait = min(30.0, 2 ** a + random.random())
            print(f"  [warn] {type(e).__name__}: {str(e)[:100]} → {wait:.1f}s", flush=True)
            time.sleep(wait)
    return {**rec, "raw": "", "_resolve": ""}


def _self_test() -> int:
    fails = 0

    def ck(c, m):
        nonlocal fails
        print(f"  {'✓' if c else '✗✗'} {m}")
        fails += 0 if c else 1

    good = """[Problem]
Marisol sold ribbons to 48 friends in April, then half as many in May. How many in total?

[Plan]
Halve the April amount, then add both months.

[Calculation]
Halve the April amount -> <<48/2=24>>
Add the two months -> <<48+24=72>>

[Answer]
#### 72"""
    r = {"raw": good, "_resolve": "72", "orig_question": "Natalia sold clips to 16 friends..."}
    ok, why = verify_one(r)
    ck(ok, f"合格件 → {why or 'ok'}")
    ck(r.get("gold") == "72", f"gold = {r.get('gold')}")

    def bad(mut, why_want, resolve="72", orig="Natalia sold clips to 16 friends"):
        rr = {"raw": mut, "_resolve": resolve, "orig_question": orig}
        o, w = verify_one(rr)
        ck((not o) and why_want in w, f"{why_want} → {w}")

    bad(good.replace("<<48/2=24>>", "<<48/2=25>>"), "算式有错")
    bad(good.replace("#### 72", "#### 73"), "与最后算式不一致")
    bad(good, "重解不一致", resolve="99")
    bad(good, "数字一个都没换", orig="Natalia sold clips to 48 friends")
    bad("no headers here", "缺四段标题")
    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gsm8k_test_0.jsonl")
    ap.add_argument("--out", default="data/gsm8k_private128.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--over", type=float, default=1.4, help="多要多少（有丢弃率 ✓）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--verify", default=None, help="只校验一个已有文件 ✓")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    if a.verify:
        rows = [json.loads(l) for l in open(a.verify, encoding="utf-8") if l.strip()]
        ok = 0
        for r in rows:
            good, why = verify_one(r)
            if good:
                ok += 1
            else:
                print(f"  ✗ {why}")
        print(f"\n  校验 {len(rows)} 条 · 通过 {ok}（{ok / max(len(rows), 1):.1%}）")
        return 0 if ok == len(rows) else 1

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("✗✗ 缺少 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2
    from openai import OpenAI

    # ★ 读原题（官方格式：question + answer ✓）
    src = []
    with open(a.src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q = d.get("question")
            if not q:
                continue
            sol = d.get("solution") or d.get("answer") or ""
            if "####" not in sol:
                continue
            src.append({"orig_question": q.strip(), "orig_solution": sol.strip()})
            if len(src) >= int(a.n * a.over):
                break
    print(f"  原题 {len(src)} 条（目标 {a.n} ✓ 丢弃余量 ×{a.over} ✓）")

    import ray

    @ray.remote(num_cpus=0.25, max_retries=2)
    def task(rec: dict, okey: str, model: str):
        c = globals().get("_CLIENT")
        if c is None or globals().get("_CK") != okey:
            c = OpenAI(api_key=okey, base_url="https://api.deepseek.com")
            globals()["_CLIENT"], globals()["_CK"] = c, okey
        return _one(c, rec, model)

    if not ray.is_initialized():
        ray.init(num_cpus=max(1, a.workers), include_dashboard=False,
                 log_to_driver=False, ignore_reinit_error=True)

    futs = [task.remote(r, key, a.model) for r in src]
    kept = 0
    reasons: dict[str, int] = {}
    t0 = time.perf_counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        pend = futs
        while pend and kept < a.n:
            ready, pend = ray.wait(pend, num_returns=min(16, len(pend)), timeout=300)
            for f in ready:
                try:
                    rec = ray.get(f)
                except Exception:                            # noqa: BLE001
                    reasons["任务失败"] = reasons.get("任务失败", 0) + 1
                    continue
                good, why = verify_one(rec)
                if not good:
                    reasons[why] = reasons.get(why, 0) + 1
                    continue
                kept += 1
                fo.write(json.dumps({"question": rec["question"],
                                     "solution": rec["solution"],
                                     "gold": rec["gold"]}, ensure_ascii=True) + "\n")
                fo.flush()
                if kept % 8 == 0 or kept == a.n:
                    el = time.perf_counter() - t0
                    print(f"  进度 {kept}/{a.n} ✓（{el:.0f}s）", flush=True)
                if kept >= a.n:
                    break
    if ray.is_initialized():
        ray.shutdown()
    print(f"\n  ✅ 合格 {kept}/{a.n} → {a.out}")
    if reasons:
        print("  丢弃原因：")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
            print(f"    {v:>4}  {k}")
    return 0 if kept >= a.n else 1


if __name__ == "__main__":
    sys.exit(main())
