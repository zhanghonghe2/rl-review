# GSM8K Three-Stage Data Regeneration — Generation Prompt (English Only)

> Goal: use a strong model to rewrite **all 7473 GSM8K train answers** into a **strict three-stage** format ✓
> Goal: teach a 1.5B model the behaviour "**plan → compute → report**" ✓ (not to memorize answers ✗)
>
> ⚠️ **Why everything is in English** ✗✓
> Mixed-language output is a **known RL failure mode** ✓✓ — DeepSeek hit it in R1-Zero and had to
> add a *language-consistency reward* to fix it ✓. GSM8K is English anyway ✓ → an English template
> removes the risk **at the source** ✓✓
>
> ⚠️ **The format below is NOT arbitrary** ✗ — it must match `ppo_llm/data.py` **exactly** ✓✓
> Otherwise the generated data **fails validation** ✓ and the run is wasted ✓

---

## 0. Hard format constraints (from the code ✓ — do not change ✗)

| Item                       | Constraint                                 | Where                              |
| -------------------------- | ------------------------------------------ | ---------------------------------- |
| Arithmetic placeholder     | must be written `<<3+4=7>>`              | `_ANN = <<([^<>=]+)=([^<>]+)>>`  |
| Expression inside `<<>>` | **only** `0-9 + - * / ( ) . space` | `_SAFE = [0-9+\-*/(). ]+`        |
| Result inside `<<>>`     | plain number (commas tolerated ✓)         | `float(claimed.replace(",",""))` |
| Final answer               | **must** be an explicit `#### N`   | `answer_strict`                  |

**★★ Three iron rules that follow** ✓✓:

1. Inside `<<>>` there must be **no** `$` `%` `,` units letters or variables ✗
   (→ that step becomes "unverifiable" ✓ i.e. the same as not writing it at all ✓)
2. The expression inside `<<>>` **must actually equal** the number on the right ✗
   (→ otherwise the process reward is 0 ✓ and worse, it teaches wrong arithmetic ✗✓)
3. Units / currency / words **always go outside** `<<>>` ✓

---

# 1. Generation prompt (give this to a strong model, one problem per call ✓)

```text
You are a meticulous elementary-school math teacher. Rewrite the solution of the problem
below into a **strict three-stage** format. Write **everything in English only**.

[Problem]
{question}

──────────────────────────── FORMAT (must follow exactly) ────────────────────────────

Output **exactly three sections**, in this order, with the headers copied verbatim:

[Plan]
   Plain English prose explaining *what to compute, what to compute first, what next*.
   **Do not write any arithmetic expression here** (words like "twice" are fine;
   things like `48+24` are not).

[Calculation]
   One line per step. Each line is **plain English describing the step, followed by
   exactly one calculation for that step**, in this exact shape:

       <what this step does, in plain English> → <<expression=result>>

   Rules for every line:
     · FIRST say what the step is doing (e.g. "halve the April amount to get May",
       "add the two months") — words only, **no arithmetic in this part**
     · THEN give exactly **one** <<expression=result>> for that description
     · the description and the expression must **match** (if it says "add", use +)
     · the expression may contain **only** digits and + - * / ( ) . and spaces
     · it must **not** contain $ % , units letters or variables
       (put any unit outside the <<>>, after it)
     · the right-hand side must be the **exact** value of that expression
       (for fractions, write a decimal with enough digits)
   The last line must produce the quantity the problem asks for.

[Answer]
   The last line must be exactly: #### number
   (a bare number — no units, no commas, no explanation)

──────────────────────────── THREE IRON RULES ────────────────────────────

1) Inside <<>> keep only bare arithmetic; exactly ONE <<>> per line.
   BAD ✗: <<48+24=72 clips>>      GOOD ✓: <<48+24=72>> clips
   BAD ✗: add them <<48+24=72>> and also <<72*2=144>>
   GOOD ✓: add the two months → <<48+24=72>>
2) The right-hand side must **really** equal the left. Recompute it yourself;
   one wrong step ruins the whole record.
3) The answer line must be exactly one line of the form `#### N`, where N is a bare number.

──────────────────────────── EXAMPLE 1 ────────────────────────────

[Problem]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips
in May. How many clips did Natalia sell altogether in April and May?

[Plan]
The number sold in May is half the number sold in April. So first compute the May
quantity, then add the two months together.

[Calculation]
Halve the April amount to get the May amount → <<48/2=24>>
Add the two months together → <<48+24=72>>

[Answer]
#### 72

──────────────────────────── EXAMPLE 2 (decimals ✓) ────────────────────────────

[Problem]
A shirt costs 12.5 dollars. I buy 3 shirts and pay with a 50 dollar bill.
How much change do I get?

[Plan]
First compute the total price of three shirts, then subtract that total from the
amount paid.

[Calculation]
Multiply the price of one shirt by three → <<12.5*3=37.5>>
Subtract the total price from the amount paid → <<50-37.5=12.5>>

[Answer]
#### 12.5

──────────────────────────── SELF-CHECK (before you answer ✓) ────────────────────────────

- [ ] All three headers present, in the order: [Plan] → [Calculation] → [Answer]
- [ ] [Plan] contains no arithmetic expression like `a+b`
- [ ] Every line of [Calculation] is: `<plain-English step> → <<...=...>>`
- [ ] Each [Calculation] line has **exactly one** <<>>, and the words match the operator
- [ ] Every expression inside <<>> uses only 0-9 + - * / ( ) . and spaces
- [ ] Every right-hand side is **actually correct** (recompute it)
- [ ] The last line is exactly `#### N` (bare number ✓ no units ✓)
- [ ] The N in [Answer] matches the final quantity computed in [Calculation]
- [ ] Everything is in **English** — no Chinese characters anywhere

Output only these three sections. No extra commentary, no markdown code fences,
no restating the problem.
```

---

# 2. Training / inference prompt template (★ must be **isomorphic** to the above ✓✓)

> **Critical** ✗: the data is three-stage ✓ → so **inference must also request three stages** ✓
> Otherwise the structure learned in training never appears at deployment ✓ → wasted ✗
>
> ★ Keep it short: our `rollout` runs **0-shot** ✓ (`--n-shot 0` ✓) →
> **training / eval / inference must all use this exact template** ✓✓
> (`sft.py` ✓ `main.py` ✓ `probe_passk.py` ✓ all share `build_messages` ✓)

```text
Answer the following math problem in a strict three-stage format.

[Problem]
{question}

Requirements:
[Plan]         Explain the solution steps in plain English (no arithmetic expressions).
[Calculation]  One line per step, shaped: <plain-English step> → <<expression=result>>
               Say what the step does first (no arithmetic in the words), then give
               exactly one <<>> for it. The expression may contain only digits and
               + - * / ( ) . and spaces.
[Answer]       The final line must be `#### number`.

Write everything in English only.
```

---

# 3. Suggested generation parameters

```python
{
  "model": "<your strong model>",
  "temperature": 0.3,        # low temp ✓ more stable structured output ✓
  "max_tokens": 512,
  "n": 1,                    # use n>1 and filter if you want more candidates ✓
}
```

**★★ Sample k=3 per problem, then filter automatically** ✓✓:

```
① all three headers present, in order ✓
② every <<>> passes data.check_arith ✓
   (only [0-9+\-*/(). ] and it really computes ✓)
③ #### N matches the official gold via data.answer_strict ✓
④ sane length (don't let the distribution blow up ✓)
⑤ ★ **纯 ASCII** ✓（英文 ✓ 数字 ✓ 无 CJK ✓ 无 ✓✗★→ 之类 ✓）
→ keep everything that passes ✓ → typically **85~95%** survive ✓✓
```

---

# 4. Ready-to-run filter script

```python
# filter_staged.py —— run the strong model's jsonl through OUR parsers
import json, re, sys
import data as D

HDR = ("[Plan]", "[Calculation]", "[Answer]")

def split_stages(text: str):
    """Return (plan, calc, answer) or None if any header is missing / out of order."""
    idx = [text.find(h) for h in HDR]
    if any(i < 0 for i in idx) or not (idx[0] < idx[1] < idx[2]):
        return None
    return (text[idx[0] + len(HDR[0]):idx[1]],
            text[idx[1] + len(HDR[1]):idx[2]],
            text[idx[2] + len(HDR[2]):])

def has_non_ascii(s: str) -> bool:
    """★ 比只查 CJK 更严 ✓✓ —— 目标数据应是**纯 ASCII** ✓

    为什么必须这么严 ✗：`✓` `✗` `★` `→` `─` `·` 这些**不在 CJK 范围** ✗
    → 只查 CJK 会漏掉它们 ✓ → 而它们会随 prompt 进模型 ✓ → 模型模仿 ✗ → 污染数据 ✓✓
    允许的字符：ASCII 里的可打印字符 + 换行 ✓（英文 + 数字 + <<>=#.-*/() ✓ 够了 ✓）
    """
    return any(ord(ch) > 127 for ch in s)

def keep(item: dict, gold: str):
    t = item["answer"]
    parts = split_stages(t)
    if parts is None:
        return False, "missing/out-of-order headers"
    plan, calc, ans = parts
    if not plan.strip():
        return False, "empty plan"
    if has_non_ascii(t):
        return False, "contains non-ASCII (mixed language / fancy symbols)"
    # ① plan must not contain arithmetic
    if re.search(r"<<|>>", plan) or re.search(r"\d\s*[+\-*/]\s*\d", plan):
        return False, "arithmetic inside [Plan]"
    # ② every <<>> must be verifiable AND correct
    tot, ok = D.process_stats(calc)
    if tot == 0:
        return False, "no verifiable calculation"
    if ok != tot:
        return False, f"wrong arithmetic ({ok}/{tot} ok)"
    # ③ answer must match gold under the STRICT criterion
    if not D.answer_strict(ans, gold):
        return False, "answer mismatch"
    return True, ""

if __name__ == "__main__":
    src, gold_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
    golds = {d["question"]: d["gold"]
             for d in (json.loads(l) for l in open(gold_file, encoding="utf-8"))}
    kept = dropped = 0
    reasons = {}
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            q = it["question"]
            if q not in golds:
                continue
            good, why = keep(it, golds[q])
            if good:
                f.write(json.dumps({"question": q,
                                    "answer": it["answer"].strip(),
                                    "gold": golds[q]}, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
                reasons[why] = reasons.get(why, 0) + 1
    print(f"kept {kept} ✓ / dropped {dropped} ✗")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"   {v:>5}  {k}")
```

---

# 5. ★ A risk worth thinking about first

```
This is "SFT on teacher data" ✓ — not RL ✓
→ does it actually "unlock potential" ✗?
   · if 1.5B can already compute but cannot structure its answer ✓
     → structured data is **very useful** ✓✓
   · if 1.5B genuinely cannot compute ✗
     → structure alone will not save it ✗ → you need **harder / more** problems ✓
→ ★ So: **first try 200 problems** ✓ → measure whether greedy beats 0.617 ✓✓
   (don't regenerate all 7473 up front ✗ — generation costs tokens and the direction is unverified ✓)
```

**★★ Recommended order** ✓✓:

1. Take **200 train problems** ✓ → generate → filter ✓ → **eyeball 5 records** ✓ (is the format right ✓)
2. Small SFT on those 200 ✓ → `sft.py --eval-only --eval-size 128 --eval-chunk 16 --vllm` ✓
   → compare with **sft_15b's 0.617** ✓✓
3. If it helps ✓ → scale to **1500** (same size as the original SFT ✓ → fair comparison ✓✓)
4. If it does not ✗ → the bottleneck is **computation, not formatting** ✓ → change the data, not the layout ✗
