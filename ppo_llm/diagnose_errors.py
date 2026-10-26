"""diagnose_errors.py —— 把"答错"的题按**错误位置**分类 ✓✓（不需要大模型 ✓）

## 为什么需要它 ✗

我们只知道"三段式数据的模型低了 7 点" ✗，但**不知道错在哪** ✓ →
而"分段奖励有没有用"**完全取决于**这一点 ✓✓：

    错在 ②【算式算错】   → 分段奖励（奖励 <<>> 全对）**直接有效** ✓✓
    错在 ③【算式都对但列错式/理解错】 → 分段奖励**没用** ✗（它奖励的项已经满分）
                                        → 那才需要 judge 去判"思路" ✗✓

## 四类（三类可自动判 ✓）

    ① 形式错    没有 `#### N`                          ← format_ok ✗
    ② 算错      某条 `<<expr=result>>` **不成立**       ← check_arith False ✓可判
    ③ 算对但答错 所有 `<<>>` 都成立 ✓ 但最后一个结果 ≠ gold  ← ✓可判（逻辑/列式错）
    ④ 抄错      `#### N` ≠ 最后一个 `<<>>` 的值         ← ✓可判
    ⑤ 其它（无算式/无法解析）                            ← 兜底

    ★ ③ 才是"真不会" ✓ —— 它占多少，直接决定该不该请 judge ✓✓

## 用法

    cd /data/ppo
    export PPO_MODEL=/data/models/Qwen2.5-1.5B-Instruct

    # 在官方 test 的前 200 题上跑，分类两个模型 ✓（各约 1 分钟 ✓）
    python3 diagnose_errors.py --adapter /data/ppo/runs/sft_off1500  --problems 200 \
        --vllm --vllm-sleep-level 0
    python3 diagnose_errors.py --adapter /data/ppo/runs/sft_stg1500  --problems 200 \
        --vllm --vllm-sleep-level 0

## 判读

    ② 占比高  → **PPO + 算式分段奖励**值得做 ✓✓（能把"算错"直接压下去 ✓）
    ③ 占比高  → 分段奖励**没用** ✗ → 要判"思路" ✓（judge ✓ 或换更强模型 ✓）
    ④ 占比高  → 只要**格式约束**就够 ✓（不需要奖励 ✓ 加一条"抄对"的规则检查 ✓✓）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from config import PPOConfig
from model import (MODEL_PATH, LMWithValueHead, empty_cache, load_tokenizer,
                   rollout)
from batch import gen_start


def _anns(txt: str) -> list[tuple[str, str]]:
    """文本里所有 (expr, claimed) ✓"""
    return D._ANN.findall(txt)


def classify(txt: str, gold: str) -> tuple[str, dict]:
    """返回 (类别, 细节) ✓"""
    info: dict = {}
    # ① 形式：必须有 #### N
    hits = re.findall(r"####\s*([^\n]*)", txt)
    if not hits:
        return "① 形式错（没有 ####）", info
    ans_v = D.normalize_number(hits[-1])
    if ans_v is None:
        return "① 形式错（#### 后没数字）", info
    g = D.normalize_number(gold)
    ok = g is not None and abs(ans_v - g) < 1e-6
    if ok:
        return "✅ 对", info

    # 到这里 = 答错 ✓
    pairs = _anns(txt)
    if not pairs:
        return "⑤ 无算式（无法定位）", info
    checked = [(e, c, D.check_arith(e, c)) for e, c in pairs]
    verifiable = [(e, c, r) for e, c, r in checked if r is not None]
    if not verifiable:
        return "⑤ 算式不可验证（格式不合规）", info
    bad = [(e, c) for e, c, r in verifiable if not r]
    info["n_ann"] = len(verifiable)
    info["n_bad"] = len(bad)
    info["bad_examples"] = bad[:2]

    # ② 算错：有算式不成立
    if bad:
        return "② 算错（<<>> 不成立）", info

    # ④ 抄错：#### 与最后一个 <<>> 的结果不一致
    last_expr, last_claimed = verifiable[-1][0], verifiable[-1][1]
    last_v = D.normalize_number(last_claimed)
    if last_v is not None and abs(last_v - ans_v) > 1e-6:
        info["last_calc"] = last_v
        info["answer_line"] = ans_v
        return "④ 抄错（#### ≠ 最后算式）", info

    # ③ 算对但答错：算式全成立，最后一个结果 = ####，但仍 ≠ gold
    return "③ 算对但答错（列式/理解错）", info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--test-file", type=str, default=None, dest="test_file",
                    help="★ 自定义测试集 jsonl ✓（question + gold 或含 #### 的 answer ✓）"
                         " 默认用官方 test ✓")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512, dest="mnt")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = 贪心（推荐 ✓ 看它到底会什么）")
    ap.add_argument("--dump", default=None, help="把分类结果写到这个 jsonl ✓")
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--vllm-kv-gb", type=float, default=1.5)
    ap.add_argument("--vllm-sleep-level", type=int, default=0, dest="vllm_sleep_level")
    a = ap.parse_args()

    tok = load_tokenizer(a.model)
    # dtype 传的是**字符串** ✗ 不是 torch.dtype（`to_pt_dtype` 会 getattr(torch, name) ✓）
    model = LMWithValueHead(a.model, device="cuda", lora=True,
                            init_lora=a.adapter, dtype="float32",
                            autocast="bfloat16")
    model.eval()

    vb = None
    if a.vllm:
        from vllm_backend import build_from_adapter
        vb = build_from_adapter(a.model, a.adapter,
                                gpu_util=a.vllm_gpu_util, kv_gb=a.vllm_kv_gb,
                                sleep_level=a.vllm_sleep_level)

    test = (D.load_test_file(a.test_file) if a.test_file
            else D.load_gsm8k("test"))[:a.problems]
    cfg = PPOConfig.preset("llm").replace(temperature=a.temperature,
                                          max_new_tokens=a.mnt, kl_coef=0.0)
    counts: dict[str, int] = {}
    rows_out = []
    n = 0
    for i in range(0, len(test), a.chunk):
        part = test[i:i + a.chunk]
        b = rollout(model, tok, part, None, cfg, seed=0, vllm_backend=vb)
        Lp = gen_start(b.mask)
        for j, pr in enumerate(part):
            txt = tok.decode(b.input_ids[j, int(Lp[j]):], skip_special_tokens=True)
            cat, info = classify(txt, pr["gold"])
            counts[cat] = counts.get(cat, 0) + 1
            n += 1
            if a.dump and cat != "✅ 对":
                rows_out.append({"i": n, "gold": pr["gold"], "cat": cat,
                                 **info, "text": txt})
        del b
        empty_cache()

    print(f"\n{'=' * 66}\n  错误分类（{n} 题 · 贪心 · adapter={a.adapter}）\n{'=' * 66}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {v:>5}  ({v / max(n, 1):>6.1%})  {k}")
    wrong = sum(v for k, v in counts.items() if k != "✅ 对")
    print(f"\n  答错 {wrong} 题 ✓")
    if wrong:
        calc = counts.get("② 算错（<<>> 不成立）", 0)
        logic = counts.get("③ 算对但答错（列式/理解错）", 0)
        copy = counts.get("④ 抄错（#### ≠ 最后算式）", 0)
        print(f"\n  ★ 判读：")
        print(f"     ② 算错 {calc} 题（{calc / wrong:.0%}）→ **PPO + 算式分段奖励**有效 ✓")
        print(f"     ③ 逻辑错 {logic} 题（{logic / wrong:.0%}）→ 分段奖励**无效** ✗ 需判思路")
        print(f"     ④ 抄错 {copy} 题（{copy / wrong:.0%}）→ 只需**格式/抄写约束** ✓")
        if logic >= calc:
            print(f"\n     → ★★ **逻辑错 ≥ 算错** → 分段奖励帮不上 ✓ 该上 judge 判思路 ✓")
        else:
            print(f"\n     → ★★ **算错 > 逻辑错** → 先用**规则式算式奖励** ✓ 更便宜 ✓")
    if a.dump and rows_out:
        with open(a.dump, "w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        print(f"\n  明细已写 {a.dump} ✓（{len(rows_out)} 条错例 ✓ 可人工看几条 ✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
