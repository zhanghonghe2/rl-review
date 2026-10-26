"""clean_staged.py —— 清洗 + 筛选"三段式"数据（已生成的旧数据也能救 ✓✓）

## 它解决什么 ✗

用**旧 prompt**（里面带 `✓` `✗` `★` `→`）生成的 jsonl ✓ 可能：
    · 正文里混进了这些符号 ✗ —— **删掉就没事** ✓✓（它们不属于数学 ✓）
    · 缺 `[Plan]/[Calculation]/[Answer]` 标题 ✗ —— 这种**救不了** ✗
    · `<<>>` 里的算式算错 ✗ —— 救不了 ✗（除非能重算 ✓ 但那要模型 ✗）

→ 所以：**先清洗，再筛选，最后报告** ✓✓
   能救的救回来 ✓ 救不了的**分类统计原因** ✓ 你再决定要不要重生成几条 ✓

## 用法

    # ① 先只扫描（不写文件 ✓ 看能救多少 ✓）
    python3 clean_staged.py --src gen.jsonl --gold train.jsonl

    # ② 扫描 + 写出清洗后的文件 ✓
    python3 clean_staged.py --src gen.jsonl --gold train.jsonl --out clean.jsonl

    # ③ 自测（不需要数据 ✓）
    python3 clean_staged.py --self-test

## 输入格式

    src  : 每行 {"question": ..., "answer": ...}
    gold : 每行 {"question": ..., "gold": ...}（可以是 GSM8K 官方 train 的 jsonl ✓）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata

import data as D

HDR = ("[Plan]", "[Calculation]", "[Answer]")


# ── 清洗 ✓ ────────────────────────────────────────────────────────
#: 这些是"排版/装饰"字符 ✓ 删掉不影响语义 ✓✓
_DECOR = {
    "\u2192": "->",     # →  箭头：换成 ASCII ✓（它可能是有意义的 ✓）
    "\u2014": "--",     # —  破折号
    "\u2013": "-",      # –  短破折号
    "\u00b7": "-",      # ·  间隔号
    "\u2022": "-",      # •  圆点
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
    "\u00a0": " ",      # 不换行空格 → 普通空格 ✓
    "\u3000": " ",      # 全角空格
    "\uff1a": ":", "\uff0c": ",", "\uff08": "(", "\uff09": ")",
}
#: 直接删掉的（纯装饰 ✓ 删了更好 ✓✓）
_DROP = set("\u2713\u2714\u2717\u2718\u2605\u2606\u26a0\u2705\u274c\u2757\u2753\u2764")  # ✓✔✗✘★☆⚠✅❌❗❓❤


def clean(text: str) -> tuple[str, dict]:
    """把一段文本清洗成**纯 ASCII** ✓ 返回 (新文本, 统计)"""
    stats = {"replaced": 0, "dropped": 0, "other": []}
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        if ch in _DROP:
            stats["dropped"] += 1
            continue
        if ch in _DECOR:
            out.append(_DECOR[ch])
            stats["replaced"] += 1
            continue
        # 剩下的（中文 ✓ 全角 ✓ 罕见符号 ✗）→ 记下来 ✓ 但不乱删 ✗
        stats["other"].append(ch)
        out.append(ch)
    return "".join(out), stats


# ── 拆三段 ✓ ──────────────────────────────────────────────────────
def extract_question(text: str) -> str:
    """★ 从"整段 prompt"里抽出**真正的题目** ✓✓

    实测的坑 ✗：生成时把**整个 prompt** 塞进了 `question` 字段 ✓
        {"question": "\nAnswer the following... [Problem]\nNatalia sold... Requirements:..."}
    → 直接拿它去 gold 表里查 ✓ → **2620 条全找不到** ✗✓
    → 所以这里把 `[Problem]` 和 `Requirements:` 之间的那段抽出来 ✓✓
    """
    t = (text or "").strip()
    if "[Problem]" not in t:
        return t
    a = t.index("[Problem]") + len("[Problem]")
    for end_mark in ("Requirements:", "[Plan]", "[Calculation]"):
        b = t.find(end_mark, a)
        if b > a:
            return t[a:b].strip()
    return t[a:].strip()


def split_stages(text: str):
    idx = [text.find(h) for h in HDR]
    if any(i < 0 for i in idx) or not (idx[0] < idx[1] < idx[2]):
        return None
    return (text[idx[0] + len(HDR[0]):idx[1]],
            text[idx[1] + len(HDR[1]):idx[2]],
            text[idx[2] + len(HDR[2]):])


def has_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)


# ── 判定 ✓ ────────────────────────────────────────────────────────
def judge(text: str, gold: str) -> tuple[bool, str]:
    """返回 (能否用, 原因) ✓ —— 原因同时用于统计 ✓"""
    parts = split_stages(text)
    if parts is None:
        return False, "缺三段标题"
    plan, calc, ans = parts
    if not plan.strip():
        return False, "思路为空"
    if has_non_ascii(text):
        return False, "清洗后仍有非 ASCII"
    if re.search(r"<<|>>", plan):
        return False, "思路里出现 <<>>"
    if re.search(r"\d\s*[+\-*/]\s*\d", plan):
        return False, "思路里出现算式"
    tot, ok = D.process_stats(calc)
    if tot == 0:
        return False, "没有可验证的算式"
    if ok != tot:
        return False, f"算式有错 ({ok}/{tot})"
    if not D.answer_strict(ans, gold):
        return False, "答案与 gold 不符"
    if not calc.strip():
        return False, "计算段为空"
    return True, ""


# ── 主流程 ✓ ──────────────────────────────────────────────────────
def run(src: str, gold_path: str, out: str | None,
        show_dropped: int = 0) -> int:
    golds = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q = d.get("question")
            if not q:
                continue
            # ★ 两种格式都认 ✓✓
            #   · 我们的预处理产物：{"question", "gold"}
            #   · **官方 GSM8K jsonl**：{"question", "answer"}  ← answer 里带 "#### N" ✗✓
            #     （第一次跑就栽在这 ✗ → "gold 表 0 条" ✓）
            if "gold" in d:
                golds[q] = d["gold"]
            elif "answer" in d:
                golds[q] = str(d["answer"]).split("####")[-1].strip()
    if not golds:
        print(f"  ✗✗ gold 表是空的 ✗ —— 检查 {gold_path} 的字段 ✓")
        print("      需要 'gold' 或 'answer'（answer 里要有 #### N）✗")
        return 2
    print(f"  gold 表 {len(golds)} 条 ✓")

    n = kept = dropped = missing = 0
    reasons: dict[str, int] = {}
    cleaned_cnt = 0
    other_chars: dict[str, int] = {}

    fout = open(out, "w", encoding="utf-8") if out else None
    try:
        with open(src, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                it = json.loads(line)
                q_raw = it.get("question", "")
                q = extract_question(q_raw)            # ★ 先抽真题 ✓
                if q not in golds and q_raw in golds:  # 退回原文 ✓
                    q = q_raw
                if q not in golds:
                    missing += 1
                    if missing <= 2:                   # 只打前两条，方便你核对 ✗
                        print(f"  ⚠️ 匹配不上（例）：{q[:70]!r}")
                    continue
                # ★ 字段名不统一 ✗ —— 生成脚本用的是 `ans` ✓ 我原来只读 `answer` ✗✓
                raw = it.get("answer") or it.get("ans") or it.get("response") or ""
                if not raw and isinstance(it.get("output"), str):
                    raw = it["output"]
                new, st = clean(raw)
                if st["replaced"] or st["dropped"]:
                    cleaned_cnt += 1
                for ch in st["other"]:
                    other_chars[ch] = other_chars.get(ch, 0) + 1
                ok, why = judge(new, golds[q])
                if ok:
                    kept += 1
                    if fout:
                        # ★★ `sft.py --data` 要的是 **`solution`** ✗✓ 不是 `answer` ✗
                        #    （`sft.py:372` 里 need = ("question", "solution") ✓
                        #      缺了就 `ValueError: --data 缺字段 ['solution']` ✓）
                        #    → 两个都写 ✓ 谁读都不会缺 ✓✓
                        # ★★ `question` 必须保留**原始**（= 模板 + 裸题 ✓✓）
                        #   我原来只写了 `q`（抽出来的裸题 ✗）→ **把模型生成时看到的
                        #   三段式请求模板剥掉了** ✗✗ → 训练输入 ≠ 生成输入 ✓
                        #   → 现在：question = 原文 ✓ · raw_question = 裸题 ✓（匹配/统计用）
                        fout.write(json.dumps({"question": q_raw.strip(),
                                               "raw_question": q,
                                               "solution": new.strip(),
                                               "answer": new.strip(),
                                               "gold": golds[q]},
                                              ensure_ascii=True) + "\n")
                else:
                    dropped += 1
                    reasons[why] = reasons.get(why, 0) + 1
                    if dropped <= show_dropped:            # ★ 打原文 ✓ 便于判 ✓
                        print(f"\n  ── 丢弃 #{dropped}（{why}）──")
                        print(f"     gold = {golds[q]!r}")
                        print(f"     {new.strip()[:600]}")
                        print("  " + "-" * 60)
    finally:
        if fout:
            fout.close()

    print(f"\n  总数 {n}")
    print(f"  被清洗过（有装饰符号 ✓）{cleaned_cnt} 条  ← 这些**救回来了** ✓✓")
    print(f"  gold 里找不到的 {missing} 条")
    print(f"  ✅ 可用 **{kept}** 条  ({kept / max(n, 1):.1%})")
    print(f"  ✗ 丢弃 {dropped} 条")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"      {v:>5}  {k}")
    if other_chars:
        top = sorted(other_chars.items(), key=lambda x: -x[1])[:8]
        print(f"\n  ⚠️ 还有无法自动处理的非 ASCII 字符（前 8 个）：")
        for ch, c in top:
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "?"
            print(f"      {ch!r} × {c}   {name}")
        print("      → 若这些是中文正文 ✗ 那这几条只能重生成 ✓")
    if out:
        print(f"\n  已写出 {out}（**纯 ASCII** ✓ 可直接喂 sft.py ✓）")
    return 0


# ── 自测 ✓ ────────────────────────────────────────────────────────
def _self_test() -> int:
    fails = 0

    def ck(cond, msg):
        nonlocal fails
        print(f"  {'✓' if cond else '✗✗'} {msg}")
        if not cond:
            fails += 1

    print("① 清洗：装饰符号 → ASCII / 删除")
    t, st = clean("Halve ✓ then add ✗ → done")
    ck(t == "Halve  then add  -> done", f"结果 {t!r}")
    ck(st["dropped"] == 2 and st["replaced"] == 1, f"统计 {st}")

    print("\n② 清洗后是纯 ASCII")
    ck(not has_non_ascii(t), "has_non_ascii=False")

    print("\n③ 中文不会被乱删（会记为 other ✗ 让人工处理 ✓）")
    t2, st2 = clean("先算五月 → 再相加")
    ck("先算五月" in t2, "中文保留 ✓")
    ck(len(st2["other"]) >= 4, f"记为待处理 other: {st2['other']}")

    GOOD = ("[Plan]\nThe number sold in May is half the number sold in April.\n\n"
            "[Calculation]\nHalve the April amount to get the May amount -> <<48/2=24>>\n"
            "Add the two months together -> <<48+24=72>>\n\n"
            "[Answer]\n#### 72")
    print("\n④ 干净的三段式通过")
    ck(judge(GOOD, "72")[0], f"{judge(GOOD, '72')}")

    print("\n⑤ 带 ✓ 的 → 清洗后通过 ✓✓（这就是【救回来】✓）")
    dirty = GOOD.replace("Halve the April", "Halve the April ✓")   # ★ 注意大小写 ✗✓
    ck(judge(dirty, "72")[0] is False, "未清洗时被拒 ✓")
    ck(judge(clean(dirty)[0], "72")[0] is True, "清洗后通过 ✓✓")

    print("\n⑥ 该拒的仍然拒 ✓")
    ck(judge(GOOD.replace("#### 72", "#### 73"), "72")[1] == "答案与 gold 不符", "答案不符")
    ck(judge(GOOD.replace("<<48/2=24>>", "<<48/2=25>>"), "72")[1].startswith("算式有错"), "算式错")
    ck(judge("no headers here", "72")[1] == "缺三段标题", "缺标题")
    ck(judge(GOOD.replace("half the number", "half of 48/2"), "72")[1] == "思路里出现算式", "思路含算式")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓ —— 旧数据可以先用它扫一遍 ✓")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="生成的 jsonl（每条 {question, answer}）")
    ap.add_argument("--gold", help="gold 表 jsonl（每条 {question, gold}）")
    ap.add_argument("--out", help="清洗后的输出 jsonl（不给就只扫描 ✓）")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--show-dropped", type=int, default=0, dest="show_dropped",
                    help="把被丢弃的前 N 条**原文**打出来 ✓（判断是模型错还是 gold 错 ✓✓）")
    ap.add_argument("--from-gsm8k", action="store_true", dest="from_gsm8k",
                    help="直接用官方 GSM8K train 当 gold（不需要 --gold ✓）")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.src:
        ap.error("需要 --src（或用 --self-test）")
    if a.from_gsm8k:
        # ★ 直接读官方 train ✓ 写成临时 gold 表 ✓（字段用 gold ✓）
        import tempfile
        tr = D.load_gsm8k("train")
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                          encoding="utf-8")
        for d in tr:
            tmp.write(json.dumps({"question": d["question"], "gold": d["gold"]},
                                 ensure_ascii=False) + "\n")
        tmp.close()
        print(f"  从官方 train 读到 {len(tr)} 条 gold ✓")
        return run(a.src, tmp.name, a.out, a.show_dropped)
    if not a.gold:
        ap.error("需要 --gold 或 --from-gsm8k")
    return run(a.src, a.gold, a.out, a.show_dropped)


if __name__ == "__main__":
    sys.exit(main())
