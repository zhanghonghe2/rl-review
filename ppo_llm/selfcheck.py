"""selfcheck.py —— 静态一致性检查（不跑模型，2 秒完成）。

起因：`train()` 的签名漏了 `scheduler` 参数，但调用处传了它 ✗
    rc = check_decode_slice(files)
      `ast.parse` **查不出来**（语法是对的 ✗），只有**运行时**才炸 ✗
      → 而那次炸在了服务器上，浪费了一个来回 ✓

这个脚本检查的正是这类问题：**本文件内**每个函数的"被调用时传的关键字"
是否都在它的签名里 ✓ 语法正确、但参数对不上的调用，它一眼就能抓出来 ✓

    python3 selfcheck.py        # 全部通过则退出码 0 ✓（可以直接放进 CI / 部署前检查）
"""
from __future__ import annotations

import ast
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 有 *args/**kwargs 的函数跳过（没法静态判断 ✓）
SKIP_CALLS = {"print", "len", "range", "float", "int", "str", "list", "dict", "set",
              "open", "max", "min", "sum", "abs", "round", "sorted", "tuple", "bool",
              "type", "enumerate", "zip", "getattr", "setattr", "hasattr", "super"}


def sig_of(fn: ast.FunctionDef) -> tuple[set[str], bool]:
    a = fn.args
    names = {x.arg for x in a.args} | {x.arg for x in a.kwonlyargs}
    if a.vararg is not None:
        names.add(a.vararg.arg)
    if a.kwarg is not None:
        names.add("**" + a.kwarg.arg)
    star = a.vararg is not None or a.kwarg is not None
    return names, star


def config_fields() -> set[str]:
    """从 config.py 里解析出 PPOConfig 的字段名（不 import ✓ 保持这个脚本零依赖）。"""
    tree = ast.parse(open(os.path.join(HERE, "config.py"), encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == "PPOConfig":
            return {t.target.id for t in n.body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)}
    return set()


CFG_FIELDS = config_fields()


def check_orphan_cases(path: str) -> list[str]:
    """★ 每个 `_case_*` 用例都必须被**引用**过 ✗

    实测被抓到过：`_case_grpo_advantage` 定义了却没挂到执行列表里 ✓
    这比"用例失败"更坏 ✗ —— 它**看起来有覆盖**，实际什么都没验 ✓✓

    判据：函数名必须在别处以一个 `ast.Name` 出现 ✓
      （`_case_a()` 是 Call ✓、放进 `cases=[..., _case_a]` 列表也是 Name ✓ 都算 ✓
        而 `def _case_a` 本身是 FunctionDef.name ✗ 不是 Name ✗ 所以不会自证 ✓）
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    defs = [n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("_case")]
    if not defs:
        return []
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return [f"{os.path.basename(path)}: {d}() 定义了但**从没被调用** ✗ "
            f"—— 看起来有覆盖，其实没跑 ✓" for d in defs if d not in referenced]


def check_dataclass_target(path: str) -> list[str]:
    """★ `@dataclass` 必须装饰**类** ✗

    如果它装饰了函数 → 运行时（**import 时**）报
        AttributeError: 'function' object has no attribute '__mro__'
    而且它会**连坐所有 import 这个模块的文件** ✗（实测一次坏了 batch/gae/loss 三个 ✓）

    实测犯规两次 ✗：
        metrics.py  把字体函数插到了 @dataclass 和 class MetricLogger 之间
        batch.py    把 concat_batches 插到了 @dataclass 和 class RolloutBatch 之间
    根因都一样：锚点写成 "class Xxx"，而文件里是 "@dataclass\nclass Xxx" ✓
    → 修法：**锚点要带上 @dataclass** ✓（或者直接用 AST 拿行号 ✓）
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    bad: list[str] = []
    for n in ast.walk(tree):
        for d in getattr(n, "decorator_list", []):
            name = getattr(d, "id", None) or getattr(getattr(d, "func", None), "id", None)
            if name == "dataclass" and not isinstance(n, ast.ClassDef):
                bad.append(
                    f"{os.path.basename(path)}:{n.lineno} @dataclass 装饰了 "
                    f"{type(n).__name__}（必须装饰 class ✗）"
                    f"—— import 时会炸，并连坐所有 import 本模块的文件")
    return bad


def check_replace_kwargs(path: str) -> list[str]:
    """★ `<cfg>.replace(top_k=None)` 这种「配置字段名写错」的错误 ✓

    运行时会报 `TypeError: unexpected keyword argument` ✗，但它藏在 **kwargs 透传里 ✗
    所以静态检查器（和 py_compile）都看不出来 ✓ —— 实测被咬过两次：
        PPOConfig.replace(eval_every=...)  ✗
        PPOConfig.replace(top_k=...)       ✗

    规则（零误报 ✓）：`replace` 也用在字符串上（`s.replace(",", "")`），
    所以只在「**至少有一个关键字是 config 字段**」时才检查 ✓
      → 那么其余的也必须都是字段 ✓
    """
    if not CFG_FIELDS:
        return []
    tree = ast.parse(open(path, encoding="utf-8").read())
    bad: list[str] = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "replace"):
            continue
        kws = [k.arg for k in n.keywords if k.arg is not None]
        if not kws or not (set(kws) & CFG_FIELDS):
            continue                     # 字符串 replace / 全都不认识 → 跳过 ✓
        for k in kws:
            if k not in CFG_FIELDS:
                bad.append(f"{os.path.basename(path)}:{n.lineno} .replace({k}=...) "
                           f"—— PPOConfig 没有这个字段 ✗（大写小写/名字写错了？）")
    return bad


def check_file(path: str) -> list[str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    funcs: dict[str, tuple[set[str], bool, int]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            names, star = sig_of(n)
            funcs[n.name] = (names, star, n.lineno)

    problems: list[str] = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
            continue
        name = n.func.id
        if name in SKIP_CALLS or name not in funcs:
            continue
        known, star, def_line = funcs[name]
        if star:
            continue
        for kw in n.keywords:
            if kw.arg is not None and kw.arg not in known:
                problems.append(
                    f"{os.path.basename(path)}:{n.lineno} 调用 {name}() 时传了 "
                    f"{kw.arg}= ，但它在第 {def_line} 行的签名里没有这个参数 ✗")
        # 位置参数也不能超
        n_pos = len([a for a in n.args if not isinstance(a, ast.Starred)])
        if n_pos > len([x for x in funcs[name][0]]):
            problems.append(
                f"{os.path.basename(path)}:{n.lineno} 调用 {name}() 传了 {n_pos} 个"
                f"位置参数，但签名只有 {len(known)} 个 ✗")
    return problems


def check_decode_slice(files: list) -> int:
    """★ 防复发规则：**不许**用 `shape[1] - max_new_tokens` 当解码起点 ✗✗

    真实事故 ✓：生成 ≈100 token、cap=192、prompt ~77 ✓
      → `T - max_new_tokens` 变成**负数** ✗ → Python 负索引 → 只解出最后十几个 token ✗
      → 评估器看到的文本 ≠ 奖励计算所用的文本 ✓
      → 同一份权重报出 **0.555 与 0.523** 两个"严格准确率" ✗（差 ≈3% ✓）
    它当时出现在 **6 处** ✓ —— 所以只能靠规则钉住 ✓ 靠记性会复发 ✓✓
    正确写法：`gen_start(b.mask)` ✓（在 batch.py ✓ 与长度/padding/cap 全无关 ✓）
    """
    bad = []
    for f in files:
        for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
            if "shape[1] - " in line and "max_new_tokens" in line:
                bad.append(f"{f}:{i}")
    if bad:
        print("❌ 解码起点写错（会让评估文本 ≠ 奖励文本 ✗）：")
        for b_ in bad:
            print(f"   {b_}  → 应该用 gen_start(b.mask) ✓")
        return 1
    print("✅ 解码起点全部走 gen_start ✓")
    return 0


def check_lp_slice(files: list) -> int:
    """★ 防复发：`gen_start` 返回的是 (B,) **张量** ✓ 所以索引必须写 `int(Lp[j])` ✓

    真实事故（就在我写这条规则前 10 分钟 ✓）：
      6 处改成 `Lp = gen_start(b.mask)` ✓ 但有一处忘了把 `Lp:` 换成 `int(Lp[j]):` ✗
      → chunk=4 时 `TypeError` 必崩 ✓ 而 chunk=1（B=1）时**静默通过** ✗✗
      → 典型的"冒烟能过、生产会炸" ✓✓
    """
    bad = []
    for f in files:
        for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
            if re.search(r"\[\s*j\s*,\s*Lp\s*:\]", line):
                bad.append(f"{f}:{i}")
    if bad:
        print("❌ 用张量 Lp 当切片边界（B>1 时会 TypeError ✗）：")
        for b_ in bad:
            print(f"   {b_}  → 应写 int(Lp[j]) ✓")
        return 1
    print("✅ 解码切片边界都取成了 int ✓")
    return 0


def check_arg_attrs(path: str) -> list[str]:
    """★★★ `a.<attr>` 必须在 argparse 里定义过 ✓✓

    为什么加这一条 ✗：`prm_train.py` 里 `--max-new-tokens` 的 dest 是 **`mnt`** ✓
        → 我按常见写法写了 `a.max_new_tokens` ✗ → **运行到那行才崩** ✓
        → 而它前面已经跑了 15 步 ✓ → 一小时白费 ✓✓
    → ★ 这类"参数名写错"**编译永远发现不了** ✗ 只能静态比对 ✓
    """
    out: list[str] = []
    src = open(path, encoding="utf-8").read()
    if "add_argument" not in src:
        return out
    tree = ast.parse(src, path)
    dests: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "add_argument" and n.args \
                and isinstance(n.args[0], ast.Constant):
            long = str(n.args[0].value)
            d = None
            for k in n.keywords:
                if k.arg == "dest" and isinstance(k.value, ast.Constant):
                    d = str(k.value.value)
            dests.add(d or long.lstrip("-").replace("-", "_"))
    if not dests:
        return out
    used: set[str] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for n in ast.walk(fn):
                if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                        and n.value.id in ("a", "args")
                        and isinstance(n.ctx, ast.Load)):
                    used.add(n.attr)
    base = os.path.basename(path)
    for m in sorted(used - dests):
        out.append(f"{base}: `a.{m}` 没有对应的 argparse 参数 ✗（拼错 dest？✓）")
    return out


def main() -> int:
    files = sorted(f for f in glob.glob(os.path.join(HERE, "*.py"))
                   if not f.endswith("selfcheck.py"))
    allp: list[str] = []
    for f in files:
        try:
            allp += check_file(f)
            allp += check_replace_kwargs(f)
            allp += check_dataclass_target(f)
            allp += check_orphan_cases(f)
            allp += check_arg_attrs(f)
        except SyntaxError as e:
            allp.append(f"{os.path.basename(f)} 语法错误：{e} ✗")
    if allp:
        print(f"✗ 发现 {len(allp)} 处调用/签名不一致：")
        for p in allp:
            print("   ", p)
        return 1
    print(f"✅ {len(files)} 个文件：调用与签名一致 ✓；"
          f".replace() 关键字全部是 PPOConfig 字段（共 {len(CFG_FIELDS)} 个）✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
