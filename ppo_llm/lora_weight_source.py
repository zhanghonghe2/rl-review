"""lora_weight_source.py —— 给 vLLM 的权重源：**惰性合并 LoRA** ✓✓

## 为什么需要它 ✗✓

vLLM 的 `ModuleSource(module)` 是「`module.named_parameters()` 的视图」✓。
而我们的训练器只有 **4.4M 的 LoRA 参数**（base 冻结 ✗）→ 直接传它，
vLLM 收到的是不完整的权重 ✗✗。

所以这里实现 `MergedLoRAWeightSource(WeightSource)` ✓：
    · `metadata()`   → 报 **base** 的形状/名字 ✓（vLLM 按这个建缓冲 ✓）
    · `iter_groups()` → 每组**现算** `W + (B @ A) · s` ✓✓
→ 效果：vLLM 得到"合并后的完整权重" ✓ 而**显存里不多一份模型** ✓（只有几 MB 在飞 ✓✓）
→ 而且**不写盘** ✓（这正是要求 ✓）

## 本文件自带 CPU 自测 ✓（不需要 GPU / 不需要 vllm ✓）

    python3 lora_weight_source.py     # 自测：合并结果 == PEFT 的前向 ✓ + 抖动测试 ✓

## 合并公式（PEFT 约定）

    W_eff = W + (lora_B @ lora_A) * scaling,   scaling = lora_alpha / r
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch


# ===================================================================== #
# 合并的核心：给定 base 参数名，找出它对应的 LoRA 对 ✓
# ===================================================================== #
@dataclass
class LoRAPair:
    """一个被 LoRA 作用的线性层 ✓"""
    A: torch.Tensor          # (r, in)    ✓ lora_A
    B: torch.Tensor          # (out, r)   ✓ lora_B
    scaling: float           # alpha / r  ✓

    def delta(self, dtype=None, device=None) -> torch.Tensor:
        """(out, in) 的增量 ✓ —— 每次现算，不缓存 ✓（省显存 ✓）"""
        d = (self.B.to(torch.float32) @ self.A.to(torch.float32)) * self.scaling
        return d.to(dtype=dtype, device=device)


def parse_lora_name(name: str) -> tuple[str, str] | None:
    """把 PEFT 的 adapter 名拆成 (base 参数名, 'A'|'B') ✓

    典型 ✓：
        base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
    → base = base_model.model.model.layers.0.self_attn.q_proj.weight ✓
    """
    for tag, which in ((".lora_A.", "A"), (".lora_B.", "B")):
        if tag in name:
            head, tail = name.split(tag, 1)
            # tail 形如 "default.weight" ✓ → 去掉 adapter 名 ✓
            if "." in tail:
                tail = tail.split(".", 1)[1]
            return f"{head}.{tail}", which
    return None


def strip_peft_prefix(name: str) -> str:
    """`base_model.model.` 前缀去掉 ✓（HF 里裸模型的名字没有它 ✓）"""
    for pre in ("base_model.model.", "base_model."):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def collect_lora_pairs(state: dict[str, torch.Tensor],
                       alpha: float | None, r: int | None,
                       strip: bool = True) -> dict[str, LoRAPair]:
    """从 state_dict 里把所有 LoRA 对收集起来 ✓ → {base 名: LoRAPair}

    ★ `strip=True`（默认 ✓）会把 `base_model.model.` 前缀剥掉 ✓✓
      —— 因为裸模型的 `named_parameters()` 给的是 `model.layers...q_proj.weight` ✓
      而 PEFT 的 adapter 键是 `base_model.model.model.layers...q_proj.lora_A...` ✗
      → 不剥就**对不上** ✗（vLLM 那边会静默跳过或抛 `no attribute base_layer` ✗✓）
    """
    As: dict[str, torch.Tensor] = {}
    Bs: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        got = parse_lora_name(k)
        if got is None:
            continue
        base, which = got
        (As if which == "A" else Bs)[base] = v
    out: dict[str, LoRAPair] = {}
    for key, A in As.items():
        # ★ 用**原始键**去配对 ✓（Bs 里的键也是原始形式 ✗ 先剥就查不到了 ✗✓）
        B = Bs.get(key)
        if B is None:
            continue
        rr = r if r is not None else A.shape[0]
        sc = (alpha / rr) if alpha is not None else 1.0
        # 只在**输出键**上剥前缀 ✓（裸模型的 named_parameters 也是这个形式 ✓）
        out[strip_peft_prefix(key) if strip else key] = LoRAPair(A=A, B=B, scaling=float(sc))
    return out


# ===================================================================== #
# WeightSource 的最小实现 ✓（不依赖 vllm ✓ 便于自测 ✓）
# ===================================================================== #
class _ParamMeta:
    """和 vllm.distributed.weight_transfer.base.ParamMeta 同构 ✓"""
    def __init__(self, name: str, dtype: torch.dtype, shape: tuple[int, ...]):
        self.name, self.dtype, self.shape = name, dtype, shape

    def __repr__(self) -> str:
        return f"ParamMeta({self.name!r}, {self.dtype}, {self.shape})"


class MergedLoRAWeightSource:
    """把 (base 模块 + LoRA) 当成一个**合并后**的权重源 ✓

    接口与 vLLM 的 `WeightSource` 一致 ✓：
        metadata()   · iter_groups() · groups() · held_names()
    """

    def __init__(self, base_module: torch.nn.Module,
                 pairs: dict[str, LoRAPair],
                 group_size: int = 16,
                 prefix_map=None,
                 peft_aware: bool | None = None):
        """
        Args:
            base_module: **合并前的**完整模型 ✓（它的 named_parameters 给出名字/形状 ✓）
            pairs:       {参数名: LoRAPair} ✓（名字要和 base_module 的**一致** ✓）
            group_size:  每次迭代产出多少张量 ✓（小一点更省峰值显存 ✓）
            prefix_map:  可选，把 base 的名字映射到 LoRA 侧的名字 ✓（None = 相同 ✓）
            peft_aware:  None（默认）= **自动检测** ✓；True/False 可强制 ✓✓
                ★ True 时对**被 PEFT 原地改写过的模型**做名字清洗 ✓✓
                · 跳过所有 `...lora_A/lora_B...` 参数 ✗（它们是增量，不是权重 ✓）
                · 把 `...q_proj.base_layer.weight` 改回 `...q_proj.weight` ✓✓
                · 理由：`get_peft_model()` 是**原地替换** ✓ → 内存里**没有**干净裸模型 ✗
                  （`get_base_model()` / 剥类型都不行 ✗ 本地实测过 ✗）
        """
        self.base = base_module
        self.pairs = pairs
        self.group_size = int(group_size)
        self.prefix_map = prefix_map or (lambda n: n)
        # ★ 自动检测 ✓：被 PEFT 原地改写过的模型，名字里一定有这些痕迹 ✓
        #   → 不用记着传参数 ✓（忘了传就会让 vLLM 收到带 base_layer 的名字 ✗ 已在真机踩过一次 ✗）
        if peft_aware is None:
            peft_aware = any((".base_layer" in n) or ("lora_" in n)
                             for n, _ in base_module.named_parameters())
            if peft_aware:
                print("  [weight-src] 检测到 PEFT 改写痕迹 ✓ → 自动启用名字清洗 ✓✓")
        self._peft_aware = bool(peft_aware)
        raw = [(n, prm) for n, prm in base_module.named_parameters()]
        if self._peft_aware:
            # ★ 跳过 lora_A/lora_B ✗ 并把 .base_layer. 抹掉 ✓✓（见 docstring 的理由）
            raw = [(n, prm) for n, prm in raw if ".lora_" not in n and "lora_A" not in n
                   and "lora_B" not in n]
            self._rename = {n: n.replace(".base_layer.", ".") for n, _ in raw}
        else:
            self._rename = {n: n for n, _ in raw}
        self._raw_names = [n for n, _ in raw]
        self._raw = {n: prm for n, prm in raw}
        self._names = [self._rename[n] for n in self._raw_names]

    # ── WeightSource 协议 ✓ ─────────────────────────────────────────
    def metadata(self) -> list[_ParamMeta]:
        out = []
        for n in self._raw_names:
            p = self._raw[n]
            out.append(_ParamMeta(self._rename[n], p.dtype, tuple(p.shape)))
        return out

    def groups(self) -> list[list[str]]:
        return [self._names[i:i + self.group_size]
                for i in range(0, len(self._names), self.group_size)]

    def held_names(self):
        return None

    def iter_groups(self):
        """★ 逐组产出 (names, tensors) ✓ —— 合并就在这里发生 ✓✓"""
        names = self._names
        params = {self._rename[rn]: self._raw[rn] for rn in self._raw_names}
        for i in range(0, len(names), self.group_size):
            chunk = names[i:i + self.group_size]
            tensors = []
            for n in chunk:
                w = params[n].detach()
                pair = self.pairs.get(self.prefix_map(n))
                if pair is not None and w.ndim == 2:
                    tensors.append(w + pair.delta(dtype=w.dtype, device=w.device))
                else:
                    tensors.append(w)                 # 没被 LoRA 作用 → 原样 ✓
            yield chunk, tensors


# ===================================================================== #
# 自测 ✓（CPU ✓ 不需要 vllm ✓）
# ===================================================================== #
def _self_test() -> int:
    torch.manual_seed(0)
    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        print(f"  {'✅' if cond else '✗✗'} {msg}")
        if not cond:
            fails += 1

    print("① 合并结果 == 手算的 W + BA·s")
    lin = torch.nn.Linear(8, 4, bias=False)
    r, alpha = 2, 4.0
    A = torch.randn(r, 8) * 0.1
    B = torch.randn(4, r) * 0.1
    s = alpha / r
    expect = lin.weight.detach() + (B @ A) * s
    got = lin.weight.detach() + LoRAPair(A, B, s).delta(lin.weight.dtype)
    check(torch.allclose(got, expect, atol=1e-6), f"最大差 {(got - expect).abs().max():.2e}")

    print("\n② 名字解析（PEFT → base）")
    n1 = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    n2 = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    b1, w1 = parse_lora_name(n1)
    b2, w2 = parse_lora_name(n2)
    check(b1 == b2, f"A/B 指向同一个 base：{b1}")
    check(w1 == "A" and w2 == "B", "A/B 识别正确")
    check(strip_peft_prefix(b1) == "model.layers.0.self_attn.q_proj.weight",
          f"去掉 peft 前缀：{strip_peft_prefix(b1)}")

    print("\n③ collect_lora_pairs 能配成对")
    sd = {n1: A, n2: B,
          "base_model.model.model.layers.0.self_attn.q_proj.weight": lin.weight}
    pairs = collect_lora_pairs(sd, alpha=alpha, r=r)
    check(len(pairs) == 1, f"配到 {len(pairs)} 对")
    bp = next(iter(pairs))
    check(abs(pairs[bp].scaling - s) < 1e-9, f"scaling = {pairs[bp].scaling}")

    print("\n④ MergedLoRAWeightSource 的 metadata 报的是 base 的形状")
    model = torch.nn.Sequential()          # 用一个最小模块覆盖"有/无 LoRA"两种情况 ✓
    model.add_module("dense", torch.nn.Linear(6, 3, bias=False))
    model.add_module("other", torch.nn.Linear(3, 2, bias=True))
    mnames = dict(model.named_parameters())
    src = MergedLoRAWeightSource(model, {})
    metas = src.metadata()
    check({m.name for m in metas} == set(mnames), f"参数名一致（{len(metas)} 个）")
    check(all(tuple(m.shape) == tuple(mnames[m.name].shape) for m in metas), "形状一致")

    print("\n⑤ 有 LoRA 的那层被合并了；没 LoRA 的原样")
    Ak, Bk = torch.randn(r, 6) * 0.1, torch.randn(3, r) * 0.1
    pairs2 = {"dense.weight": LoRAPair(Ak, Bk, s)}
    src2 = MergedLoRAWeightSource(model, pairs2, group_size=1)
    seen = {n: t for names, ts in src2.iter_groups() for n, t in zip(names, ts)}
    want_dense = model.dense.weight.detach() + (Bk @ Ak) * s
    check(torch.allclose(seen["dense.weight"], want_dense, atol=1e-6),
          f"dense 已合并（差 {(seen['dense.weight'] - want_dense).abs().max():.2e}）")
    check(torch.equal(seen["other.weight"], model.other.weight.detach()), "other 原样 ✓")
    check(torch.equal(seen["other.bias"], model.other.bias.detach()), "bias 原样 ✓")

    print("\n⑥ ★ 惰性：base 改了之后，**下一次迭代**能读到新值")
    with torch.no_grad():
        model.dense.weight.add_(1.0)
    seen2 = {n: t for names, ts in MergedLoRAWeightSource(model, pairs2).iter_groups()
             for n, t in zip(names, ts)}
    delta_expect = model.dense.weight.detach() + (Bk @ Ak) * s
    check(torch.allclose(seen2["dense.weight"], delta_expect, atol=1e-6),
          "重迭代拿到的是**当前**权重 ✓（这就是 send_weights() 每轮 replay 的前提 ✓）")

    print("\n⑦ 分组不改变整体内容")
    big = torch.nn.Sequential(*[torch.nn.Linear(4, 4, bias=False) for _ in range(5)])
    for gs in (1, 2, 3, 100):
        got = {n: t for names, ts in MergedLoRAWeightSource(big, {}, group_size=gs).iter_groups()
               for n, t in zip(names, ts)}
        ref = dict(big.named_parameters())
        ok = set(got) == set(ref) and all(torch.equal(got[k], ref[k].detach()) for k in ref)
        check(ok, f"group_size={gs} → {len(got)} 个张量一致")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓ —— 合并逻辑可以交给 vLLM 用了 ✓✓")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
