"""diag_vllm_bf16.py —— 查两件事（10 分钟 ✓）

① `IPCTrainerInitInfo` 到底要哪些参数 ✓（上一轮缺 `rank` ✗）
② ★★ vLLM 的 **bf16 vs fp32** 在同一 prompt 上的表现 ✗✓
   （上一轮 bf16 答了 '6' 且 34s/token ✗ 而 fp32 答了 '391' ✓ —— 必须分清 ✓）

## 它怎么判

    同一 prompt ✓ 三种配置各生成一次（**先热身再计时** ✓ 排除首调用开销 ✗）：
        HF  fp32   ← 真值 ✓
        vLLM fp32  ← 之前答对的那个 ✓
        vLLM bf16  ← 可疑的那个 ✗
    → 若 fp32 对 ✓ 而 bf16 错 ✗ → **bf16 路径有问题** ✗（那就用 fp32 ✓ 慢一点但正确）
    → 若两者都对但都慢 ✗ → 是**显存挤/首调用** ✓（热身后再看 ✓）

## 跑法

    cd /data/ppo
    export PPO_MODEL=/data/models/Qwen2.5-1.5B-Instruct
    python3 diag_vllm_bf16.py
"""
from __future__ import annotations

import inspect
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("PPO_MODEL", "/data/models/Qwen2.5-1.5B-Instruct")


def probe_ipc_init_info() -> None:
    print("=" * 72)
    print("① IPCTrainerInitInfo / NCCLTrainerInitInfo 的字段")
    print("=" * 72)
    import dataclasses
    import vllm.distributed.weight_transfer.base as B
    base_cls = getattr(B, "TrainerInitInfo", None)
    for modname in ("ipc_engine", "nccl_engine"):
        try:
            mod = __import__(f"vllm.distributed.weight_transfer.{modname}", fromlist=["*"])
        except Exception as e:                                 # noqa: BLE001
            print(f"  {modname}: 导入失败 {e}")
            continue
        for name, obj in vars(mod).items():
            if not (isinstance(obj, type) and base_cls and issubclass(obj, base_cls)):
                continue
            if obj is base_cls:
                continue
            print(f"\n  ── {name}()  backend={getattr(obj, 'backend', '?')}")
            try:
                print(f"     签名 {inspect.signature(obj)}")
            except (ValueError, TypeError):
                pass
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    d = (" = " + repr(f.default)) if f.default is not dataclasses.MISSING else ""
                    print(f"     · {f.name}: {f.type}{d}")
            doc = (inspect.getdoc(obj) or "").strip().splitlines()
            for line in doc[:5]:
                print(f"     │ {line}")


def run_case(dtype: str, prompt_ids, tok, n=16):
    """起一个 vLLM 实例，热身两次，再计时 ✓"""
    from vllm import LLM, SamplingParams
    print(f"\n── vLLM dtype={dtype}")
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, dtype=dtype, enforce_eager=True, max_model_len=512,
              seed=0, gpu_memory_utilization=0.40)
    print(f"   加载 {time.perf_counter() - t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=n, seed=0)

    def gen(tag):
        t = time.perf_counter()
        try:
            out = llm.generate([{"prompt_token_ids": prompt_ids}], sp)
        except (TypeError, ValueError):
            out = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sp)
        dt = time.perf_counter() - t
        ids = list(out[0].outputs[0].token_ids)
        print(f"   {tag:<8} {dt:6.2f}s  → {tok.decode(ids)[:50]!r}")
        return ids

    gen("热身1")                       # ★ 排除首调用 ✗
    gen("热身2")
    ids = gen("计时")
    del llm
    import torch
    torch.cuda.empty_cache()
    return ids


def main() -> int:
    probe_ipc_init_info()

    import torch
    from transformers import AutoModelForCausalLM
    from model import load_tokenizer
    tok = load_tokenizer(MODEL)

    msgs = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
    prompt_ids = tok(tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)).input_ids
    print(f"\nprompt {len(prompt_ids)} token")

    print("\n" + "=" * 72)
    print("② HF fp32 真值")
    print("=" * 72)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             device_map="cuda").eval()
    cur = torch.tensor([prompt_ids], device="cuda")
    hf_out = []
    t0 = time.perf_counter()
    for _ in range(16):
        with torch.no_grad():
            nxt = int(hf(input_ids=cur, use_cache=False).logits[0, -1].argmax())
        hf_out.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)
        if nxt == tok.eos_token_id:
            break
    print(f"  HF fp32 ({time.perf_counter() - t0:.1f}s) → {tok.decode(hf_out)[:50]!r}")
    del hf
    torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("③ vLLM 两档 dtype（热身后再计时 ✓）")
    print("=" * 72)
    r32 = run_case("float32", prompt_ids, tok)
    r16 = run_case("bfloat16", prompt_ids, tok)

    print("\n" + "=" * 72)
    print("结论")
    print("=" * 72)
    ok32 = r32[:len(hf_out)] == hf_out
    ok16 = r16[:len(hf_out)] == hf_out
    print(f"  HF fp32      : {tok.decode(hf_out)[:40]!r}")
    print(f"  vLLM fp32    : {tok.decode(r32)[:40]!r}   {'✅ 对' if ok32 else '✗ 错'}")
    print(f"  vLLM bf16    : {tok.decode(r16)[:40]!r}   {'✅ 对' if ok16 else '✗ 错'}")
    print()
    if ok32 and ok16:
        print("  ✅ 两档都对 ✓ → 上一轮的 34s 只是**首调用/显存挤** ✓（热身就好 ✓✓）")
    elif ok32 and not ok16:
        print("  ✗✗ **bf16 路径有问题** ✗ → 训练时用 **fp32 的 vLLM** ✓（权重 6.25G ✓ 装得下 ✓）")
        print("      或者查：是不是 bf16 下 FlashAttention-2 的某个 kernel 在这个 prompt 上退化 ✗")
    elif ok16 and not ok32:
        print("  ⚠️ 反过来了 ✗ → 这说明本轮的 fp32 才是异常 ✓ → 再跑一次确认 ✓")
    else:
        print("  ✗✗ 两档都错 → **vLLM 的输出本身不可信** ✗✓")
        print("      → 别急着接线 ✓ 这不是「慢」的问题 ✗ 是「错」的问题 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
