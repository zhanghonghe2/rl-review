"""diag_vllm_attn.py —— 用**构造参数**强制 attention backend，验证 bf16 能不能救回来 ✓

## 为什么上一轮没生效 ✗

vLLM 0.31 是 **V1 引擎** ✓ → 它在启动日志里打的是
    `Using FLASH_ATTN attention backend out of potential backends: [...]`
而 `VLLM_ATTENTION_BACKEND` 这个老环境变量 V1 可能**已经不读** ✗✓
→ 必须走 **`LLM(...)` 的构造参数** ✓（具体名字不确定 ✗ → 这里先探测 ✓）

## 判据

    只要哪一档给出 '391' ✓ → 就用那一档 ✓✓
    全部都是 '6' ✗ → 那 bf16 在这台机器上就是不行 ✓ → 用 fp32 ✓（6.25G ✓ 装得下 ✓）

## 跑法

    cd /data/ppo
    export PPO_MODEL=/data/models/Qwen2.5-1.5B-Instruct
    python3 diag_vllm_attn.py 2>&1 | grep -vE "^\\s*$" | tail -60
"""
from __future__ import annotations

import inspect
import os
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("PPO_MODEL", "/data/models/Qwen2.5-1.5B-Instruct")


def attn_kwargs() -> list[str]:
    """LLM 构造函数里跟 attention 有关的参数名 ✓"""
    from vllm import LLM
    in_llm = [p for p in inspect.signature(LLM.__init__).parameters
              if any(k in p.lower() for k in ("att_", "attention", "backend", "sched", "kernel"))]
    return in_llm


def main() -> int:
    try:
        import torch
        from vllm import SamplingParams
    except Exception as e:                                    # noqa: BLE001
        print(f"❌ 导入失败：{e}")
        return 1

    from model import load_tokenizer
    tok = load_tokenizer(MODEL)
    msgs = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
    prompt_ids = tok(tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)).input_ids

    print("=" * 66)
    print("LLM 构造里可用的相关参数")
    print("=" * 66)
    print(f"  {attn_kwargs()}")
    # 顺便看看有没有 AttentionConfig 之类 ✓
    try:
        import vllm.config as C
        print(f"  vllm.config 里带 attention 的类："
              f"{[n for n in dir(C) if 'ttention' in n or 'ttn' in n]}")
    except Exception:                                        # noqa: BLE001
        pass

    cases = [
        ("bf16 默认",            "bfloat16", {}),
        ("bf16 attention_backend", "bfloat16", {"attention_backend": "TRITON_ATTN"}),
        ("bf16 attn_backend",      "bfloat16", {"attn_backend": "TRITON_ATTN"}),
        ("fp32 默认（对照 ✓）",    "float32",  {}),
    ]

    print("\n" + "=" * 66)
    print("逐档生成（同一 prompt ✓ 贪心 ✓）")
    print("=" * 66)
    for tag, dtype, extra in cases:
        try:
            llm = LLM(model=MODEL, dtype=dtype, enforce_eager=True, max_model_len=512,
                      seed=0, gpu_memory_utilization=0.35, **extra)
        except TypeError as e:
            print(f"\n── {tag}: 构造参数不被接受 ✗（{e}）")
            continue
        except Exception as e:                               # noqa: BLE001
            print(f"\n── {tag}: 构造失败 ✗ {type(e).__name__}: {e}")
            continue

        sp = SamplingParams(temperature=0.0, max_tokens=16, seed=0)
        try:
            out = llm.generate([{"prompt_token_ids": prompt_ids}], sp)
        except (TypeError, ValueError):
            out = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sp)
        txt = tok.decode(list(out[0].outputs[0].token_ids))
        ok = "391" in txt
        print(f"\n── {tag}")
        print(f"   {txt[:40]!r}   {'✅ 对 ✓' if ok else '✗ 错 ✗'}")
        del llm
        torch.cuda.empty_cache()

    print("\n" + "=" * 66)
    print("结论")
    print("=" * 66)
    print("  · 若「bf16 + 某个参数」给出 391 ✓ → 接线时带上那个参数 ✓✓")
    print("  · 若全是 6 ✗ → **用 fp32 的 vLLM** ✓（6.25G ✓ 与训练共存仍可行 ✓）")
    print("  · 无论哪种 ✓，记得把「fp32/bf16 + 后端」定死写进配置 ✓（默认值不可信 ✗）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
