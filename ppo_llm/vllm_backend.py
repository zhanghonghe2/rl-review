"""vllm_backend.py —— vLLM 作为 rollout 后端的**唯一入口**（自包含 ✓ 可自测 ✓）

## 设计原则 ✓

1. **默认不用它** ✓ —— `main.py` 里 `--rollout-backend hf` 是默认值 ✓，行为完全不变 ✓✓
2. **只替"生成"那一步** ✗ —— `logprobs_old` 仍然由**训练器重算** ✓✓
   → 所以 `assert_ratio_starts_at_one` 依旧有效 ✓（ratio 起手 = 1 ✓）
3. **每步必须同步权重** ✗✓ —— 否则采到的是**旧策略**的样本 ✓ 而 ratio 断言**看不见** ✗✗
   → 所以 `sync()` 是**强制**的 ✓ + 每 N 步做一次**一致性断言** ✓✓

## 用法（在 main.py 里）

    from vllm_backend import VLLMBackend
    be = VLLMBackend(model_path, merge_from=(base_module, lora_pairs), dtype="float32")
    be.start()
    ...
    for step in ...:
        be.sync()                      # ★ 每步 ✓
        toks = be.generate(prompt_id_lists, temperature=1.0, max_tokens=192)
    be.check_consistent(tok, probe_prompt_ids)   # ★ 每 N 步验一次 ✓✓

## 自测（不需要训练器 ✓）

    cd /data/ppo && export PPO_MODEL=/data/models/Qwen2.5-1.5B-Instruct
    python3 vllm_backend.py                     # 起引擎 → 同步 → 生成 → 一致性 + 计时 ✓
    python3 vllm_backend.py --lora /data/ppo/runs/rft_g8p48
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# ★★ 让 V1 引擎在**本进程**里跑 ✓✓
#    V1 默认把推理放进独立的 EngineCore 子进程 ✗ → 那时内部模型句柄拿不到 ✗
#    → 这就是 find_weight_sink 找不到 / apply_model 要 pickle / update_weights 要建通道的根因 ✓
#    关掉多进程后，`llm.llm_engine...driver_worker.model_runner.model` 就能直接访问 ✓✓
#    （代价：生成是同步阻塞的 ✓ —— 对我们一步一等的循环正好 ✓）
# ★ 必须**强制** ✓ 不能用 setdefault ✗ —— 若外部已把它设成 1 ✗
#   setdefault 就**不会覆盖** ✓ → 引擎照样跑子进程 ✗ → find_weight_sink 静默失败 ✗✓
_prev = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
if _prev not in (None, "", "0"):
    print(f"  [vllm] ⚠️ 外部把它设成了 {_prev!r} ✗ → 强制改回 0 ✓")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# fp32：已验证正确 ✓（bf16 会走 FlashAttention-2，在 sm_120 上**算错** ✗✗ 见 diag_vllm_bf16.py）
DEFAULT_DTYPE = "float32"


class VLLMBackend:
    """vLLM 生成 + 权重热同步 ✓（fp32 ✓ 正确性优先 ✓）"""

    def __init__(self, model_path: str, *, dtype: str = DEFAULT_DTYPE,
                 gpu_memory_utilization: float = 0.35,
                 max_model_len: int = 1024,
                 lora_base=None, lora_pairs=None,
                 lora_prefix_map=None, sync_group_size: int = 8,
                 kv_cache_bytes: int | None = None,
                 sleep_level: int = 1):
        self.model_path = model_path
        self.dtype = dtype
        self.gpu_mem = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.lora_base = lora_base
        self.lora_pairs = lora_pairs or {}
        self.lora_prefix_map = lora_prefix_map
        self.sync_group_size = sync_group_size
        # ★ KV 只要 ~1G（我们的并发很小 ✗ 不要默认的 8.6G ✓ 见 kv_cache_bytes 的注释）
        self.kv_cache_bytes = kv_cache_bytes
        # ★ 生成结束后把 KV 还回去 ✓（level 1 = 只还 KV ✓ 权重仍驻留 ✓）
        self.sleep_level = int(sleep_level)
        self._asleep = False
        self.llm = None
        self.engine = None                       # TrainerWeightTransferEngine ✓
        self._client = None                     # _SyncClient ✓
        self._sink = None                       # vLLM 内部可写 module ✓
        self._name_checked = False              # 名字体检只做一次 ✓
        self._src = None
        self.last_sync_s = float("nan")

    # ── 生命周期 ─────────────────────────────────────────────────────
    def start(self) -> None:
        # ★★ 硬保护：只许 fp32 ✓✓ —— 放在 import 之前 ✓（fail fast ✓ 本地也能验 ✓）
        #   实测（diag_vllm_bf16.py）：
        #     fp32 → TRITON_ATTN  → '391' ✓ 正确
        #     bf16 → FLASH_ATTN(FA2) → '6'  ✗ **静默算错** ✗✗（sm_120 上 FA2 的 kernel 有问题）
        #   → 这种错**不会报错** ✗ → RL 会拿"错策略"的样本训练 ✗✗ 必须拦住 ✓
        # ★★ 允许 fp32 ✓ 和 bf16 ✓（其它一律拒 ✗）
        #   实测（diag_vllm_bf16.py）：
        #     **sm_120（5090）**：bf16 → FA2 → **静默算错** ✗✗（17*23 答 '6' ✗）
        #     **sm_89（L40S）** ：bf16 → **正确** ✓✓（答 '391' ✓ 2024 亲测 ✓）
        #   → 所以：默认仍 fp32 ✓；L40S 上可以显式用 bf16 ✓ 省一半权重显存 ✓✓
        if str(self.dtype) not in ("float32", "bfloat16"):
            raise RuntimeError(f"vLLM dtype 只允许 float32 / bfloat16 ✓（收到 {self.dtype!r}）")
        if str(self.dtype) == "bfloat16":
            print("  [vllm] ★ dtype=**bfloat16** ✓（需确认这块卡不是 sm_120 ✗）\n"
                  "         sm_120 上 bf16 会走 FA2 → **静默算错** ✗✗（先跑 diag_vllm_bf16.py ✓）")
        from vllm import LLM
        t0 = time.perf_counter()
        cfg = _make_weight_transfer_config("ipc")
        base_kw = dict(model=self.model_path, dtype=self.dtype, enforce_eager=True,
                       gpu_memory_utilization=self.gpu_mem,
                       max_model_len=self.max_model_len, seed=0)
        # ★★ 显式给 KV cache 的大小 ✓✓ —— 否则 vLLM 会把"剩余显存"全占掉 ✗
        #    我们峰值并发 ≈ 128 × 250 token ≈ 32k ✓ → 1.5G 足够 ✓（实测默认给了 8.6G ✗）
        if self.kv_cache_bytes:
            _nm = _engine_arg_name(("kv_cache_memory_bytes", "kv_cache_memory",
                                    "kv_cache_size_bytes", "kv_cache_size"))
            if _nm:
                base_kw[_nm] = int(self.kv_cache_bytes)
                print(f"  [vllm] {_nm} = {self.kv_cache_bytes / 1e9:.1f} GB ✓"
                      f"（显式给 ✓ 不让它自己算 ✗）")
            else:
                print(f"  [vllm] ⚠️ EngineArgs 里找不到 KV 大小的参数 ✗ "
                      f"→ 退回自动分配 ✗ → 它会占掉剩余显存（约 8.6G ✗）✓")
                print(f"  [vllm]    对策：把 --vllm-gpu-util 调小（比如 0.25 ✗ 但那时可能放不下）")
                print(f"  [vllm]    或者查 `EngineArgs` 的字段：python3 -c \""
                      f"import inspect;from vllm.engine.arg_utils import EngineArgs as E;"
                      f"print([p for p in inspect.signature(E.__init__).parameters if 'kv' in p.lower()])\"")
            print(f"  [vllm] gpu_memory_utilization={self.gpu_mem} 只是【预算上限】✓ "
                  f"真占的是 权重+KV ✓（看日志的 Actual usage ✓）")
        self.llm = None
        for extra in (({"weight_transfer_config": cfg} if cfg is not None else {}),
                      {}):
            try:
                self.llm = LLM(**base_kw, **extra)
                if "weight_transfer_config" in extra:
                    print("  [vllm] 已把 weight_transfer_config 交给引擎 ✓")
                break
            except TypeError as e:
                msg = str(e)
                if "weight_transfer_config" in msg:
                    print(f"  [vllm] LLM 不接受 weight_transfer_config ✗（{e}）→ 退回不带 ✓")
                    continue
                # ★ 别的 TypeError（比如某个名字不对 ✗）→ 把那个键去掉重试 ✓ 不要整个挂掉 ✗
                import re as _re
                m = _re.search(r"unexpected keyword argument '([^']+)'", msg)
                if m and m.group(1) in base_kw:
                    bad = m.group(1)
                    print(f"  [vllm] ⚠️ EngineArgs 不认 {bad!r} ✗ → 去掉它重试 ✓（显存会变大 ✗）")
                    base_kw.pop(bad, None)
                    extra = {}
                    self.llm = LLM(**base_kw)
                    break
                raise
        if self.llm is None:
            raise RuntimeError("引擎没建起来 ✗")
        # ★★ 显式打印「它落在哪张卡」—— vLLM 没有 device 参数
        #   它跟 torch.cuda.current_device() 走；训练器是显式 .to(cuda:1)，不改 current
        #   → 所以 vLLM = cuda:0 · 训练器 = cuda:1
        _cur = "?"
        try:
            import torch as _t
            _cur = f"cuda:{_t.cuda.current_device()}"
        except Exception:
            pass
        print(f"  [vllm] 引擎 {time.perf_counter() - t0:.1f}s · dtype={self.dtype} · "
              f"落在 **{_cur}** ✓（跟 current_device 走，没有 device 参数）")
        # ★ sleep(level=1) 到底释放什么 ✗ 我**不确定**（各版本不同 ✗）
        #   → 直接把它自己的文档打出来 ✓✓ 不猜 ✓
        try:
            import inspect as _i
            _doc = (_i.getdoc(self.llm.sleep) or "").strip().splitlines()
            print("  [vllm] sleep 的语义（以它自己为准 ✓）：")
            for _l in _doc[:8]:
                print(f"      │ {_l}")
        except Exception:                                     # noqa: BLE001
            pass

    def _build_source(self):
        """惰性合并的权重源 ✓（只在我们自己的模块里依赖它 ✓）"""
        if self._src is not None:
            return self._src
        from lora_weight_source import MergedLoRAWeightSource
        if self.lora_base is None:
            raise RuntimeError("没有 base 模型 → 无法建权重源 ✗")
        self._src = MergedLoRAWeightSource(
            self.lora_base, self.lora_pairs,
            group_size=self.sync_group_size,
            prefix_map=self.lora_prefix_map or (lambda n: n))
        return self._src

    def _build_engine(self):
        if self.engine is not None:
            return self.engine
        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
        src = self._build_source()
        # ★ 找 ipc 的 init_info 子类 ✓（rank 是必需参数 ✗ → 给 0 ✓）
        init_info = None
        infos = _find_init_infos()
        for cls, backend in infos:
            if "ipc" not in str(backend).lower():
                continue
            for kwargs in ({"rank": 0}, {"rank": 0, "world_size": 1}, {}):
                try:
                    init_info = cls(**kwargs)
                    break
                except TypeError:
                    continue
            if init_info is not None:
                break
        if init_info is None:
            raise RuntimeError(f"没找到可用的 ipc init_info ✗（候选 {infos}）")
        # ★ client 必须包一层 ✓（见 _SyncClient 的注释 ✓）
        self._client = _SyncClient(self.llm)
        self.engine = WeightTransferTrainerFactory.trainer_init(
            init_info, client=self._client, source=src)
        pk = getattr(init_info, "packed", None)
        print(f"  [vllm] 权重通道已建 ✓（{type(init_info).__name__} · "
              f"packed={type(pk).__name__}）")
        print(f"  [vllm] 协议调用：{self._client.calls}")
        return self.engine

    def _flatten(self):
        """把 WeightSource 的 (names, tensors) 组拍平成 [(name, tensor)] ✓"""
        src = self._build_source()
        return [(n, t) for names, ts in src.iter_groups() for n, t in zip(names, ts)]

    def _iter_cast_groups(self):
        """★ 逐组产出 (names, tensors)，**按需 cast** ✓✓

        ✗ 之前是一口气 `[(n, t.to(bf16)) for ...]` ✓
          → 等于在整个 fp32 权重（3B = 12.4G）之外**再拷一整份 bf16（6.2G）** ✗✗
          → 峰值 18.6G ✗ → 3B 直接 OOM ✓（实测就崩在 `t.to(self.dtype)` ✓）
        ✓ 现在：**一组**（默认 8 个张量 ✓）拷完就 load ✓ → 峰值只多几十 MB ✓✓
        """
        import torch
        src = self._build_source()
        # ★★ `t.to("bfloat16")` 会被当成**设备名** ✗（实测：Invalid device string: 'bfloat16'）
        #   → 必须 `t.to(dtype=torch.bfloat16)` ✓✓（字符串先转成 torch.dtype ✓）
        _dt = getattr(torch, str(self.dtype))
        for names, ts in src.iter_groups():
            if str(self.dtype) != "float32":
                yield names, [t.to(dtype=_dt) for t in ts]
            else:
                yield names, list(ts)

    def sync(self) -> float:
        """★ 每步调用 ✓ 把训练器当前权重推进 vLLM（**内存** ✓ 不写盘 ✓ 不 pickle ✓）"""
        import torch
        t0 = time.perf_counter()
        self.wake()          # ★★ 先唤醒 ✓✓ —— 若 level 1 会把权重卸到 CPU ✗
                             #    那么"睡着的时候 load_weights"就写不进去 ✗（而且不报错 ✗✓）

        if self._sink is None:
            self._sink, path = find_weight_sink(self.llm) if self.llm is not None else (None, "")
            if self._sink is not None:
                print(f"  [vllm] 找到权重句柄 ✓ `{path}`")
        if self._sink is None:
            # ★ 兜底：如果对象图里找不到（说明多进程没关掉 ✗ 或版本封装变了 ✗）
            #   → 退回 vLLM 官方的 weight_transfer 通道 ✓（IPCTrainerInitInfo + send_weights ✓）
            print("  [vllm] 没找到内部句柄 ✗ → 尝试官方 weight_transfer 通道 ✓")
            eng = self._build_engine()
            eng.send_weights()
            try:
                torch.cuda.synchronize()
            except Exception:                                # noqa: BLE001
                pass
            self.last_sync_s = time.perf_counter() - t0
            return self.last_sync_s

        # ① 名字体检：**只取名字** ✓（不碰张量 ✓ 便宜 ✓）—— 只做一次 ✓
        if not self._name_checked:
            self._name_checked = True
            names = [n for ns, _ in self._build_source().iter_groups() for n in ns]
            bad = [n for n in names
                   if n.startswith(("base_model.", "module."))
                   or ".base_layer" in n or ".lora_" in n or ".modules_to_save" in n]
            print(f"  [vllm] 权重名体检：{len(names)} 个 · 前 2 个 = {names[:2]}")
            if bad:
                raise RuntimeError(
                    "权重名里混进了 PEFT 包装 ✗（例 %r）\n"
                    "  → 必须用**裸模型**当 base ✓：AutoModelForCausalLM.from_pretrained(...)"
                    % bad[0])
            if not all(n.startswith("model.") for n in names):
                print("  [vllm] ⚠️ 有名字不以 model. 开头 ✓（可能不是 Qwen ✓，先继续 ✓）")

        # ② ★★ 流式：逐组 cast + load ✓✓（峰值只多"一组"的量 ✓ 而不是整份 ✓✓）
        n_loaded = 0
        for names, ts in self._iter_cast_groups():
            self._sink.load_weights(list(zip(names, ts)))
            n_loaded += len(names)
            del ts
        if n_loaded == 0:
            raise RuntimeError("一组权重都没灌进去 ✗")
        self.last_sync_s = time.perf_counter() - t0
        return self.last_sync_s

    def sleep(self) -> None:
        """★ 释放 KV cache ✓（level 1 = 权重仍驻留显卡 ✓ 所以同步照样能推 ✓✓）"""
        if self.llm is None or self._asleep or self.sleep_level <= 0:
            return
        try:
            self.llm.sleep(level=self.sleep_level)
            self._asleep = True
        except Exception as e:                                # noqa: BLE001
            # ★ 不能静默失败 ✗ —— 否则显存会一直占着 ✓ 而你以为释放了 ✗✓
            print(f"  [vllm] ⚠️ sleep 失败 ✗（{type(e).__name__}: {e}）→ 显存放不掉 ✗")

    def wake(self) -> None:
        """★ 生成前把 KV 恢复 ✓（必须在任何 generate 之前 ✓）"""
        if self.llm is None or not self._asleep:
            return
        try:
            self.llm.wake_up()
            self._asleep = False
        except Exception as e:                                # noqa: BLE001
            print(f"  [vllm] ⚠️ wake_up 失败 ✗（{type(e).__name__}: {e}）")

    def shutdown(self) -> None:
        try:
            if self.engine is not None:
                self.engine.shutdown()
        except Exception:                                    # noqa: BLE001
            pass
        self.engine = None
        self.llm = None

    # ── 生成（和 model.rollout 的接口对齐 ✓）─────────────────────────
    def generate(self, prompt_id_lists: list[list[int]], *,
                 temperature: float = 1.0, max_tokens: int = 192,
                 top_p: float = 1.0, seed: int = 0) -> list[list[int]]:
        """给定一批 prompt 的 token ids ✓ → 返回每条生成的 token ids ✓

        ★ 采样参数必须**显式**给全 ✓（F3 那个教训：HF 的 generation_config 里
          藏着 repetition_penalty=1.1 ✗；vLLM 也有自己的默认值 ✗）
          这里显式写死：temperature ✓ top_p=1.0 ✓ top_k=-1 ✓ repetition_penalty=1.0 ✓
        """
        from vllm import SamplingParams

        def _mk(i: int):
            return SamplingParams(temperature=float(temperature),
                                  top_p=float(top_p),
                                  top_k=-1,
                                  repetition_penalty=1.0,
                                  max_tokens=int(max_tokens),
                                  # ★★★ 每个请求**必须**有自己的种子 ✗✓
                                  #   vLLM 的 seed 是**逐请求**的 ✓ —— 若全传同一个 ✗
                                  #   同一条题的 G 条采样会**完全相同** ✗✗
                                  #   → GRPO 组内无差异 → A ≡ 0 → **梯度恒 0** ✗✓
                                  #   （实测：reward 非零 32 但 A 非零 0 ✗ grad_norm=0.000 ✗）
                                  #   HF 那条路不受影响 ✓（整批一个 torch.manual_seed ✓ 每行独立抽 ✓）
                                  #   ★ 只有 temperature==0（贪心 ✓）时才无所谓 ✓
                                  seed=int(seed) + i)
        self.wake()
        prompts = [{"prompt_token_ids": list(ids)} for ids in prompt_id_lists]
        sps = [_mk(i) for i in range(len(prompts))]
        try:
            outs = self.llm.generate(prompts, sps)
        except (TypeError, ValueError):
            outs = self.llm.generate(prompt_token_ids=prompt_id_lists,
                                     sampling_params=sps)
        out_ids = [list(o.outputs[0].token_ids) for o in outs]
        self.sleep()                      # ★ 用完立刻把 KV 还回去 ✓✓
        return out_ids

    # ── ★★ 守卫：一致性断言 ✓✓ ───────────────────────────────────────
    def check_consistent(self, tok, probe_prompt_ids: list[int],
                         trainer_greedy=None, n: int = 12) -> bool:
        """★ 同步之后，vLLM 的贪心必须和**训练器**的贪心一致 ✓✓

        这是整个接线里**唯一**能发现"权重没同步/映射错了/后端算错了"的检查 ✗✓
        → 和 `assert_ratio_starts_at_one` 同一个哲学：**静默的错误比崩溃危险得多** ✓✓

        Args:
            tok: tokenizer ✓（用于解码打印 ✓）
            probe_prompt_ids: 一条固定的探针 prompt ✓（要短 ✓ 快 ✓）
            trainer_greedy: 可选的 callable(prompt_ids, n) → list[int] ✓
                            （由 main.py 用训练器算 ✓ 不传则只打印 vLLM 侧 ✓）
        """
        v = self.generate([probe_prompt_ids], temperature=0.0, max_tokens=n)[0]
        ok = True
        if trainer_greedy is not None:
            t = trainer_greedy(probe_prompt_ids, n)
            # 只比到较短的那一截 ✓（EOS 位置可能不同 ✓）
            m = min(len(v), len(t))
            ok = v[:m] == t[:m]
            print(f"  [vllm] 一致性探针：vLLM {tok.decode(v)[:28]!r} vs "
                  f"训练器 {tok.decode(t)[:28]!r} → {'✅ 一致 ✓' if ok else '✗✗ 不一致'} ")
            if not ok:
                print("  [vllm] ✗✗ **权重没同步上 / 名字映射错 / 后端算错** ✗")
                print("         → 继续训练会采到**错误策略**的样本 ✓ 而 ratio 断言**看不见** ✗✗")
                # ★★ 直接抛错 ✗✓ —— 静默继续的代价是"训一晚上全是垃圾" ✗✗
                raise RuntimeError(
                    "vLLM 与训练器的贪心输出不一致 ✗ → 立即停止（样本不可信 ✓）")
        else:
            print(f"  [vllm] 探针（仅 vLLM）：{tok.decode(v)[:40]!r}")
        return ok


def bare_base(model):
    """★ 取出**真正的裸模型** ✓✓（不含任何 peft 包装 ✗）

    为什么不能用 `get_base_model()` ✗（本地用真 peft 实测过 ✓）：
        get_base_model() 返回的仍是 **LoraModel** ✓ → 它的 named_parameters 给的是
            model.layers.0.q_proj.**base_layer**.weight   ✗ vLLM 会抛 `no attribute base_layer` ✗✗
            model.layers.0.q_proj.lora_A.default.weight    ✗
        而真正的裸模型（`model.base_model.model`）给的是
            model.layers.0.q_proj.weight                   ✓✓ 正是 vLLM 要的 ✓

    兼容：PeftModel ✓ / 已经裸的模型 ✓ / 多层包装 ✓
    """
    # ★ 规则：只剥**包装层**，不剥模型自己的子模块 ✓✓
    #   我第一次写成了"盲剥 4 层" ✗ → 把 Qwen 自己的 `.model` 也剥了 ✗
    #   → 名字从 `model.layers…` 变成 `layers…` ✗✓（本地用真 peft 立刻测出来了 ✓）
    m = model
    for _ in range(3):
        tname = type(m).__name__
        # ① PeftModel / PeftModelForCausalLM → 剥 base_model ✓
        if tname.startswith("PeftModel") and getattr(m, "base_model", None) is not None:
            m = m.base_model
            continue
        # ② peft 的 LoraModel → 剥它外面那层 .model ✓（它才是包装 ✗）
        if tname == "LoraModel" and getattr(m, "model", None) is not None:
            m = m.model
            continue
        break                                 # 已经是真模型了 ✓ 停 ✓
    return m


def live_lora_pairs(model):
    """★ 从**训练器里活着的**模型取 LoRA 对 ✓✓

    为什么不能用 `collect_lora_pairs(state_dict, alpha, r)` ✗：
        · α/r 要猜 ✗（新版 PEFT 用 rsLoRA 时是 α/√r ✗✗ → 猜错就合并错 ✓ 而且**不报错** ✗）
        · 更要紧的是：`state_dict()` 可能返回**副本** ✗ → 每步读到旧值 ✗
    → 所以这里：① 从 peft 模块上直接读 **scaling** ✓（它自己的值 ✓ 一定对 ✓）
                ② 直接持有**参数张量本身** ✓ → 优化器原地更新后**自动可见** ✓✓

    返回：{裸模型参数名: LoRAPair} ✓（键与 `bare_base(model).named_parameters()` 一致 ✓）

    ★ 配套用 `MergedLoRAWeightSource(bare_base(model), pairs)` ✓✓
    """
    from lora_weight_source import LoRAPair, parse_lora_name, strip_peft_prefix

    # ① scaling：从任一 peft 包装层的 .scaling 字典里拿 ✓（不需要知道 α/r ✓）
    scaling = None
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and sc:
            scaling = float(next(iter(sc.values())))
            break
    if scaling is None:
        raise RuntimeError("这个模型上没有 peft 的 `.scaling` ✗ —— 是不是没挂 LoRA？")

    # ② A/B 张量：直接拿参数对象 ✓（不 detach 成副本 ✓ 保持引用 ✓）
    As, Bs = {}, {}
    for name, prm in model.named_parameters():
        got = parse_lora_name(name)
        if got is None:
            continue
        base, which = got
        key = strip_peft_prefix(base)
        (As if which == "A" else Bs)[key] = prm.data          # ★ .data：原地更新可见 ✓

    pairs = {}
    for k, A in As.items():
        B = Bs.get(k)
        if B is None:
            continue
        pairs[k] = LoRAPair(A=A, B=B, scaling=scaling)
    if not pairs:
        raise RuntimeError("没找到任何 LoRA 对 ✗")
    return pairs


def find_weight_sink(root, max_depth: int = 6):
    """★ 在 vLLM 的对象图里找"能写权重"的那个 module ✓✓

    判据：同时有 `load_weights` 和 `named_parameters` ✓（就是那个 nn.Module ✓）
    → 不写死路径 ✓（V0/V1/各版本都不一样 ✗）；找不到就返回 (None, "") ✓
    """
    seen: set[int] = set()

    def walk(obj, path: str, depth: int):
        if depth > max_depth or id(obj) in seen:
            return None
        seen.add(id(obj))
        if hasattr(obj, "load_weights") and hasattr(obj, "named_parameters"):
            return obj, path
        for attr in ("model_runner", "model", "worker", "driver_worker",
                     "model_executor", "llm_engine", "engine_core",
                     "engine_core_client", "executor", "compilation_config"):
            child = getattr(obj, attr, None)
            if child is None or isinstance(child, (str, bytes, int, float, dict)):
                continue
            got = walk(child, f"{path}.{attr}", depth + 1)
            if got:
                return got
        for attr in ("_model_runner", "_model", "_driver_worker", "_worker"):
            child = getattr(obj, attr, None)
            if child is not None and not isinstance(child, (str, bytes, int, float)):
                got = walk(child, f"{path}.{attr}", depth + 1)
                if got:
                    return got
        try:
            if isinstance(obj, (list, tuple)) and obj:
                return walk(obj[0], f"{path}[0]", depth + 1)
        except Exception:                                    # noqa: BLE001
            pass
        return None

    return walk(root, "llm", 0)


def _engine_arg_name(cands: tuple[str, ...]) -> str | None:
    """★ 在 `EngineArgs` 的构造签名里找一个存在的参数名 ✓✓

    为什么要探测 ✗：vLLM 的**命令行**叫 `--kv-cache-memory` ✓
    但 `EngineArgs` 的**字段名**可能是 `kv_cache_memory_bytes` / `kv_cache_size` ✗ 等等 ✓
    → 实测报错：`EngineArgs.__init__() got an unexpected keyword argument 'kv_cache_memory'` ✗✓
    → 所以：列一组候选 ✓ 用存在的那个 ✓ 并**打出来** ✓✓
    """
    try:
        import inspect as _i
        from vllm.engine.arg_utils import EngineArgs
        params = set(_i.signature(EngineArgs.__init__).parameters)
    except Exception:                                        # noqa: BLE001
        return None
    for c in cands:
        if c in params:
            return c
    return None


def _make_weight_transfer_config(backend: str = "ipc"):
    """★ 造引擎侧的 WeightTransferConfig ✓（vLLM 要求引擎启动时配置它 ✗）

    报错原文：
        RuntimeError: Weight transfer not configured.
                      Please set weight_transfer_config to enable weight transfer.
    → 而 `trainer_init` 的文档说"**训练器侧**不需要它" ✓ → 所以是**引擎侧**要 ✓

    这个类名/字段在不同小版本里可能不同 ✗ → 这里多形状尝试 ✓ 并打印用了哪个 ✓✓
    """
    import vllm.config as C
    cands = [n for n in dir(C) if "WeightTransfer" in n and n.endswith("Config")]
    for name in cands:
        cls = getattr(C, name)
        for kwargs in ({"backend": backend}, {"backend": backend, "enabled": True},
                       {"enabled": True}, {}):
            try:
                cfg = cls(**kwargs)
                print(f"  [vllm] weight_transfer_config = {name}({kwargs}) ✓")
                return cfg
            except TypeError:
                continue
    print(f"  [vllm] ⚠️ 没造出 weight_transfer_config ✗（vllm.config 里的候选：{cands}）")
    return None


class _DictAsRequest:
    """★ 把 ipc_engine 传来的 dict 包成 LLM 认的对象 ✓

    证据（vLLM 0.31.0 源码）✓：
        ipc_engine.py:303   engine.client.init_weight_transfer_engine({"packed": init_info.packed})
        llm.py:880          request["init_info"] if isinstance(request, dict) else request.init_info
    → 两边**协议不一致** ✗（引擎侧要 "init_info" 键/属性 ✓ 训练器侧给的是 {"packed": ...} ✗）
    → 所以包一层：让对象**同时**带 init_info 和 update_info 两个属性 ✓✓
      （它读哪个都对 ✓ 而且不改变载荷本身 ✓）
    """

    __slots__ = ("init_info", "update_info")

    def __init__(self, payload: dict):
        self.init_info = payload
        self.update_info = payload

    def __repr__(self) -> str:
        return f"_DictAsRequest(keys={sorted(self.init_info)})"


class _SyncClient:
    """★ 薄适配层：满足 `VLLMWeightSyncClient` 协议 ✓ 但把载荷包对 ✓

    文档说 client 只需四个方法 ✓（`LLM` 自己正好有那四个 ✓）——
    只是它要的形状不同 ✗ → 这里转一手 ✓✓
    """

    def __init__(self, llm):
        self._llm = llm
        self.calls: list[str] = []

    def init_weight_transfer_engine(self, payload: dict) -> None:
        self.calls.append("init")
        self._llm.init_weight_transfer_engine(_DictAsRequest(payload))

    def start_weight_update(self) -> None:
        self.calls.append("start")
        self._llm.start_weight_update()

    def update_weights(self, payload) -> None:
        self.calls.append("update")
        if isinstance(payload, list):
            self._llm.update_weights([_DictAsRequest(p) for p in payload])
        else:
            self._llm.update_weights(_DictAsRequest(payload))

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self.calls.append("finish")
        self._llm.finish_weight_update(weight_version)


def build_from_adapter(model_path: str, adapter_path: str | None = None, *,
                       dtype: str = DEFAULT_DTYPE,
                       gpu_util: float = 0.45, kv_gb: float = 1.5,
                       sleep_level: int = 1, verbose: bool = True,
                       max_len: int | None = 4096) -> "VLLMBackend":
    """★ 纯推理用：从一个 adapter 直接建好可用的后端 ✓✓

    适用：`sample_rft.py`（拒绝采样 ✓）· `probe_passk.py`（pass@k ✓）· `sft.py --eval-only` ✓
    —— 它们**只生成** ✓ 权重**全程不变** ✗ → 所以**只 sync 一次** ✓✓（不需要每步同步 ✓）

    做法（和自测里验过的那套一致 ✓）：
        ① **裸 base**（CPU ✓ 只为名字/形状/数值 ✓）← 绝不用 PEFT 包装 ✗（名字会带 base_layer ✗）
        ② adapter 的 LoRA 对（CPU ✓ 10MB ✓）
        ③ VLLMBackend（惰性合并 ✓）+ start() + sync() 一次 ✓
    """
    import torch
    from transformers import AutoModelForCausalLM

    base = None
    pairs = {}
    if adapter_path:
        from peft import PeftModel
        from lora_weight_source import collect_lora_pairs
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
                                                    device_map="cpu").eval()
        hf_cpu = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
                                                 device_map="cpu"),
            adapter_path)
        cfg = hf_cpu.peft_config[list(hf_cpu.peft_config)[0]]
        sd = {k: v.detach() for k, v in hf_cpu.state_dict().items() if "lora_" in k}
        pairs = collect_lora_pairs(sd, alpha=cfg.lora_alpha, r=cfg.r)
        if verbose:
            print(f"  [vllm] adapter {adapter_path} · LoRA {len(pairs)} 组 · "
                  f"r={cfg.r} α={cfg.lora_alpha} ✓")
        del hf_cpu

    be = VLLMBackend(model_path, dtype=dtype, gpu_memory_utilization=gpu_util,
                     # ★★★ 原来写死 **1024** ✗✗ —— PRM 的题干最长 **1,143 token** ✗
                     #   → 直接 `VLLMValidationError: prompt 1048 > 1024` 崩 ✓（实测 ✓）
                     #   默认改成 **None** ✓ → 交给 vLLM 用模型自己的上限（Qwen2.5 = 32768 ✓）
                     #   ★ 注意：`max_model_len` 必须 ≥ 题干 + max_new_tokens ✗
                     #     （PRM 要 1,143 + 1,024 ≈ 2,170 → **2048 都不够** ✗）
                     # ★★ 默认 **4096** ✓✓（不是 None ✗ 也不是 1024 ✗）
                     #   · 1024 ✗ → PRM 题干 1,143 token 直接崩 ✓
                     #   · None ✗ → vLLM 取模型上限 **32768** ✓ → KV 要 1.75 GiB
                     #             > 我们给的 1.4 GiB ✗ → 起不来 ✓（实测 ✓）
                     #   · 4096 ✓ → 够 1,143(题干) + 1,024(生成) = 2,167 ✓ 还留余量 ✓
                     #             KV ≈ 112~224 MiB ✓（1.4 GiB 里随便放 ✓✓）
                     max_model_len=max_len, lora_base=base, lora_pairs=pairs,
                     kv_cache_bytes=int(kv_gb * 1e9), sleep_level=sleep_level)
    be.start()
    # ★★ adapter=None（= 评**基座** ✓）时**不需要 sync** ✗✓
    #   引擎本来就是用 `model_path` 的权重建的 ✓ → 再 sync 反而报"没有 base 模型" ✗
    #   （实测：`--no-adapter --vllm` 就崩在这里 ✓）
    if adapter_path:
        be.sync()                   # ★ 只此一次 ✓ 权重不变 ✓✓
    else:
        if verbose:
            print("  [vllm] 无 adapter → 引擎直接用基座权重 ✓ 跳过 sync ✓")
    if verbose:
        print(f"  [vllm] 就绪 ✓（同步 {be.last_sync_s:.2f}s ✓）")
    return be


def _find_init_infos():
    """枚举 TrainerInitInfo 子类 ✓ + 各自的 backend ✓"""
    out = []
    try:
        import vllm.distributed.weight_transfer.base as B
        base_cls = getattr(B, "TrainerInitInfo", None)
        if base_cls is None:
            return out
        for modname in ("base", "ipc_engine", "nccl_engine", "m2n_engine",
                        "sharded_rdt_engine"):
            try:
                mod = __import__(f"vllm.distributed.weight_transfer.{modname}",
                                 fromlist=["*"])
            except Exception:                                # noqa: BLE001
                continue
            for _n, obj in vars(mod).items():
                if (isinstance(obj, type) and issubclass(obj, base_cls)
                        and obj is not base_cls and obj not in [c for c, _ in out]):
                    out.append((obj, getattr(obj, "backend", "?")))
    except Exception:                                        # noqa: BLE001
        pass
    return out


# ===================================================================== #
# 自测 ✓（不需要训练器 ✓）
# ===================================================================== #
def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PPO_MODEL",
                    "/data/models/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--lora", default=os.environ.get("PPO_LORA", ""))
    ap.add_argument("--dtype", default=DEFAULT_DTYPE)
    ap.add_argument("--n", type=int, default=32, help="一次生成多少条（测吞吐 ✓）")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from model import load_tokenizer
    tok = load_tokenizer(args.model)

    base, pairs, pmap = None, {}, None
    if args.lora:
        from peft import PeftModel
        from lora_weight_source import collect_lora_pairs
        # ★ ① 用**裸模型**（CPU ✓ 只为名字/形状/权重 ✓）—— 绝不带 PEFT 包装 ✗
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                    device_map="cpu").eval()
        print(f"  裸 base 在 CPU ✓（名字如 {next(iter(base.named_parameters()))[0]} ✓）")
        # ★ ② adapter 的 LoRA 对：从 PEFT 模型读（用 CPU ✓ 不占显存 ✓）
        hf_cpu = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 device_map="cpu"),
            args.lora)
        cfg = hf_cpu.peft_config[list(hf_cpu.peft_config)[0]]
        sd = {k: v.detach() for k, v in hf_cpu.state_dict().items() if "lora_" in k}
        # ★ collect_lora_pairs 产出的键已剥掉 `base_model.model.` ✓ → 与裸模型名字一致 ✓✓
        pairs = collect_lora_pairs(sd, alpha=cfg.lora_alpha, r=cfg.r)
        print(f"  LoRA {len(pairs)} 组 · r={cfg.r} α={cfg.lora_alpha} ✓  "
              f"例：{next(iter(pairs))}")
        bnames = {n for n, _ in base.named_parameters()}
        miss = [k for k in pairs if k not in bnames]
        print(f"  名字对齐：{len(pairs) - len(miss)}/{len(pairs)} 组能对上裸模型 "
              f"{'✓✓' if not miss else '✗ 例 ' + miss[0]}")
        del hf_cpu

    be = VLLMBackend(args.model, dtype=args.dtype,
                     lora_base=base, lora_pairs=pairs, lora_prefix_map=pmap)
    be.start()

    msgs = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
    probe = tok(tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)).input_ids

    print("\n── 首次同步")
    dt = be.sync()
    print(f"  send_weights() **{dt:.2f}s** "
          f"{'✓✓ 内存通道' if dt < 2 else '✗ 偏慢（可能退化成 pickle ✗）'}")

    print("\n── 生成一条（贪心，验答案）")
    v = be.generate([probe], temperature=0.0, max_tokens=16)[0]
    txt = tok.decode(v)
    print(f"  {txt[:40]!r}   {'✅ 含 391' if '391' in txt else '✗ 不含 391 ✗'}")

    # ★★★ 语义验证：必须和 **HF+adapter** 的贪心一致 ✓✓
    #   这是唯一能发现「合并算错了 / 名字没对上 / 权重没推上去」的检查 ✗✓
    #   （上一轮 load_weights 返回 ok ✓ 但可能**静默跳过了一半权重** ✗）
    print("\n── ★ 和 HF+adapter 逐 token 比对（最关键 ✓✓）")
    if args.lora:
        hf_gpu = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 device_map="cuda"), args.lora).eval()
        cur = torch.tensor([probe], device="cuda")
        ref = []
        for _ in range(16):
            with torch.no_grad():
                nxt = int(hf_gpu(input_ids=cur, use_cache=False).logits[0, -1].argmax())
            ref.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)
            if nxt == tok.eos_token_id:
                break
        m = min(len(v), len(ref))
        same = sum(a == b for a, b in zip(v[:m], ref[:m]))
        print(f"  vLLM    : {tok.decode(v)[:40]!r}")
        print(f"  HF+LoRA : {tok.decode(ref)[:40]!r}")
        print(f"  相同 **{same}/{m}** "
              f"{'✅ 合并与传输都对 ✓✓' if same == m else '✗✗ 不一致 ✗ 不要接线 ✗'}")
        if same != m:
            print("  ★ 不一致的典型原因 ✗：")
            print("     ① load_weights 静默跳过了没匹配上的名字 ✗")
            print("     ② LoRA 的 scaling ≠ α/r ✗（新版 PEFT 用 rsLoRA 时是 α/√r ✗✗）")
            print("     ③ 推的是没合并的 base ✗（collect_lora_pairs 配到 0 对时就会这样 ✗）")
        del hf_gpu
        torch.cuda.empty_cache()
    else:
        print("  （没设 --lora ✓ 跳过 ✓）")

    print(f"\n── 吞吐（{args.n} 条 × ≤128 token）")
    t0 = time.perf_counter()
    outs = be.generate([probe] * args.n, temperature=1.0, max_tokens=128)
    dt = time.perf_counter() - t0
    ntok = sum(len(o) for o in outs)
    print(f"  {dt:.2f}s · {ntok} token · **{ntok / dt:.0f} tok/s**")

    print("\n── 权重同步开销（连续 3 次 ✓ 模拟训练循环）")
    for i in range(3):
        print(f"  第 {i + 1} 次：{be.sync():.2f}s")

    be.shutdown()
    print("\n★ 若 send_weights 稳定 < 2s ✓ 且答案含 391 ✓ → 可以接进 main.py 了 ✓✓")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
