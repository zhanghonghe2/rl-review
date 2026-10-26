"""RolloutWorker：GPU 0 的 vLLM 推理 Worker（Ray Actor）

职责：只负责生成（rollout），不训练。
- 用 vLLM 批量推理
- 支持 LoRA 动态加载（每轮 PPO 更新后加载新权重）
- 返回 responses + log_probs（PPO/GRPO 都需要）
"""
import os
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


@ray.remote(num_gpus=1)
class RolloutWorker:
    def __init__(self, model_path: str, max_lora_rank: int = 8):
        self.model_path = model_path
        self.lora_path = None          # 当前加载的 LoRA 目录

        # vLLM 引擎：单卡、开 LoRA 支持
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,    # 留 10% 余量
            enable_lora=True,              # 关键：支持动态加载 LoRA
            max_lora_rank=max_lora_rank,
            max_loras=1,                   # 同时只挂一个 adapter
        )

        # 从 vLLM 拿 tokenizer（或自己加载）
        from vllm.transformers_utils.tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer(model_path)

    def generate(self, prompts, lora_path=None, max_tokens=256, temperature=1.0):
        """
        批量生成回复 + log_probs

        prompts: list[str]
        lora_path: 最新的 LoRA adapter 目录（None = 用基座模型）
        返回: (responses, log_probs, token_ids)
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=False,   
            include_stop_str_in_output=True,
            spaces_between_special_tokens=False,
            logprobs=1,                  # 返回 top-1 logprob，OPD/PPO 需要
        )

        # 构造 chat template prompt
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        # LoRA 请求（如果给了新权重）
        lora_request = None
        if lora_path is not None:
            if lora_path != self.lora_path:
                print(f"[Rollout] 加载新 LoRA: {lora_path}")
                self.lora_path = lora_path
            lora_request = LoRARequest("ppo_adapter", 1, lora_path)

        outputs = self.llm.generate(
            prompt_texts,
            sampling_params,
            lora_request=lora_request,
        )

        responses = []
        log_probs = []        # list of tensor(r_len,)
        token_ids = []        # list of list[int]，完整序列（prompt+response）
        gen_lens = []         # list of int，每个样本生成的 token 数

        for out in outputs:
            resp = out.outputs[0]
            responses.append(resp.text)

            # 逐 token logprob
            sample_lp = []
            for step_lp in resp.logprobs:
                token_lp = list(step_lp.values())[0].logprob
                sample_lp.append(token_lp)
            log_probs.append(torch.tensor(sample_lp))

            # 关键：拼接 prompt + 生成，才是完整序列
            full_ids = list(out.prompt_token_ids) + list(resp.token_ids)
            token_ids.append(full_ids)
            gen_lens.append(len(resp.logprobs))     # 生成部分长度

        return responses, log_probs, token_ids, gen_lens

    def update_lora(self, lora_path):
        """显式通知权重更新（下一轮 generate 生效）"""
        self.lora_path = lora_path
        return True

    def get_weights_info(self):
        """调试用：当前加载的权重状态"""
        return {
            "model": self.model_path,
            "lora": self.lora_path,
        }


if __name__ == "__main__":
    # 单独测试（不开 Ray，直接本地跑）
    worker = RolloutWorker("Qwen2.5-1.5B-Instruct")
    responses, lps = worker.generate(
        ["什么是相对论？", "1+1等于几？", "写一句诗"],
        lora_path=None,          # 先用基座
    )
    for r, lp in zip(responses, lps):
        print(f"回复: {r[:50]}...  log_probs: {lp.shape}")
