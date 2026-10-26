import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from critic import Critic

class Actor:
    def __init__(self, modelPath, loraConfig: LoraConfig):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.actorModel = AutoModelForCausalLM.from_pretrained(modelPath, torch_dtype=torch.bfloat16, device_map=self.device)
        self.referenceModel = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.lora_config = loraConfig
        self.actorModel = get_peft_model(self.actorModel, self.lora_config)



    def inference(self, prompts):
        """生成回复 + 用 forward_log_probs 算 old log_probs（保证格式一致）"""
        messages = [
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": p}]
            for p in prompts
        ]
        texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(self.device)

        with torch.no_grad():
            generated = self.actorModel.generate(
                **model_inputs, max_new_tokens=512,
                do_sample=True, temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = model_inputs.input_ids.shape[1]
        response_ids = generated[:, prompt_len:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # 用 forward_log_probs 算 old，保证和后续 new 格式完全一致
        log_probs_old = self.forward_log_probs(prompts, responses, detach=True)
        return responses, log_probs_old


    def forward_log_probs_from_tokens(self, full_token_ids, gen_lens, detach=False):
        """
        用 vLLM 原始 token 序列算 log_probs，不重新分词（根治 off-by-N）
        full_token_ids: list of list[int]，vLLM 返回的完整序列（prompt + response）
        gen_lens: list of int，每个样本生成的 token 数（= len(vLLM logprobs)）
        """
        # pad 到相同长度
        max_len = max(len(ids) for ids in full_token_ids)
        padded = torch.tensor(
            [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in full_token_ids],
            dtype=torch.long, device=self.device,
        )
        attention_mask = (padded != self.tokenizer.pad_token_id).long()

        outputs = self.actorModel(input_ids=padded, attention_mask=attention_mask)
        logits = torch.log_softmax(outputs.logits, dim=-1)   # (batch, seq, vocab)

        log_probs_new = []
        for i, ids in enumerate(full_token_ids):
            G = gen_lens[i]
            # 生成部分在序列末尾 G 个 token；logits[t] 预测 token[t+1]
            resp_logits = logits[i, -G - 1 : -1, :]        # (G, vocab)
            resp_ids = torch.tensor(ids[-G:], device=self.device)   # (G,)
            lp = resp_logits.gather(1, resp_ids.unsqueeze(-1)).squeeze(-1)
            log_probs_new.append(lp.detach() if detach else lp)

        return log_probs_new

    def forward_log_probs(self, prompts, responses, detach=False):
        """一次分词完整对话 → 前向 → 抠 response 部分的 log_prob"""
        messages = [
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": p},
             {"role": "assistant", "content": r}]
            for p, r in zip(prompts, responses)
        ]
        texts = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        full_ids = inputs["input_ids"]                              # (batch, padded_seq)

        outputs = self.actorModel(**inputs)
        logits = torch.log_softmax(outputs.logits, dim=-1)

        # suffix 长度：模板末尾 <eos>\n 的 token 数
        suffix = self.tokenizer.eos_token + "\n"
        suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))

        resp_lens = [len(self.tokenizer.encode(r, add_special_tokens=False)) for r in responses]

        log_probs_new = []
        for i, r_len in enumerate(resp_lens):
            # 内容在 [ -r_len - suffix_len , -suffix_len )
            content_logits = logits[i, -r_len - suffix_len - 1 : -suffix_len - 1, :]  # (r_len, vocab)
            content_ids    = full_ids[i, -r_len - suffix_len : -suffix_len]            # (r_len,)
            lp = content_logits.gather(1, content_ids.unsqueeze(-1)).squeeze(-1)
            log_probs_new.append(lp.detach() if detach else lp)

        return log_probs_new

    def ref_log_probs_from_tokens(self, full_token_ids, gen_lens):
        """
        用 vLLM 原始 token 序列算 reference 的 log_probs——和 forward_log_probs_from_tokens 对齐
        """
        max_len = max(len(ids) for ids in full_token_ids)
        padded = torch.tensor(
            [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in full_token_ids],
            dtype=torch.long, device=self.device,
        )
        attention_mask = (padded != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.referenceModel(input_ids=padded, attention_mask=attention_mask)
            logits = torch.log_softmax(outputs.logits, dim=-1)

        log_probs_ref = []
        for i, ids in enumerate(full_token_ids):
            G = gen_lens[i]
            resp_logits = logits[i, -G - 1 : -1, :]
            resp_ids = torch.tensor(ids[-G:], device=self.device)
            log_probs_ref.append(resp_logits.gather(1, resp_ids.unsqueeze(-1)).squeeze(-1))

        return log_probs_ref

    def ref_log_probs(self, prompts, responses):
        messages = [
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": p},
             {"role": "assistant", "content": r}]
            for p, r in zip(prompts, responses)
        ]
        texts = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)  # ref 在 CPU，inputs 也在 CPU
        full_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.referenceModel(**inputs)
            logits = torch.log_softmax(outputs.logits, dim=-1)

        suffix = self.tokenizer.eos_token + "\n"
        suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))
        resp_lens = [len(self.tokenizer.encode(r, add_special_tokens=False)) for r in responses]

        log_probs_ref = []
        for i, r_len in enumerate(resp_lens):
            lp = logits[i, -r_len - suffix_len - 1 : -suffix_len - 1, :]
            ids = full_ids[i, -r_len - suffix_len : -suffix_len]
            log_probs_ref.append(lp.gather(1, ids.unsqueeze(-1)).squeeze(-1))

        return log_probs_ref


if __name__ == "__main__":
    loraConfigA = LoraConfig(
        r=8,                    
        lora_alpha=16,          
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    actor = Actor(modelPath="Qwen2.5-0.5B-Instruct", loraConfig=loraConfigA)

    loraConfigC = LoraConfig(
            r=8,                    
            lora_alpha=16,          
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05,
    )
    critic = Critic(modelPath="Qwen2.5-0.5B-Instruct", loraConfig=loraConfigC)

    responses, log_probs_old = actor.inference(["什么是相对论", "什么是黑洞"])
    # print(responses)
    log_probs_new = actor.forward_log_probs(["什么是相对论", "什么是黑洞"], responses)
    print(log_probs_old[0].shape)
    print(log_probs_new[0].shape)
    # print(log_probs_new)
    stateV = critic.getValues(["什么是相对论", "什么是黑洞"], responses)
    print(stateV[0].shape)