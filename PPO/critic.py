import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class CriticModule(nn.Module):
    """Critic 网络：共享 Transformer backbone + 标量输出头

    输入: prompt + response 的 token 序列
    输出: 每个 token 位置的价值 V(s)，形状 (batch, seq_len)
    """
    def __init__(self, modelPath, loraConfig: LoraConfig):
        super().__init__()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # backbone: 只取 Transformer，不要 lm_head
        base_model = AutoModel.from_pretrained(
            modelPath, torch_dtype=torch.float32   # MPS float16 矩阵乘有兼容问题
        )
        hidden_size = base_model.config.hidden_size

        # Critic 不做生成，task_type 改成 FEATURE_EXTRACTION
        loraConfig.task_type = "FEATURE_EXTRACTION"
        self.backbone = get_peft_model(base_model.to(self.device), loraConfig)

        # 标量输出头: hidden → 1
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, input_ids, attention_mask=None):
        """返回每个 token 位置的 V(s)"""
        outputs = self.backbone(
            input_ids=input_ids.to(self.device),
            attention_mask = attention_mask.to(self.device, dtype=torch.float16) if attention_mask is not None else None,
            output_hidden_states=True,
        )
        # hidden_states[-1]: (batch, seq_len, hidden_size)
        values = self.value_head(outputs.last_hidden_state).squeeze(-1)  # (batch, seq_len)
        return values

class Critic:
    def __init__(self, modelPath, loraConfig: LoraConfig):
        self.criticModel = CriticModule(modelPath, loraConfig)
        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"


    def getValuesFromTokens(self, full_token_ids, gen_lens):
        """
        用 vLLM 原始 token 序列算 V(s)——和 actor.forward_log_probs_from_tokens 对齐
        """
        max_len = max(len(ids) for ids in full_token_ids)
        padded = torch.tensor(
            [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in full_token_ids],
            dtype=torch.long, device="cpu",
        )
        attention_mask = (padded != self.tokenizer.pad_token_id).long()

        outputs = self.criticModel(input_ids=padded, attention_mask=attention_mask)
        all_values = outputs.squeeze(-1)   # (batch, seq)

        values, valuesNext = [], []
        for i, ids in enumerate(full_token_ids):
            G = gen_lens[i]
            # V(s_t) 在 token 前一个位置（和 logits 偏移一致）
            values.append(all_values[i, -G - 1 : -1])     # (G,)
            valuesNext.append(all_values[i, -G :])        # (G,)

        return values, valuesNext

    def getValues(self, prompts, responses):
        """和 actor.forward_log_probs 同样的分刺和提取逻辑"""
        messages = [
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": p},
             {"role": "assistant", "content": r}]
            for p, r in zip(prompts, responses)
        ]
        texts = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to("cpu")
        outputs = self.criticModel(**model_inputs)                # (batch, padded_seq)

        suffix = self.tokenizer.eos_token + "\n"
        suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))
        resp_lens = [len(self.tokenizer.encode(r, add_special_tokens=False)) for r in responses]

        values = []
        veluesNext = []
        for i, r_len in enumerate(resp_lens):
            # 和 actor logits 对齐偏移：V(s_t) 在 token 前一个位置
            values.append(outputs[i, -r_len - suffix_len - 1: -suffix_len - 1])
            veluesNext.append(outputs[i, -r_len - suffix_len: -suffix_len])

        return values, veluesNext


if __name__ == "__main__":
    loraConfig = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    critic = Critic(modelPath="qwen2.5-instruct", loraConfig=loraConfig)
    v = critic.getValues(["什么是相对论"], ["相对论是爱因斯坦提出的..."])
    print(v[0].shape)