import transformers
import torch
import random
from actor import Actor
from critic import Critic
from reward import RewardModel
from peft import LoraConfig, get_peft_model

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class MathDataset(Dataset):
    """将 GSM8K 包装成 PyTorch Dataset"""
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["question"]

class MathDatasetSFT(Dataset):
    """将 GSM8K 包装成 PyTorch Dataset"""
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input": self.data[idx]["question"], "output": self.data[idx]["answer"]}



def collate_fn(batch):
    """batch 就是 list of strings，直接返回"""
    return batch


def PPO(loraConfigActor: LoraConfig, loraConfigCritic: LoraConfig, epoches=3, batch_size=4):
    reward_pool = ThreadPoolExecutor(max_workers=24)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")

    ds_train = ds_train.shuffle(seed=42)
    split = len(ds_train) // 2
    ds_rl = ds_train.select(range(split))        # PPO 采样用
    ds_sft = ds_train.select(range(split, split * 2))  # SFT loss 用

    train_dataset_rl = MathDataset(ds_rl)
    train_dataset_sft = MathDatasetSFT(ds_sft)
    
    test_dataset = MathDataset(ds_test)

    dataLoaderTrainRL = DataLoader(train_dataset_rl, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataLoaderTrainSFT = DataLoader(train_dataset_sft, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # dataLoaderTest = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


    actor = Actor(modelPath="Qwen2.5-1.5B-Instruct", loraConfig=loraConfigActor)
    critic = Critic(modelPath="Qwen2.5-1.5B-Instruct", loraConfig=loraConfigCritic)
    rewardModel = RewardModel()
    
    actorOptimizer = torch.optim.AdamW(actor.actorModel.parameters(), lr=1e-5)
    criticOptimizer = torch.optim.AdamW(critic.criticModel.parameters(), lr=1e-5)

    T_max = int(len(train_dataset_rl) / batch_size + 1) * epoches
    actorScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actorOptimizer, T_max=T_max)
    criticScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(criticOptimizer, T_max=T_max)

    def evaluate(total_games=30, log_file="eval_log.txt"):
        indices = random.sample(range(len(test_dataset)), total_games)
        prompts = [test_dataset[i] for i in indices]
        with torch.no_grad():
            responses, _ = actor.inference(prompts)

        results = list(reward_pool.map(
            lambda x: rewardModel.get_reward(x[0], x[1]),
            zip(prompts, responses)
        ))
        scores = [r[0] for r in results]
        avg = float(np.mean(scores))

        # 写入文件
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Step {env_step}, 平均得分: {avg:.4f}\n")
            f.write(f"{'='*60}\n")
            for i in range(total_games):
                s, reason = results[i]
                f.write(f"[{s:.2f}] {reason}\n")
                f.write(f"  Prompt:   {prompts[i]}\n")
                f.write(f"  Response: {responses[i]}\n")
                f.write(f"  {'*'*60}\n")

        print(f"[eval] Step {env_step}, 平均得分: {avg:.4f}")
        return avg

    env_step = 0


    prePrompts, preSFTData, preResponses, preLogProbsOld, preRewards = None, None, None, None, None

    for epoch in range(epoches):
        for prompts, SFTData in tqdm(zip(dataLoaderTrainRL, dataLoaderTrainSFT), total=len(dataLoaderTrainRL)):
            env_step += 1
            
            responses, logProbsOld = actor.inference(prompts)
            

            futureRewards = reward_pool.map(
                lambda x: rewardModel.get_reward(*x)[0],
                zip(prompts, responses)
            )
            
            if prePrompts is not None:
                preStateValue, preNextStateValue = critic.getValues(prePrompts, preResponses)
                preReturns, preAdvantages = [], []
                for idx in range(batch_size):
                    G = preNextStateValue[idx][-1].detach() 
                    preRewardNow = preRewards[idx]
                    preReturnNow, preAdvantageNow = [], []
                    for i in range(preStateValue[idx].shape[0] -1, -1, -1):
                        G = preRewardNow + 0.99 * G
                        preRewardNow = 0
                        with torch.no_grad():
                            A = G - preStateValue[idx][i]
                        preAdvantageNow.insert(0, A)
                        preReturnNow.insert(0, G)
                    preReturnNow = torch.stack(preReturnNow)
                    preAdvantageNow = torch.stack(preAdvantageNow)
                    preAdvantageNow = (preAdvantageNow - preAdvantageNow.mean(dim=0, keepdim=True)) / (preAdvantageNow.std(dim=0, keepdim=True) + 1e-8)
                    preReturns.append(preReturnNow)
                    preAdvantages.append(preAdvantageNow)

                preLogProbsRef = actor.ref_log_probs(prePrompts, preResponses)
                for repeat in range(4):
                    sft_batch = batch_size // 4
                    preLogProbsNew = actor.forward_log_probs(prePrompts, preResponses)
                    preStateValue, preNextStateValue = critic.getValues(prePrompts, preResponses)
                    loss = 0

                    for idx in range(batch_size):
                        criticLoss = torch.nn.functional.mse_loss(preReturns[idx], preStateValue[idx])
                        ratio = (preLogProbsNew[idx] - preLogProbsOld[idx]).exp()
                        surr1 = ratio * preAdvantages[idx]
                        surr2 = torch.clamp(ratio, 0.9, 1.1) * preAdvantages[idx]
                        actor_loss = -torch.mean(torch.min(surr1, surr2))
                        kl = (preLogProbsNew[idx] - preLogProbsRef[idx].to(device)).mean()
                        loss += (criticLoss + actor_loss) + 0.2 * kl

                    loss_sft = actor.sft_loss([item["input"] for item in preSFTData[repeat * sft_batch: (repeat + 1) * sft_batch]], 
                        [item["output"] for item in preSFTData[repeat * sft_batch: (repeat + 1) * sft_batch]])

                    loss += 0.2 * loss_sft

                    criticOptimizer.zero_grad()
                    actorOptimizer.zero_grad()
                    loss.backward()
                    criticOptimizer.step()
                    actorOptimizer.step()

                torch.cuda.empty_cache()
                actorScheduler.step()
                criticScheduler.step()
            
            rewards = list(futureRewards)

            preRewards = rewards
            prePrompts = prompts
            preResponses = responses
            preSFTData = SFTData
            preLogProbsOld = logProbsOld

            if env_step % 30 == 0:
                print(f"当前是第 {env_step} 步, 当前的得分是 {evaluate(40)}")

            if env_step % 100 == 0:
                actor.actorModel.save_pretrained(f"actor_model_{env_step}")

    actor.actorModel.save_pretrained(f"actor_model_{env_step}")

if __name__ == "__main__":
    loraConfigActor = LoraConfig(
        r=8,                    
        lora_alpha=16,          
        target_modules=["q_proj"], 
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    loraConfigCritic = LoraConfig(
        r=8,                    
        lora_alpha=16,          
        target_modules=["q_proj"], 
        lora_dropout=0.05,
    )

    PPO(loraConfigActor, loraConfigCritic)
