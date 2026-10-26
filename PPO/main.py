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


def collate_fn(batch):
    """batch 就是 list of strings，直接返回"""
    return batch

    


def PPO(loraConfigActor: LoraConfig, loraConfigCritic: LoraConfig, epoches=3, batch_size=4):
    reward_pool = ThreadPoolExecutor(max_workers=24)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")

    # 打乱后切成不相交的两半：一半 RL，一半 SFT
    ds_train = ds_train.shuffle(seed=42)
    split = len(ds_train) // 2
    ds_rl = ds_train.select(range(split))        # PPO 采样用
    ds_sft = ds_train.select(range(split, len(ds_train)))  # SFT loss 用

    train_dataset = MathDataset(ds_rl)
    test_dataset = MathDataset(ds_test)

    dataLoaderTrain = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # dataLoaderTest = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


    actor = Actor(modelPath="Qwen2.5-0.5B-Instruct", loraConfig=loraConfigActor)
    critic = Critic(modelPath="Qwen2.5-0.5B-Instruct", loraConfig=loraConfigCritic)
    rewardModel = RewardModel()
    
    actorOptimizer = torch.optim.AdamW(actor.actorModel.parameters(), lr=1e-5)
    criticOptimizer = torch.optim.AdamW(critic.criticModel.parameters(), lr=1e-5)

    T_max = int(len(train_dataset) / batch_size + 1) * epoches
    actorScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actorOptimizer, T_max=T_max)
    criticScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(criticOptimizer, T_max=T_max)

    env_step = 0
    
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

    for epoch in range(epoches):
        for prompts in tqdm(dataLoaderTrain):
            env_step += 1

            responses, logProbsOld = actor.inference(prompts)
            # responses: len(response) = 4

            # reward
            # with ThreadPoolExecutor(max_workers=batch_size) as pool:
            reward = list(reward_pool.map(
                lambda x: rewardModel.get_reward(*x)[0],
                zip(prompts, responses)
            ))
            
            # critic
            stateValue, nextStateValue = critic.getValues(prompts, responses) 
            returns, advantages = [], []
            for idx in range(batch_size):
                G = nextStateValue[idx][-1].detach() 
                rewardNow = reward[idx]
                returnNow, advantageNow = [], []
                for i in range(stateValue[idx].shape[0] -1, -1, -1):
                    G = rewardNow + 0.99 * G
                    rewardNow = 0
                    with torch.no_grad():
                        A = G - stateValue[idx][i]
                    advantageNow.insert(0, A)
                    returnNow.insert(0, G)
                returnNow = torch.stack(returnNow)
                advantageNow = torch.stack(advantageNow)
                advantageNow = (advantageNow - advantageNow.mean(dim=0, keepdim=True)) / (advantageNow.std(dim=0, keepdim=True) + 1e-8)
                returns.append(returnNow)
                advantages.append(advantageNow)

            # actor
            logProbsRef = actor.ref_log_probs(prompts, responses)
            for _ in range(4):
                logProbsNew = actor.forward_log_probs(prompts, responses)
                stateValue, nextStateValue = critic.getValues(prompts, responses)
                loss = 0
                for idx in range(batch_size):
                    criticLoss = torch.nn.functional.mse_loss(returns[idx], stateValue[idx])
                    ratio = (logProbsNew[idx] - logProbsOld[idx]).exp()
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[idx]
                    actor_loss = -torch.mean(torch.min(surr1, surr2))
                    kl = (logProbsNew[idx] - logProbsRef[idx].to(device)).mean()
                    loss += (criticLoss + actor_loss) + 0.02 * kl

                
                loss = loss / batch_size
                criticOptimizer.zero_grad()
                actorOptimizer.zero_grad()
                loss.backward()
                criticOptimizer.step()
                actorOptimizer.step()

            torch.mps.empty_cache()
            actorScheduler.step()
            criticScheduler.step()
            
            if env_step % 20 == 0:
                print(f"当前是第 {env_step} 步, 当前的得分是 {evaluate(20)}")

            if env_step % 100 == 0:
                actor.actorModel.save_pretrained(f"actor_model_{env_step}")


    
def GRPO(loraConfigActor: LoraConfig, loraConfigCritic: LoraConfig, epoches=3, batch_size=4, group_size=4):
    reward_pool = ThreadPoolExecutor(max_workers=24)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")

    ds_train = ds_train.shuffle(seed=42)
    split = len(ds_train) // 2
    ds_rl = ds_train.select(range(split))        # PPO 采样用
    ds_sft = ds_train.select(range(split, len(ds_train))) 

    train_dataset = MathDataset(ds_rl)
    test_dataset = MathDataset(ds_test)

    dataLoaderTrain = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    actor = Actor(modelPath="Qwen2.5-0.5B-Instruct", loraConfig=loraConfigActor)
    rewardModel = RewardModel()
    actorOptimizer = torch.optim.AdamW(actor.actorModel.parameters(), lr=1e-5)
    T_max = int(len(train_dataset) / batch_size + 1) * epoches
    actorScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actorOptimizer, T_max=T_max)

    env_step = 0

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

    for epoch in range(epoches):
        for prompts in tqdm(dataLoaderTrain):
            env_step += 1

            expandedPrompts = [p for p in prompts for _ in range(group_size)]
            responses, logProbsOld = actor.inference(expandedPrompts)

            rewards = list(reward_pool.map(
                lambda x: rewardModel.get_reward(*x)[0],
                zip(expandedPrompts, responses)
            ))

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).view(batch_size, group_size)
            groupMean = rewards.mean(dim=1, keepdim=True)
            groupStd = rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards - groupMean) / groupStd # [batch_size, group_size]

            for _ in range(4):
                actorLoss = 0
                logProbsNew = actor.forward_log_probs(expandedPrompts, responses)
                for idx in range(batch_size * group_size):
                    ratio = (logProbsNew[idx] - logProbsOld[idx]).exp()
                    surr1 = ratio * advantages.view(-1)[idx]
                    surr2 = torch.clamp(ratio, 0.9, 1.1) * advantages.view(-1)[idx]
                    actorLoss += -torch.min(surr1, surr2).mean()

                
                actorLoss = actorLoss / (batch_size * group_size)
                actorOptimizer.zero_grad()
                actorLoss.backward()
                actorOptimizer.step()

            torch.mps.empty_cache()
            actorScheduler.step()
            
            if env_step % 20 == 0:
                print(f"当前是第 {env_step} 步, 当前的得分是 {evaluate(20)}")

            if env_step % 100 == 0:
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

    # PPO(loraConfigActor, loraConfigCritic)
    GRPO(loraConfigActor, loraConfigCritic)


# '''
# actor_inference_old reward critic_advantage reference_inference actor_inference_new 
                
# '''