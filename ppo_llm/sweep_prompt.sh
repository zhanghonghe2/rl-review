#!/bin/bash
# ★ 三步对照：只改 prompt ✓ 不训练 ✓
#   每次还会打 **三段式率** ✓ → 它是"公平性前提" ✓（若很低 ✗ 那这组实验要重做 ✓）
cd /data/ppo
export PPO_MODEL=/data/models/Qwen2.5-3B-Instruct
export VLLM_LOGGING_LEVEL=WARNING

for st in box plan three; do
  echo "════ style=$st ════"
  python3 sft.py --no-adapter --test-file data/math500_$st.jsonl \
      --eval-size 99999 --eval-chunk 8 --max-new-tokens 1024 \
      --reward-mode math --vllm --vllm-sleep-level 0 2>&1 | grep 评估
done
