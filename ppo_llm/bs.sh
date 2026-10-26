#!/bin/bash
# ★ PRM（B 组）启动脚本 ✓✓
#
#   和 A 组（普通 GRPO）**唯一的差别**是 `--reward-shape paper` ✓
#   （A 组就是 main.py 的普通 GRPO ✓ 已跑出 T=1 0.664 → 0.718 ✓）
#
#   用法：
#       export DEEPSEEK_API_KEY=sk-xxxx      # ★ key 从环境变量读 ✓ 别写进文件 ✗
#       bash bs.sh                           # 启动
#       pkill -f prm_train.py                # 停止
#
set -euo pipefail

cd /data/ppo

: "${DEEPSEEK_API_KEY:?✗ 请先 export DEEPSEEK_API_KEY=sk-xxxx}"
MODEL="${MODEL:-/data/models/Qwen2.5-Math-1.5B-Instruct}"
POOL="${POOL:-data/pool_1to7.jsonl}"
OUT="${OUT:-runs/prm_mlp}"
MNT="${MNT:-2048}"
export PPO_MODEL="$MODEL"
export VLLM_LOGGING_LEVEL=WARNING
export PPO_SYSTEM=none          # ★★ 关键 ✓ 不加会被 system 逼着写 `#### <number>` ✗
export CUDA_VISIBLE_DEVICES=0,1
# ★★★ 防碎片 ✓✓ —— 报错信息自己建议的 ✓
#   症状：**跑很久才 OOM** ✓ 且 `reserved but unallocated` 有几 GiB ✗
#   → 不是某一步特别大 ✗ 是**分配器把内存切成碎片、越攒越多** ✓
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ★ 用法：`bash bs.sh`（正式 45 步）· `ITERS=2 EVAL_EVERY=1 bash bs.sh`（6 分钟验证 ✓）
ITERS="${ITERS:-45}"
EVAL_EVERY="${EVAL_EVERY:-15}"

LOG="${OUT}.log"
ERR="${OUT}.err"
rm -f "$LOG" "$ERR"

nohup python3 prm_train.py \
    --model "$PPO_MODEL" \
    --device cuda:1 \
    --pool-file "$POOL" \
    --test-file data/math500.jsonl \
    --problems 8 \
    --eval-every $EVAL_EVERY \
    --eval-size 500 \
    --eval-chunk 64 \
    --group-size 4 \
    --iters $ITERS \
    --ckpt-every 5 \
    --judge api \
    --judge-model deepseek-chat \
    --beta 0.15 \
    --adv-norm batch \
    --segment-mode paragraph \
    --reward-shape paper \
    --max-new-tokens "$MNT" \
    --max-len 2048 \
    --rollout-chunk 16 \
    --rows-per-update 1 \
    --inner-epochs 2 \
    --lr 3e-5 \
    --kl-coef 0.02 \
    --clip-eps 0.1 \
    --grad-clip 0.5 \
    --vllm \
    --vllm-gpu-util 0.85 \
    --vllm-kv-gb 6 \
    --out "$OUT" \
    > "$LOG" 2> "$ERR" &

echo "pid = $!"
echo "日志：tail -f $LOG"
echo "关键行：grep -E '步 |段判对|ratio|judge|OutOfMemory' $LOG | head -12"
