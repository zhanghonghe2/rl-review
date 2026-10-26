# data/ 说明

仓库里**只放测试集与小型派生集**（约 11 MB）。可再生成的大文件已排除在 `.gitignore` 里。

## 已包含

| 文件 | 用途 |
|---|---|
| `gsm8k_test.jsonl` / `gsm8k_test_0.jsonl` / `gsm8k_test_3.jsonl` | GSM8K 官方 test（1319 题）|
| `gsm8k_private128.jsonl` / `gsm8k_test_fmt128.jsonl` | 私有重写测试集 / 格式诊断子集 |
| `math500.jsonl` | MATH-500（500 题，主评测集）|
| `math500_{box,plan,three,prm}.jsonl` | 三种 prompt 变体 / PRM 格式版 |
| `math_test.jsonl` | MATH 官方 test（5,000 题）|
| `*.hits.jsonl` | 池子筛选记录（每题 8 条里对几条）|
| `math_prm_train.jsonl.rejected.jsonl` | 生成 5 段数据时被验证器拒掉的样本 |

## 未包含（可再生成）

| 文件 | 怎么生成 |
|---|---|
| `math_train.jsonl` / `math_test.jsonl` | `python3 math_data.py convert_hf`（HuggingFace MATH）|
| `math_train_hard.jsonl` | 由 `math_train.jsonl` 按 level≥3 过滤 |
| `gsm8k_train*.jsonl` | HuggingFace GSM8K（train 7,473），按 1500/5973 切分 |
| `math_prm_train.jsonl`（24 MB）| `python3 get_math_prm.py --src data/math_train.jsonl --out data/math_prm_train.jsonl` |
| `pool_1to7*.jsonl` / `hard500/hard684/hard266` | `python3 filter_pool.py --pool ... --lo 1 --hi 7` |
| `prm_split/{sft,rl,dev}.jsonl` | `python3 split_prm.py` |

**★ 口径提醒**：所有指标用**严格判据**（`\boxed{}` 符号等价）· 500 题 · `pass@1` = T=1 单样本。
