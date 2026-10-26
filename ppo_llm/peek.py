"""peek.py —— 同一批题，看**基座 / 两个 SFT**各自输出什么 ✓（6 题 ✓ ~30 秒 ✓）"""
import argparse, os, sys
# ★ 用**脚本自己的目录** ✗ 不要写死 /data/ppo ✗（换机器就废 ✓）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from config import PPOConfig
from model import MODEL_PATH, LMWithValueHead, load_tokenizer, rollout, empty_cache
from batch import gen_start

ap = argparse.ArgumentParser()
ap.add_argument("--test-file", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--adapter", nargs="*", default=[None],
                help="可以给多个：'基座' 用 None（写成 none ✓）")
ap.add_argument("--max-new-tokens", type=int, default=512, dest="mnt")
a = ap.parse_args()
adapters = [None if x in (None, "none", "") else x for x in a.adapter]

tok = load_tokenizer(MODEL_PATH)
test = D.load_test_file(a.test_file)[: a.n]
for ad in adapters:
    m = LMWithValueHead(MODEL_PATH, device="cuda", lora=True, init_lora=ad,
                        dtype="float32", autocast="bfloat16")
    m.eval()
    cfg = PPOConfig.preset("llm").replace(temperature=0.0, max_new_tokens=a.mnt,
                                          kl_coef=0.0)
    b = rollout(m, tok, test, None, cfg, seed=0)
    Lp = gen_start(b.mask)
    print(f"\n{'='*72}\n  ★ adapter = {ad or '**基座（无 adapter）**'}\n{'='*72}")
    for j, pr in enumerate(test):
        txt = tok.decode(b.input_ids[j, int(Lp[j]):], skip_special_tokens=True)
        ok = int(D.reward_fn_base(txt, pr["gold"]))
        print(f"\n  ── 题{j+1} · gold={pr['gold']} · {'✅ 对' if ok else '✗ 错'} · "
              f"{len(txt)} 字符 ──")
        print("  " + txt.replace("\n", "\n  ")[:900])
    del b
    empty_cache()
