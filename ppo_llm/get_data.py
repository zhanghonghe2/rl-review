import os
import json
from pathlib import Path
records = []


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/gsm8k_test.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:          # 空行跳过，不然 json.loads("") 报错
            continue
        records.append(json.loads(line))




for idx in range(len(records)):
    question = f"""Answer the following math problem in a strict three-stage format.

[Problem]
{records[idx]["question"]}

Requirements:
[Plan]
Explain the solution steps in plain English (no arithmetic expressions).

[Calculation]
One line per step, shaped: <plain-English step> -> <<expression=result>> Say what the
step does first (no arithmetic in the words), then give exactly one <<>> for it. The
expression may contain only digits and + - * / ( ) . and spaces.

[Answer]
The final line must be `#### number`.

Write everything in English only.
"""

    records[idx]["question"] = question

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/gsm8k_test_3.jsonl"), "w", encoding="utf-8") as f:
    for item in records:
        # ensure_ascii=False 可以正确保留中文字符
        json_str = json.dumps(item, ensure_ascii=False)
        f.write(json_str + "\n")