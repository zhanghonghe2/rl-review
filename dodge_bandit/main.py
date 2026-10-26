#!/usr/bin/env python3
"""统一入口：把两个 RL 小 demo 放在一条命令后面。

    python main.py bandit          # 多臂老虎机：4 种策略对照
    python main.py dodge           # 躲避老虎：表格 Q-learning 追逃
    python main.py all             # 两个都跑
    python main.py bandit --help   # 各自的参数
"""

from __future__ import annotations

import sys

import bandits
import dodge

USAGE = """用法：
  python main.py bandit [参数]   多臂老虎机策略对照
  python main.py dodge  [参数]   躲避老虎 Q-learning

例子：
  python main.py bandit --steps 2000 --runs 300
  python main.py dodge --episodes 6000 --render
  python main.py all
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "bandit":
        return bandits.main(rest)
    if cmd == "dodge":
        return dodge.main(rest)
    if cmd == "all":
        print("=" * 72)
        print("① 多臂老虎机")
        print("=" * 72)
        bandits.main(["--steps", "1000", "--runs", "200", *rest])
        print()
        print("=" * 72)
        print("② 躲避老虎")
        print("=" * 72)
        return dodge.main(["--episodes", "4000", *rest])

    print(f"未知子命令：{cmd}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
