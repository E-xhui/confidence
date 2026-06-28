#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen shard JSONL files.")
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "phase2_main"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "predictions_qwen_full.jsonl"
    shard_paths = sorted(run_dir.glob("predictions_qwen_shard_*.jsonl"))
    if not shard_paths:
        raise SystemExit("No shard outputs found.")
    total = 0
    with out.open("w", encoding="utf-8") as w:
        for path in shard_paths:
            with path.open("r", encoding="utf-8") as r:
                n = 0
                for line in r:
                    if line.strip():
                        w.write(line)
                        n += 1
                print(f"{path.name}: {n}")
                total += n
    print(f"merged: {out} ({total} rows)")


if __name__ == "__main__":
    main()
