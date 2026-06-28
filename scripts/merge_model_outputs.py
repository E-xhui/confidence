#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge model shard JSONL files.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prefix", required=True, help="For predictions_<prefix>_shard_*.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / f"predictions_{args.prefix}_full.jsonl"
    shard_paths = sorted(run_dir.glob(f"predictions_{args.prefix}_shard_*.jsonl"))
    if not shard_paths:
        raise SystemExit(f"No shard outputs found for prefix={args.prefix} in {run_dir}")
    total = 0
    with out.open("w", encoding="utf-8") as w:
        for path in shard_paths:
            n = 0
            with path.open("r", encoding="utf-8") as r:
                for line in r:
                    if line.strip():
                        w.write(line)
                        n += 1
            print(f"{path.name}: {n}")
            total += n
    print(f"merged: {out} ({total} rows)")


if __name__ == "__main__":
    main()
