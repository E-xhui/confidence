#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize condition reader support output.")
    parser.add_argument("--reader-output", required=True)
    parser.add_argument("--condition", choices=["sup", "mis"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.reader_output).open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("No rows found.")
    n = len(rows)
    gold_key = f"{args.condition}_reader_supports_gold"
    wrong_key = f"{args.condition}_reader_supports_wrong"
    support_gold = sum(int(row.get(gold_key, 0)) for row in rows)
    support_wrong = sum(int(row.get(wrong_key, 0)) for row in rows)
    reader_models = sorted({row.get("reader_model", "") for row in rows if row.get("reader_model")})
    examples_not_supporting_gold = [
        {
            "question": row.get("question", ""),
            "gold_answer": row.get("gold_answer", ""),
            "wrong_answer": row.get("wrong_answer", ""),
            "reader_answer": row.get("reader_answer", ""),
        }
        for row in rows
        if int(row.get(gold_key, 0)) == 0
    ][:10]
    examples_supporting_wrong = [
        {
            "question": row.get("question", ""),
            "gold_answer": row.get("gold_answer", ""),
            "wrong_answer": row.get("wrong_answer", ""),
            "reader_answer": row.get("reader_answer", ""),
        }
        for row in rows
        if int(row.get(wrong_key, 0)) == 1
    ][:10]
    summary = {
        "condition": args.condition,
        "n": n,
        "support_gold_rate": support_gold / n,
        "support_wrong_rate": support_wrong / n,
        "reader_model": reader_models,
        "examples_not_supporting_gold": examples_not_supporting_gold,
        "examples_supporting_wrong": examples_supporting_wrong,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
