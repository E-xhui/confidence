#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced T3 cross-family context subset.")
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--questions-per-bin", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    rows = []
    by_qid: dict[str, list[dict]] = defaultdict(list)
    qid_bin: dict[str, str] = {}
    with Path(args.contexts).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            by_qid[qid].append(row)
            qid_bin[qid] = str(row["popularity_bin"])

    rng = random.Random(args.seed)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for pop_bin in ["low", "mid", "high"]:
        qids = sorted(qid for qid, b in qid_bin.items() if b == pop_bin)
        if len(qids) < args.questions_per_bin:
            raise SystemExit(f"Not enough {pop_bin} questions: {len(qids)} < {args.questions_per_bin}")
        picked = rng.sample(qids, args.questions_per_bin)
        selected.extend(picked)
        counts[pop_bin] = len(picked)

    selected_set = set(selected)
    condition_order = {"closed": 0, "irr": 1, "same_entity_irr": 2, "sup": 3, "mis": 4}
    for qid in sorted(selected_set):
        rows.extend(sorted(by_qid[qid], key=lambda r: condition_order.get(str(r["condition"]), 99)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source_contexts": args.contexts,
        "out": str(out),
        "seed": args.seed,
        "questions_per_bin": args.questions_per_bin,
        "selected_questions": len(selected_set),
        "rows": len(rows),
        "popularity_counts": counts,
    }
    (out.parent / "phi3_subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
