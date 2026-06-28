#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one five-condition context file for cross-model runs.")
    parser.add_argument("--phase2", default="runs/phase2_unique/contexts_phase2.jsonl")
    parser.add_argument("--irr", default="runs/phase2_unique_irr/contexts_phase2.jsonl")
    parser.add_argument("--same-entity-irr", default="runs/phase2_same_entity_irr/contexts_phase2.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_five_condition")
    args = parser.parse_args()

    rows = []
    for path in [args.phase2, args.irr, args.same_entity_irr]:
        rows.extend(read_jsonl(path))
    order = {"closed": 0, "irr": 1, "same_entity_irr": 2, "sup": 3, "mis": 4}
    rows.sort(key=lambda r: (str(r["question_id"]), order.get(str(r["condition"]), 99)))
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "contexts_phase2.jsonl", rows)
    summary = {
        "n_rows": len(rows),
        "n_questions": len({str(r["question_id"]) for r in rows}),
        "condition_counts": {},
        "sources": {
            "phase2": args.phase2,
            "irr": args.irr,
            "same_entity_irr": args.same_entity_irr,
        },
    }
    for row in rows:
        cond = str(row["condition"])
        summary["condition_counts"][cond] = summary["condition_counts"].get(cond, 0) + 1
    (out_dir / "context_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
