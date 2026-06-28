#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.data import build_workset, make_context_rows, write_jsonl, write_mis_examples
from ragcalib.spec_config import SPEC


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PopQA/ConflictQA Phase 2 contexts.")
    parser.add_argument("--out-dir", required=True, help="Output run directory.")
    parser.add_argument("--n-total", type=int, default=SPEC.n_total)
    parser.add_argument("--conflict-file", default="conflictQA-popQA-qwen7b.json")
    parser.add_argument(
        "--evidence-policy",
        choices=["auto", "spec_fields"],
        default="auto",
        help="auto detects true vs misleading evidence; spec_fields follows raw field names.",
    )
    parser.add_argument(
        "--unique-by",
        choices=["question", "question_id", "question_and_gold", "none"],
        default="question",
        help="Deduplicate selected workset. Use question for strict unique user-facing items.",
    )
    parser.add_argument("--seed", type=int, default=SPEC.random_seed)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workset, diagnostics = build_workset(
        n_total=args.n_total,
        conflict_file=args.conflict_file,
        evidence_policy=args.evidence_policy,
        seed=args.seed,
        unique_by=args.unique_by,
    )
    context_rows, context_diag = make_context_rows(workset)
    diagnostics.update(context_diag)

    write_jsonl(out_dir / "workset.jsonl", workset)
    write_jsonl(out_dir / "contexts_phase2.jsonl", context_rows)
    write_mis_examples(out_dir / "mis_examples.md", workset, n=5)
    (out_dir / "data_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
