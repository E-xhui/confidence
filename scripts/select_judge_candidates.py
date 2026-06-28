#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import load_predictions
from ragcalib.text_utils import normalize_text


def token_set(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "to",
        "is",
        "was",
        "were",
        "by",
        "with",
        "film",
        "movie",
        "genre",
    }
    return {tok for tok in normalize_text(text).split() if tok and tok not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def answer_candidates(row: dict) -> list[str]:
    vals = []
    raw = row.get("gold_answers", [])
    if isinstance(raw, list):
        vals.extend(str(v) for v in raw)
    elif isinstance(raw, str):
        vals.append(raw)
    vals.append(str(row.get("gold_answer", "")))
    out = []
    seen = set()
    for val in vals:
        val = val.strip()
        norm = normalize_text(val)
        if val and norm and norm not in seen:
            out.append(val)
            seen.add(norm)
    return out


def likely_boundary(row: dict) -> tuple[bool, str, float]:
    pred = str(row.get("model_answer", "") or "").strip()
    if not pred or bool(row.get("is_abstain", False)):
        return False, "empty_or_abstain", 0.0
    pred_norm = normalize_text(pred)
    if len(pred_norm.split()) > 14:
        return False, "too_long", 0.0
    best = 0.0
    best_reason = "low_overlap"
    for gold in answer_candidates(row):
        gold_norm = normalize_text(gold)
        if not gold_norm:
            continue
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return True, "substring_or_superset", 1.0
        score = jaccard(token_set(pred), token_set(gold))
        if score > best:
            best = score
            best_reason = "token_overlap"
    if best >= 0.25:
        return True, best_reason, best
    # High-confidence sup/closed misses are often alias or granularity failures.
    if row.get("condition") in {"closed", "sup"}:
        try:
            if float(row.get("conf_ptrue", 0.0)) >= 0.9 and len(pred_norm.split()) <= 8:
                return True, "high_conf_closed_or_sup_miss", best
        except Exception:
            pass
    return False, best_reason, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Select likely EM-boundary rows for LLM judge review.")
    parser.add_argument("--predictions", default="runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--out", default="runs/phase2_with_same_entity_irr/judge_candidates.jsonl")
    parser.add_argument("--max-per-condition", type=int, default=500)
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    rows = []
    per_condition = {}
    for row in df.to_dict("records"):
        if int(row.get("correct", 0)) != 0:
            continue
        keep, reason, score = likely_boundary(row)
        if not keep:
            continue
        cond = row.get("condition", "")
        count = per_condition.get(cond, 0)
        if count >= args.max_per_condition:
            continue
        out_row = {
            "candidate_id": len(rows),
            "question_id": row.get("question_id"),
            "condition": cond,
            "sample_idx": row.get("sample_idx"),
            "question": row.get("question"),
            "context": row.get("context", ""),
            "gold_answer": row.get("gold_answer", ""),
            "gold_answers": row.get("gold_answers", []),
            "model_answer": row.get("model_answer", ""),
            "conf_ptrue": row.get("conf_ptrue"),
            "conf_verbalized": row.get("conf_verbalized"),
            "selection_reason": reason,
            "overlap_score": score,
        }
        rows.append(out_row)
        per_condition[cond] = count + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "n": len(rows),
        "max_per_condition": args.max_per_condition,
        "per_condition": per_condition,
        "out": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
