#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.data import read_jsonl, write_jsonl
from ragcalib.text_utils import entity_count, extract_entities, normalize_text, safe_ratio, token_len


def target_entities(item: dict) -> set[str]:
    parts = [item.get("question", ""), item.get("gold_answer", ""), item.get("wrong_answer", "")]
    parts.extend(item.get("gold_answers", []))
    return set().union(*(extract_entities(part) for part in parts))


def answer_norms(item: dict) -> list[str]:
    answers = list(item.get("gold_answers", []))
    if item.get("gold_answer"):
        answers.append(item["gold_answer"])
    if item.get("wrong_answer"):
        answers.append(item["wrong_answer"])
    return [normalize_text(answer) for answer in answers if normalize_text(answer)]


def has_answer_leak_norm(context_norm: str, answers_norm: list[str]) -> bool:
    return any(answer and answer in context_norm for answer in answers_norm)


def prepare_items(workset: list[dict]) -> list[dict]:
    prepared = []
    for item in workset:
        sup_context = item.get("sup_context", "")
        prepared.append(
            {
                **item,
                "_target_entities": target_entities(item),
                "_answer_norms": answer_norms(item),
                "_sup_len": token_len(sup_context),
                "_sup_entities": extract_entities(sup_context),
                "_sup_norm": normalize_text(sup_context),
                "_sup_entity_count": entity_count(sup_context),
            }
        )
    return prepared


def choose_irr(item: dict, donors: list[dict], rng: random.Random, max_scan: int) -> tuple[dict, dict]:
    target_ents = item["_target_entities"]
    target_len = item["_sup_len"]
    answer_norm_list = item["_answer_norms"]
    best = None
    best_score = None
    shuffled = donors[:]
    rng.shuffle(shuffled)
    for donor in shuffled[:max_scan]:
        if donor["question_id"] == item["question_id"]:
            continue
        context = donor.get("sup_context", "")
        if not context.strip():
            continue
        if has_answer_leak_norm(donor["_sup_norm"], answer_norm_list):
            continue
        overlap = target_ents & donor["_sup_entities"]
        if overlap:
            continue
        length_diff = abs(donor["_sup_len"] - target_len)
        score = (length_diff, rng.random())
        if best is None or score < best_score:
            best = donor
            best_score = score
    if best is None:
        # Rare fallback: preserve no answer leakage, but allow entity overlap.
        for donor in shuffled:
            if donor["question_id"] != item["question_id"] and not has_answer_leak_norm(donor["_sup_norm"], answer_norm_list):
                best = donor
                break
    if best is None:
        raise RuntimeError(f"No irr donor found for question_id={item['question_id']}")
    diagnostics = {
        "irr_source_question_id": best["question_id"],
        "irr_source_question": best["question"],
        "irr_len_tokens": best["_sup_len"],
        "sup_len_tokens": target_len,
        "irr_over_sup_len_ratio": safe_ratio(best["_sup_len"], target_len),
        "irr_entity_count": best["_sup_entity_count"],
        "target_entity_overlap_count": len(target_ents & best["_sup_entities"]),
        "gold_or_wrong_answer_leak": has_answer_leak_norm(best["_sup_norm"], answer_norm_list),
    }
    return best, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build irr condition contexts from unrelated donor evidence.")
    parser.add_argument("--workset", required=True)
    parser.add_argument("--out", required=True, help="Output irr contexts JSONL.")
    parser.add_argument("--diagnostics-out", required=True)
    parser.add_argument("--examples-out", required=True)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--max-scan", type=int, default=1500)
    args = parser.parse_args()

    workset = prepare_items(read_jsonl(args.workset))
    rng = random.Random(args.seed)
    rows = []
    diagnostics = []
    for item in workset:
        donor, diag = choose_irr(item, workset, rng, args.max_scan)
        diagnostics.append({"question_id": item["question_id"], **diag})
        context = donor["sup_context"]
        rows.append(
            {
                "question_id": item["question_id"],
                "popularity_bin": item["popularity_bin"],
                "popularity_raw": item["popularity_raw"],
                "condition": "irr",
                "context": context,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_answers": item["gold_answers"],
                "wrong_answer": "",
                "ctx_len_tokens": token_len(context),
                "ctx_entity_count": entity_count(context),
                "irr_source_question_id": donor["question_id"],
                "irr_source_question": donor["question"],
                "irr_source": "other_popqa_sup_context",
            }
        )

    write_jsonl(args.out, rows)
    leak_count = sum(int(d["gold_or_wrong_answer_leak"]) for d in diagnostics)
    overlap_count = sum(int(d["target_entity_overlap_count"] > 0) for d in diagnostics)
    ratios = [d["irr_over_sup_len_ratio"] for d in diagnostics if d["irr_over_sup_len_ratio"] is not None]
    summary = {
        "n": len(rows),
        "source": "other_popqa_sup_context",
        "answer_leak_count": leak_count,
        "answer_leak_rate": leak_count / len(rows) if rows else None,
        "entity_overlap_count": overlap_count,
        "entity_overlap_rate": overlap_count / len(rows) if rows else None,
        "irr_over_sup_len_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "irr_over_sup_len_ratio_min": min(ratios) if ratios else None,
        "irr_over_sup_len_ratio_max": max(ratios) if ratios else None,
        "per_item": diagnostics,
    }
    Path(args.diagnostics_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# IRR Examples", ""]
    for idx, row in enumerate(rows[:5], 1):
        lines.extend(
            [
                f"## Example {idx}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold answer: {row['gold_answer']}",
                "",
                f"IRR source question: {row['irr_source_question']}",
                "",
                "IRR context:",
                "",
                row["context"],
                "",
            ]
        )
    Path(args.examples_out).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_item"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
