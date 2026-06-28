#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.modeling import (
    answer_prompt,
    clean_answer,
    confidence_prompt,
    generate_text,
    is_abstention,
    load_model_bundle,
    parse_confidence,
    ptrue_prompt,
    ptrue_score,
    score_answer,
    seqlik_to_unit_interval,
    sequence_mean_logprob,
)
from ragcalib.spec_config import SPEC


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 sampling for one model.")
    parser.add_argument("--contexts", required=True, help="contexts_phase2.jsonl")
    parser.add_argument("--out", required=True, help="Prediction JSONL path.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=SPEC.k_samples)
    parser.add_argument("--temperature", type=float, default=SPEC.temperature)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    context_rows = read_jsonl(args.contexts)
    if args.limit_questions is not None:
        keep_ids = []
        for row in context_rows:
            qid = row["question_id"]
            if qid not in keep_ids:
                keep_ids.append(qid)
            if len(keep_ids) >= args.limit_questions:
                break
        keep = set(keep_ids)
        context_rows = [r for r in context_rows if r["question_id"] in keep]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for row_idx, row in enumerate(context_rows):
            for sample_idx in range(args.k):
                seed = SPEC.random_seed + row_idx * 100 + sample_idx
                prompt = answer_prompt(bundle.tokenizer, row["context"], row["question"])
                answer_raw, prompt_ids, answer_ids = generate_text(
                    bundle,
                    prompt,
                    max_new_tokens=SPEC.max_new_answer_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    seed=seed,
                )
                answer = clean_answer(answer_raw)
                mean_logprob = sequence_mean_logprob(bundle, prompt_ids, answer_ids)

                conf_prompt = confidence_prompt(bundle.tokenizer, row["context"], row["question"], answer)
                conf_raw, _, _ = generate_text(
                    bundle,
                    conf_prompt,
                    max_new_tokens=SPEC.max_new_conf_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
                ptrue = ptrue_score(bundle, ptrue_prompt(bundle.tokenizer, row["context"], row["question"], answer))

                result = {
                    "question_id": row["question_id"],
                    "popularity_bin": row["popularity_bin"],
                    "popularity_raw": row["popularity_raw"],
                    "condition": row["condition"],
                    "context": row["context"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_answers": row["gold_answers"],
                    "wrong_answer": row.get("wrong_answer", ""),
                    "model_name": args.model,
                    "sample_idx": sample_idx,
                    "model_answer": answer,
                    "model_answer_raw": answer_raw,
                    "correct": score_answer(answer, row["gold_answers"]),
                    "is_abstain": is_abstention(answer),
                    "conf_verbalized": parse_confidence(conf_raw),
                    "conf_verbalized_raw": conf_raw,
                    "conf_ptrue": ptrue,
                    "conf_seqlik": seqlik_to_unit_interval(mean_logprob),
                    "mean_answer_logprob": mean_logprob,
                    "ctx_len_tokens": row["ctx_len_tokens"],
                    "ctx_entity_count": row["ctx_entity_count"],
                    "temperature": args.temperature,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
