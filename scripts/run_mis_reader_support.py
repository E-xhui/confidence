#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.modeling import answer_prompt, clean_answer, generate_text, load_model_bundle
from ragcalib.text_utils import answer_in_text, normalize_text


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def supports_wrong(reader_answer: str, wrong_answer: str, gold_answers: list[str]) -> tuple[int, int]:
    answer_norm = normalize_text(reader_answer)
    if not answer_norm:
        return 0, 0
    supports_gold = int(answer_in_text(reader_answer, gold_answers))
    wrong_hit = answer_in_text(wrong_answer, [reader_answer]) or answer_in_text(reader_answer, [wrong_answer])
    return int(wrong_hit and not supports_gold), supports_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether mis contexts lead an independent reader to the wrong answer.")
    parser.add_argument("--workset", required=True, help="workset.jsonl")
    parser.add_argument("--out", required=True, help="reader support JSONL")
    parser.add_argument("--model", required=True, help="Reader model path or HF id.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    workset = read_jsonl(args.workset)
    if args.limit is not None:
        workset = workset[: args.limit]

    bundle = load_model_bundle(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in workset:
            prompt = answer_prompt(bundle.tokenizer, item["mis_context"], item["question"])
            raw, _, _ = generate_text(
                bundle,
                prompt,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
            )
            answer = clean_answer(raw)
            supports, supports_gold = supports_wrong(answer, item.get("wrong_answer", ""), item.get("gold_answers", []))
            row = {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_answer": item.get("gold_answer", ""),
                "gold_answers": item.get("gold_answers", []),
                "wrong_answer": item.get("wrong_answer", ""),
                "reader_answer": answer,
                "reader_answer_raw": raw,
                "mis_reader_supports_wrong": supports,
                "mis_reader_supports_gold": supports_gold,
                "reader_model": args.model,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

