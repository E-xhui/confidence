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

from ragcalib.modeling import generate_text, load_model_bundle


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def judge_prompt(tokenizer, row: dict) -> str:
    gold_answers = row.get("gold_answers", [])
    if not isinstance(gold_answers, list):
        gold_answers = [str(gold_answers)]
    user = (
        "Decide whether the model answer should be counted as correct for the question.\n"
        "Accept aliases, abbreviations, more specific answers, and answers that include the gold answer.\n"
        "Reject answers that name a different person/place/work/type, even if related.\n\n"
        f"Question: {row.get('question', '')}\n"
        f"Gold answers: {gold_answers}\n"
        f"Model answer: {row.get('model_answer', '')}\n\n"
        "Reply in exactly this JSON format: {\"correct\": true/false, \"reason\": \"short reason\"}"
    )
    messages = [
        {"role": "system", "content": "You are a strict but fair answer equivalence judge."},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user


def parse_judge(raw: str) -> tuple[int | None, str]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data.get("correct"), bool):
                return int(data["correct"]), str(data.get("reason", ""))
        except Exception:
            pass
    lower = text.lower()
    if re.search(r"\btrue\b|\byes\b|\bcorrect\b", lower) and not re.search(r"\bfalse\b|\bincorrect\b|\bno\b", lower):
        return 1, text[:200]
    if re.search(r"\bfalse\b|\bincorrect\b|\bno\b", lower):
        return 0, text[:200]
    return None, text[:200]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM judge over selected answer-equivalence candidates.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    if args.limit is not None:
        rows = rows[: args.limit]
    bundle = load_model_bundle(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            raw, _, _ = generate_text(
                bundle,
                judge_prompt(bundle.tokenizer, row),
                max_new_tokens=96,
                temperature=0.0,
                do_sample=False,
            )
            correct, reason = parse_judge(raw)
            out = {
                **row,
                "judge_correct": correct,
                "judge_reason": reason,
                "judge_raw": raw,
                "judge_model": args.model,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
