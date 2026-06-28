#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.modeling import generate_text, load_model_bundle


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def audit_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    user = (
        "You are auditing whether a retrieval context is a plausible evidence stimulus for a QA experiment.\n"
        "You are NOT told the experimental condition. Evaluate only the question, context, target answer, and gold answers.\n\n"
        f"Question: {row.get('question', '')}\n"
        f"Context:\n{row.get('context', '')}\n\n"
        f"Target answer to evaluate: {row.get('target_answer', '')}\n"
        f"Gold answers: {row.get('gold_answers', [])}\n\n"
        "Return exactly one JSON object with these fields:\n"
        "{\n"
        '  "supports_target": true/false,\n'
        '  "contradicts_gold": true/false,\n'
        '  "naturalness": integer 1-5,\n'
        '  "template_likeness": integer 1-5,\n'
        '  "explicit_answer_anchor": true/false,\n'
        '  "misleading_plausibility": integer 1-5,\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "Definitions:\n"
        "- supports_target: the context supports or strongly suggests the target answer.\n"
        "- contradicts_gold: the context asserts or implies an answer incompatible with the gold answers.\n"
        "- naturalness: 1=very artificial, 5=natural retrieved-style passage.\n"
        "- template_likeness: 1=not templated, 5=obviously templated or formulaic.\n"
        "- explicit_answer_anchor: the target answer appears explicitly or near-explicitly in the context.\n"
        "- misleading_plausibility: 1=not plausible as evidence, 5=plausible but potentially misleading evidence.\n"
    )
    messages = [
        {"role": "system", "content": "You are a strict, concise evidence-quality audit judge."},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user


def parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "supports_target": data.get("supports_target"),
                "contradicts_gold": data.get("contradicts_gold"),
                "naturalness": data.get("naturalness"),
                "template_likeness": data.get("template_likeness"),
                "explicit_answer_anchor": data.get("explicit_answer_anchor"),
                "misleading_plausibility": data.get("misleading_plausibility"),
                "reason": str(data.get("reason", ""))[:500],
                "parse_ok": True,
            }
        except Exception:
            pass
    return {
        "supports_target": None,
        "contradicts_gold": None,
        "naturalness": None,
        "template_likeness": None,
        "explicit_answer_anchor": None,
        "misleading_plausibility": None,
        "reason": text[:500],
        "parse_ok": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blind stimulus-quality audit over sampled sup/mis contexts.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.sample)
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
                audit_prompt(bundle.tokenizer, row),
                max_new_tokens=192,
                temperature=0.0,
                do_sample=False,
            )
            parsed = parse_json(raw)
            out = {
                **row,
                **parsed,
                "audit_raw": raw,
                "audit_model": args.model,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
