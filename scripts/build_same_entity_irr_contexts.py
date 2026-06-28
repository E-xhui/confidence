#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.text_utils import answer_in_text, entity_count, normalize_text, token_len


INITIAL_DOT = "<DOT>"
ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Jr.", "Sr.", "U.S.", "U.K.")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def target_entity(question: str) -> str:
    q = question.strip().rstrip("?")
    patterns = [
        r"^Who was the (?:composer|producer|director|screenwriter) (?:of|for) (.+)$",
        r"^Who is the (?:author|father|mother) of (.+)$",
        r"^In what (?:city|country) (?:was|is) (.+?)(?: born)?$",
        r"^What is (.+?)'s occupation$",
        r"^What is the capital of (.+)$",
        r"^What is (.+?) the capital of$",
        r"^What genre is (.+)$",
        r"^What sport does (.+) play$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            return match.group(1).strip(" \"'")
    return q


def protect_sentence_periods(text: str) -> str:
    out = text
    for abbr in ABBREVIATIONS:
        out = out.replace(abbr, abbr.replace(".", INITIAL_DOT))
    out = re.sub(r"\b([A-Z])\.", rf"\1{INITIAL_DOT}", out)
    return out


def split_sentences(text: str) -> list[str]:
    protected = protect_sentence_periods(re.sub(r"\s+", " ", text or "").strip())
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [part.replace(INITIAL_DOT, ".").strip() for part in parts if part.strip()]


def wrong_answer_terms(wrong_answer: str) -> list[str]:
    wrong = str(wrong_answer or "").strip()
    if not wrong:
        return []
    terms = [wrong]
    patterns = [
        r"\bis\s+(.+?)[.]?$",
        r"\bwas\s+(.+?)[.]?$",
        r"\bwas born in\s+(.+?)[.]?$",
        r"\bwas produced by\s+(.+?)[.]?$",
        r"\bwas directed by\s+(.+?)[.]?$",
        r"\bwas written by\s+(.+?)[.]?$",
        r"\bby\s+(.+?)[.]?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, wrong, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .\"'")
            if candidate and len(candidate.split()) <= 8:
                terms.append(candidate)
    return terms


def answer_terms(item: dict, entity: str) -> list[str]:
    terms: list[str] = []
    for value in item.get("gold_answers", []):
        if str(value).strip():
            terms.append(str(value).strip())
    for key in ("gold_answer",):
        if str(item.get(key, "")).strip():
            terms.append(str(item[key]).strip())
    terms.extend(wrong_answer_terms(str(item.get("wrong_answer", ""))))
    expanded = list(terms)
    for term in terms:
        tokens = [tok.strip(" ,.") for tok in str(term).split() if tok.strip(" ,.")]
        person_like = 2 <= len(tokens) <= 4 and sum(int(tok[:1].isupper()) for tok in tokens) >= 2
        if person_like:
            last = tokens[-1]
            if len(normalize_text(last)) > 3:
                expanded.append(last)
    terms = expanded
    deduped = []
    seen = set()
    entity_norm = normalize_text(entity)
    for term in terms:
        norm = normalize_text(term)
        if entity_norm and (norm in entity_norm or entity_norm in norm):
            continue
        if norm and norm not in seen:
            deduped.append(term)
            seen.add(norm)
    return deduped


def relation_cues(question: str) -> list[str]:
    q = normalize_text(question)
    cue_sets = [
        (("composer",), ["composer", "composed by", "music composed by", "score by", "soundtrack by"]),
        (("producer",), ["producer", "produced by"]),
        (("director",), ["director", "directed by"]),
        (("screenwriter",), ["screenwriter", "written by", "writer", "script", "screenplay"]),
        (("author",), ["author", "written by", "writer", "novel by"]),
        (("father",), ["father", "son of", "daughter of"]),
        (("mother",), ["mother", "son of", "daughter of"]),
        (("born", "city"), ["born", "birthplace", "place of birth"]),
        (("country",), ["country", "located in", "is in"]),
        (("occupation",), ["occupation", "is a", "was a", "profession"]),
        (("genre",), ["genre", "is a", "was a"]),
        (("capital",), ["capital of", "capital"]),
        (("sport",), ["sport", "plays", "player"]),
    ]
    cues: list[str] = []
    for triggers, values in cue_sets:
        if any(trigger in q for trigger in triggers):
            cues.extend(values)
    return cues


def has_relation_cue(sentence: str, cues: list[str]) -> bool:
    sent = normalize_text(sentence)
    return any(normalize_text(cue) in sent for cue in cues)


def source_chunks(item: dict) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for source, key in (("sup", "sup_context"), ("mis", "mis_context")):
        chunks.extend((source, sent) for sent in split_sentences(str(item.get(key, ""))))
    return chunks


def build_context(item: dict, max_tokens: int) -> tuple[str, dict]:
    entity = target_entity(item["question"])
    terms = answer_terms(item, entity)
    cues = relation_cues(item["question"])
    kept: list[tuple[str, str]] = []
    rejected_answer = 0
    rejected_relation = 0
    for source, sent in source_chunks(item):
        if answer_in_text(sent, terms):
            rejected_answer += 1
            continue
        if has_relation_cue(sent, cues):
            rejected_relation += 1
            continue
        kept.append((source, sent))

    selected: list[tuple[str, str]] = []
    total = 0
    for source, sent in kept:
        sent_len = token_len(sent)
        if selected and total + sent_len > max_tokens:
            break
        selected.append((source, sent))
        total += sent_len
        if total >= 40:
            break

    fallback = False
    if selected:
        body = " ".join(sent for _, sent in selected)
        if normalize_text(entity) not in normalize_text(body):
            context = f"Background about {entity}: {body}"
        else:
            context = body
    else:
        fallback = True
        context = (
            f"Background about {entity}: This passage concerns {entity}, but it does not state "
            "the requested relation or provide the answer to the question."
        )

    # Final guard: strip any selected sentence that creates a cross-boundary answer leak.
    if answer_in_text(context, terms) and selected:
        safe_selected: list[tuple[str, str]] = []
        for source, sent in selected:
            trial_body = " ".join(s for _, s in safe_selected + [(source, sent)])
            trial_context = (
                trial_body
                if normalize_text(entity) in normalize_text(trial_body)
                else f"Background about {entity}: {trial_body}"
            )
            if not answer_in_text(trial_context, terms):
                safe_selected.append((source, sent))
        if safe_selected:
            selected = safe_selected
            body = " ".join(sent for _, sent in selected)
            context = body if normalize_text(entity) in normalize_text(body) else f"Background about {entity}: {body}"
        else:
            fallback = True
            selected = []
            context = (
                f"Background about {entity}: This passage concerns {entity}, but it does not state "
                "the requested relation or provide the answer to the question."
            )

    diag = {
        "target_entity": entity,
        "selected_sources": sorted({source for source, _ in selected}) if selected else ["fallback"],
        "selected_sentence_count": len(selected),
        "rejected_answer_sentence_count": rejected_answer,
        "rejected_relation_sentence_count": rejected_relation,
        "fallback_used": fallback,
        "same_entity_answer_leak": answer_in_text(context, terms),
        "same_entity_len_tokens": token_len(context),
        "same_entity_entity_count": entity_count(context),
        "subject_string_present": normalize_text(entity) in normalize_text(context),
    }
    return context, diag


def main() -> None:
    parser = argparse.ArgumentParser(description="Build same-entity irrelevant contexts from answer-stripped local evidence.")
    parser.add_argument("--workset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--diagnostics-out", required=True)
    parser.add_argument("--examples-out", required=True)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    workset = read_jsonl(args.workset)
    rows = []
    per_item = []
    for item in workset:
        context, diag = build_context(item, max_tokens=args.max_tokens)
        rows.append(
            {
                "question_id": item["question_id"],
                "popularity_bin": item["popularity_bin"],
                "popularity_raw": item["popularity_raw"],
                "condition": "same_entity_irr",
                "context": context,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_answers": item["gold_answers"],
                "wrong_answer": "",
                "ctx_len_tokens": token_len(context),
                "ctx_entity_count": entity_count(context),
                "same_entity_target": diag["target_entity"],
                "same_entity_source": "+".join(diag["selected_sources"]),
            }
        )
        per_item.append({"question_id": item["question_id"], "question": item["question"], **diag})

    write_jsonl(args.out, rows)
    lengths = [d["same_entity_len_tokens"] for d in per_item]
    summary = {
        "n": len(rows),
        "source": "answer_stripped_same_item_sup_mis_evidence",
        "answer_leak_count": sum(int(d["same_entity_answer_leak"]) for d in per_item),
        "answer_leak_rate": sum(int(d["same_entity_answer_leak"]) for d in per_item) / len(per_item) if per_item else None,
        "fallback_count": sum(int(d["fallback_used"]) for d in per_item),
        "fallback_rate": sum(int(d["fallback_used"]) for d in per_item) / len(per_item) if per_item else None,
        "subject_string_present_rate": sum(int(d["subject_string_present"]) for d in per_item) / len(per_item) if per_item else None,
        "len_tokens_mean": mean(lengths) if lengths else None,
        "len_tokens_min": min(lengths) if lengths else None,
        "len_tokens_max": max(lengths) if lengths else None,
        "source_counts": {},
        "per_item": per_item,
    }
    for row in rows:
        summary["source_counts"][row["same_entity_source"]] = summary["source_counts"].get(row["same_entity_source"], 0) + 1
    Path(args.diagnostics_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Same-Entity IRR Examples", ""]
    for idx, row in enumerate(rows[:8], 1):
        lines.extend(
            [
                f"## Example {idx}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold answer: {row['gold_answer']}",
                "",
                f"Target entity: {row['same_entity_target']}",
                "",
                f"Source: {row['same_entity_source']}",
                "",
                "Context:",
                "",
                row["context"],
                "",
            ]
        )
    Path(args.examples_out).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_item"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
