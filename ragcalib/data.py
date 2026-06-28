from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .spec_config import SPEC
from .text_utils import answer_in_text, entity_count, normalize_text, parse_answer_list, safe_ratio, token_len


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_popqa() -> pd.DataFrame:
    ds = load_dataset(SPEC.dataset_popqa, split="test")
    df = ds.to_pandas()
    df["gold_answers"] = df["possible_answers"].apply(parse_answer_list)
    df["popularity_raw"] = df["s_pop"].astype(float)
    return df


def load_conflictqa_popqa(conflict_file: str) -> pd.DataFrame:
    path = hf_hub_download(SPEC.dataset_conflictqa, conflict_file, repo_type="dataset")
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    df["gold_answers"] = df["ground_truth"].apply(parse_answer_list)
    df["popularity_raw"] = df["popularity"].astype(float)
    return df


def add_popularity_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = list(SPEC.popularity_bins)
    out["popularity_bin"] = pd.qcut(
        out["popularity_raw"].rank(method="first"),
        q=3,
        labels=labels,
    ).astype(str)
    return out


def _candidate_has_gold(answer: str, evidence: str, gold_answers: list[str]) -> bool:
    return answer_in_text(answer, gold_answers) or answer_in_text(evidence, gold_answers)


def select_evidence(row: pd.Series, evidence_policy: str) -> dict[str, Any] | None:
    gold_answers = row["gold_answers"]
    candidates = [
        {
            "role": "parametric",
            "answer": row.get("memory_answer", ""),
            "evidence": row.get("parametric_memory_aligned_evidence", ""),
        },
        {
            "role": "counter",
            "answer": row.get("counter_answer", ""),
            "evidence": row.get("counter_memory_aligned_evidence", ""),
        },
    ]
    for cand in candidates:
        cand["has_gold"] = _candidate_has_gold(cand["answer"], cand["evidence"], gold_answers)

    if evidence_policy == "spec_fields":
        sup = candidates[0]
        mis = candidates[1]
    elif evidence_policy == "auto":
        sup_options = [c for c in candidates if c["has_gold"]]
        mis_options = [c for c in candidates if not c["has_gold"]]
        if not sup_options or not mis_options:
            return None
        sup = sup_options[0]
        mis = mis_options[0]
    else:
        raise ValueError(f"Unknown evidence_policy: {evidence_policy}")

    if not str(sup["evidence"]).strip() or not str(mis["evidence"]).strip():
        return None

    return {
        "sup_context": sup["evidence"],
        "sup_source_role": sup["role"],
        "mis_context": mis["evidence"],
        "mis_source_role": mis["role"],
        "wrong_answer": mis["answer"],
        "field_gold_flags": {c["role"]: bool(c["has_gold"]) for c in candidates},
    }


def build_workset(
    n_total: int,
    conflict_file: str,
    evidence_policy: str = "auto",
    seed: int = SPEC.random_seed,
    unique_by: str = "question",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    popqa = add_popularity_bins(load_popqa())
    conflict = add_popularity_bins(load_conflictqa_popqa(conflict_file))

    # ConflictQA is PopQA-derived; question text is the most stable join key.
    pop_cols = ["question", "id", "subj", "prop", "obj", "possible_answers", "s_pop", "o_pop"]
    merged = conflict.merge(popqa[pop_cols], on="question", how="left", suffixes=("", "_popqa"))
    merged["question_id"] = merged["id"].fillna(pd.Series(range(len(merged)))).astype(str)

    selected_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rng = random.Random(seed)
    seen_unique_keys: set[str] = set()

    per_bin = n_total // len(SPEC.popularity_bins)
    remainder = n_total % len(SPEC.popularity_bins)
    targets = {
        bin_name: per_bin + (1 if i < remainder else 0)
        for i, bin_name in enumerate(SPEC.popularity_bins)
    }

    for bin_name in SPEC.popularity_bins:
        bin_df = merged[merged["popularity_bin"] == bin_name].sample(
            frac=1.0,
            random_state=seed,
        )
        kept = 0
        for _, row in bin_df.iterrows():
            evidence = select_evidence(row, evidence_policy=evidence_policy)
            if evidence is None:
                rejection_counts[f"{bin_name}:missing_valid_sup_or_mis"] += 1
                continue
            qid = str(row["question_id"])
            gold_answers = parse_answer_list(row["gold_answers"])
            if unique_by == "question":
                unique_key = normalize_text(row["question"])
            elif unique_by == "question_id":
                unique_key = qid
            elif unique_by == "question_and_gold":
                unique_key = normalize_text(row["question"]) + "::" + normalize_text(gold_answers[0] if gold_answers else "")
            elif unique_by == "none":
                unique_key = f"{qid}::{len(selected_rows)}"
            else:
                raise ValueError(f"Unknown unique_by: {unique_by}")
            if unique_key in seen_unique_keys:
                rejection_counts[f"{bin_name}:duplicate_{unique_by}"] += 1
                continue
            seen_unique_keys.add(unique_key)
            source_counts[f"sup:{evidence['sup_source_role']}"] += 1
            source_counts[f"mis:{evidence['mis_source_role']}"] += 1
            selected_rows.append(
                {
                    "question_id": qid,
                    "question": row["question"],
                    "gold_answer": gold_answers[0] if gold_answers else "",
                    "gold_answers": gold_answers,
                    "wrong_answer": evidence["wrong_answer"],
                    "popularity_bin": bin_name,
                    "popularity_raw": float(row["popularity_raw"]),
                    "sup_context": evidence["sup_context"],
                    "mis_context": evidence["mis_context"],
                    "sup_source_role": evidence["sup_source_role"],
                    "mis_source_role": evidence["mis_source_role"],
                    "field_gold_flags": evidence["field_gold_flags"],
                }
            )
            kept += 1
            if kept >= targets[bin_name]:
                break
        if kept < targets[bin_name]:
            rejection_counts[f"{bin_name}:shortfall"] = targets[bin_name] - kept

    rng.shuffle(selected_rows)
    diagnostics = {
        "n_requested": n_total,
        "n_selected": len(selected_rows),
        "targets": targets,
        "bin_counts": dict(Counter(r["popularity_bin"] for r in selected_rows)),
        "rejection_counts": dict(rejection_counts),
        "evidence_policy": evidence_policy,
        "conflict_file": conflict_file,
        "unique_by": unique_by,
        "unique_question_count": len({normalize_text(r["question"]) for r in selected_rows}),
        "unique_question_id_count": len({r["question_id"] for r in selected_rows}),
        "evidence_source_counts": dict(source_counts),
    }
    return selected_rows, diagnostics


def make_context_rows(workset: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    surface: list[dict[str, Any]] = []
    for item in workset:
        sup_len = token_len(item["sup_context"])
        mis_len = token_len(item["mis_context"])
        sup_ent = entity_count(item["sup_context"])
        mis_ent = entity_count(item["mis_context"])
        surface.append(
            {
                "question_id": item["question_id"],
                "len_ratio_mis_over_sup": safe_ratio(mis_len, sup_len),
                "entity_diff_mis_minus_sup": mis_ent - sup_ent,
                "sup_len_tokens": sup_len,
                "mis_len_tokens": mis_len,
                "sup_entity_count": sup_ent,
                "mis_entity_count": mis_ent,
            }
        )
        for condition in SPEC.phase2_conditions:
            context = ""
            wrong_answer = ""
            if condition == "sup":
                context = item["sup_context"]
            elif condition == "mis":
                context = item["mis_context"]
                wrong_answer = item["wrong_answer"]
            rows.append(
                {
                    "question_id": item["question_id"],
                    "popularity_bin": item["popularity_bin"],
                    "popularity_raw": item["popularity_raw"],
                    "condition": condition,
                    "context": context,
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "gold_answers": item["gold_answers"],
                    "wrong_answer": wrong_answer,
                    "ctx_len_tokens": token_len(context),
                    "ctx_entity_count": entity_count(context),
                    "sup_source_role": item["sup_source_role"],
                    "mis_source_role": item["mis_source_role"],
                }
            )

    surface_df = pd.DataFrame(surface)
    diagnostics = {
        "surface_matching": {
            "len_ratio_mis_over_sup_mean": float(surface_df["len_ratio_mis_over_sup"].mean())
            if not surface_df.empty
            else None,
            "len_ratio_mis_over_sup_p10": float(surface_df["len_ratio_mis_over_sup"].quantile(0.10))
            if not surface_df.empty
            else None,
            "len_ratio_mis_over_sup_p90": float(surface_df["len_ratio_mis_over_sup"].quantile(0.90))
            if not surface_df.empty
            else None,
            "entity_diff_mis_minus_sup_mean": float(surface_df["entity_diff_mis_minus_sup"].mean())
            if not surface_df.empty
            else None,
            "entity_diff_mis_minus_sup_p10": float(surface_df["entity_diff_mis_minus_sup"].quantile(0.10))
            if not surface_df.empty
            else None,
            "entity_diff_mis_minus_sup_p90": float(surface_df["entity_diff_mis_minus_sup"].quantile(0.90))
            if not surface_df.empty
            else None,
        }
    }
    return rows, diagnostics


def write_mis_examples(out_path: str | Path, workset: list[dict[str, Any]], n: int = 5) -> None:
    lines = ["# Misleading Evidence Examples", ""]
    for i, item in enumerate(workset[:n], 1):
        lines.extend(
            [
                f"## Example {i}",
                "",
                f"Question: {item['question']}",
                "",
                f"Gold answer: {item['gold_answer']}",
                "",
                f"Wrong answer candidate: {item['wrong_answer']}",
                "",
                "Mis context:",
                "",
                item["mis_context"],
                "",
                "Sup context:",
                "",
                item["sup_context"],
                "",
            ]
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
