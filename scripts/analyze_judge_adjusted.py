#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import condition_summary, load_predictions
from scripts.analyze_b2_same_entity_irr import support_stage_decomposition


def read_jsonl_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in p.open(encoding="utf-8") if line.strip()])


def write_report(path: Path, summary: pd.DataFrame, rho: dict, judge_summary: dict) -> None:
    lines = [
        "# Judge-Adjusted EM Boundary Report",
        "",
        "Judge corrections are applied only to selected EM-boundary candidates.",
        "",
        "## Judge Summary",
        "",
    ]
    for key, value in judge_summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Adjusted P(True) Condition Summary", ""])
    lines.append("| condition | acc | conf | delta_acc | delta_conf | FG | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    prim = summary[summary["confidence"] == "conf_ptrue"]
    order = ["closed", "irr", "same_entity_irr", "sup", "mis"]
    prim = prim.assign(condition=pd.Categorical(prim["condition"], order, ordered=True)).sort_values("condition")
    for row in prim.to_dict("records"):
        lines.append(
            f"| {row['condition']} | {row['acc']:.4f} | {row['conf']:.4f} | {row['delta_acc']:.4f} | "
            f"{row['delta_conf']:.4f} | {row['FG']:.4f} | {int(row['n_questions'])} |"
        )
    lines.extend(["", "## Adjusted Staged Rho", ""])
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in ["rho_topicality", "rho_answer_support", "rho_veracity", "rho_P", "rho_PE", "rho_eps"]:
        lines.append(f"| {key} | {rho.get(key):.4f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LLM-judge corrections to selected EM-boundary candidates.")
    parser.add_argument("--predictions", default="runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--judge", default="runs/phase2_with_same_entity_irr/judge_outputs.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_with_same_entity_irr/analysis_judge_adjusted")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    df = df.reset_index().rename(columns={"index": "row_index"})
    df["question_id"] = df["question_id"].astype(str)
    judges = read_jsonl_df(args.judge)
    if judges.empty:
        raise SystemExit("No judge rows found.")
    key_cols = ["question_id", "condition", "sample_idx"]
    judges["question_id"] = judges["question_id"].astype(str)
    judges = judges.dropna(subset=["judge_correct"]).copy()
    judges["judge_correct"] = judges["judge_correct"].astype(int)
    keep = key_cols + ["judge_correct", "judge_reason", "selection_reason"]
    merged = df.merge(judges[keep], on=key_cols, how="left")
    merged["correct_original"] = merged["correct"].astype(int)
    judge_correct_filled = merged["judge_correct"].fillna(0).astype(int)
    mask = merged["judge_correct"].notna() & (judge_correct_filled == 1)
    merged["correct"] = merged["correct_original"]
    merged.loc[mask, "correct"] = 1
    adjusted_path = out_dir / "predictions_judge_adjusted.jsonl"
    with adjusted_path.open("w", encoding="utf-8") as f:
        for row in merged.drop(columns=["row_index"]).to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = []
    for conf in ["conf_ptrue", "conf_verbalized", "conf_seqlik"]:
        if conf in merged.columns:
            s = condition_summary(merged, conf)
            s["confidence"] = conf
            summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "condition_summary_judge_adjusted.csv", index=False)
    rho = support_stage_decomposition(merged, "conf_ptrue")
    (out_dir / "staged_rho_judge_adjusted.json").write_text(json.dumps(rho, ensure_ascii=False, indent=2), encoding="utf-8")
    judge_summary = {
        "judge_rows": int(len(judges)),
        "judge_positive_rate": float(judges["judge_correct"].mean()),
        "corrected_sample_count": int(mask.sum()),
        "original_correct_mean": float(merged["correct_original"].mean()),
        "adjusted_correct_mean": float(merged["correct"].mean()),
    }
    (out_dir / "judge_adjustment_summary.json").write_text(json.dumps(judge_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "judge_adjusted_report.md", summary, rho, judge_summary)
    print(json.dumps({"out_dir": str(out_dir), **judge_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
