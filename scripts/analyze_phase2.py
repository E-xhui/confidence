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

from ragcalib.metrics import (
    available_confidence_columns,
    closed_book_distribution,
    condition_summary,
    faithfulness_beta,
    load_predictions,
    stratified_ece_auroc,
    variance_decomposition_reml,
)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(out_path: Path, metrics: dict, examples: pd.DataFrame) -> None:
    lines = ["# Phase 2 Gate Report", ""]
    for conf_col, block in metrics["by_confidence"].items():
        lines.extend([f"## Confidence: `{conf_col}`", ""])
        lines.append("### Minimal result table")
        lines.append("")
        lines.append("| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in block["condition_summary"]:
            lines.append(
                "| {model_name} | {condition} | {acc} | {conf} | {delta_acc} | {delta_conf} | {FG} | {n_questions} |".format(
                    model_name=row["model_name"],
                    condition=row["condition"],
                    acc=_fmt(row["acc"]),
                    conf=_fmt(row["conf"]),
                    delta_acc=_fmt(row["delta_acc"]),
                    delta_conf=_fmt(row["delta_conf"]),
                    FG=_fmt(row["FG"]),
                    n_questions=row["n_questions"],
                )
            )
        lines.extend(["", "### Gate quantities", ""])
        lines.append(json.dumps(block["gate"], ensure_ascii=False, indent=2))
        lines.extend(["", "### Variance decomposition", ""])
        lines.append(json.dumps(block["variance_decomposition"], ensure_ascii=False, indent=2))
        lines.extend([""])

    lines.extend(["## Diagnostics", ""])
    lines.append("Closed-book correct-rate distribution:")
    lines.append("")
    lines.append(json.dumps(metrics["closed_book_correct_rate"], ensure_ascii=False, indent=2))
    lines.extend(["", "## Misleading examples", ""])
    for i, row in enumerate(examples.to_dict("records"), 1):
        lines.extend(
            [
                f"### Example {i}",
                "",
                f"Question: {row.get('question', '')}",
                "",
                f"Gold answer: {row.get('gold_answer', '')}",
                "",
                f"Wrong answer: {row.get('wrong_answer', '')}",
                "",
                "Mis context:",
                "",
                str(row.get("context", "")),
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 2 predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    conf_cols = available_confidence_columns(df)
    if not conf_cols:
        raise ValueError("No usable confidence columns found.")

    metrics = {
        "prediction_path": str(args.predictions),
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "models": sorted(df["model_name"].dropna().unique().tolist()),
        "confidence_columns": conf_cols,
        "closed_book_correct_rate": closed_book_distribution(df),
        "by_confidence": {},
    }

    for conf_col in conf_cols:
        summary = condition_summary(df, conf_col)
        beta = faithfulness_beta(df, conf_col)
        vd = variance_decomposition_reml(df, conf_col)
        gate = {
            "FG_mis": None,
            "delta_acc_mis": None,
            "delta_conf_mis": None,
            "faithfulness_beta": beta,
        }
        mis_rows = summary[summary["condition"] == "mis"]
        if not mis_rows.empty:
            gate["FG_mis"] = float(mis_rows["FG"].mean())
            gate["delta_acc_mis"] = float(mis_rows["delta_acc"].mean())
            gate["delta_conf_mis"] = float(mis_rows["delta_conf"].mean())
        ece = stratified_ece_auroc(df, conf_col)
        summary.to_csv(out_dir / f"condition_summary_{conf_col}.csv", index=False)
        ece.to_csv(out_dir / f"ece_by_condition_popularity_{conf_col}.csv", index=False)
        metrics["by_confidence"][conf_col] = {
            "condition_summary": summary.to_dict("records"),
            "gate": gate,
            "variance_decomposition": vd,
        }

    examples = (
        df[df["condition"] == "mis"]
        .drop_duplicates("question_id")
        .head(5)[["question", "gold_answer", "wrong_answer", "context"]]
    )
    (out_dir / "phase2_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir / "phase2_gate_report.md", metrics, examples)
    print(json.dumps({"out_dir": str(out_dir), "confidence_columns": conf_cols}, indent=2))


if __name__ == "__main__":
    main()
