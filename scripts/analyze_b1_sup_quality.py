#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import load_predictions


def read_jsonl_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    return pd.DataFrame(rows)


def _first(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else None


def per_item_predictions(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    return (
        df.groupby(["model_name", "question_id", "condition"], as_index=False)
        .agg(
            ybar=("correct", "mean"),
            cbar=(conf_col, "mean"),
            answers=("model_answer", lambda s: " || ".join(sorted({str(v) for v in s if str(v).strip()}))[:500]),
            question=("question", _first),
            gold_answer=("gold_answer", _first),
            wrong_answer=("wrong_answer", _first),
        )
        .dropna(subset=["cbar"])
    )


def write_markdown(out_path: Path, summary: dict[str, Any], anomaly_examples: pd.DataFrame) -> None:
    lines = [
        "# B1 Sup Reader Quality Check",
        "",
        f"Primary confidence: `{summary['confidence']}`",
        "",
        "## Sup Reader",
        "",
        f"- n: `{summary.get('sup_reader_n')}`",
        f"- support gold rate: `{summary.get('sup_reader_support_gold_rate'):.4f}`"
        if summary.get("sup_reader_support_gold_rate") is not None
        else "- support gold rate: `NA`",
        f"- support wrong rate: `{summary.get('sup_reader_support_wrong_rate'):.4f}`"
        if summary.get("sup_reader_support_wrong_rate") is not None
        else "- support wrong rate: `NA`",
        "",
        "## Sup-On-Known",
        "",
        f"- closed_known questions: `{summary['closed_known_questions']}`",
        f"- sup accuracy on closed_known: `{summary['sup_acc_on_closed_known']:.4f}`",
        f"- closed accuracy on closed_known: `{summary['closed_acc_on_closed_known']:.4f}`",
        f"- sup worse than closed rate: `{summary['sup_worse_than_closed_rate']:.4f}`",
        f"- sup mostly wrong rate: `{summary['sup_mostly_wrong_rate']:.4f}`",
        f"- mean P(True) for sup mostly wrong: `{summary['sup_mostly_wrong_mean_conf']:.4f}`"
        if summary.get("sup_mostly_wrong_mean_conf") is not None
        else "- mean P(True) for sup mostly wrong: `NA`",
        "",
        "## Sup-On-Known By Reader Result",
        "",
    ]
    reader_breakdown = summary.get("sup_on_known_by_reader", [])
    if reader_breakdown:
        lines.append("| sup_reader_supports_gold | n | sup_acc | sup_conf | sup_mostly_wrong_rate |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in reader_breakdown:
            lines.append(
                f"| {row['sup_reader_supports_gold']} | {row['n_questions']} | "
                f"{row['sup_acc']:.4f} | {row['sup_conf']:.4f} | {row['sup_mostly_wrong_rate']:.4f} |"
            )
    else:
        lines.append("Reader output not available yet.")

    lines.extend(["", "## Anomaly Examples", ""])
    for i, row in enumerate(anomaly_examples.to_dict("records"), 1):
        lines.extend(
            [
                f"### Example {i}",
                "",
                f"Question: {row.get('question', '')}",
                "",
                f"Gold: {row.get('gold_answer', '')}",
                "",
                f"Wrong: {row.get('wrong_answer', '')}",
                "",
                f"Closed ybar/conf: {row.get('closed_ybar'):.4f} / {row.get('closed_cbar'):.4f}",
                "",
                f"Sup ybar/conf: {row.get('sup_ybar'):.4f} / {row.get('sup_cbar'):.4f}",
                "",
                f"Sup answers: {row.get('sup_answers', '')}",
                "",
                f"Sup reader answer: {row.get('reader_answer', '')}",
                "",
                f"Sup reader supports gold: {row.get('sup_reader_supports_gold', 'NA')}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze B1 sup reader quality and sup-on-known anomalies.")
    parser.add_argument("--predictions", default="runs/phase2_unique_with_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--sup-reader", default="runs/phase2_unique/sup_reader_support.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_unique/analysis_b1_sup_quality")
    parser.add_argument("--confidence", default="conf_ptrue")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(args.predictions)
    item = per_item_predictions(pred, args.confidence)
    closed = item[item["condition"] == "closed"][
        ["model_name", "question_id", "ybar", "cbar", "answers"]
    ].rename(columns={"ybar": "closed_ybar", "cbar": "closed_cbar", "answers": "closed_answers"})
    sup = item[item["condition"] == "sup"][
        [
            "model_name",
            "question_id",
            "ybar",
            "cbar",
            "answers",
            "question",
            "gold_answer",
            "wrong_answer",
        ]
    ].rename(columns={"ybar": "sup_ybar", "cbar": "sup_cbar", "answers": "sup_answers"})
    joined = sup.merge(closed, on=["model_name", "question_id"], how="inner")
    joined["closed_known"] = joined["closed_ybar"] >= 0.6
    joined["sup_worse_than_closed"] = joined["sup_ybar"] < joined["closed_ybar"]
    joined["sup_mostly_wrong"] = joined["sup_ybar"] < 0.5

    reader = read_jsonl_df(args.sup_reader)
    reader_summary: dict[str, Any] = {
        "sup_reader_n": None,
        "sup_reader_support_gold_rate": None,
        "sup_reader_support_wrong_rate": None,
    }
    if not reader.empty:
        reader_cols = [
            "question_id",
            "reader_answer",
            "sup_reader_supports_gold",
            "sup_reader_supports_wrong",
        ]
        joined = joined.merge(reader[reader_cols], on="question_id", how="left")
        reader_summary = {
            "sup_reader_n": int(len(reader)),
            "sup_reader_support_gold_rate": float(reader["sup_reader_supports_gold"].mean()),
            "sup_reader_support_wrong_rate": float(reader["sup_reader_supports_wrong"].mean()),
        }
    else:
        joined["reader_answer"] = ""
        joined["sup_reader_supports_gold"] = pd.NA
        joined["sup_reader_supports_wrong"] = pd.NA

    known = joined[joined["closed_known"]].copy()
    summary: dict[str, Any] = {
        "confidence": args.confidence,
        "predictions": args.predictions,
        "sup_reader": args.sup_reader,
        "closed_known_questions": int(known["question_id"].nunique()),
        "closed_acc_on_closed_known": float(known["closed_ybar"].mean()),
        "sup_acc_on_closed_known": float(known["sup_ybar"].mean()),
        "sup_conf_on_closed_known": float(known["sup_cbar"].mean()),
        "sup_worse_than_closed_rate": float(known["sup_worse_than_closed"].mean()),
        "sup_mostly_wrong_rate": float(known["sup_mostly_wrong"].mean()),
        "sup_mostly_wrong_mean_conf": float(known.loc[known["sup_mostly_wrong"], "sup_cbar"].mean())
        if known["sup_mostly_wrong"].any()
        else None,
        **reader_summary,
    }

    if not reader.empty:
        breakdown = (
            known.dropna(subset=["sup_reader_supports_gold"])
            .groupby("sup_reader_supports_gold", as_index=False)
            .agg(
                n_questions=("question_id", "nunique"),
                sup_acc=("sup_ybar", "mean"),
                sup_conf=("sup_cbar", "mean"),
                sup_mostly_wrong_rate=("sup_mostly_wrong", "mean"),
            )
        )
        summary["sup_on_known_by_reader"] = breakdown.to_dict("records")
        breakdown.to_csv(out_dir / "sup_on_known_by_reader.csv", index=False)
    else:
        summary["sup_on_known_by_reader"] = []

    anomalies = known.sort_values(["sup_ybar", "sup_cbar"], ascending=[True, False])
    anomaly_cols = [
        "question_id",
        "question",
        "gold_answer",
        "wrong_answer",
        "closed_ybar",
        "closed_cbar",
        "closed_answers",
        "sup_ybar",
        "sup_cbar",
        "sup_answers",
        "reader_answer",
        "sup_reader_supports_gold",
        "sup_reader_supports_wrong",
    ]
    anomalies[anomaly_cols].to_csv(out_dir / "sup_on_known_anomalies.csv", index=False)
    joined.to_csv(out_dir / "sup_quality_joined_items.csv", index=False)
    (out_dir / "b1_sup_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(out_dir / "b1_sup_quality_report.md", summary, anomalies.head(12)[anomaly_cols])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
