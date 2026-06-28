#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PRIMARY = "conf_ptrue"
CONDITION_ORDER = ["closed", "irr", "same_entity_irr", "sup", "mis"]
H3_METHODS = ["identity", "platt_logistic", "isotonic_relabel"]


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_model(entry: str) -> dict[str, Any]:
    try:
        label, b2_dir, h3_dir, pred_path = entry.split("=", 3)
    except ValueError as exc:
        raise SystemExit(
            "--model expects label=b2_dir=h3_dir=predictions.jsonl, "
            f"got: {entry}"
        ) from exc
    b2 = Path(b2_dir)
    h3 = Path(h3_dir)
    pred = Path(pred_path)
    return {
        "label": label,
        "b2_dir": b2,
        "h3_dir": h3,
        "pred_path": pred,
        "n_rows": count_jsonl(pred),
        "condition": pd.read_csv(b2 / "condition_summary_with_same_entity.csv"),
        "rho": pd.read_csv(b2 / "staged_rho.csv"),
        "h3": pd.read_csv(h3 / "h3_calibration_results.csv"),
    }


def write_report(models: list[dict[str, Any]], out: Path, note: str | None) -> None:
    lines: list[str] = [
        "# Cross-Model Robustness Summary",
        "",
        f"Primary confidence: `{PRIMARY}`",
        "",
    ]
    if note:
        lines.extend([note, ""])
    lines.extend(["## Prediction Files", "", "| model | rows | path |", "|---|---:|---|"])
    for model in models:
        lines.append(f"| {model['label']} | {model['n_rows']} | `{model['pred_path']}` |")

    lines.extend(
        [
            "",
            "## Condition Summary",
            "",
            "| model | condition | acc | P(True) | delta_acc | delta_conf | FG | n_questions |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        table = model["condition"]
        table = table[table["confidence"] == PRIMARY].copy()
        table["condition"] = pd.Categorical(table["condition"], CONDITION_ORDER, ordered=True)
        for row in table.sort_values("condition").to_dict("records"):
            lines.append(
                f"| {model['label']} | {row['condition']} | {fmt(row.get('acc'))} | "
                f"{fmt(row.get('conf'))} | {fmt(row.get('delta_acc'))} | "
                f"{fmt(row.get('delta_conf'))} | {fmt(row.get('FG'))} | "
                f"{int(row.get('n_questions', 0))} |"
            )

    lines.extend(
        [
            "",
            "## Staged Rho",
            "",
            "| model | rho_topicality | rho_answer_support | rho_veracity | rho_P | rho_PE | rho_eps |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        rho = model["rho"]
        row = rho[rho["confidence"] == PRIMARY].iloc[0].to_dict()
        lines.append(
            f"| {model['label']} | {fmt(row.get('rho_topicality'))} | "
            f"{fmt(row.get('rho_answer_support'))} | {fmt(row.get('rho_veracity'))} | "
            f"{fmt(row.get('rho_P'))} | {fmt(row.get('rho_PE'))} | {fmt(row.get('rho_eps'))} |"
        )

    lines.extend(
        [
            "",
            "## H3 Calibration",
            "",
            "| model | method | ECE | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        h3 = model["h3"]
        h3 = h3[(h3["source_confidence"] == PRIMARY) & (h3["method"].isin(H3_METHODS))]
        order = {name: idx for idx, name in enumerate(H3_METHODS)}
        h3 = h3.sort_values("method", key=lambda s: s.map(order))
        for row in h3.to_dict("records"):
            lines.append(
                f"| {model['label']} | {row['method']} | {fmt(row.get('ece'))} | "
                f"{fmt(row.get('FG_mis'))} | {fmt(row.get('rho_topicality'))} | "
                f"{fmt(row.get('rho_answer_support'))} | {fmt(row.get('rho_veracity'))} |"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cross-model robustness from existing analysis outputs.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="label=b2_dir=h3_dir=predictions.jsonl",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    models = [load_model(entry) for entry in args.model]
    write_report(models, Path(args.out), args.note)
    print(json.dumps({"out": args.out, "models": [m["label"] for m in models]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
