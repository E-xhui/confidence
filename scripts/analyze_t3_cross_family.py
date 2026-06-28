#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


CONF_COLS = ["conf_ptrue", "conf_verbalized"]
STAGED_CONDITIONS = ["irr", "same_entity_irr", "sup", "mis"]
TOPICALITY = {"irr": -0.75, "same_entity_irr": 0.25, "sup": 0.25, "mis": 0.25}
ANSWER_SUPPORT = {"irr": 0.0, "same_entity_irr": -2.0 / 3.0, "sup": 1.0 / 3.0, "mis": 1.0 / 3.0}
VERACITY = {"irr": 0.0, "same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5}


def load_predictions(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No predictions found in {path}")
    return pd.DataFrame(rows)


def condition_summary(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    per_item = (
        df.groupby(["model_name", "question_id", "condition"], as_index=False)
        .agg(
            ybar=("correct", "mean"),
            cbar=(conf_col, "mean"),
            n=("correct", "size"),
            abstain_rate=("is_abstain", "mean"),
        )
        .dropna(subset=["cbar"])
    )
    closed = per_item[per_item["condition"] == "closed"][
        ["model_name", "question_id", "ybar", "cbar"]
    ].rename(columns={"ybar": "ybar_closed", "cbar": "cbar_closed"})
    joined = per_item.merge(closed, on=["model_name", "question_id"], how="left")
    joined["delta_acc"] = joined["ybar"] - joined["ybar_closed"]
    joined["delta_conf"] = joined["cbar"] - joined["cbar_closed"]
    joined["fg"] = joined["delta_conf"] - joined["delta_acc"]
    return (
        joined.groupby(["model_name", "condition"], as_index=False)
        .agg(
            acc=("ybar", "mean"),
            conf=("cbar", "mean"),
            abstain_rate=("abstain_rate", "mean"),
            delta_acc=("delta_acc", "mean"),
            delta_conf=("delta_conf", "mean"),
            FG=("fg", "mean"),
            n_questions=("question_id", "nunique"),
        )
        .sort_values(["model_name", "condition"])
    )


def finite(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return num / den


def ci(vals: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in vals if math.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def cells_from_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question_id"] = df["question_id"].astype(str)
    agg: dict[str, tuple[str, str]] = {
        "correct": ("correct", "mean"),
        "n_samples": ("correct", "size"),
        "popularity_raw": ("popularity_raw", "mean"),
        "popularity_bin": ("popularity_bin", "first"),
    }
    for col in CONF_COLS + ["conf_seqlik"]:
        if col in df.columns:
            agg[col] = (col, "mean")
    return df.groupby(["question_id", "condition"], as_index=False).agg(**agg)


def bootstrap(cells: pd.DataFrame, func: Callable[[pd.DataFrame], dict[str, float]], metrics: list[str], b: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(cells["question_id"].astype(str).unique()))
    grouped = {qid: g for qid, g in cells.groupby("question_id", sort=False)}
    values = {metric: [] for metric in metrics}
    for _ in range(b):
        parts = []
        for draw_idx, qid in enumerate(rng.choice(qids, size=len(qids), replace=True)):
            part = grouped[str(qid)].copy()
            part["question_id"] = part["question_id"].astype(str) + f"__b{draw_idx}"
            parts.append(part)
        res = func(pd.concat(parts, ignore_index=True))
        for metric in metrics:
            values[metric].append(finite(res.get(metric)))
    return {metric: ci(v) for metric, v in values.items()}


def staged_rho(cells: pd.DataFrame, conf_col: str) -> dict[str, float]:
    d = cells[cells["condition"].isin(STAGED_CONDITIONS)].dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(float(x), SPEC.logit_eps))
    d["topicality"] = d["condition"].map(TOPICALITY).astype(float)
    d["answer_support"] = d["condition"].map(ANSWER_SUPPORT).astype(float)
    d["veracity"] = d["condition"].map(VERACITY).astype(float)
    x = np.column_stack([
        np.ones(len(d)),
        d["topicality"].to_numpy(float),
        d["answer_support"].to_numpy(float),
        d["veracity"].to_numpy(float),
    ])
    y = d["z"].to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    sigmas = {}
    for name, coef, coding in [
        ("topicality", beta[1], TOPICALITY),
        ("answer_support", beta[2], ANSWER_SUPPORT),
        ("veracity", beta[3], VERACITY),
    ]:
        vals = np.array([float(coef) * coding[c] for c in STAGED_CONDITIONS], dtype=float)
        sigmas[f"sigma2_{name}"] = float(np.var(vals, ddof=0))
    d["fixed"] = x @ beta
    d["resid_fixed"] = d["z"] - d["fixed"]
    q_mean = d.groupby("question_id")["resid_fixed"].mean()
    sigma_p = float(np.var(q_mean, ddof=1)) if len(q_mean) > 1 else 0.0
    sigma_resid = float(np.var(d["resid_fixed"] - d["question_id"].map(q_mean), ddof=1)) if len(d) > 1 else 0.0
    total = sigma_p + sigma_resid + sum(sigmas.values())
    rho_top = sigmas["sigma2_topicality"] / total if total else float("nan")
    rho_as = sigmas["sigma2_answer_support"] / total if total else float("nan")
    rho_v = sigmas["sigma2_veracity"] / total if total else float("nan")
    return {
        "rho_topicality": rho_top,
        "rho_answer_support": rho_as,
        "rho_veracity": rho_v,
        "rho_v_over_as": ratio(rho_v, rho_as),
        "rho_v_over_top": ratio(rho_v, rho_top),
    }


def paired(cells: pd.DataFrame, conf_col: str) -> dict[str, float]:
    wide = cells[cells["condition"].isin(["sup", "mis"])].pivot_table(
        index="question_id", columns="condition", values=[conf_col, "correct"], aggfunc="mean"
    ).dropna()
    sup_conf = wide[(conf_col, "sup")].to_numpy(float)
    mis_conf = wide[(conf_col, "mis")].to_numpy(float)
    sup_acc = wide[("correct", "sup")].to_numpy(float)
    mis_acc = wide[("correct", "mis")].to_numpy(float)
    delta_conf = mis_conf - sup_conf
    delta_acc = mis_acc - sup_acc
    sup_gt = sup_conf - mis_conf
    directional_auc = float(np.mean(sup_gt > 0) + 0.5 * np.mean(sup_gt == 0))
    return {
        "delta_conf_mis_minus_sup": float(np.mean(delta_conf)),
        "delta_acc_mis_minus_sup": float(np.mean(delta_acc)),
        "gap_mis_minus_sup": float(np.mean(delta_conf - delta_acc)),
        "directional_auc_sup_gt_mis": directional_auc,
        "separable_auc": max(directional_auc, 1.0 - directional_auc),
    }


def summarize_metric(
    cells: pd.DataFrame,
    conf_col: str,
    name: str,
    func: Callable[[pd.DataFrame, str], dict[str, float]],
    metrics: list[str],
    b: int,
    seed: int,
) -> pd.DataFrame:
    point = func(cells, conf_col)
    boot = bootstrap(cells, lambda x: func(x, conf_col), metrics, b=b, seed=seed)
    rows = []
    for metric in metrics:
        lo, hi = boot[metric]
        rows.append(
            {
                "confidence": conf_col,
                "block": name,
                "metric": metric,
                "value": finite(point.get(metric)),
                "ci_low": lo,
                "ci_high": hi,
                "n_questions": int(cells["question_id"].nunique()),
                "n_cells": int(len(cells)),
                "bootstrap_B": b,
            }
        )
    return pd.DataFrame(rows)


def closed_accuracy_by_popularity(cells: pd.DataFrame) -> pd.DataFrame:
    closed = cells[cells["condition"] == "closed"].copy()
    rows = []
    for pop_bin, part in closed.groupby("popularity_bin", sort=True):
        rows.append(
            {
                "popularity_bin": pop_bin,
                "closed_acc": float(part["correct"].mean()),
                "closed_conf_ptrue": float(part["conf_ptrue"].mean()) if "conf_ptrue" in part else float("nan"),
                "closed_conf_verbalized": float(part["conf_verbalized"].mean()) if "conf_verbalized" in part else float("nan"),
                "n_questions": int(part["question_id"].nunique()),
                "n_cells": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, rho: pd.DataFrame, paired_df: pd.DataFrame, fg: pd.DataFrame, closed_pop: pd.DataFrame, rows: int) -> None:
    def fmt(x: object) -> str:
        v = finite(x)
        return "NA" if not math.isfinite(v) else f"{v:.4f}"

    lines = [
        "# T3 Cross-Family Phi-3 300-Question Report",
        "",
        "Model: `microsoft/Phi-3-mini-4k-instruct` with native transformers (`--no-trust-remote-code`).",
        "",
        f"Prediction rows: {rows}. Unit for analysis: question x condition cell mean. Confidence channels: `conf_ptrue`, `conf_verbalized`.",
        "",
        "## Staged Rho",
        "",
        "| confidence | rho_veracity | rho_answer_support | rho_topicality | rho_v/rho_as | n_questions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for conf in CONF_COLS:
        sub = rho[(rho["confidence"] == conf) & (rho["block"] == "staged_rho")].set_index("metric")
        lines.append(
            f"| {conf} | {fmt(sub.loc['rho_veracity','value'])} | {fmt(sub.loc['rho_answer_support','value'])} | "
            f"{fmt(sub.loc['rho_topicality','value'])} | {fmt(sub.loc['rho_v_over_as','value'])} | {int(sub.iloc[0]['n_questions'])} |"
        )

    lines += [
        "",
        "## Sup-vs-Mis Paired",
        "",
        "| confidence | delta_conf(mis-sup) | delta_acc(mis-sup) | gap | directional AUC P(conf_sup>conf_mis) | separable AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for conf in CONF_COLS:
        sub = paired_df[(paired_df["confidence"] == conf) & (paired_df["block"] == "paired_sup_mis")].set_index("metric")
        lines.append(
            f"| {conf} | {fmt(sub.loc['delta_conf_mis_minus_sup','value'])} | {fmt(sub.loc['delta_acc_mis_minus_sup','value'])} | "
            f"{fmt(sub.loc['gap_mis_minus_sup','value'])} | {fmt(sub.loc['directional_auc_sup_gt_mis','value'])} | {fmt(sub.loc['separable_auc','value'])} |"
        )

    lines += [
        "",
        "## FG(mis)",
        "",
        "| confidence | acc(mis) | conf(mis) | delta_acc vs closed | delta_conf vs closed | FG(mis) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in fg[fg["condition"] == "mis"].iterrows():
        lines.append(
            f"| {row['confidence']} | {fmt(row['acc'])} | {fmt(row['conf'])} | {fmt(row['delta_acc'])} | "
            f"{fmt(row['delta_conf'])} | {fmt(row['FG'])} |"
        )

    lines += [
        "",
        "## Closed Accuracy By Popularity",
        "",
        "| popularity_bin | closed_acc | closed_conf_ptrue | closed_conf_verbalized | n_questions |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in closed_pop.iterrows():
        lines.append(
            f"| {row['popularity_bin']} | {fmt(row['closed_acc'])} | {fmt(row['closed_conf_ptrue'])} | "
            f"{fmt(row['closed_conf_verbalized'])} | {int(row['n_questions'])} |"
        )

    lines += [
        "",
        "##判定",
        "",
        "Phi-3 cross-family subset is intended as directional robustness, not a replacement for the full Qwen mainline. Interpret against the fixed claim: confidence may carry weak nonzero sup-vs-mis separability, but veracity should explain a much smaller share of confidence variation than answer-bearing support, while misleading evidence should still produce positive high-confidence error gaps.",
        "",
    ]
    (out_dir / "phi3_cross_family_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze T3 Phi-3 cross-family subset.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    cells = cells_from_predictions(df)
    cells.to_csv(out_dir / "phi3_cell_means.csv", index=False)

    rho_rows = []
    paired_rows = []
    for idx, conf_col in enumerate(CONF_COLS):
        rho_rows.append(
            summarize_metric(
                cells,
                conf_col,
                "staged_rho",
                staged_rho,
                ["rho_veracity", "rho_answer_support", "rho_topicality", "rho_v_over_as", "rho_v_over_top"],
                b=args.bootstrap,
                seed=args.seed + idx,
            )
        )
        paired_rows.append(
            summarize_metric(
                cells,
                conf_col,
                "paired_sup_mis",
                paired,
                [
                    "delta_conf_mis_minus_sup",
                    "delta_acc_mis_minus_sup",
                    "gap_mis_minus_sup",
                    "directional_auc_sup_gt_mis",
                    "separable_auc",
                ],
                b=args.bootstrap,
                seed=args.seed + 100 + idx,
            )
        )

    rho = pd.concat(rho_rows, ignore_index=True)
    paired_df = pd.concat(paired_rows, ignore_index=True)
    rho.to_csv(out_dir / "phi3_staged_rho.csv", index=False)
    paired_df.to_csv(out_dir / "phi3_paired_sup_mis.csv", index=False)

    fg_rows = []
    for conf_col in CONF_COLS:
        summary = condition_summary(df, conf_col)
        summary["confidence"] = conf_col
        fg_rows.append(summary)
    fg = pd.concat(fg_rows, ignore_index=True)
    fg.to_csv(out_dir / "phi3_condition_summary_fg.csv", index=False)

    closed_pop = closed_accuracy_by_popularity(cells)
    closed_pop.to_csv(out_dir / "phi3_closed_acc_by_popularity.csv", index=False)

    meta = {
        "predictions": args.predictions,
        "prediction_rows": int(len(df)),
        "n_questions": int(cells["question_id"].nunique()),
        "n_cells": int(len(cells)),
        "bootstrap_B": args.bootstrap,
        "seed": args.seed,
    }
    (out_dir / "phi3_cross_family_analysis_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, rho, paired_df, fg, closed_pop, rows=len(df))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
