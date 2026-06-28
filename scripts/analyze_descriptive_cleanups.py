#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import available_confidence_columns, load_predictions
from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


PRIMARY = "conf_ptrue"
STAGED_CONDITIONS = ["irr", "same_entity_irr", "sup", "mis"]
TOPICALITY = {"irr": -0.75, "same_entity_irr": 0.25, "sup": 0.25, "mis": 0.25}
ANSWER_SUPPORT = {"irr": 0.0, "same_entity_irr": -2.0 / 3.0, "sup": 1.0 / 3.0, "mis": 1.0 / 3.0}
VERACITY = {"irr": 0.0, "same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5}


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def per_item(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    agg = (
        df.groupby(["model_name", "question_id", "condition"], as_index=False)
        .agg(
            ybar=("correct", "mean"),
            cbar=(conf_col, "mean"),
            ctx_len_tokens=("ctx_len_tokens", "mean"),
            ctx_entity_count=("ctx_entity_count", "mean"),
        )
        .dropna(subset=["cbar"])
    )
    return agg


def ols_record(y: pd.Series, x: pd.DataFrame) -> dict[str, Any]:
    model = sm.OLS(y.astype(float), sm.add_constant(x.astype(float), has_constant="add")).fit(cov_type="HC3")
    row: dict[str, Any] = {"n": int(model.nobs), "r2": float(model.rsquared)}
    for name in model.params.index:
        key = "intercept" if name == "const" else name
        row[f"{key}_coef"] = float(model.params[name])
        row[f"{key}_p"] = float(model.pvalues[name])
    return row


def answer_support_surface_control(df: pd.DataFrame, conf_cols: list[str]) -> pd.DataFrame:
    rows = []
    for conf_col in conf_cols:
        item = per_item(df, conf_col)
        sub = item[item["condition"].isin(["same_entity_irr", "sup", "mis"])].copy()
        sub["answer_bearing"] = sub["condition"].isin(["sup", "mis"]).astype(float)
        sub["is_mis"] = (sub["condition"] == "mis").astype(float)
        for outcome_name, y_col in [("confidence", "cbar"), ("accuracy", "ybar")]:
            part = sub.dropna(subset=[y_col, "ctx_len_tokens", "ctx_entity_count"])
            row = {
                "confidence": conf_col,
                "outcome": outcome_name,
                "comparison": "answer_bearing_vs_same_entity_irr_control_len_entities",
            }
            row.update(ols_record(part[y_col], part[["answer_bearing", "ctx_len_tokens", "ctx_entity_count"]]))
            rows.append(row)
            row2 = {
                "confidence": conf_col,
                "outcome": outcome_name,
                "comparison": "answer_bearing_plus_mis_indicator_control_len_entities",
            }
            row2.update(
                ols_record(part[y_col], part[["answer_bearing", "is_mis", "ctx_len_tokens", "ctx_entity_count"]])
            )
            rows.append(row2)
    return pd.DataFrame(rows)


def staged_moment_rho(df: pd.DataFrame, conf_col: str) -> dict[str, float]:
    d = df[df["condition"].isin(STAGED_CONDITIONS)].dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
    d["topicality"] = d["condition"].map(TOPICALITY).astype(float)
    d["answer_support"] = d["condition"].map(ANSWER_SUPPORT).astype(float)
    d["veracity"] = d["condition"].map(VERACITY).astype(float)
    x = sm.add_constant(d[["topicality", "answer_support", "veracity"]].astype(float), has_constant="add")
    model = sm.OLS(d["z"].astype(float), x).fit()
    betas = {name: float(model.params.get(name, 0.0)) for name in ["topicality", "answer_support", "veracity"]}
    sigmas = {}
    for name, coding in [("topicality", TOPICALITY), ("answer_support", ANSWER_SUPPORT), ("veracity", VERACITY)]:
        vals = np.array([betas[name] * coding[c] for c in STAGED_CONDITIONS], dtype=float)
        sigmas[f"sigma2_{name}"] = float(np.var(vals, ddof=0))
    d["fixed"] = np.asarray(x @ model.params)
    d["resid_fixed"] = d["z"] - d["fixed"]
    q_mean = d.groupby("question_id")["resid_fixed"].mean()
    cell = (
        d.groupby(["question_id", "condition"], as_index=False)
        .agg(resid_cell=("resid_fixed", "mean"), z_cell=("z", "mean"))
    )
    cell = cell.join(q_mean.rename("q_mean"), on="question_id")
    sigma_p = float(np.var(q_mean, ddof=1)) if len(q_mean) > 1 else 0.0
    sigma_pe = float(np.var(cell["resid_cell"] - cell["q_mean"], ddof=1)) if len(cell) > 1 else 0.0
    d = d.merge(cell[["question_id", "condition", "z_cell"]], on=["question_id", "condition"], how="left")
    sigma_eps = float(np.var(d["z"] - d["z_cell"], ddof=1)) if len(d) > 1 else 0.0
    total = sigma_p + sigma_pe + sigma_eps + sum(sigmas.values())
    return {
        "rho_topicality": sigmas["sigma2_topicality"] / total if total else np.nan,
        "rho_answer_support": sigmas["sigma2_answer_support"] / total if total else np.nan,
        "rho_veracity": sigmas["sigma2_veracity"] / total if total else np.nan,
        "rho_P": sigma_p / total if total else np.nan,
        "rho_PE": sigma_pe / total if total else np.nan,
        "rho_eps": sigma_eps / total if total else np.nan,
        "beta_topicality": betas["topicality"],
        "beta_answer_support": betas["answer_support"],
        "beta_veracity": betas["veracity"],
        "n_questions": int(d["question_id"].nunique()),
    }


def bootstrap_rho_ci(df: pd.DataFrame, conf_col: str, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = df[df["condition"].isin(STAGED_CONDITIONS)].dropna(subset=[conf_col]).copy()
    qids = np.array(sorted(base["question_id"].astype(str).unique()))
    base["question_id"] = base["question_id"].astype(str)
    rng = np.random.default_rng(seed)
    point = staged_moment_rho(base, conf_col)
    rows = []
    grouped = {qid: part for qid, part in base.groupby("question_id", sort=False)}
    for i in range(n_boot):
        sample_qids = rng.choice(qids, size=len(qids), replace=True)
        parts = []
        for j, qid in enumerate(sample_qids):
            part = grouped[qid].copy()
            part["question_id"] = f"{qid}__boot{j}"
            parts.append(part)
        boot_df = pd.concat(parts, ignore_index=True)
        row = staged_moment_rho(boot_df, conf_col)
        row["bootstrap_idx"] = i
        rows.append(row)
    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["rho_topicality", "rho_answer_support", "rho_veracity", "rho_P", "rho_PE", "rho_eps"]:
        vals = boot[metric].dropna().to_numpy()
        ci_rows.append(
            {
                "confidence": conf_col,
                "metric": metric,
                "point_moment": point.get(metric),
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else None,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else None,
                "bootstrap_n": int(len(vals)),
                "estimator": "question_bootstrap_moment",
            }
        )
    return pd.DataFrame(ci_rows), {"point": point, "bootstrap_rows": rows}


def merge_reml_points(ci: pd.DataFrame, reml_path: str | Path) -> pd.DataFrame:
    reml = pd.read_csv(reml_path)
    if reml.empty:
        return ci
    rows = []
    for row in ci.to_dict("records"):
        match = reml[reml["confidence"] == row["confidence"]]
        metric = row["metric"]
        row["point_reml"] = None
        if not match.empty and metric in match.columns:
            row["point_reml"] = float(match.iloc[0][metric])
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_path: Path, surface: pd.DataFrame, ci: pd.DataFrame, primary: str) -> None:
    lines = [
        "# Descriptive Cleanups",
        "",
        f"Primary confidence: `{primary}`",
        "",
        "## Answer-Support Surface Control",
        "",
        "| confidence | outcome | comparison | answer_bearing coef | p | ctx_len coef | ctx_entity coef | n | r2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in surface.to_dict("records"):
        if row["confidence"] != primary:
            continue
        lines.append(
            f"| {row['confidence']} | {row['outcome']} | {row['comparison']} | "
            f"{_fmt(row.get('answer_bearing_coef'))} | {_fmt(row.get('answer_bearing_p'))} | "
            f"{_fmt(row.get('ctx_len_tokens_coef'))} | {_fmt(row.get('ctx_entity_count_coef'))} | "
            f"{int(row['n'])} | {_fmt(row['r2'])} |"
        )
    lines.extend(["", "## Staged Rho Bootstrap CI", ""])
    lines.append("| metric | REML point | moment point | 95% CI low | 95% CI high | bootstrap n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in ci[ci["confidence"] == primary].to_dict("records"):
        lines.append(
            f"| {row['metric']} | {_fmt(row.get('point_reml'))} | {_fmt(row.get('point_moment'))} | "
            f"{_fmt(row.get('ci_low'))} | {_fmt(row.get('ci_high'))} | {int(row['bootstrap_n'])} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean descriptive diagnostics: surface controls and rho CIs.")
    parser.add_argument("--predictions", default="runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_with_same_entity_irr/analysis_descriptive_cleanups")
    parser.add_argument("--reml-rho", default="runs/phase2_with_same_entity_irr/analysis_b2_same_entity_irr/staged_rho.csv")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    conf_cols = available_confidence_columns(df)
    if PRIMARY in conf_cols:
        conf_cols = [PRIMARY] + [c for c in conf_cols if c != PRIMARY]
    surface = answer_support_surface_control(df, conf_cols)
    surface.to_csv(out_dir / "answer_support_surface_control.csv", index=False)

    ci_parts = []
    boot_payload: dict[str, Any] = {}
    for conf_col in conf_cols:
        ci, payload = bootstrap_rho_ci(df, conf_col, args.bootstrap, args.seed + conf_cols.index(conf_col))
        ci_parts.append(ci)
        boot_payload[conf_col] = payload
    ci = pd.concat(ci_parts, ignore_index=True)
    ci = merge_reml_points(ci, args.reml_rho)
    ci.to_csv(out_dir / "staged_rho_bootstrap_ci.csv", index=False)
    (out_dir / "staged_rho_bootstrap_payload.json").write_text(
        json.dumps(boot_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    primary = PRIMARY if PRIMARY in conf_cols else conf_cols[0]
    write_report(out_dir / "descriptive_cleanups_report.md", surface, ci, primary)
    print(json.dumps({"out_dir": str(out_dir), "confidence_columns": conf_cols, "bootstrap": args.bootstrap}, indent=2))


if __name__ == "__main__":
    main()
