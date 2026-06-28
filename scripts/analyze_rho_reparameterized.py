#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import available_confidence_columns, load_predictions, variance_decomposition_reml
from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


CONDITIONS = ("sup", "mis", "irr")
RELEVANCE = {"sup": 1.0 / 3.0, "mis": 1.0 / 3.0, "irr": -2.0 / 3.0}
VERACITY = {"sup": 0.5, "mis": -0.5, "irr": 0.0}


def read_jsonl_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in p.open(encoding="utf-8") if line.strip()])


def closed_known_ids(df: pd.DataFrame) -> set[str]:
    closed = (
        df[df["condition"] == "closed"]
        .groupby("question_id", as_index=False)
        .agg(ybar=("correct", "mean"))
    )
    return set(closed.loc[closed["ybar"] >= 0.6, "question_id"].astype(str))


def sup_reader_gold_ids(path: str | Path) -> set[str]:
    reader = read_jsonl_df(path)
    if reader.empty or "sup_reader_supports_gold" not in reader.columns:
        return set()
    return set(reader.loc[reader["sup_reader_supports_gold"].astype(int) == 1, "question_id"].astype(str))


def reparameterized_decomposition(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    d = df[df["condition"].isin(CONDITIONS)].dropna(subset=[conf_col]).copy()
    if d.empty:
        return {"estimator": "unavailable", "reason": f"No rows for {conf_col}."}
    d["condition"] = d["condition"].astype(str)
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
    d["relevance_c"] = d["condition"].map(RELEVANCE).astype(float)
    d["veracity"] = d["condition"].map(VERACITY).astype(float)
    d["qe"] = d["question_id"].astype(str) + "::" + d["condition"].astype(str)
    try:
        model = MixedLM.from_formula(
            "z ~ relevance_c + veracity",
            groups="question_id",
            re_formula="1",
            vc_formula={"question_condition": "0 + C(qe)"},
            data=d,
        )
        result = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
        beta_relevance = float(result.fe_params.get("relevance_c", 0.0))
        beta_veracity = float(result.fe_params.get("veracity", 0.0))
        relevance_values = np.array([beta_relevance * RELEVANCE[c] for c in CONDITIONS], dtype=float)
        veracity_values = np.array([beta_veracity * VERACITY[c] for c in CONDITIONS], dtype=float)
        sigma_p = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
        sigma_pe = float(result.vcomp[0]) if len(result.vcomp) else 0.0
        sigma_eps = float(result.scale)
        sigma_relevance = float(np.var(relevance_values, ddof=0))
        sigma_veracity = float(np.var(veracity_values, ddof=0))
        total = sigma_p + sigma_relevance + sigma_veracity + sigma_pe + sigma_eps
        return {
            "estimator": "REML MixedLM reparameterized",
            "conditions": list(CONDITIONS),
            "coding": {
                "relevance_c": RELEVANCE,
                "veracity": VERACITY,
            },
            "beta_relevance": beta_relevance,
            "beta_veracity": beta_veracity,
            "sigma2_P": sigma_p,
            "sigma2_relevance": sigma_relevance,
            "sigma2_veracity": sigma_veracity,
            "sigma2_PE": sigma_pe,
            "sigma2_eps": sigma_eps,
            "rho_P": sigma_p / total if total else None,
            "rho_relevance": sigma_relevance / total if total else None,
            "rho_veracity": sigma_veracity / total if total else None,
            "rho_PE": sigma_pe / total if total else None,
            "rho_eps": sigma_eps / total if total else None,
            "mixedlm_converged": bool(result.converged),
        }
    except Exception as exc:
        return {"estimator": "unavailable", "reason": str(exc)}


def direct_sup_mis_decomposition(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    sub = df[df["condition"].isin(["sup", "mis"])].copy()
    if sub.empty:
        return {"estimator": "unavailable", "reason": "No sup/mis rows."}
    vd = variance_decomposition_reml(sub, conf_col)
    out = {
        "direct_conditions": ["sup", "mis"],
        "direct_rho_veracity_from_condition_main_effect": vd.get("rho_E"),
        "direct_sigma2_veracity": vd.get("sigma2_E"),
        "direct_estimator": vd.get("estimator"),
    }
    if vd.get("reason"):
        out["direct_reason"] = vd.get("reason")
    return out


def write_report(out_path: Path, rows: pd.DataFrame, primary: str) -> None:
    lines = [
        "# Rho Reparameterized Analysis",
        "",
        f"Primary confidence: `{primary}`",
        "",
        "Coding: `relevance_c={sup:1/3, mis:1/3, irr:-2/3}`; `veracity={sup:0.5, mis:-0.5, irr:0}`.",
        "",
        "## Main Table",
        "",
        "| subset | confidence | rho_relevance | rho_veracity | rho_P | rho_PE | rho_eps | beta_relevance | beta_veracity | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.to_dict("records"):
        lines.append(
            "| {subset} | {confidence} | {rho_relevance:.4f} | {rho_veracity:.4f} | {rho_P:.4f} | "
            "{rho_PE:.4f} | {rho_eps:.4f} | {beta_relevance:.4f} | {beta_veracity:.4f} | {n_questions} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Direct Sup-vs-Mis Check",
            "",
            "`direct_rho_veracity_from_condition_main_effect` is the older two-condition `{sup, mis}` main-effect decomposition, included as a sanity check.",
            "",
            "| subset | confidence | direct rho veracity |",
            "|---|---|---:|",
        ]
    )
    for row in rows.to_dict("records"):
        val = row.get("direct_rho_veracity_from_condition_main_effect")
        sval = "NA" if pd.isna(val) else f"{float(val):.4f}"
        lines.append(f"| {row['subset']} | {row['confidence']} | {sval} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate relevance and veracity variance for RAG confidence.")
    parser.add_argument("--predictions", default="runs/phase2_unique_with_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--sup-reader", default="runs/phase2_unique/sup_reader_support.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_unique_with_irr/analysis_rho_reparameterized")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    df["question_id"] = df["question_id"].astype(str)
    conf_cols = available_confidence_columns(df)
    if "conf_ptrue" in conf_cols:
        conf_cols = ["conf_ptrue"] + [c for c in conf_cols if c != "conf_ptrue"]
    known_ids = closed_known_ids(df)
    sup_gold_ids = sup_reader_gold_ids(args.sup_reader)
    subset_defs: list[tuple[str, set[str] | None]] = [
        ("all", None),
        ("closed_known", known_ids),
    ]
    if sup_gold_ids:
        subset_defs.extend(
            [
                ("sup_reader_gold_pass", sup_gold_ids),
                ("closed_known_and_sup_reader_gold_pass", known_ids & sup_gold_ids),
            ]
        )

    rows = []
    for subset_name, qids in subset_defs:
        sub = df if qids is None else df[df["question_id"].isin(qids)].copy()
        if sub["question_id"].nunique() < 10:
            continue
        for conf_col in conf_cols:
            rep = reparameterized_decomposition(sub, conf_col)
            direct = direct_sup_mis_decomposition(sub, conf_col)
            row = {
                "subset": subset_name,
                "confidence": conf_col,
                "n_questions": int(sub["question_id"].nunique()),
                **rep,
                **direct,
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "rho_reparameterized.csv", index=False)
    (out_dir / "rho_reparameterized.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    primary = "conf_ptrue" if "conf_ptrue" in conf_cols else conf_cols[0]
    write_report(out_dir / "rho_reparameterized_report.md", out, primary)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
