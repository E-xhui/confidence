from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.regression.mixed_linear_model import MixedLM

from .spec_config import SPEC
from .text_utils import logit


CONFIDENCE_COLUMNS = ["conf_verbalized", "conf_ptrue", "conf_seqlik"]


def load_predictions(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No predictions found in {path}")
    return pd.DataFrame(rows)


def available_confidence_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in CONFIDENCE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            cols.append(col)
    return cols


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
    summary = (
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
    return summary


def faithfulness_beta(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    per_item = (
        df.groupby(["model_name", "question_id", "condition"], as_index=False)
        .agg(ybar=("correct", "mean"), cbar=(conf_col, "mean"))
        .dropna(subset=["cbar"])
    )
    closed = per_item[per_item["condition"] == "closed"][
        ["model_name", "question_id", "ybar", "cbar"]
    ].rename(columns={"ybar": "ybar_closed", "cbar": "cbar_closed"})
    reg = per_item[per_item["condition"] != "closed"].merge(
        closed, on=["model_name", "question_id"], how="inner"
    )
    reg["delta_acc"] = reg["ybar"] - reg["ybar_closed"]
    reg["delta_conf"] = reg["cbar"] - reg["cbar_closed"]
    if len(reg) < 3:
        return {"beta_F": None, "p_value": None, "n": len(reg)}
    x = sm.add_constant(reg["delta_acc"])
    model = sm.OLS(reg["delta_conf"], x).fit(cov_type="HC3")
    return {
        "alpha_F": float(model.params["const"]),
        "beta_F": float(model.params["delta_acc"]),
        "beta_F_p_value": float(model.pvalues["delta_acc"]),
        "n": int(len(reg)),
    }


def ece_score(y_true: np.ndarray, y_conf: np.ndarray, bins: int = SPEC.ece_bins) -> float:
    valid = ~np.isnan(y_conf)
    y_true = y_true[valid]
    y_conf = y_conf[valid]
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (y_conf >= lo) & (y_conf <= hi)
        else:
            mask = (y_conf >= lo) & (y_conf < hi)
        if mask.any():
            ece += (mask.sum() / len(y_conf)) * abs(y_true[mask].mean() - y_conf[mask].mean())
    return float(ece)


def stratified_ece_auroc(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    rows = []
    group_cols = ["model_name", "condition", "popularity_bin"]
    for keys, part in df.dropna(subset=[conf_col]).groupby(group_cols):
        y = part["correct"].astype(float).to_numpy()
        c = part[conf_col].astype(float).to_numpy()
        try:
            auroc = float(roc_auc_score(y, c)) if len(set(y)) == 2 else None
        except Exception:
            auroc = None
        rows.append(
            {
                "model_name": keys[0],
                "condition": keys[1],
                "popularity_bin": keys[2],
                "ece": ece_score(y, c),
                "auroc": auroc,
                "n": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _fixed_effect_values(result: Any, conditions: list[str]) -> np.ndarray:
    vals = []
    params = result.fe_params
    for cond in conditions:
        value = float(params.get("Intercept", 0.0))
        term = f"C(condition)[T.{cond}]"
        value += float(params.get(term, 0.0))
        vals.append(value)
    return np.array(vals, dtype=float)


def variance_decomposition_reml(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    evidence_conditions = [
        c
        for c in SPEC.evidence_conditions_for_rho
        if c in set(df["condition"].astype(str))
    ]
    if len(evidence_conditions) < 2:
        return {
            "estimator": "unavailable",
            "reason": "Need at least two evidence conditions excluding closed.",
        }
    d = df[df["condition"].isin(evidence_conditions)].dropna(subset=[conf_col]).copy()
    if d.empty:
        return {"estimator": "unavailable", "reason": f"No non-null {conf_col}."}
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
    d["qe"] = d["question_id"].astype(str) + "::" + d["condition"].astype(str)
    try:
        model = MixedLM.from_formula(
            "z ~ C(condition)",
            groups="question_id",
            re_formula="1",
            vc_formula={"question_condition": "0 + C(qe)"},
            data=d,
        )
        result = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
        sigma_p = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
        sigma_pe = float(result.vcomp[0]) if len(result.vcomp) else 0.0
        sigma_eps = float(result.scale)
        sigma_e = float(np.var(_fixed_effect_values(result, evidence_conditions), ddof=0))
        total = sigma_p + sigma_e + sigma_pe + sigma_eps
        return {
            "estimator": "REML MixedLM",
            "evidence_conditions": evidence_conditions,
            "sigma2_P": sigma_p,
            "sigma2_E": sigma_e,
            "sigma2_PE": sigma_pe,
            "sigma2_eps": sigma_eps,
            "rho_P": sigma_p / total if total else None,
            "rho_E": sigma_e / total if total else None,
            "rho_PE": sigma_pe / total if total else None,
            "rho_eps": sigma_eps / total if total else None,
            "mixedlm_converged": bool(result.converged),
        }
    except Exception as exc:
        return variance_decomposition_moments(d, conf_col, reason=f"REML failed: {exc}")


def variance_decomposition_moments(df: pd.DataFrame, conf_col: str, reason: str = "") -> dict[str, Any]:
    d = df.dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
    evidence_conditions = sorted(d["condition"].unique().tolist())
    cell = d.groupby(["question_id", "condition"], as_index=False).agg(z_cell=("z", "mean"))
    q_mean = cell.groupby("question_id")["z_cell"].mean()
    e_mean = cell.groupby("condition")["z_cell"].mean()
    grand = cell["z_cell"].mean()
    sigma_p = float(np.var(q_mean, ddof=1)) if len(q_mean) > 1 else 0.0
    sigma_e = float(np.var(e_mean, ddof=0)) if len(e_mean) > 1 else 0.0
    merged = cell.join(q_mean.rename("q_mean"), on="question_id").join(
        e_mean.rename("e_mean"), on="condition"
    )
    interaction = merged["z_cell"] - merged["q_mean"] - merged["e_mean"] + grand
    sigma_pe = float(np.var(interaction, ddof=1)) if len(interaction) > 1 else 0.0
    resid = d.merge(cell, on=["question_id", "condition"], how="left")
    sigma_eps = float(np.var(resid["z"] - resid["z_cell"], ddof=1)) if len(resid) > 1 else 0.0
    total = sigma_p + sigma_e + sigma_pe + sigma_eps
    return {
        "estimator": "method_of_moments_fallback",
        "reason": reason,
        "evidence_conditions": evidence_conditions,
        "sigma2_P": sigma_p,
        "sigma2_E": sigma_e,
        "sigma2_PE": sigma_pe,
        "sigma2_eps": sigma_eps,
        "rho_P": sigma_p / total if total else None,
        "rho_E": sigma_e / total if total else None,
        "rho_PE": sigma_pe / total if total else None,
        "rho_eps": sigma_eps / total if total else None,
    }


def closed_book_distribution(df: pd.DataFrame) -> dict[str, Any]:
    closed = (
        df[df["condition"] == "closed"]
        .groupby(["model_name", "question_id"], as_index=False)
        .agg(closed_book_correct_rate=("correct", "mean"))
    )
    if closed.empty:
        return {}
    return {
        "mean": float(closed["closed_book_correct_rate"].mean()),
        "p10": float(closed["closed_book_correct_rate"].quantile(0.10)),
        "p50": float(closed["closed_book_correct_rate"].quantile(0.50)),
        "p90": float(closed["closed_book_correct_rate"].quantile(0.90)),
        "counts": {
            str(k): int(v)
            for k, v in closed["closed_book_correct_rate"].value_counts().sort_index().items()
        },
    }

