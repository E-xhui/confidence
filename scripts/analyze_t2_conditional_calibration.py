#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import ece_score, load_predictions
from ragcalib.spec_config import SPEC
from scripts.analyze_b2_same_entity_irr import support_stage_decomposition
from scripts.analyze_h3_calibration import staged_moment_rho


PRIMARY = "conf_ptrue"
STAGED_CONDITIONS = ["irr", "same_entity_irr", "sup", "mis"]
MONOTONIC_METHODS = ["temperature_scaling", "platt_logistic", "isotonic_relabel"]
CTX_SURFACE = ["ctx_len_tokens", "ctx_entity_count"]
CONF_FEATURES = ["conf_ptrue", "conf_verbalized", "conf_seqlik", "mean_answer_logprob"]
PRIOR_FEATURES = ["popularity_raw", "closed_book_correct_rate"]
SELFCONS_FEATURES = [
    "answer_agreement",
    "conf_ptrue_group_mean",
    "conf_ptrue_group_var",
    "conf_verbalized_group_mean",
    "conf_verbalized_group_var",
    "conf_seqlik_group_mean",
    "conf_seqlik_group_var",
]
FEATURE_GROUPS = {
    "conf_only": CONF_FEATURES,
    "conf_plus_prior": CONF_FEATURES + PRIOR_FEATURES,
    "conf_plus_surface": CONF_FEATURES + CTX_SURFACE,
    "conf_plus_selfcons": CONF_FEATURES + SELFCONS_FEATURES,
    "all_no_ctx_surface": CONF_FEATURES + PRIOR_FEATURES + SELFCONS_FEATURES,
    "all_no_selfcons": CONF_FEATURES + PRIOR_FEATURES + CTX_SURFACE,
    "all": CONF_FEATURES + PRIOR_FEATURES + CTX_SURFACE + SELFCONS_FEATURES,
}


def clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_array(x: np.ndarray) -> np.ndarray:
    c = clip01(x)
    return np.log(c / (1.0 - c))


def normalize_answer(value: object) -> str:
    return str(value or "").strip().lower()


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> dict[str, float | bool]:
    logits = logit_array(scores)

    def objective(log_t: float) -> float:
        pred = sigmoid(logits / math.exp(log_t))
        return float(log_loss(y, clip01(pred), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-4})
    return {"temperature": float(math.exp(result.x)), "success": bool(result.success)}


def apply_temperature(scores: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return clip01(sigmoid(logit_array(scores) / float(params["temperature"])))


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logit_array(scores).reshape(-1, 1), y.astype(int))
    return model


def apply_platt(scores: np.ndarray, model: LogisticRegression) -> np.ndarray:
    return clip01(model.predict_proba(logit_array(scores).reshape(-1, 1))[:, 1])


def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(clip01(scores), y.astype(float))
    return model


def apply_isotonic(scores: np.ndarray, model: IsotonicRegression) -> np.ndarray:
    return clip01(model.predict(clip01(scores)))


def make_gbm(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed,
    )


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["question_id"] = out["question_id"].astype(str)
    closed_rate = out[out["condition"] == "closed"].groupby("question_id")["correct"].mean()
    out = out.join(closed_rate.rename("closed_book_correct_rate"), on="question_id")

    rows = []
    for (qid, condition), group in out.groupby(["question_id", "condition"], sort=False):
        answers = [normalize_answer(v) for v in group["model_answer"].tolist()]
        counts = Counter(answers)
        row: dict[str, Any] = {
            "question_id": qid,
            "condition": condition,
            "answer_agreement": max(counts.values()) / len(answers) if answers else np.nan,
        }
        for col in ["conf_ptrue", "conf_verbalized", "conf_seqlik"]:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_group_mean"] = float(vals.mean())
            row[f"{col}_group_var"] = float(vals.var(ddof=0))
        rows.append(row)
    out = out.merge(pd.DataFrame(rows), on=["question_id", "condition"], how="left")
    for col in sorted({c for group in FEATURE_GROUPS.values() for c in group}):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fg_mis(df: pd.DataFrame, conf_col: str) -> dict[str, float | None]:
    item = (
        df.groupby(["question_id", "condition"], as_index=False)
        .agg(ybar=("correct", "mean"), cbar=(conf_col, "mean"))
        .dropna(subset=["cbar"])
    )
    closed = item[item["condition"] == "closed"][["question_id", "ybar", "cbar"]].rename(
        columns={"ybar": "closed_ybar", "cbar": "closed_cbar"}
    )
    joined = item.merge(closed, on="question_id", how="inner")
    mis = joined[joined["condition"] == "mis"].copy()
    if mis.empty:
        return {"FG_mis": np.nan, "delta_acc_mis": np.nan, "delta_conf_mis": np.nan}
    mis["delta_acc"] = mis["ybar"] - mis["closed_ybar"]
    mis["delta_conf"] = mis["cbar"] - mis["closed_cbar"]
    return {
        "FG_mis": float((mis["delta_conf"] - mis["delta_acc"]).mean()),
        "delta_acc_mis": float(mis["delta_acc"].mean()),
        "delta_conf_mis": float(mis["delta_conf"].mean()),
    }


def b2_rho(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return support_stage_decomposition(df, conf_col)
        except Exception as exc:
            return {
                "rho_topicality": np.nan,
                "rho_answer_support": np.nan,
                "rho_veracity": np.nan,
                "rho_P": np.nan,
                "rho_PE": np.nan,
                "rho_eps": np.nan,
                "mixedlm_converged": False,
                "mixedlm_error": repr(exc),
            }


def metrics_row(df: pd.DataFrame, method: str, conf_col: str = "cal_conf", feature_group: str | None = None) -> dict[str, Any]:
    valid = df.dropna(subset=[conf_col]).copy()
    y = valid["correct"].astype(int).to_numpy()
    c = clip01(valid[conf_col].to_numpy(dtype=float))
    row: dict[str, Any] = {
        "method": method,
        "feature_group": feature_group,
        "ece": ece_score(y.astype(float), c, bins=SPEC.ece_bins),
        "nll": float(log_loss(y, c, labels=[0, 1])),
        "brier": float(brier_score_loss(y, c)),
        "auroc": float(roc_auc_score(y, c)) if len(np.unique(y)) == 2 else np.nan,
        "mean_conf": float(np.mean(c)),
        "acc": float(np.mean(y)),
        "n_samples": int(len(valid)),
        "n_questions": int(valid["question_id"].nunique()),
    }
    row.update(fg_mis(valid, conf_col))
    main = b2_rho(valid, conf_col)
    sanity = staged_moment_rho(valid, conf_col)
    for key in ["rho_topicality", "rho_answer_support", "rho_veracity", "rho_P", "rho_PE", "rho_eps"]:
        row[f"b2_{key}"] = main.get(key)
        row[f"h3_moment_{key}"] = sanity.get(key)
    row["b2_mixedlm_converged"] = main.get("mixedlm_converged")
    row["b2_mixedlm_error"] = main.get("mixedlm_error")
    return row


def group_folds(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = df["correct"].astype(int).to_numpy()
    groups = df["question_id"].astype(str).to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(df, y, groups))


def oof_identity(df: pd.DataFrame, source_col: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    out["cal_conf"] = clip01(out[source_col].to_numpy(dtype=float))
    return out, []


def oof_monotonic(df: pd.DataFrame, source_col: str, method: str, n_splits: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts = []
    params = []
    for fold, (train_idx, test_idx) in enumerate(group_folds(df, n_splits)):
        train = df.iloc[train_idx].dropna(subset=[source_col]).copy()
        test = df.iloc[test_idx].copy()
        y_train = train["correct"].astype(int).to_numpy()
        train_scores = train[source_col].to_numpy(dtype=float)
        test_scores = test[source_col].to_numpy(dtype=float)
        if method == "temperature_scaling":
            model_params = fit_temperature(train_scores, y_train)
            pred = apply_temperature(test_scores, model_params)
        elif method == "platt_logistic":
            model = fit_platt(train_scores, y_train)
            pred = apply_platt(test_scores, model)
            model_params = {"coef": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}
        elif method == "isotonic_relabel":
            model = fit_isotonic(train_scores, y_train)
            pred = apply_isotonic(test_scores, model)
            model_params = {"threshold_count": int(len(model.X_thresholds_))}
        else:
            raise ValueError(f"unknown monotonic method: {method}")
        test["cal_conf"] = pred
        parts.append(test)
        params.append({"fold": fold, **model_params})
    return pd.concat(parts, ignore_index=True), params


def oof_gbm(df: pd.DataFrame, features: list[str], n_splits: int, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts = []
    params = []
    for fold, (train_idx, test_idx) in enumerate(group_folds(df, n_splits)):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        model = make_gbm(seed + fold)
        model.fit(train[features].to_numpy(dtype=float), train["correct"].astype(int).to_numpy())
        test["cal_conf"] = clip01(model.predict_proba(test[features].to_numpy(dtype=float))[:, 1])
        parts.append(test)
        params.append({"fold": fold, "features": features})
    return pd.concat(parts, ignore_index=True), params


def add_ctx_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = df.copy()
    supmis = out[out["condition"].isin(["sup", "mis"])]
    for col in CTX_SURFACE:
        ranked = supmis[col].rank(method="first")
        bins = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
        out.loc[supmis.index, f"{col}_bin"] = bins.astype(float).to_numpy()
    out["ctx_match_cell"] = out["ctx_len_tokens_bin"].astype("Int64").astype(str) + "::" + out[
        "ctx_entity_count_bin"
    ].astype("Int64").astype(str)
    return out


def matched_surface_subset(df: pd.DataFrame, n_bins: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    binned = add_ctx_bins(df, n_bins)
    supmis = binned[binned["condition"].isin(["sup", "mis"])].copy()
    keep_indices: list[int] = []
    cell_rows = []
    for cell, group in supmis.groupby("ctx_match_cell", sort=False):
        sup_idx = group[group["condition"] == "sup"].index.to_numpy()
        mis_idx = group[group["condition"] == "mis"].index.to_numpy()
        n = min(len(sup_idx), len(mis_idx))
        if n == 0:
            continue
        keep_sup = rng.choice(sup_idx, size=n, replace=False)
        keep_mis = rng.choice(mis_idx, size=n, replace=False)
        keep_indices.extend([int(i) for i in keep_sup])
        keep_indices.extend([int(i) for i in keep_mis])
        cell_rows.append({"cell": cell, "n_per_condition": int(n), "sup_pool": int(len(sup_idx)), "mis_pool": int(len(mis_idx))})
    keep = set(keep_indices)
    matched_qids = set(binned.loc[list(keep), "question_id"].astype(str))
    include = (
        binned.index.isin(keep)
        | (binned["question_id"].astype(str).isin(matched_qids) & binned["condition"].isin(["closed", "irr", "same_entity_irr"]))
    )
    matched = binned[include].copy()
    diag = {
        "n_bins": n_bins,
        "matched_supmis_rows": int(len(keep)),
        "matched_questions": int(len(matched_qids)),
        "cells_used": int(len(cell_rows)),
        "condition_counts": {str(k): int(v) for k, v in matched["condition"].value_counts().to_dict().items()},
        "cell_rows": cell_rows,
    }
    return matched, diag


def condition_internal_surface_tests(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    rows = []
    for condition in ["sup", "mis"]:
        sub = df[df["condition"] == condition].copy()
        x = sub[CTX_SURFACE].to_numpy(dtype=float)
        y = sub["correct"].astype(int).to_numpy()
        groups = sub["question_id"].astype(str).to_numpy()
        fold_outputs = {}
        fold_coefs = {}
        for class_weight in [None, "balanced"]:
            key = "unweighted" if class_weight is None else "balanced"
            preds = np.full(len(sub), np.nan)
            coefs = []
            for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(sub, y, groups)):
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x[train_idx])
                x_test = scaler.transform(x[test_idx])
                model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight=class_weight, random_state=seed + fold)
                model.fit(x_train, y[train_idx])
                preds[test_idx] = model.predict_proba(x_test)[:, 1]
                coefs.append(model.coef_[0] / scaler.scale_)
            auc = float(roc_auc_score(y, preds)) if len(np.unique(y)) == 2 else np.nan
            fold_outputs[key] = auc
            fold_coefs[key] = np.mean(np.vstack(coefs), axis=0)
        coef = fold_coefs["balanced"]
        ctx_len_auc = float(roc_auc_score(y, x[:, 0])) if len(np.unique(y)) == 2 else np.nan
        ctx_entity_auc = float(roc_auc_score(y, x[:, 1])) if len(np.unique(y)) == 2 else np.nan
        rows.append(
            {
                "condition": condition,
                "n_samples": int(len(sub)),
                "n_questions": int(sub["question_id"].nunique()),
                "n_positive": int(y.sum()),
                "positive_rate": float(np.mean(y)),
                "surface_cv_auroc": fold_outputs["balanced"],
                "surface_cv_auroc_unweighted": fold_outputs["unweighted"],
                "surface_cv_auroc_balanced": fold_outputs["balanced"],
                "surface_cv_auroc_separable": float(max(fold_outputs["balanced"], 1.0 - fold_outputs["balanced"])),
                "ctx_len_tokens_auc": ctx_len_auc,
                "ctx_entity_count_auc": ctx_entity_auc,
                "coef_ctx_len_tokens": float(coef[0]),
                "coef_ctx_entity_count": float(coef[1]),
            }
        )
    return pd.DataFrame(rows)


def fold_permutation_importance(df: pd.DataFrame, features: list[str], n_splits: int, seed: int) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(group_folds(df, n_splits)):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        model = make_gbm(seed + fold)
        model.fit(train[features].to_numpy(dtype=float), train["correct"].astype(int).to_numpy())
        result = permutation_importance(
            model,
            test[features].to_numpy(dtype=float),
            test["correct"].astype(int).to_numpy(),
            scoring="roc_auc",
            n_repeats=5,
            random_state=seed + 100 + fold,
        )
        for idx, feature in enumerate(features):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance_mean": float(result.importances_mean[idx]),
                    "importance_std": float(result.importances_std[idx]),
                }
            )
    table = pd.DataFrame(rows)
    return (
        table.groupby("feature", as_index=False)
        .agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"))
        .sort_values("importance_mean", ascending=False)
    )


def pdp_surface(df: pd.DataFrame, features: list[str], seed: int, max_rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    model = make_gbm(seed)
    model.fit(df[features].to_numpy(dtype=float), df["correct"].astype(int).to_numpy())
    if len(df) > max_rows:
        sample_idx = rng.choice(df.index.to_numpy(), size=max_rows, replace=False)
        base = df.loc[sample_idx, features].copy()
    else:
        base = df[features].copy()
    rows = []
    for feature in CTX_SURFACE:
        values = np.quantile(df[feature].to_numpy(dtype=float), [0.1, 0.25, 0.5, 0.75, 0.9])
        for value in values:
            perturbed = base.copy()
            perturbed[feature] = float(value)
            pred = model.predict_proba(perturbed.to_numpy(dtype=float))[:, 1]
            rows.append({"feature": feature, "value": float(value), "mean_pred": float(np.mean(pred))})
    return pd.DataFrame(rows)


def write_report(out_path: Path, main: pd.DataFrame, ablation: pd.DataFrame, d1: pd.DataFrame, d2: pd.DataFrame, d3: pd.DataFrame) -> None:
    def fmt(value: object) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    lines = ["# T2 Conditional Calibration", "", "## Main Comparison", ""]
    lines.append("| method | feature_group | B2 rho_veracity | H3 moment rho_veracity | ECE | FG(mis) | AUROC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in main.to_dict("records"):
        lines.append(
            f"| {row['method']} | {row.get('feature_group') or ''} | {fmt(row.get('b2_rho_veracity'))} | "
            f"{fmt(row.get('h3_moment_rho_veracity'))} | {fmt(row.get('ece'))} | {fmt(row.get('FG_mis'))} | {fmt(row.get('auroc'))} |"
        )
    lines.extend(["", "## Feature Ablation", ""])
    lines.append("| feature_group | B2 rho_veracity | ECE | FG(mis) | AUROC |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in ablation.to_dict("records"):
        lines.append(
            f"| {row['feature_group']} | {fmt(row.get('b2_rho_veracity'))} | {fmt(row.get('ece'))} | "
            f"{fmt(row.get('FG_mis'))} | {fmt(row.get('auroc'))} |"
        )
    lines.extend(["", "## D1 Matched Surface", ""])
    lines.append("| method | B2 rho_veracity | ECE | FG(mis) | AUROC | n samples | n questions |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in d1.to_dict("records"):
        lines.append(
            f"| {row['method']} | {fmt(row.get('b2_rho_veracity'))} | {fmt(row.get('ece'))} | "
            f"{fmt(row.get('FG_mis'))} | {fmt(row.get('auroc'))} | {int(row.get('n_samples', 0))} | {int(row.get('n_questions', 0))} |"
        )
    lines.extend(["", "## D2 Condition-Internal Surface Predictiveness", ""])
    lines.append("| condition | positive_rate | n_pos | balanced AUROC | unweighted AUROC | separable AUROC | ctx_len AUC | ctx_entity AUC | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in d2.to_dict("records"):
        lines.append(
            f"| {row['condition']} | {fmt(row.get('positive_rate'))} | {int(row.get('n_positive', 0))} | "
            f"{fmt(row.get('surface_cv_auroc_balanced'))} | {fmt(row.get('surface_cv_auroc_unweighted'))} | "
            f"{fmt(row.get('surface_cv_auroc_separable'))} | {fmt(row.get('ctx_len_tokens_auc'))} | "
            f"{fmt(row.get('ctx_entity_count_auc'))} | {int(row.get('n_samples', 0))} |"
        )
    lines.extend(["", "## D3 Top Permutation Importances", ""])
    lines.append("| feature | importance_mean | importance_std |")
    lines.append("|---|---:|---:|")
    for row in d3.head(10).to_dict("records"):
        lines.append(f"| {row['feature']} | {fmt(row.get('importance_mean'))} | {fmt(row.get('importance_std'))} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 conditional non-monotonic calibration with artifact diagnostics.")
    parser.add_argument("--predictions", default="runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--out-dir", default="runs/phase3_construct_validation/t2_conditional")
    parser.add_argument("--source-confidence", default=PRIMARY)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--match-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = enrich_features(load_predictions(args.predictions))
    source = args.source_confidence

    main_rows = []
    params: dict[str, Any] = {
        "predictions": args.predictions,
        "source_confidence": source,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "feature_groups": FEATURE_GROUPS,
        "forbidden_features": ["condition", "gold_answer", "wrong_answer", "correct"],
    }

    identity_df, _ = oof_identity(df, source)
    main_rows.append(metrics_row(identity_df, "identity", feature_group="source_confidence"))
    for method in MONOTONIC_METHODS:
        cal_df, method_params = oof_monotonic(df, source, method, args.n_splits)
        main_rows.append(metrics_row(cal_df, method, feature_group="source_confidence"))
        params[method] = method_params
    for group_name in ["all", "all_no_ctx_surface"]:
        cal_df, method_params = oof_gbm(df, FEATURE_GROUPS[group_name], args.n_splits, args.seed)
        main_rows.append(metrics_row(cal_df, "conditional_gbm", feature_group=group_name))
        params[f"conditional_gbm_{group_name}"] = method_params
    main = pd.DataFrame(main_rows)

    ablation_rows = []
    for group_name, features in FEATURE_GROUPS.items():
        cal_df, _ = oof_gbm(df, features, args.n_splits, args.seed)
        ablation_rows.append(metrics_row(cal_df, "conditional_gbm", feature_group=group_name))
    ablation = pd.DataFrame(ablation_rows)

    matched_df, match_diag = matched_surface_subset(df, args.match_bins, args.seed)
    d1_identity, _ = oof_identity(matched_df, source)
    d1_gbm, _ = oof_gbm(matched_df, FEATURE_GROUPS["all"], args.n_splits, args.seed)
    d1 = pd.DataFrame(
        [
            metrics_row(d1_identity, "matched_identity", feature_group="source_confidence"),
            metrics_row(d1_gbm, "matched_conditional_gbm", feature_group="all"),
        ]
    )
    params["d1_matching"] = match_diag

    d2 = condition_internal_surface_tests(df, args.n_splits, args.seed)
    d3 = fold_permutation_importance(df, FEATURE_GROUPS["all"], args.n_splits, args.seed)
    pdp = pdp_surface(df, FEATURE_GROUPS["all"], args.seed)

    main.to_csv(out_dir / "main_comparison.csv", index=False)
    ablation.to_csv(out_dir / "feature_ablation.csv", index=False)
    d1.to_csv(out_dir / "d1_matched_surface.csv", index=False)
    d2.to_csv(out_dir / "d2_condition_internal_surface.csv", index=False)
    d3.to_csv(out_dir / "d3_feature_importance.csv", index=False)
    pdp.to_csv(out_dir / "d3_surface_pdp.csv", index=False)
    (out_dir / "t2_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "t2_conditional_report.md", main, ablation, d1, d2, d3)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "main_rows": len(main),
                "ablation_rows": len(ablation),
                "d1_rows": len(d1),
                "d2_rows": len(d2),
                "d3_rows": len(d3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
