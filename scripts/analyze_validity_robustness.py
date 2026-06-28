#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import load_predictions
from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


CONFIDENCE_COLUMNS = ["conf_ptrue", "conf_verbalized", "conf_seqlik"]
STAGED_CONDITIONS = ["irr", "same_entity_irr", "sup", "mis"]
TOPICALITY = {"irr": -0.75, "same_entity_irr": 0.25, "sup": 0.25, "mis": 0.25}
ANSWER_SUPPORT = {"irr": 0.0, "same_entity_irr": -2.0 / 3.0, "sup": 1.0 / 3.0, "mis": 1.0 / 3.0}
VERACITY = {"irr": 0.0, "same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5}


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return num / den


def ci(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def load_cell_means(path: str, model_label: str) -> pd.DataFrame:
    df = load_predictions(path)
    df["question_id"] = df["question_id"].astype(str)
    agg: dict[str, tuple[str, str]] = {
        "correct": ("correct", "mean"),
        "n_samples": ("correct", "size"),
    }
    for col in CONFIDENCE_COLUMNS:
        if col in df.columns:
            agg[col] = (col, "mean")
    for col in ["popularity_raw", "ctx_len_tokens", "ctx_entity_count"]:
        if col in df.columns:
            agg[col] = (col, "mean")
    out = df.groupby(["question_id", "condition"], as_index=False).agg(**agg)
    out["model"] = model_label
    return out


def bootstrap_questions(
    cells: pd.DataFrame,
    func: Callable[[pd.DataFrame], dict[str, float]],
    metrics: list[str],
    b: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(cells["question_id"].astype(str).unique()))
    boot: dict[str, list[float]] = {m: [] for m in metrics}
    grouped = {qid: g for qid, g in cells.groupby("question_id", sort=False)}
    for _ in range(b):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        parts = []
        for draw_idx, qid in enumerate(sampled):
            part = grouped[str(qid)].copy()
            # Keep duplicate bootstrap draws distinct for grouped operations.
            part["question_id"] = part["question_id"].astype(str) + f"__boot{draw_idx}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        result = func(sample)
        for metric in metrics:
            boot[metric].append(finite_float(result.get(metric)))
    return {metric: ci(vals) for metric, vals in boot.items()}


def staged_logit_rho(cells: pd.DataFrame, conf_col: str) -> dict[str, float]:
    d = cells[cells["condition"].isin(STAGED_CONDITIONS)].dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(float(x), SPEC.logit_eps))
    d["topicality"] = d["condition"].map(TOPICALITY).astype(float)
    d["answer_support"] = d["condition"].map(ANSWER_SUPPORT).astype(float)
    d["veracity"] = d["condition"].map(VERACITY).astype(float)
    x = np.column_stack(
        [
            np.ones(len(d)),
            d["topicality"].to_numpy(dtype=float),
            d["answer_support"].to_numpy(dtype=float),
            d["veracity"].to_numpy(dtype=float),
        ]
    )
    y = d["z"].to_numpy(dtype=float)
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
        "rho_v_over_as": safe_ratio(rho_v, rho_as),
        "rho_v_over_top": safe_ratio(rho_v, rho_top),
    }


def paired_sup_mis(cells: pd.DataFrame, conf_col: str) -> dict[str, float]:
    wide = cells[cells["condition"].isin(["sup", "mis"])].pivot_table(
        index="question_id", columns="condition", values=[conf_col, "correct"], aggfunc="mean"
    )
    sub = wide.dropna()
    sup_conf = sub[(conf_col, "sup")].to_numpy(dtype=float)
    mis_conf = sub[(conf_col, "mis")].to_numpy(dtype=float)
    sup_acc = sub[("correct", "sup")].to_numpy(dtype=float)
    mis_acc = sub[("correct", "mis")].to_numpy(dtype=float)
    delta_conf = mis_conf - sup_conf
    delta_acc = mis_acc - sup_acc
    diff_sup_gt_mis = sup_conf - mis_conf
    p_sup_gt_mis = float(np.mean(diff_sup_gt_mis > 0) + 0.5 * np.mean(diff_sup_gt_mis == 0))
    sep = max(p_sup_gt_mis, 1.0 - p_sup_gt_mis)
    sd = float(np.std(delta_conf, ddof=1)) if len(delta_conf) > 1 else float("nan")
    return {
        "delta_conf_mis_minus_sup": float(np.mean(delta_conf)),
        "delta_acc_mis_minus_sup": float(np.mean(delta_acc)),
        "gap_mis_minus_sup": float(np.mean(delta_conf - delta_acc)),
        "directional_auc_sup_gt_mis": p_sup_gt_mis,
        "separable_auc": sep,
        "rank_biserial_sup_gt_mis": 2.0 * p_sup_gt_mis - 1.0,
        "dz_mis_minus_sup": safe_ratio(float(np.mean(delta_conf)), sd),
    }


def axis_sensitivity(cells: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    wide = cells[cells["condition"].isin(STAGED_CONDITIONS)].pivot_table(
        index="question_id", columns="condition", values=[conf_col, "correct"], aggfunc="mean"
    )
    rows = []
    definitions = {
        "topicality": wide[(conf_col, "same_entity_irr")] - wide[(conf_col, "irr")],
        "answer_support": (wide[(conf_col, "sup")] + wide[(conf_col, "mis")]) / 2.0 - wide[(conf_col, "same_entity_irr")],
        "veracity": wide[(conf_col, "sup")] - wide[(conf_col, "mis")],
    }
    for axis, diff_series in definitions.items():
        diff = diff_series.dropna().to_numpy(dtype=float)
        if len(diff) == 0:
            rows.append(
                {
                    "axis": axis,
                    "standardized_contrast_abs": float("nan"),
                    "directional_auc": float("nan"),
                    "separable_auc": float("nan"),
                }
            )
            continue
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
        directional = float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))
        rows.append(
            {
                "axis": axis,
                "standardized_contrast_abs": abs(safe_ratio(float(np.mean(diff)), sd)),
                "directional_auc": directional,
                "separable_auc": max(directional, 1.0 - directional),
            }
        )
    return pd.DataFrame(rows)


def axis_sensitivity_metric(cells: pd.DataFrame, conf_col: str, axis: str, metric: str) -> dict[str, float]:
    table = axis_sensitivity(cells, conf_col).set_index("axis")
    return {f"{axis}_{metric}": finite_float(table.loc[axis, metric])}


def axis_sensitivity_flat(cells: pd.DataFrame, conf_col: str) -> dict[str, float]:
    table = axis_sensitivity(cells, conf_col)
    out = {}
    for _, row in table.iterrows():
        axis = str(row["axis"])
        for metric in ["standardized_contrast_abs", "directional_auc", "separable_auc"]:
            out[f"{axis}_{metric}"] = finite_float(row[metric])
    return out


def conf_matrix(cells: pd.DataFrame, conf_col: str, conditions: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    wide = cells[cells["condition"].isin(conditions)].pivot_table(
        index="question_id", columns="condition", values=[conf_col, "correct"], aggfunc="mean"
    )
    needed = [(conf_col, c) for c in conditions] + [("correct", c) for c in conditions]
    wide = wide.dropna(subset=needed)
    conf = np.column_stack([wide[(conf_col, c)].to_numpy(dtype=float) for c in conditions])
    acc = np.column_stack([wide[("correct", c)].to_numpy(dtype=float) for c in conditions])
    return conf, acc, wide.index


def logit_np(x: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(x, dtype=float), SPEC.logit_eps, 1.0 - SPEC.logit_eps)
    return np.log(c / (1.0 - c))


def staged_logit_rho_matrix(conf: np.ndarray) -> dict[str, float]:
    z = logit_np(conf)
    codes = np.array(
        [[TOPICALITY[c], ANSWER_SUPPORT[c], VERACITY[c]] for c in STAGED_CONDITIONS],
        dtype=float,
    )
    x = np.column_stack([np.ones(z.size), np.tile(codes, (z.shape[0], 1))])
    y = z.reshape(-1)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    sigmas = {}
    for idx, name in enumerate(["topicality", "answer_support", "veracity"], start=1):
        vals = beta[idx] * codes[:, idx - 1]
        sigmas[f"sigma2_{name}"] = float(np.var(vals, ddof=0))
    fixed = (np.column_stack([np.ones(len(codes)), codes]) @ beta).reshape(1, -1)
    resid = z - fixed
    q_mean = resid.mean(axis=1)
    sigma_p = float(np.var(q_mean, ddof=1)) if len(q_mean) > 1 else 0.0
    sigma_resid = float(np.var(resid - q_mean[:, None], ddof=1)) if resid.size > 1 else 0.0
    total = sigma_p + sigma_resid + sum(sigmas.values())
    rho_top = sigmas["sigma2_topicality"] / total if total else float("nan")
    rho_as = sigmas["sigma2_answer_support"] / total if total else float("nan")
    rho_v = sigmas["sigma2_veracity"] / total if total else float("nan")
    return {
        "rho_topicality": rho_top,
        "rho_answer_support": rho_as,
        "rho_veracity": rho_v,
        "rho_v_over_as": safe_ratio(rho_v, rho_as),
        "rho_v_over_top": safe_ratio(rho_v, rho_top),
    }


def paired_sup_mis_matrix(conf: np.ndarray, acc: np.ndarray) -> dict[str, float]:
    # Matrix columns are ["sup", "mis"].
    sup_conf, mis_conf = conf[:, 0], conf[:, 1]
    sup_acc, mis_acc = acc[:, 0], acc[:, 1]
    delta_conf = mis_conf - sup_conf
    delta_acc = mis_acc - sup_acc
    diff_sup_gt_mis = sup_conf - mis_conf
    p_sup_gt_mis = float(np.mean(diff_sup_gt_mis > 0) + 0.5 * np.mean(diff_sup_gt_mis == 0))
    sd = float(np.std(delta_conf, ddof=1)) if len(delta_conf) > 1 else float("nan")
    return {
        "delta_conf_mis_minus_sup": float(np.mean(delta_conf)),
        "delta_acc_mis_minus_sup": float(np.mean(delta_acc)),
        "gap_mis_minus_sup": float(np.mean(delta_conf - delta_acc)),
        "directional_auc_sup_gt_mis": p_sup_gt_mis,
        "separable_auc": max(p_sup_gt_mis, 1.0 - p_sup_gt_mis),
        "rank_biserial_sup_gt_mis": 2.0 * p_sup_gt_mis - 1.0,
        "dz_mis_minus_sup": safe_ratio(float(np.mean(delta_conf)), sd),
    }


def axis_sensitivity_matrix(conf: np.ndarray) -> dict[str, float]:
    # Matrix columns are ["irr", "same_entity_irr", "sup", "mis"].
    irr, same, sup, mis = conf[:, 0], conf[:, 1], conf[:, 2], conf[:, 3]
    diffs = {
        "topicality": same - irr,
        "answer_support": (sup + mis) / 2.0 - same,
        "veracity": sup - mis,
    }
    out: dict[str, float] = {}
    for axis, diff in diffs.items():
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
        directional = float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))
        out[f"{axis}_standardized_contrast_abs"] = abs(safe_ratio(float(np.mean(diff)), sd))
        out[f"{axis}_directional_auc"] = directional
        out[f"{axis}_separable_auc"] = max(directional, 1.0 - directional)
    return out


def bootstrap_matrix(
    n: int,
    func: Callable[[np.ndarray], dict[str, float]],
    metrics: list[str],
    b: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        result = func(idx)
        for metric in metrics:
            boot[metric].append(finite_float(result.get(metric)))
    return {metric: ci(vals) for metric, vals in boot.items()}


def run_part_a(models: list[dict[str, str]], out_dir: Path, seed: int, paired_b: int, rho_b: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a1_rows = []
    a2_rows = []
    a6_rows = []
    params = {
        "seed": seed,
        "paired_bootstrap_b": paired_b,
        "rho_bootstrap_b": rho_b,
        "unit": "question_condition_cell_mean",
        "part": "A1_A2_A6",
    }
    for model_idx, model in enumerate(models):
        cells = load_cell_means(model["path"], model["label"])
        conf_cols = [c for c in CONFIDENCE_COLUMNS if c in cells.columns]
        n_questions = int(cells["question_id"].nunique())
        n_cells = int(len(cells))
        for conf_idx, conf_col in enumerate(conf_cols):
            base_seed = seed + model_idx * 10000 + conf_idx * 1000
            staged_conf, _, staged_qids = conf_matrix(cells, conf_col, STAGED_CONDITIONS)
            point = staged_logit_rho_matrix(staged_conf)
            rho_metrics = ["rho_topicality", "rho_answer_support", "rho_veracity", "rho_v_over_as", "rho_v_over_top"]
            rho_ci = bootstrap_matrix(
                len(staged_qids),
                lambda idx, m=staged_conf: staged_logit_rho_matrix(m[idx]),
                rho_metrics,
                rho_b,
                base_seed,
            )
            row = {
                "model": model["label"],
                "confidence": conf_col,
                "subset": "full",
                "estimator": "cell_mean_staged_logit_moment",
                "n_questions": n_questions,
                "n_cells": n_cells,
                "bootstrap_b": rho_b,
                **point,
            }
            for metric in rho_metrics:
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = rho_ci[metric]
            a1_rows.append(row)

            pair_conf, pair_acc, pair_qids = conf_matrix(cells, conf_col, ["sup", "mis"])
            pair = paired_sup_mis_matrix(pair_conf, pair_acc)
            pair_metrics = list(pair.keys())
            pair_ci = bootstrap_matrix(
                len(pair_qids),
                lambda idx, cm=pair_conf, am=pair_acc: paired_sup_mis_matrix(cm[idx], am[idx]),
                pair_metrics,
                paired_b,
                base_seed + 101,
            )
            pair_row = {
                "model": model["label"],
                "confidence": conf_col,
                "subset": "full",
                "n_questions": n_questions,
                "n_cells": n_cells,
                "bootstrap_b": paired_b,
                **pair,
            }
            for metric in pair_metrics:
                pair_row[f"{metric}_ci_low"], pair_row[f"{metric}_ci_high"] = pair_ci[metric]
            a2_rows.append(pair_row)

            axis_point_dict = axis_sensitivity_matrix(staged_conf)
            axis_point = pd.DataFrame(
                [
                    {
                        "axis": axis,
                        "standardized_contrast_abs": axis_point_dict[f"{axis}_standardized_contrast_abs"],
                        "directional_auc": axis_point_dict[f"{axis}_directional_auc"],
                        "separable_auc": axis_point_dict[f"{axis}_separable_auc"],
                    }
                    for axis in ["topicality", "answer_support", "veracity"]
                ]
            )
            axis_metrics = [
                f"{axis}_{metric}"
                for axis in ["topicality", "answer_support", "veracity"]
                for metric in ["standardized_contrast_abs", "directional_auc", "separable_auc"]
            ]
            all_axis_ci = bootstrap_matrix(
                len(staged_qids),
                lambda idx, m=staged_conf: axis_sensitivity_matrix(m[idx]),
                axis_metrics,
                paired_b,
                base_seed + 202,
            )
            for _, axis_row in axis_point.iterrows():
                axis = str(axis_row["axis"])
                metrics = ["standardized_contrast_abs", "directional_auc", "separable_auc"]
                out = {
                    "model": model["label"],
                    "confidence": conf_col,
                    "subset": "full",
                    "axis": axis,
                    "n_questions": n_questions,
                    "n_cells": n_cells,
                    "bootstrap_b": paired_b,
                    "standardized_contrast_abs": finite_float(axis_row["standardized_contrast_abs"]),
                    "directional_auc": finite_float(axis_row["directional_auc"]),
                    "separable_auc": finite_float(axis_row["separable_auc"]),
                }
                for metric in metrics:
                    out[f"{metric}_ci_low"], out[f"{metric}_ci_high"] = all_axis_ci[f"{axis}_{metric}"]
                a6_rows.append(out)
    a1 = pd.DataFrame(a1_rows)
    a2 = pd.DataFrame(a2_rows)
    a6 = pd.DataFrame(a6_rows)
    a1.to_csv(out_dir / "part_a1_staged_logit_rho.csv", index=False)
    a2.to_csv(out_dir / "part_a2_sup_mis_paired.csv", index=False)
    a6.to_csv(out_dir / "part_a6_cross_axis_sensitivity.csv", index=False)
    (out_dir / "validity_robustness_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    write_part_a_report(out_dir / "part_a_report.md", a1, a2, a6)


def condition_level_table(cells: pd.DataFrame, model_label: str) -> pd.DataFrame:
    rows = []
    for conf_col in [c for c in CONFIDENCE_COLUMNS if c in cells.columns]:
        sub = cells[cells["condition"].isin(["same_entity_irr", "sup", "mis"])].copy()
        for condition, group in sub.groupby("condition", sort=False):
            rows.append(
                {
                    "model": model_label,
                    "confidence": conf_col,
                    "condition": condition,
                    "acc": float(group["correct"].mean()),
                    "conf": float(group[conf_col].mean()),
                    "n_questions": int(group["question_id"].nunique()),
                    "n_cells": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def question_centered_design(
    cells: pd.DataFrame,
    conf_col: str,
    scale: str,
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, pd.DataFrame]:
    if model_kind == "c3a":
        conditions = ["same_entity_irr", "sup", "mis"]
        codes = {
            "answer_bearing": {"same_entity_irr": 0.0, "sup": 1.0, "mis": 1.0},
            "veracity": {"same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5},
        }
        feature_names = ["answer_bearing", "veracity", "ctx_len_tokens", "ctx_entity_count"]
    elif model_kind == "c3b":
        conditions = STAGED_CONDITIONS
        codes = {
            "topicality": TOPICALITY,
            "answer_support": ANSWER_SUPPORT,
            "veracity": VERACITY,
        }
        feature_names = ["topicality", "answer_support", "veracity", "ctx_len_tokens", "ctx_entity_count"]
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")
    d = cells[cells["condition"].isin(conditions)].dropna(subset=[conf_col]).copy()
    if scale == "logit":
        d["y"] = d[conf_col].astype(float).map(lambda x: logit(float(x), SPEC.logit_eps))
    elif scale == "raw":
        d["y"] = d[conf_col].astype(float)
    else:
        raise ValueError(f"unknown scale: {scale}")
    for name, mapping in codes.items():
        d[name] = d["condition"].map(mapping).astype(float)
    for name in ["ctx_len_tokens", "ctx_entity_count"]:
        d[name] = pd.to_numeric(d[name], errors="coerce").fillna(0.0)
    cols = ["y", *feature_names]
    centered_parts = []
    for _, group in d.groupby("question_id", sort=False):
        part = group[["question_id", "condition", *cols]].copy()
        part[cols] = part[cols] - part[cols].mean(axis=0)
        centered_parts.append(part)
    centered = pd.concat(centered_parts, ignore_index=True)
    x = centered[feature_names].to_numpy(dtype=float)
    y = centered["y"].to_numpy(dtype=float)
    return x, y, feature_names, int(d["question_id"].nunique()), int(len(d)), d


def question_centered_matrices(
    cells: pd.DataFrame,
    conf_col: str,
    scale: str,
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    if model_kind == "c3a":
        conditions = ["same_entity_irr", "sup", "mis"]
        code_maps = {
            "answer_bearing": {"same_entity_irr": 0.0, "sup": 1.0, "mis": 1.0},
            "veracity": {"same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5},
        }
        feature_names = ["answer_bearing", "veracity", "ctx_len_tokens", "ctx_entity_count"]
    elif model_kind == "c3b":
        conditions = STAGED_CONDITIONS
        code_maps = {
            "topicality": TOPICALITY,
            "answer_support": ANSWER_SUPPORT,
            "veracity": VERACITY,
        }
        feature_names = ["topicality", "answer_support", "veracity", "ctx_len_tokens", "ctx_entity_count"]
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")
    d = cells[cells["condition"].isin(conditions)].dropna(subset=[conf_col]).copy()
    for name, mapping in code_maps.items():
        d[name] = d["condition"].map(mapping).astype(float)
    for name in ["ctx_len_tokens", "ctx_entity_count"]:
        d[name] = pd.to_numeric(d[name], errors="coerce").fillna(0.0)
    if scale == "logit":
        d["y"] = d[conf_col].astype(float).map(lambda x: logit(float(x), SPEC.logit_eps))
    elif scale == "raw":
        d["y"] = d[conf_col].astype(float)
    else:
        raise ValueError(f"unknown scale: {scale}")
    y_wide = d.pivot_table(index="question_id", columns="condition", values="y", aggfunc="mean")
    feature_wides = {
        name: d.pivot_table(index="question_id", columns="condition", values=name, aggfunc="mean") for name in feature_names
    }
    needed = conditions
    qids = y_wide.dropna(subset=needed).index
    for wide in feature_wides.values():
        qids = qids.intersection(wide.dropna(subset=needed).index)
    y_mat = np.column_stack([y_wide.loc[qids, c].to_numpy(dtype=float) for c in conditions])
    x_tensor = np.stack(
        [
            np.column_stack([feature_wides[name].loc[qids, c].to_numpy(dtype=float) for name in feature_names])
            for c in conditions
        ],
        axis=1,
    )
    y_centered = y_mat - y_mat.mean(axis=1, keepdims=True)
    x_centered = x_tensor - x_tensor.mean(axis=1, keepdims=True)
    return x_centered, y_centered, feature_names, int(len(qids)), int(len(qids) * len(conditions))


def ols_beta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def design_diagnostics(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"condition_number": float("nan"), "max_vif": float("nan")}
    # Standardize for comparable condition-number and VIF diagnostics.
    sd = x.std(axis=0, ddof=1)
    keep = sd > 1e-12
    xs = x[:, keep].copy()
    if xs.shape[1] == 0:
        return {"condition_number": float("inf"), "max_vif": float("inf")}
    xs = (xs - xs.mean(axis=0)) / xs.std(axis=0, ddof=1)
    condition_number = float(np.linalg.cond(xs))
    vifs = []
    for j in range(xs.shape[1]):
        target = xs[:, j]
        others = np.delete(xs, j, axis=1)
        if others.shape[1] == 0:
            vifs.append(1.0)
            continue
        pred = others @ np.linalg.lstsq(others, target, rcond=None)[0]
        ss_res = float(np.sum((target - pred) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        vifs.append(1.0 / max(1.0 - r2, 1e-12))
    return {"condition_number": condition_number, "max_vif": float(max(vifs))}


def c3_regression_one(cells: pd.DataFrame, conf_col: str, model_label: str, model_kind: str, scale: str, b: int, seed: int) -> pd.DataFrame:
    x_blocks, y_blocks, feature_names, n_questions, n_cells = question_centered_matrices(cells, conf_col, scale, model_kind)
    x = x_blocks.reshape(-1, x_blocks.shape[-1])
    y = y_blocks.reshape(-1)
    beta = ols_beta(x, y)
    diagnostics = design_diagnostics(x)
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in feature_names}
    n = x_blocks.shape[0]
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        bx = x_blocks[idx].reshape(-1, x_blocks.shape[-1])
        by = y_blocks[idx].reshape(-1)
        bbeta = ols_beta(bx, by)
        for name, value in zip(feature_names, bbeta):
            boot[name].append(float(value))
    rows = []
    status = "diagnostic" if diagnostics["condition_number"] > 30 or diagnostics["max_vif"] > 10 else "primary"
    for idx, name in enumerate(feature_names):
        lo, hi = ci(boot[name])
        rows.append(
            {
                "model": model_label,
                "confidence": conf_col,
                "regression": model_kind,
                "scale": scale,
                "term": name,
                "coef": float(beta[idx]),
                "ci_low": lo,
                "ci_high": hi,
                "condition_number": diagnostics["condition_number"],
                "max_vif": diagnostics["max_vif"],
                "status": status,
                "n_questions": n_questions,
                "n_cells": n_cells,
                "bootstrap_b": b,
            }
        )
    return pd.DataFrame(rows)


def item_prior_axis(cells: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    d = cells[cells["condition"].isin(STAGED_CONDITIONS)].dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(float(x), SPEC.logit_eps))
    d["topicality"] = d["condition"].map(TOPICALITY).astype(float)
    d["answer_support"] = d["condition"].map(ANSWER_SUPPORT).astype(float)
    d["veracity"] = d["condition"].map(VERACITY).astype(float)
    x = np.column_stack(
        [
            np.ones(len(d)),
            d["topicality"].to_numpy(dtype=float),
            d["answer_support"].to_numpy(dtype=float),
            d["veracity"].to_numpy(dtype=float),
        ]
    )
    beta = np.linalg.lstsq(x, d["z"].to_numpy(dtype=float), rcond=None)[0]
    d["resid"] = d["z"] - x @ beta
    uq = d.groupby("question_id")["resid"].mean().rename("u_q").reset_index()
    closed = cells[cells["condition"] == "closed"].copy()
    proxies = closed[["question_id", "popularity_raw", "correct", conf_col]].rename(
        columns={
            "correct": "closed_book_acc",
            conf_col: "closed_book_mean_conf",
        }
    )
    proxies["closed_book_correct_rate"] = proxies["closed_book_acc"]
    return uq.merge(proxies, on="question_id", how="inner")


def corr_pair(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if method == "spearman":
        xx = rankdata(xx)
        yy = rankdata(yy)
    return float(np.corrcoef(xx, yy)[0, 1])


def bootstrap_corr(df: pd.DataFrame, x_col: str, y_col: str, method: str, b: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    arr_x = df[x_col].to_numpy(dtype=float)
    arr_y = df[y_col].to_numpy(dtype=float)
    n = len(df)
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        vals.append(corr_pair(arr_x[idx], arr_y[idx], method))
    return ci(vals)


def e1_correlations(cells: pd.DataFrame, conf_col: str, model_label: str, b: int, seed: int) -> pd.DataFrame:
    data = item_prior_axis(cells, conf_col)
    proxies = ["popularity_raw", "closed_book_correct_rate", "closed_book_mean_conf", "closed_book_acc"]
    rows = []
    for proxy_idx, proxy in enumerate(proxies):
        for method_idx, method in enumerate(["pearson", "spearman"]):
            estimate = corr_pair(data["u_q"].to_numpy(dtype=float), data[proxy].to_numpy(dtype=float), method)
            lo, hi = bootstrap_corr(data, "u_q", proxy, method, b, seed + proxy_idx * 100 + method_idx)
            rows.append(
                {
                    "model": model_label,
                    "confidence": conf_col,
                    "proxy": proxy,
                    "method": method,
                    "correlation": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_questions": int(len(data)),
                    "n_cells": int(len(data)),
                    "bootstrap_b": b,
                    "note": "closed_book_correct_rate and closed_book_acc are the same behavioral proxy in this run"
                    if proxy in {"closed_book_correct_rate", "closed_book_acc"}
                    else "",
                }
            )
    return pd.DataFrame(rows)


def run_parts_ce(models: list[dict[str, str]], out_dir: Path, seed: int, b: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    c1_rows = []
    c3_rows = []
    e1_rows = []
    for model_idx, model in enumerate(models):
        cells = load_cell_means(model["path"], model["label"])
        c1_rows.append(condition_level_table(cells, model["label"]))
        conf_cols = [c for c in CONFIDENCE_COLUMNS if c in cells.columns]
        for conf_idx, conf_col in enumerate(conf_cols):
            base_seed = seed + model_idx * 10000 + conf_idx * 1000
            for reg_idx, reg in enumerate(["c3a", "c3b"]):
                for scale_idx, scale in enumerate(["logit", "raw"]):
                    c3_rows.append(
                        c3_regression_one(
                            cells,
                            conf_col,
                            model["label"],
                            reg,
                            scale,
                            b,
                            base_seed + reg_idx * 100 + scale_idx * 10,
                        )
                    )
            e1_rows.append(e1_correlations(cells, conf_col, model["label"], b, base_seed + 700))
    c1 = pd.concat(c1_rows, ignore_index=True)
    c3 = pd.concat(c3_rows, ignore_index=True)
    e1 = pd.concat(e1_rows, ignore_index=True)
    c1.to_csv(out_dir / "part_c1_condition_anchor_table.csv", index=False)
    c3.to_csv(out_dir / "part_c3_staged_regression.csv", index=False)
    e1.to_csv(out_dir / "part_e1_item_prior_correlations.csv", index=False)
    write_part_ce_report(out_dir / "part_ce_report.md", c1, c3, e1)
    params = {
        "seed": seed,
        "bootstrap_b": b,
        "unit": "question_condition_cell_mean",
        "parts": "C,E",
        "c3_status_rule": "diagnostic if condition_number > 30 or max_vif > 10",
    }
    (out_dir / "validity_robustness_params_ce.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in p.open(encoding="utf-8") if line.strip()])


def reader_flags(sup_reader_path: str, mis_reader_path: str) -> tuple[pd.DataFrame, set[str]]:
    sup = read_jsonl_df(sup_reader_path)
    mis = read_jsonl_df(mis_reader_path)
    rows = []
    if not sup.empty:
        rows.append(
            {
                "flag": "sup_reader_supports_gold",
                "n_questions": int(sup["question_id"].astype(str).nunique()),
                "rate": float(sup["sup_reader_supports_gold"].astype(int).mean()),
                "count": int(sup["sup_reader_supports_gold"].astype(int).sum()),
                "reader_is_automated_audit": True,
            }
        )
    if not mis.empty:
        for col in ["mis_reader_supports_wrong", "mis_reader_supports_gold"]:
            rows.append(
                {
                    "flag": col,
                    "n_questions": int(mis["question_id"].astype(str).nunique()),
                    "rate": float(mis[col].astype(int).mean()),
                    "count": int(mis[col].astype(int).sum()),
                    "reader_is_automated_audit": True,
                }
            )
    clean: set[str] = set()
    if not sup.empty and not mis.empty:
        sup_ok = set(sup.loc[sup["sup_reader_supports_gold"].astype(int).eq(1), "question_id"].astype(str))
        mis_ok = set(
            mis.loc[
                mis["mis_reader_supports_wrong"].astype(int).eq(1)
                & mis["mis_reader_supports_gold"].astype(int).eq(0),
                "question_id",
            ].astype(str)
        )
        clean = sup_ok & mis_ok
        all_q = set(sup["question_id"].astype(str)) | set(mis["question_id"].astype(str))
        rows.append(
            {
                "flag": "clean_intersection",
                "n_questions": int(len(all_q)),
                "rate": float(len(clean) / len(all_q)) if all_q else float("nan"),
                "count": int(len(clean)),
                "reader_is_automated_audit": True,
            }
        )
    return pd.DataFrame(rows), clean


def closed_known_ids(cells: pd.DataFrame) -> set[str]:
    closed = cells[cells["condition"] == "closed"].copy()
    return set(closed.loc[closed["correct"].astype(float).ge(0.6), "question_id"].astype(str))


def subset_part_a_headline(cells: pd.DataFrame, conf_col: str, subset_name: str, qids: set[str], b_pair: int, b_rho: int, seed: int) -> dict[str, Any]:
    sub = cells[cells["question_id"].astype(str).isin(qids)].copy()
    n_questions = int(sub["question_id"].nunique())
    n_cells = int(len(sub))
    row: dict[str, Any] = {
        "subset": subset_name,
        "confidence": conf_col,
        "n_questions": n_questions,
        "n_cells": n_cells,
        "underpowered": bool(n_questions < 50),
        "paired_bootstrap_b": b_pair if n_questions >= 50 else 0,
        "rho_bootstrap_b": b_rho if n_questions >= 50 else 0,
    }
    if n_questions < 3:
        return row
    staged_conf, _, staged_qids = conf_matrix(sub, conf_col, STAGED_CONDITIONS)
    rho = staged_logit_rho_matrix(staged_conf)
    row.update({f"a1_{k}": v for k, v in rho.items()})
    if n_questions >= 50:
        rho_metrics = list(rho.keys())
        rho_ci = bootstrap_matrix(
            len(staged_qids),
            lambda idx, m=staged_conf: staged_logit_rho_matrix(m[idx]),
            rho_metrics,
            b_rho,
            seed,
        )
        for metric in rho_metrics:
            row[f"a1_{metric}_ci_low"], row[f"a1_{metric}_ci_high"] = rho_ci[metric]

    pair_conf, pair_acc, pair_qids = conf_matrix(sub, conf_col, ["sup", "mis"])
    pair = paired_sup_mis_matrix(pair_conf, pair_acc)
    row.update({f"a2_{k}": v for k, v in pair.items()})
    axis = axis_sensitivity_matrix(staged_conf)
    for key, value in axis.items():
        if key.startswith("answer_support_") or key.startswith("veracity_"):
            row[f"a6_{key}"] = value
    if n_questions >= 50:
        pair_metrics = list(pair.keys())
        pair_ci = bootstrap_matrix(
            len(pair_qids),
            lambda idx, cm=pair_conf, am=pair_acc: paired_sup_mis_matrix(cm[idx], am[idx]),
            pair_metrics,
            b_pair,
            seed + 101,
        )
        for metric in pair_metrics:
            row[f"a2_{metric}_ci_low"], row[f"a2_{metric}_ci_high"] = pair_ci[metric]
    return row


def write_audit_sample(predictions: str, out_path: Path, seed: int) -> None:
    df = load_predictions(predictions)
    df["question_id"] = df["question_id"].astype(str)
    cells = (
        df[df["condition"].isin(["sup", "mis"])]
        .sort_values(["question_id", "condition", "sample_idx"])
        .groupby(["question_id", "condition"], as_index=False)
        .first()
    )
    closed = df[df["condition"] == "closed"].groupby("question_id", as_index=False).agg(closed_book_correct_rate=("correct", "mean"))
    cells = cells.merge(closed, on="question_id", how="left")
    cells["closed_known"] = cells["closed_book_correct_rate"].fillna(0).ge(0.6)
    cells["popularity_bin"] = cells.get("popularity_bin", "unknown")
    rng = np.random.default_rng(seed)
    selected = []
    for condition in ["sup", "mis"]:
        part = cells[cells["condition"] == condition].copy()
        strata = []
        for pop in sorted(part["popularity_bin"].astype(str).unique()):
            for known in [False, True]:
                strata.append((pop, known))
        base = 50 // max(1, len(strata))
        remainder = 50 - base * len(strata)
        for idx, (pop, known) in enumerate(strata):
            target = base + (1 if idx < remainder else 0)
            pool = part[(part["popularity_bin"].astype(str) == pop) & (part["closed_known"].eq(known))]
            if pool.empty:
                continue
            take = min(target, len(pool))
            selected.append(pool.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
        got = sum(len(x) for x in selected if not x.empty and x.iloc[0]["condition"] == condition)
        if got < 50:
            already = set(pd.concat(selected)["question_id"].astype(str) + "::" + pd.concat(selected)["condition"].astype(str)) if selected else set()
            pool = part[~(part["question_id"].astype(str) + "::" + part["condition"].astype(str)).isin(already)]
            if not pool.empty:
                selected.append(pool.sample(n=min(50 - got, len(pool)), random_state=int(rng.integers(0, 1_000_000))))
    sample = pd.concat(selected, ignore_index=True).head(100)
    rows = []
    for _, row in sample.iterrows():
        target = row.get("gold_answer", "") if row["condition"] == "sup" else row.get("wrong_answer", "")
        rows.append(
            {
                "audit_id": f"{row['question_id']}::{row['condition']}",
                "question_id": row["question_id"],
                "hidden_condition": row["condition"],
                "question": row.get("question", ""),
                "context": row.get("context", ""),
                "target_answer": target,
                "gold_answer": row.get("gold_answer", ""),
                "gold_answers": row.get("gold_answers", []),
                "wrong_answer": row.get("wrong_answer", ""),
                "popularity_bin": row.get("popularity_bin", ""),
                "popularity_raw": row.get("popularity_raw", None),
                "closed_known": bool(row.get("closed_known", False)),
                "closed_book_correct_rate": float(row.get("closed_book_correct_rate", np.nan)),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_part_d_local(models: list[dict[str, str]], out_dir: Path, seed: int, b_pair: int, b_rho: int, sup_reader: str, mis_reader: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flags, clean = reader_flags(sup_reader, mis_reader)
    flags.to_csv(out_dir / "part_d1_reader_flags.csv", index=False)
    rows = []
    for model_idx, model in enumerate(models):
        cells = load_cell_means(model["path"], model["label"])
        all_q = set(cells["question_id"].astype(str).unique())
        known = closed_known_ids(cells)
        subset_map = {
            "full": all_q,
            "clean": clean & all_q,
            "closed_known": known,
            "clean_and_closed_known": clean & known,
        }
        for conf_idx, conf_col in enumerate([c for c in CONFIDENCE_COLUMNS if c in cells.columns]):
            for sub_idx, (subset_name, qids) in enumerate(subset_map.items()):
                row = subset_part_a_headline(
                    cells,
                    conf_col,
                    subset_name,
                    qids,
                    b_pair,
                    b_rho,
                    seed + model_idx * 10000 + conf_idx * 1000 + sub_idx * 100,
                )
                row["model"] = model["label"]
                rows.append(row)
    d2 = pd.DataFrame(rows)
    d2.to_csv(out_dir / "part_d2_subset_headline.csv", index=False)
    write_audit_sample(models[0]["path"], out_dir / "part_d3_blind_audit_sample.jsonl", seed)
    params = {
        "seed": seed,
        "paired_bootstrap_b": b_pair,
        "rho_bootstrap_b": b_rho,
        "sup_reader": sup_reader,
        "mis_reader": mis_reader,
        "d3_sample": "part_d3_blind_audit_sample.jsonl",
        "human_20_spot_check": "deferred limitation",
    }
    (out_dir / "validity_robustness_params_d.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")


def write_part_ce_report(path: Path, c1: pd.DataFrame, c3: pd.DataFrame, e1: pd.DataFrame) -> None:
    def fmt(value: object) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    lines = [
        "# Validity Robustness Parts C and E",
        "",
        "## C1 Condition-Level Anchor Table",
        "",
        "| model | confidence | condition | acc | conf | n_questions | n_cells |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in c1.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {row['condition']} | {fmt(row['acc'])} | "
            f"{fmt(row['conf'])} | {int(row['n_questions'])} | {int(row['n_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## C2 Wording",
            "",
            "sup and mis both contain answer-bearing spans, so the first-order answer-bearing effect is controlled; residual differences may still reflect answer identity/prior, which is inspected separately.",
            "",
            "## C3 Staged-Coding Regression",
            "",
            "| model | confidence | regression | scale | term | coef | CI low | CI high | max VIF | cond # | status |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    focus_terms = {"answer_bearing", "answer_support", "topicality", "veracity"}
    for row in c3[c3["term"].isin(focus_terms)].to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {row['regression']} | {row['scale']} | {row['term']} | "
            f"{fmt(row['coef'])} | {fmt(row['ci_low'])} | {fmt(row['ci_high'])} | "
            f"{fmt(row['max_vif'])} | {fmt(row['condition_number'])} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## E1 Item/Prior Axis Correlations",
            "",
            "| model | confidence | proxy | method | corr | CI low | CI high | n_questions |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in e1.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {row['proxy']} | {row['method']} | "
            f"{fmt(row['correlation'])} | {fmt(row['ci_low'])} | {fmt(row['ci_high'])} | {int(row['n_questions'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_part_a_report(path: Path, a1: pd.DataFrame, a2: pd.DataFrame, a6: pd.DataFrame) -> None:
    def fmt(value: object) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    lines = [
        "# Validity Robustness Part A",
        "",
        "Unit: question x condition cell mean. Confidence intervals use by-question cluster bootstrap.",
        "",
        "## A1 Staged Logit Rho",
        "",
        "| model | confidence | rho_veracity | rho_answer_support | rho_topicality | rho_v/rho_as | rho_v/rho_top | n_questions | n_cells |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in a1.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {fmt(row['rho_veracity'])} | {fmt(row['rho_answer_support'])} | "
            f"{fmt(row['rho_topicality'])} | {fmt(row['rho_v_over_as'])} | {fmt(row['rho_v_over_top'])} | "
            f"{int(row['n_questions'])} | {int(row['n_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## A2 Sup vs Mis Paired",
            "",
            "| model | confidence | delta_conf(mis-sup) | delta_acc(mis-sup) | gap | P(conf_sup>conf_mis) | separable AUROC | rank-biserial | dz |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in a2.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {fmt(row['delta_conf_mis_minus_sup'])} | "
            f"{fmt(row['delta_acc_mis_minus_sup'])} | {fmt(row['gap_mis_minus_sup'])} | "
            f"{fmt(row['directional_auc_sup_gt_mis'])} | {fmt(row['separable_auc'])} | "
            f"{fmt(row['rank_biserial_sup_gt_mis'])} | {fmt(row['dz_mis_minus_sup'])} |"
        )
    lines.extend(
        [
            "",
            "## A6 Cross-Axis Absolute Sensitivity",
            "",
            "| model | confidence | axis | abs standardized contrast | separable AUROC | directional AUROC | n_questions | n_cells |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in a6.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {row['axis']} | {fmt(row['standardized_contrast_abs'])} | "
            f"{fmt(row['separable_auc'])} | {fmt(row['directional_auc'])} | {int(row['n_questions'])} | {int(row['n_cells'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_models(raw: list[str]) -> list[dict[str, str]]:
    models = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"model argument must be label=path, got: {item}")
        label, path = item.split("=", 1)
        models.append({"label": label, "path": path})
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Validity robustness analyses for RAG calibration.")
    parser.add_argument(
        "--model",
        action="append",
        default=[
            "Qwen7=runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl",
            "Qwen14=runs/phase2_five_condition/predictions_qwen14_full.jsonl",
        ],
        help="Model input as label=predictions.jsonl. Can be repeated.",
    )
    parser.add_argument("--out-dir", default="runs/phase3_construct_validation/validity_robustness")
    parser.add_argument("--parts", default="A", help="Supports A or C,E. D is intentionally separate.")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--paired-bootstrap-b", type=int, default=1000)
    parser.add_argument("--rho-bootstrap-b", type=int, default=300)
    parser.add_argument("--sup-reader", default="runs/phase2_unique/sup_reader_support.jsonl")
    parser.add_argument("--mis-reader", default="runs/phase2_unique/mis_reader_support.jsonl")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    models = parse_models(args.model)
    parts = {p.strip().upper() for p in args.parts.split(",") if p.strip()}
    if parts == {"A"}:
        run_part_a(models, out_dir, args.seed, args.paired_bootstrap_b, args.rho_bootstrap_b)
    elif parts == {"C", "E"} or parts == {"CE"}:
        run_parts_ce(models, out_dir, args.seed, args.paired_bootstrap_b)
    elif parts == {"D-LOCAL"} or parts == {"D_LOCAL"}:
        run_part_d_local(
            models,
            out_dir,
            args.seed,
            args.paired_bootstrap_b,
            args.rho_bootstrap_b,
            args.sup_reader,
            args.mis_reader,
        )
    else:
        raise SystemExit("Supported batches are --parts A, --parts C,E, or --parts D-local. Part D judge runs separately.")
    print(json.dumps({"out_dir": str(out_dir), "parts": sorted(parts), "models": models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
