#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import available_confidence_columns, ece_score, load_predictions
from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


PRIMARY = "conf_ptrue"
CONDITION_ORDER = ["closed", "irr", "same_entity_irr", "sup", "mis"]
STAGED_CONDITIONS = ["irr", "same_entity_irr", "sup", "mis"]
TOPICALITY = {"irr": -0.75, "same_entity_irr": 0.25, "sup": 0.25, "mis": 0.25}
ANSWER_SUPPORT = {"irr": 0.0, "same_entity_irr": -2.0 / 3.0, "sup": 1.0 / 3.0, "mis": 1.0 / 3.0}
VERACITY = {"irr": 0.0, "same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    logits = np.log(clip01(scores) / (1.0 - clip01(scores)))

    def objective(log_t: float) -> float:
        t = math.exp(log_t)
        pred = sigmoid(logits / t)
        return float(log_loss(y, clip01(pred), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-4})
    return {"temperature": float(math.exp(result.x)), "success": bool(result.success)}


def apply_temperature(scores: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    logits = np.log(clip01(scores) / (1.0 - clip01(scores)))
    return clip01(sigmoid(logits / float(params["temperature"])))


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    logits = np.log(clip01(scores) / (1.0 - clip01(scores))).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logits, y.astype(int))
    return model


def apply_platt(scores: np.ndarray, model: LogisticRegression) -> np.ndarray:
    logits = np.log(clip01(scores) / (1.0 - clip01(scores))).reshape(-1, 1)
    return clip01(model.predict_proba(logits)[:, 1])


def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(clip01(scores), y.astype(float))
    return model


def apply_isotonic(scores: np.ndarray, model: IsotonicRegression) -> np.ndarray:
    return clip01(model.predict(clip01(scores)))


def condition_dummies(df: pd.DataFrame) -> np.ndarray:
    cond = df["condition"].astype(str)
    return np.column_stack([(cond == name).astype(float).to_numpy() for name in CONDITION_ORDER[1:]])


def staged_codes(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["condition"].map(TOPICALITY).fillna(0.0).astype(float).to_numpy(),
            df["condition"].map(ANSWER_SUPPORT).fillna(0.0).astype(float).to_numpy(),
            df["condition"].map(VERACITY).fillna(0.0).astype(float).to_numpy(),
        ]
    )


def feature_matrix(df: pd.DataFrame, scores: np.ndarray, mode: str) -> np.ndarray:
    z = np.log(clip01(scores) / (1.0 - clip01(scores))).reshape(-1, 1)
    if mode == "condition":
        return np.column_stack([z, condition_dummies(df)])
    if mode == "staged_codes":
        return np.column_stack([z, staged_codes(df)])
    if mode == "condition_only":
        return condition_dummies(df)
    raise ValueError(f"unknown feature mode: {mode}")


def fit_feature_logistic(train: pd.DataFrame, scores: np.ndarray, y: np.ndarray, mode: str) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(feature_matrix(train, scores, mode), y.astype(int))
    return model


def apply_feature_logistic(eval_df: pd.DataFrame, scores: np.ndarray, model: LogisticRegression, mode: str) -> np.ndarray:
    return clip01(model.predict_proba(feature_matrix(eval_df, scores, mode))[:, 1])


def staged_moment_rho(df: pd.DataFrame, conf_col: str) -> dict[str, float]:
    d = df[df["condition"].isin(STAGED_CONDITIONS)].dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
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
    beta_topicality, beta_answer_support, beta_veracity = [float(v) for v in beta[1:]]
    sigmas = {}
    for name, coef, coding in [
        ("topicality", beta_topicality, TOPICALITY),
        ("answer_support", beta_answer_support, ANSWER_SUPPORT),
        ("veracity", beta_veracity, VERACITY),
    ]:
        vals = np.array([coef * coding[c] for c in STAGED_CONDITIONS], dtype=float)
        sigmas[f"sigma2_{name}"] = float(np.var(vals, ddof=0))
    d["fixed"] = x @ beta
    d["resid_fixed"] = d["z"] - d["fixed"]
    q_mean = d.groupby("question_id")["resid_fixed"].mean()
    cell = d.groupby(["question_id", "condition"], as_index=False).agg(
        resid_cell=("resid_fixed", "mean"),
        z_cell=("z", "mean"),
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
        "beta_topicality": beta_topicality,
        "beta_answer_support": beta_answer_support,
        "beta_veracity": beta_veracity,
    }


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
        return {"FG_mis": None, "delta_acc_mis": None, "delta_conf_mis": None}
    mis["delta_acc"] = mis["ybar"] - mis["closed_ybar"]
    mis["delta_conf"] = mis["cbar"] - mis["closed_cbar"]
    return {
        "FG_mis": float((mis["delta_conf"] - mis["delta_acc"]).mean()),
        "delta_acc_mis": float(mis["delta_acc"].mean()),
        "delta_conf_mis": float(mis["delta_conf"].mean()),
    }


def eval_metrics(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    valid = df.dropna(subset=[conf_col]).copy()
    y = valid["correct"].astype(int).to_numpy()
    c = clip01(valid[conf_col].to_numpy(dtype=float))
    out = {
        "ece": ece_score(y.astype(float), c, bins=SPEC.ece_bins),
        "nll": float(log_loss(y, c, labels=[0, 1])),
        "brier": float(brier_score_loss(y, c)),
        "mean_conf": float(np.mean(c)),
        "acc": float(np.mean(y)),
        "n_samples": int(len(valid)),
        "n_questions": int(valid["question_id"].nunique()),
    }
    out.update(fg_mis(valid, conf_col))
    out.update(staged_moment_rho(valid, conf_col))
    return out


def split_questions(df: pd.DataFrame, seed: int, train_frac: float) -> tuple[set[str], set[str]]:
    qids = np.array(sorted(df["question_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    n_train = int(round(len(qids) * train_frac))
    return set(qids[:n_train]), set(qids[n_train:])


def calibrate_column(train: pd.DataFrame, eval_df: pd.DataFrame, source_col: str) -> list[tuple[str, pd.DataFrame, dict[str, Any]]]:
    train_valid = train.dropna(subset=[source_col]).copy()
    y = train_valid["correct"].astype(int).to_numpy()
    scores = clip01(train_valid[source_col].to_numpy(dtype=float))
    eval_scores = clip01(eval_df[source_col].to_numpy(dtype=float))
    outputs: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []

    ident = eval_df.copy()
    ident["cal_conf"] = eval_scores
    outputs.append(("identity", ident, {}))

    temp_params = fit_temperature(scores, y)
    temp = eval_df.copy()
    temp["cal_conf"] = apply_temperature(eval_scores, temp_params)
    outputs.append(("temperature_scaling", temp, temp_params))

    platt_model = fit_platt(scores, y)
    platt = eval_df.copy()
    platt["cal_conf"] = apply_platt(eval_scores, platt_model)
    outputs.append(
        (
            "platt_logistic",
            platt,
            {
                "coef": float(platt_model.coef_[0][0]),
                "intercept": float(platt_model.intercept_[0]),
            },
        )
    )

    isotonic_model = fit_isotonic(scores, y)
    iso = eval_df.copy()
    iso["cal_conf"] = apply_isotonic(eval_scores, isotonic_model)
    outputs.append(("isotonic_relabel", iso, {"threshold_count": int(len(isotonic_model.X_thresholds_))}))

    # These are deliberately stronger than output-only calibration. They get
    # evidence-condition features, so they test whether a method can repair the
    # blind spot once veracity/topicality information is supplied externally.
    for method, mode, feature_names in [
        (
            "condition_oracle_logistic",
            "condition",
            ["logit_conf", *[f"is_{name}" for name in CONDITION_ORDER[1:]]],
        ),
        (
            "staged_oracle_logistic",
            "staged_codes",
            ["logit_conf", "topicality_code", "answer_support_code", "veracity_code"],
        ),
        (
            "condition_only_oracle",
            "condition_only",
            [f"is_{name}" for name in CONDITION_ORDER[1:]],
        ),
    ]:
        model = fit_feature_logistic(train_valid, scores, y, mode)
        cal = eval_df.copy()
        cal["cal_conf"] = apply_feature_logistic(eval_df, eval_scores, model, mode)
        outputs.append(
            (
                method,
                cal,
                {
                    "feature_mode": mode,
                    "feature_names": feature_names,
                    "coef": [float(v) for v in model.coef_[0]],
                    "intercept": float(model.intercept_[0]),
                },
            )
        )
    return outputs


def write_report(out_path: Path, table: pd.DataFrame, primary: str) -> None:
    lines = [
        "# H3 Calibration Before/After",
        "",
        "Question-level split: calibration methods are fit on the calibration half and reported on the held-out evaluation half.",
        "",
        f"Primary confidence: `{primary}`",
        "",
        "## Primary Table",
        "",
        "| source | method | ECE | NLL | Brier | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    prim = table[table["source_confidence"] == primary]
    for row in prim.to_dict("records"):
        lines.append(
            f"| {row['source_confidence']} | {row['method']} | {row['ece']:.4f} | {row['nll']:.4f} | "
            f"{row['brier']:.4f} | {row['FG_mis']:.4f} | {row['rho_topicality']:.4f} | "
            f"{row['rho_answer_support']:.4f} | {row['rho_veracity']:.4f} |"
        )
    lines.extend(["", "## All Confidence Channels", ""])
    lines.append("| source | method | ECE | rho_veracity | FG(mis) | n samples |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in table.to_dict("records"):
        lines.append(
            f"| {row['source_confidence']} | {row['method']} | {row['ece']:.4f} | "
            f"{row['rho_veracity']:.4f} | {row['FG_mis']:.4f} | {int(row['n_samples'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `identity`, `temperature_scaling`, `platt_logistic`, and `isotonic_relabel` are output-only calibrators: they see the confidence score and correctness labels, but not evidence condition.",
            "- `condition_oracle_logistic`, `staged_oracle_logistic`, and `condition_only_oracle` are upper-bound condition-aware controls: they are allowed to use condition labels or the staged topicality/answer-support/veracity codes.",
            "- If output-only methods reduce ECE while leaving `rho_veracity` near zero, but condition-aware controls can introduce a veracity component, H3 should be framed as an output-channel failure rather than a failure of correctness calibration alone.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="H3: test whether calibration fixes ECE but not veracity blindness.")
    parser.add_argument("--predictions", default="runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_with_same_entity_irr/analysis_h3_calibration")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--train-frac", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    df["question_id"] = df["question_id"].astype(str)
    train_ids, eval_ids = split_questions(df, args.seed, args.train_frac)
    train = df[df["question_id"].isin(train_ids)].copy()
    eval_df = df[df["question_id"].isin(eval_ids)].copy()
    conf_cols = available_confidence_columns(df)
    if PRIMARY in conf_cols:
        conf_cols = [PRIMARY] + [c for c in conf_cols if c != PRIMARY]
    rows = []
    params: dict[str, Any] = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "train_questions": len(train_ids),
        "eval_questions": len(eval_ids),
        "source_confidence_columns": conf_cols,
    }
    for col in conf_cols:
        params[col] = {}
        for method, cal_df, method_params in calibrate_column(train, eval_df, col):
            metrics = eval_metrics(cal_df, "cal_conf")
            rows.append(
                {
                    "source_confidence": col,
                    "method": method,
                    **metrics,
                }
            )
            params[col][method] = method_params
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "h3_calibration_results.csv", index=False)
    (out_dir / "h3_calibration_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = PRIMARY if PRIMARY in conf_cols else conf_cols[0]
    write_report(out_dir / "h3_calibration_report.md", table, primary)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(table), **params}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
