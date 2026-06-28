#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragcalib.metrics import load_predictions
from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


CONDITIONS = ["closed", "irr", "same_entity_irr", "sup", "mis"]
EVIDENCE_CONDITIONS = ["irr", "same_entity_irr", "sup", "mis"]
CONFIDENCES = ["conf_ptrue", "conf_verbalized", "conf_seqlik"]
MODEL_CONFIGS = [
    {
        "label": "Qwen2.5-7B",
        "predictions": "runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl",
        "b2": "runs/phase2_with_same_entity_irr/analysis_b2_same_entity_irr",
        "h3": "runs/phase2_with_same_entity_irr/analysis_h3_calibration",
        "phase25": "runs/phase2_unique_with_irr/analysis_phase25_qwen",
    },
    {
        "label": "Qwen2.5-14B",
        "predictions": "runs/phase2_five_condition/predictions_qwen14_full.jsonl",
        "b2": "runs/phase2_five_condition/analysis_qwen14_b2",
        "h3": "runs/phase2_five_condition/analysis_qwen14_h3",
        "phase25": "runs/phase2_five_condition/analysis_qwen14_phase25",
    },
]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)

COLORS = {
    "Qwen2.5-7B": "#0072B2",
    "Qwen2.5-14B": "#D55E00",
    "acc": "#0072B2",
    "conf": "#D55E00",
    "rho_topicality": "#56B4E9",
    "rho_answer_support": "#009E73",
    "rho_veracity": "#D55E00",
    "rho_P": "#CC79A7",
    "rho_PE": "#E69F00",
    "rho_eps": "#8C8C8C",
}


def fmt(value: object, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def per_item_for_uq(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    item = (
        df[df["condition"].isin(EVIDENCE_CONDITIONS)]
        .dropna(subset=[conf_col])
        .groupby(["question_id", "condition"], as_index=False)
        .agg(
            cbar=(conf_col, "mean"),
            popularity_raw=("popularity_raw", "first"),
            popularity_bin=("popularity_bin", "first"),
        )
    )
    item["z"] = item["cbar"].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
    item["qe"] = item["question_id"].astype(str) + "::" + item["condition"].astype(str)
    item["question_id"] = item["question_id"].astype(str)
    return item


def extract_random_intercept(re_value: Any) -> float:
    try:
        if hasattr(re_value, "iloc"):
            return float(re_value.iloc[0])
        if isinstance(re_value, dict):
            return float(next(iter(re_value.values())))
        return float(np.asarray(re_value).ravel()[0])
    except Exception:
        return float("nan")


def construct_validation_one(df: pd.DataFrame, model_label: str, conf_col: str) -> tuple[dict[str, Any], pd.DataFrame]:
    item = per_item_for_uq(df, conf_col)
    model = MixedLM.from_formula(
        "z ~ C(condition)",
        groups="question_id",
        re_formula="1",
        vc_formula={"question_condition": "0 + C(qe)"},
        data=item,
    )
    result = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
    rows = []
    for qid, re_value in result.random_effects.items():
        part = item[item["question_id"] == str(qid)]
        if part.empty:
            continue
        rows.append(
            {
                "model": model_label,
                "confidence": conf_col,
                "question_id": str(qid),
                "u_q": extract_random_intercept(re_value),
                "popularity_raw": float(part["popularity_raw"].iloc[0]),
                "popularity_bin": part["popularity_bin"].iloc[0],
            }
        )
    uq = pd.DataFrame(rows).dropna(subset=["u_q", "popularity_raw"])
    uq["log1p_popularity"] = np.log1p(uq["popularity_raw"].astype(float))
    pearson = pearsonr(uq["u_q"], uq["log1p_popularity"])
    spearman = spearmanr(uq["u_q"], uq["popularity_raw"])
    summary = {
        "model": model_label,
        "confidence": conf_col,
        "n_questions": int(len(uq)),
        "pearson_r_uq_log_popularity": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r_uq_popularity": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "mixedlm_converged": bool(result.converged),
        "u_q_mean": float(uq["u_q"].mean()),
        "u_q_std": float(uq["u_q"].std(ddof=1)),
    }
    return summary, uq


def load_condition_summaries() -> pd.DataFrame:
    rows = []
    for cfg in MODEL_CONFIGS:
        table = pd.read_csv(Path(cfg["b2"]) / "condition_summary_with_same_entity.csv")
        table = table[table["confidence"].isin(CONFIDENCES)].copy()
        table["model"] = cfg["label"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def load_rho() -> pd.DataFrame:
    rows = []
    for cfg in MODEL_CONFIGS:
        table = pd.read_csv(Path(cfg["b2"]) / "staged_rho.csv")
        table["model"] = cfg["label"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def load_h3() -> pd.DataFrame:
    rows = []
    for cfg in MODEL_CONFIGS:
        table = pd.read_csv(Path(cfg["h3"]) / "h3_calibration_results.csv")
        table["model"] = cfg["label"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def load_closed_known() -> pd.DataFrame:
    rows = []
    for cfg in MODEL_CONFIGS:
        table = pd.read_csv(Path(cfg["phase25"]) / "a1_closed_strata.csv")
        table["model"] = cfg["label"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def plot_condition_summary(cond: pd.DataFrame, out_dir: Path) -> None:
    d = cond[cond["confidence"] == "conf_ptrue"].copy()
    d["condition"] = pd.Categorical(d["condition"], CONDITIONS, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True)
    x = np.arange(len(CONDITIONS))
    width = 0.36
    for i, model in enumerate(["Qwen2.5-7B", "Qwen2.5-14B"]):
        sub = d[d["model"] == model].sort_values("condition")
        offset = (i - 0.5) * width
        axes[0].bar(x + offset, sub["acc"], width * 0.92, label=model, color=COLORS[model], alpha=0.9)
        axes[1].bar(x + offset, sub["conf"], width * 0.92, label=model, color=COLORS[model], alpha=0.9)
    for ax, ylabel, title in zip(axes, ["Accuracy", "P(True)",], ["Correctness by condition", "Confidence by condition"]):
        ax.set_xticks(x)
        ax.set_xticklabels(["closed", "irr", "same\nentity", "sup", "mis"])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.02))
    save_fig(fig, out_dir, "fig_condition_acc_conf")


def plot_rho(rho: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["rho_topicality", "rho_answer_support", "rho_veracity", "rho_PE", "rho_eps"]
    labels = ["topicality", "answer\nsupport", "veracity", "q x evidence", "sample\nnoise"]
    d = rho[rho["confidence"].isin(["conf_ptrue", "conf_verbalized"])].copy()
    d["bar"] = d["model"] + "\n" + d["confidence"].map({"conf_ptrue": "P(True)", "conf_verbalized": "verbalized"})
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    bottom = np.zeros(len(d))
    x = np.arange(len(d))
    for metric, label in zip(metrics, labels):
        vals = d[metric].fillna(0).to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=COLORS.get(metric, "#999999"), edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(d["bar"], rotation=0)
    ax.set_ylabel("Variance share")
    ax.set_title("Staged variance decomposition")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, out_dir, "fig_rho_decomposition")


def plot_h3(h3: pd.DataFrame, out_dir: Path) -> None:
    methods = ["identity", "platt_logistic", "isotonic_relabel"]
    method_labels = ["raw", "Platt", "isotonic"]
    d = h3[(h3["source_confidence"] == "conf_ptrue") & (h3["method"].isin(methods))].copy()
    d["method"] = pd.Categorical(d["method"], methods, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True)
    x = np.arange(len(methods))
    for model in ["Qwen2.5-7B", "Qwen2.5-14B"]:
        sub = d[d["model"] == model].sort_values("method")
        axes[0].plot(x, sub["ece"], marker="o", label=model, color=COLORS[model])
        axes[1].plot(x, sub["rho_veracity"], marker="o", label=model, color=COLORS[model])
    for ax, ylabel, title in zip(axes, ["ECE", r"$\rho_{\mathrm{veracity}}$"], ["Calibration error drops", "Veracity share stays near zero"]):
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    save_fig(fig, out_dir, "fig_h3_calibration")


def plot_closed_known(closed: pd.DataFrame, out_dir: Path) -> None:
    d = closed[(closed["confidence"] == "conf_ptrue") & (closed["closed_status"] == "closed_known")]
    d = d[d["condition"].isin(["closed", "sup", "mis"])].copy()
    d["condition"] = pd.Categorical(d["condition"], ["closed", "sup", "mis"], ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True)
    x = np.arange(3)
    width = 0.36
    for i, model in enumerate(["Qwen2.5-7B", "Qwen2.5-14B"]):
        sub = d[d["model"] == model].sort_values("condition")
        offset = (i - 0.5) * width
        axes[0].bar(x + offset, sub["acc"], width * 0.92, label=model, color=COLORS[model])
        axes[1].bar(x + offset, sub["FG"], width * 0.92, label=model, color=COLORS[model])
    for ax, ylabel, title in zip(axes, ["Accuracy", "Faithfulness gap"], ["Closed-known accuracy collapses", "Misleading evidence gap"]):
        ax.set_xticks(x)
        ax.set_xticklabels(["closed", "sup", "mis"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].set_ylim(0, 1.02)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.02))
    save_fig(fig, out_dir, "fig_closed_known")


def plot_construct(uq: pd.DataFrame, out_dir: Path) -> None:
    d = uq[uq["confidence"] == "conf_ptrue"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, model in zip(axes, ["Qwen2.5-7B", "Qwen2.5-14B"]):
        sub = d[d["model"] == model]
        ax.scatter(sub["log1p_popularity"], sub["u_q"], s=10, alpha=0.35, color=COLORS[model], edgecolors="none")
        if len(sub) > 1:
            coef = np.polyfit(sub["log1p_popularity"], sub["u_q"], deg=1)
            xx = np.linspace(sub["log1p_popularity"].min(), sub["log1p_popularity"].max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#222222", linewidth=1.2)
        ax.set_title(model)
        ax.set_xlabel("log(1 + popularity)")
    axes[0].set_ylabel(r"Question random intercept $u_q$")
    save_fig(fig, out_dir, "fig_construct_uq_popularity")


def write_report(path: Path, construct: pd.DataFrame) -> None:
    lines = [
        "# Construct Validation and Figure Bundle",
        "",
        "Construct validation fits the spec-aligned mixed model over evidence conditions:",
        "",
        "`logit(confidence_qe) ~ C(condition) + u_q + delta_qe + eps`",
        "",
        "Then it correlates the fitted question random intercept `u_q` with `log1p(popularity_raw)`.",
        "",
        "## u_q vs popularity",
        "",
        "| model | confidence | n | Pearson r | Pearson p | Spearman rho | Spearman p | converged |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in construct.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['confidence']} | {int(row['n_questions'])} | "
            f"{fmt(row['pearson_r_uq_log_popularity'])} | {fmt(row['pearson_p'])} | "
            f"{fmt(row['spearman_r_uq_popularity'])} | {fmt(row['spearman_p'])} | "
            f"{row['mixedlm_converged']} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/fig_condition_acc_conf.pdf`: condition-level accuracy and P(True).",
            "- `figures/fig_rho_decomposition.pdf`: staged variance decomposition.",
            "- `figures/fig_h3_calibration.pdf`: ECE before/after and rho_veracity under calibration.",
            "- `figures/fig_closed_known.pdf`: closed-known collapse under misleading evidence.",
            "- `figures/fig_construct_uq_popularity.pdf`: construct validation scatter.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct validation and publication figures for RAG calibration.")
    parser.add_argument("--out-dir", default="runs/phase3_construct_validation")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    construct_rows = []
    uq_rows = []
    for cfg in MODEL_CONFIGS:
        df = load_predictions(cfg["predictions"])
        for conf_col in ["conf_ptrue", "conf_verbalized"]:
            summary, uq = construct_validation_one(df, cfg["label"], conf_col)
            construct_rows.append(summary)
            uq_rows.append(uq)
    construct = pd.DataFrame(construct_rows)
    uq = pd.concat(uq_rows, ignore_index=True)
    construct.to_csv(out_dir / "construct_validation_uq_popularity.csv", index=False)
    uq.to_csv(out_dir / "construct_validation_random_intercepts.csv", index=False)

    cond = load_condition_summaries()
    rho = load_rho()
    h3 = load_h3()
    closed = load_closed_known()
    cond.to_csv(out_dir / "figure_condition_summary_source.csv", index=False)
    rho.to_csv(out_dir / "figure_rho_source.csv", index=False)
    h3.to_csv(out_dir / "figure_h3_source.csv", index=False)
    closed.to_csv(out_dir / "figure_closed_known_source.csv", index=False)

    plot_condition_summary(cond, fig_dir)
    plot_rho(rho, fig_dir)
    plot_h3(h3, fig_dir)
    plot_closed_known(closed, fig_dir)
    plot_construct(uq, fig_dir)
    write_report(out_dir / "construct_validation_report.md", construct)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "fig_dir": str(fig_dir),
                "construct_rows": int(len(construct)),
                "figures": sorted(p.name for p in fig_dir.glob("fig_*.pdf")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
