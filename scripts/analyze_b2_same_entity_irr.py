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

from ragcalib.metrics import available_confidence_columns, condition_summary, load_predictions
from ragcalib.spec_config import SPEC
from ragcalib.text_utils import logit


PRIMARY = "conf_ptrue"
ORDER = ["closed", "irr", "same_entity_irr", "sup", "mis"]


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
    return (
        df.groupby(["model_name", "question_id", "condition"], as_index=False)
        .agg(ybar=("correct", "mean"), cbar=(conf_col, "mean"))
        .dropna(subset=["cbar"])
    )


def key_contrasts(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    item = per_item(df, conf_col)
    wide = item.pivot_table(index=["model_name", "question_id"], columns="condition", values=["ybar", "cbar"])
    rows = []
    for model_name in item["model_name"].dropna().unique():
        part = item[item["model_name"] == model_name]
        qids = sorted(part["question_id"].unique())
        for left, right in [
            ("same_entity_irr", "irr"),
            ("same_entity_irr", "closed"),
            ("same_entity_irr", "sup"),
            ("same_entity_irr", "mis"),
            ("mis", "sup"),
        ]:
            vals = []
            accs = []
            for qid in qids:
                try:
                    l_conf = wide.loc[(model_name, qid), ("cbar", left)]
                    r_conf = wide.loc[(model_name, qid), ("cbar", right)]
                    l_acc = wide.loc[(model_name, qid), ("ybar", left)]
                    r_acc = wide.loc[(model_name, qid), ("ybar", right)]
                except KeyError:
                    continue
                if pd.notna(l_conf) and pd.notna(r_conf):
                    vals.append(float(l_conf - r_conf))
                if pd.notna(l_acc) and pd.notna(r_acc):
                    accs.append(float(l_acc - r_acc))
            rows.append(
                {
                    "confidence": conf_col,
                    "model_name": model_name,
                    "contrast": f"{left}_minus_{right}",
                    "delta_conf": float(np.mean(vals)) if vals else None,
                    "delta_acc": float(np.mean(accs)) if accs else None,
                    "n_questions": len(vals),
                }
            )
    return pd.DataFrame(rows)


def support_stage_decomposition(df: pd.DataFrame, conf_col: str) -> dict[str, Any]:
    # Orthogonal-ish staged coding over {irr, same_entity_irr, sup, mis}:
    # topicality separates unrelated from same-entity/answer-bearing evidence;
    # answer_support separates answer-bearing evidence from same-entity non-answer evidence;
    # veracity separates sup from mis.
    conditions = ["irr", "same_entity_irr", "sup", "mis"]
    topicality = {"irr": -0.75, "same_entity_irr": 0.25, "sup": 0.25, "mis": 0.25}
    answer_support = {"irr": 0.0, "same_entity_irr": -2.0 / 3.0, "sup": 1.0 / 3.0, "mis": 1.0 / 3.0}
    veracity = {"irr": 0.0, "same_entity_irr": 0.0, "sup": 0.5, "mis": -0.5}
    d = df[df["condition"].isin(conditions)].dropna(subset=[conf_col]).copy()
    d["z"] = d[conf_col].astype(float).map(lambda x: logit(x, SPEC.logit_eps))
    d["topicality"] = d["condition"].map(topicality).astype(float)
    d["answer_support"] = d["condition"].map(answer_support).astype(float)
    d["veracity"] = d["condition"].map(veracity).astype(float)
    d["qe"] = d["question_id"].astype(str) + "::" + d["condition"].astype(str)
    model = MixedLM.from_formula(
        "z ~ topicality + answer_support + veracity",
        groups="question_id",
        re_formula="1",
        vc_formula={"question_condition": "0 + C(qe)"},
        data=d,
    )
    result = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
    betas = {name: float(result.fe_params.get(name, 0.0)) for name in ["topicality", "answer_support", "veracity"]}
    sigmas = {}
    for name, coding in [("topicality", topicality), ("answer_support", answer_support), ("veracity", veracity)]:
        vals = np.array([betas[name] * coding[c] for c in conditions], dtype=float)
        sigmas[f"sigma2_{name}"] = float(np.var(vals, ddof=0))
    sigma_p = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
    sigma_pe = float(result.vcomp[0]) if len(result.vcomp) else 0.0
    sigma_eps = float(result.scale)
    total = sigma_p + sigma_pe + sigma_eps + sum(sigmas.values())
    out = {
        "estimator": "REML MixedLM staged",
        "conditions": conditions,
        "coding": {"topicality": topicality, "answer_support": answer_support, "veracity": veracity},
        **{f"beta_{k}": v for k, v in betas.items()},
        "sigma2_P": sigma_p,
        **sigmas,
        "sigma2_PE": sigma_pe,
        "sigma2_eps": sigma_eps,
        "rho_P": sigma_p / total if total else None,
        "rho_topicality": sigmas["sigma2_topicality"] / total if total else None,
        "rho_answer_support": sigmas["sigma2_answer_support"] / total if total else None,
        "rho_veracity": sigmas["sigma2_veracity"] / total if total else None,
        "rho_PE": sigma_pe / total if total else None,
        "rho_eps": sigma_eps / total if total else None,
        "mixedlm_converged": bool(result.converged),
    }
    return out


def write_report(out: Path, summary: pd.DataFrame, contrasts: pd.DataFrame, rho_rows: list[dict[str, Any]], primary: str) -> None:
    lines = ["# B2 Same-Entity IRR Analysis", "", f"Primary confidence: `{primary}`", ""]
    lines.extend(["## Condition Summary", ""])
    lines.append("| condition | acc | conf | delta_acc | delta_conf | FG | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    prim = summary[summary["confidence"] == primary].copy()
    prim["condition"] = pd.Categorical(prim["condition"], ORDER, ordered=True)
    for row in prim.sort_values("condition").to_dict("records"):
        lines.append(
            f"| {row['condition']} | {_fmt(row['acc'])} | {_fmt(row['conf'])} | {_fmt(row['delta_acc'])} | "
            f"{_fmt(row['delta_conf'])} | {_fmt(row['FG'])} | {int(row['n_questions'])} |"
        )
    lines.extend(["", "## Key Contrasts", ""])
    lines.append("| contrast | delta_conf | delta_acc | n |")
    lines.append("|---|---:|---:|---:|")
    for row in contrasts[contrasts["confidence"] == primary].to_dict("records"):
        lines.append(f"| {row['contrast']} | {_fmt(row['delta_conf'])} | {_fmt(row['delta_acc'])} | {int(row['n_questions'])} |")
    lines.extend(["", "## Staged Rho", ""])
    lines.append("| confidence | rho_topicality | rho_answer_support | rho_veracity | rho_P | rho_PE | rho_eps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rho_rows:
        lines.append(
            f"| {row['confidence']} | {_fmt(row.get('rho_topicality'))} | {_fmt(row.get('rho_answer_support'))} | "
            f"{_fmt(row.get('rho_veracity'))} | {_fmt(row.get('rho_P'))} | {_fmt(row.get('rho_PE'))} | {_fmt(row.get('rho_eps'))} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze same-entity irrelevant evidence control.")
    parser.add_argument("--predictions", default="runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl")
    parser.add_argument("--out-dir", default="runs/phase2_with_same_entity_irr/analysis_b2_same_entity_irr")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    conf_cols = available_confidence_columns(df)
    if PRIMARY in conf_cols:
        conf_cols = [PRIMARY] + [c for c in conf_cols if c != PRIMARY]
    summaries = []
    contrast_rows = []
    rho_rows = []
    for conf_col in conf_cols:
        s = condition_summary(df, conf_col)
        s["confidence"] = conf_col
        summaries.append(s)
        contrast_rows.append(key_contrasts(df, conf_col))
        rho = support_stage_decomposition(df, conf_col)
        rho["confidence"] = conf_col
        rho_rows.append(rho)
    summary = pd.concat(summaries, ignore_index=True)
    contrasts = pd.concat(contrast_rows, ignore_index=True)
    summary.to_csv(out_dir / "condition_summary_with_same_entity.csv", index=False)
    contrasts.to_csv(out_dir / "key_contrasts.csv", index=False)
    Path(out_dir / "staged_rho.json").write_text(json.dumps(rho_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rho_rows).to_csv(out_dir / "staged_rho.csv", index=False)
    primary = PRIMARY if PRIMARY in conf_cols else conf_cols[0]
    write_report(out_dir / "b2_same_entity_irr_report.md", summary, contrasts, rho_rows, primary)
    print(json.dumps({"out_dir": str(out_dir), "confidence_columns": conf_cols}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
