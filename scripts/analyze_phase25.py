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

from ragcalib.metrics import (
    available_confidence_columns,
    load_predictions,
    variance_decomposition_reml,
)
from ragcalib.text_utils import answer_in_text, normalize_text


PRIMARY_CONF = "conf_ptrue"
CONF_LABELS = {
    "conf_ptrue": "P(True)",
    "conf_verbalized": "verbalized",
    "conf_seqlik": "sequence_likelihood",
}


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _ordered_conf_cols(df: pd.DataFrame) -> list[str]:
    cols = available_confidence_columns(df)
    if PRIMARY_CONF in cols:
        cols = [PRIMARY_CONF] + [c for c in cols if c != PRIMARY_CONF]
    return cols


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else None


def read_jsonl_df(path: str | Path) -> pd.DataFrame:
    rows = []
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def enrich_prediction_metadata(df: pd.DataFrame, context_paths: list[str]) -> pd.DataFrame:
    missing_cols = [
        col
        for col in [
            "mis_source_role",
            "sup_source_role",
            "irr_source_question_id",
            "irr_source_question",
            "irr_source",
        ]
        if col not in df.columns
    ]
    if not missing_cols:
        return df

    meta_frames = []
    for path in context_paths:
        meta = read_jsonl_df(path)
        if meta.empty:
            continue
        keep = ["question_id", "condition"] + [c for c in missing_cols if c in meta.columns]
        if len(keep) > 2:
            meta_frames.append(meta[keep].drop_duplicates(["question_id", "condition"]))
    if not meta_frames:
        return df
    meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(["question_id", "condition"])
    return df.merge(meta, on=["question_id", "condition"], how="left")


def per_item_table(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    cols = {
        "correct": "mean",
        conf_col: "mean",
        "sample_idx": "size",
        "popularity_bin": _first_non_null,
        "popularity_raw": _first_non_null,
        "ctx_len_tokens": "mean",
        "ctx_entity_count": "mean",
        "mis_source_role": _first_non_null,
        "sup_source_role": _first_non_null,
    }
    present = {k: v for k, v in cols.items() if k in df.columns}
    out = (
        df.groupby(["model_name", "question_id", "condition"], as_index=False)
        .agg(**{k: (k, v) for k, v in present.items()})
        .rename(columns={"correct": "ybar", conf_col: "cbar", "sample_idx": "n_samples"})
        .dropna(subset=["cbar"])
    )
    closed = out[out["condition"] == "closed"][
        ["model_name", "question_id", "ybar", "cbar"]
    ].rename(columns={"ybar": "ybar_closed", "cbar": "cbar_closed"})
    out = out.merge(closed, on=["model_name", "question_id"], how="left")
    out["delta_acc"] = out["ybar"] - out["ybar_closed"]
    out["delta_conf"] = out["cbar"] - out["cbar_closed"]
    out["FG"] = out["delta_conf"] - out["delta_acc"]
    out["closed_known"] = out["ybar_closed"] >= 0.6
    out["closed_status"] = np.where(out["closed_known"], "closed_known", "closed_unknown")
    return out


def summarize_items(
    items: pd.DataFrame,
    group_cols: list[str],
    conf_col: str,
) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    return (
        items.groupby(group_cols + ["model_name", "condition"], as_index=False)
        .agg(
            acc=("ybar", "mean"),
            conf=("cbar", "mean"),
            delta_acc=("delta_acc", "mean"),
            delta_conf=("delta_conf", "mean"),
            FG=("FG", "mean"),
            n_questions=("question_id", "nunique"),
        )
        .assign(confidence=conf_col)
        .sort_values(group_cols + ["model_name", "condition"])
    )


def closed_strata_summary(df: pd.DataFrame, conf_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    flips = []
    for conf_col in conf_cols:
        items = per_item_table(df, conf_col)
        summaries.append(summarize_items(items, ["closed_status"], conf_col))
        mis = items[(items["condition"] == "mis") & (items["closed_known"])].copy()
        if not mis.empty:
            flipped = mis["ybar"] < 0.5
            flips.append(
                {
                    "confidence": conf_col,
                    "model_name": _first_non_null(mis["model_name"]),
                    "closed_known_questions": int(mis["question_id"].nunique()),
                    "flip_to_mostly_wrong_rate": float(flipped.mean()),
                    "mean_mis_error_rate": float((1.0 - mis["ybar"]).mean()),
                    "mean_conf_flipped": float(mis.loc[flipped, "cbar"].mean()) if flipped.any() else None,
                    "mean_conf_not_flipped": float(mis.loc[~flipped, "cbar"].mean()) if (~flipped).any() else None,
                }
            )
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(flips)


def mis_source_summary(df: pd.DataFrame, conf_cols: list[str]) -> pd.DataFrame:
    rows = []
    label = {
        "parametric": "confirming_wrong_prior",
        "counter": "conflicting_counter_memory",
    }
    for conf_col in conf_cols:
        items = per_item_table(df, conf_col)
        mis = items[items["condition"] == "mis"].copy()
        if mis.empty or "mis_source_role" not in mis.columns:
            continue
        mis["mis_source_group"] = mis["mis_source_role"].map(label).fillna(mis["mis_source_role"])
        mis["confidence"] = conf_col
        rows.append(
            mis.groupby(["confidence", "model_name", "mis_source_role", "mis_source_group"], as_index=False)
            .agg(
                acc=("ybar", "mean"),
                conf=("cbar", "mean"),
                delta_acc=("delta_acc", "mean"),
                delta_conf=("delta_conf", "mean"),
                FG=("FG", "mean"),
                n_questions=("question_id", "nunique"),
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def popularity_summary(df: pd.DataFrame, conf_cols: list[str]) -> pd.DataFrame:
    rows = []
    for conf_col in conf_cols:
        items = per_item_table(df, conf_col)
        items["confidence"] = conf_col
        rows.append(
            items.groupby(["confidence", "model_name", "popularity_bin", "condition"], as_index=False)
            .agg(acc=("ybar", "mean"), conf=("cbar", "mean"), n_questions=("question_id", "nunique"))
        )
    return pd.concat(rows, ignore_index=True)


def _ols_record(y: pd.Series, x: pd.DataFrame, prefix: str) -> dict[str, Any]:
    model = sm.OLS(y.astype(float), sm.add_constant(x.astype(float), has_constant="add")).fit(cov_type="HC3")
    row: dict[str, Any] = {"n": int(model.nobs), "r2": float(model.rsquared)}
    for name in model.params.index:
        key = "intercept" if name == "const" else name
        row[f"{prefix}{key}_coef"] = float(model.params[name])
        row[f"{prefix}{key}_p"] = float(model.pvalues[name])
    return row


def surface_regressions(df: pd.DataFrame, conf_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    within_rows = []
    contrast_rows = []
    for conf_col in conf_cols:
        items = per_item_table(df, conf_col)
        for (model_name, condition), part in items.groupby(["model_name", "condition"]):
            x_cols = ["ctx_len_tokens", "ctx_entity_count"]
            part = part.dropna(subset=["cbar"] + x_cols)
            if len(part) < 10 or part[x_cols].nunique().max() <= 1:
                continue
            row = {
                "confidence": conf_col,
                "model_name": model_name,
                "condition": condition,
            }
            row.update(_ols_record(part["cbar"], part[x_cols], ""))
            within_rows.append(row)

        sm_part = items[items["condition"].isin(["sup", "mis"])].dropna(
            subset=["cbar", "ctx_len_tokens", "ctx_entity_count"]
        )
        if len(sm_part) >= 10:
            sm_part = sm_part.copy()
            sm_part["is_mis"] = (sm_part["condition"] == "mis").astype(float)
            row = {"confidence": conf_col, "model_name": _first_non_null(sm_part["model_name"])}
            row.update(_ols_record(sm_part["cbar"], sm_part[["is_mis", "ctx_len_tokens", "ctx_entity_count"]], ""))
            contrast_rows.append(row)
    return pd.DataFrame(within_rows), pd.DataFrame(contrast_rows)


def rho_subgroups(df: pd.DataFrame, conf_cols: list[str]) -> pd.DataFrame:
    rows = []
    base = per_item_table(df, conf_cols[0])
    known_ids = set(base.loc[(base["condition"] == "closed") & (base["closed_known"]), "question_id"])
    unknown_ids = set(base.loc[(base["condition"] == "closed") & (~base["closed_known"]), "question_id"])
    subgroup_defs = {
        "closed_known": known_ids,
        "closed_unknown": unknown_ids,
    }
    if "mis_source_role" in base.columns:
        role_by_q = (
            base[base["condition"] == "mis"][["question_id", "mis_source_role"]]
            .drop_duplicates()
            .dropna()
        )
        for role, part in role_by_q.groupby("mis_source_role"):
            subgroup_defs[f"mis_source_{role}"] = set(part["question_id"])

    for conf_col in conf_cols:
        for subgroup, qids in subgroup_defs.items():
            sub = df[df["question_id"].isin(qids)]
            if sub["question_id"].nunique() < 10:
                continue
            vd = variance_decomposition_reml(sub, conf_col)
            rows.append(
                {
                    "confidence": conf_col,
                    "subgroup": subgroup,
                    "n_questions": int(sub["question_id"].nunique()),
                    **vd,
                }
            )
    return pd.DataFrame(rows)


def irr_behavior(df: pd.DataFrame) -> pd.DataFrame:
    closed_answers = (
        df[df["condition"] == "closed"]
        .groupby(["model_name", "question_id"])["model_answer"]
        .apply(lambda s: sorted({normalize_text(v) for v in s if normalize_text(v)}))
        .reset_index(name="closed_answer_norms")
    )
    irr = df[df["condition"] == "irr"].merge(closed_answers, on=["model_name", "question_id"], how="left").copy()
    if irr.empty:
        return pd.DataFrame()

    def classify(row: pd.Series) -> str:
        answer = str(row.get("model_answer", "") or "")
        answer_norm = normalize_text(answer)
        closed_norms = row.get("closed_answer_norms") or []
        if bool(row.get("is_abstain", False)) or answer_norm in {"", "unknown", "i don t know"}:
            return "abstain"
        if answer_norm in closed_norms or bool(row.get("correct", False)):
            return "fallback_to_prior_or_gold"
        if answer_norm and answer_in_text(str(row.get("context", "") or ""), [answer]):
            return "answer_from_irrelevant_context"
        return "other_or_hallucinated"

    irr["irr_behavior"] = irr.apply(classify, axis=1)
    sample = (
        irr.groupby(["model_name", "irr_behavior"], as_index=False)
        .agg(n_samples=("question_id", "size"), acc=("correct", "mean"))
    )
    sample["sample_rate"] = sample["n_samples"] / sample.groupby("model_name")["n_samples"].transform("sum")
    item = (
        irr.groupby(["model_name", "question_id", "irr_behavior"], as_index=False)
        .size()
        .sort_values(["model_name", "question_id", "size"], ascending=[True, True, False])
        .drop_duplicates(["model_name", "question_id"])
        .groupby(["model_name", "irr_behavior"], as_index=False)
        .agg(n_questions=("question_id", "nunique"))
    )
    item["question_rate"] = item["n_questions"] / item.groupby("model_name")["n_questions"].transform("sum")
    return sample.merge(item, on=["model_name", "irr_behavior"], how="outer").fillna(0)


def write_report(out_dir: Path, outputs: dict[str, pd.DataFrame], conf_cols: list[str]) -> None:
    primary = PRIMARY_CONF if PRIMARY_CONF in conf_cols else conf_cols[0]
    lines = [
        "# Phase 2.5 A-Group Analysis",
        "",
        f"Primary confidence: `{primary}` ({CONF_LABELS.get(primary, primary)})",
        "",
        "## A1 Closed-Known Gate",
        "",
    ]
    a1 = outputs["closed_strata"]
    primary_a1 = a1[a1["confidence"] == primary]
    lines.append("| subset | condition | acc | conf | Δacc | Δconf | FG | n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in primary_a1.to_dict("records"):
        lines.append(
            f"| {row['closed_status']} | {row['condition']} | {_fmt(row['acc'])} | {_fmt(row['conf'])} | "
            f"{_fmt(row['delta_acc'])} | {_fmt(row['delta_conf'])} | {_fmt(row['FG'])} | {int(row['n_questions'])} |"
        )
    lines.extend(["", "Flip diagnostics:", ""])
    flips = outputs["flip"]
    for row in flips[flips["confidence"] == primary].to_dict("records"):
        lines.append(
            f"- closed_known n={int(row['closed_known_questions'])}; "
            f"flip_to_mostly_wrong={_fmt(row['flip_to_mostly_wrong_rate'])}; "
            f"mean_mis_error_rate={_fmt(row['mean_mis_error_rate'])}; "
            f"mean_conf_flipped={_fmt(row['mean_conf_flipped'])}."
        )

    lines.extend(["", "## A2 Mis Source Split", ""])
    a2 = outputs["mis_source"]
    lines.append("| source | group | acc | conf | Δacc | Δconf | FG | n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in a2[a2["confidence"] == primary].to_dict("records"):
        lines.append(
            f"| {row['mis_source_role']} | {row['mis_source_group']} | {_fmt(row['acc'])} | "
            f"{_fmt(row['conf'])} | {_fmt(row['delta_acc'])} | {_fmt(row['delta_conf'])} | "
            f"{_fmt(row['FG'])} | {int(row['n_questions'])} |"
        )

    lines.extend(["", "## A3 Popularity Axis", ""])
    a3 = outputs["popularity"]
    lines.append("| popularity | condition | acc | conf | n |")
    lines.append("|---|---|---:|---:|---:|")
    for row in a3[a3["confidence"] == primary].sort_values(["popularity_bin", "condition"]).to_dict("records"):
        lines.append(
            f"| {row['popularity_bin']} | {row['condition']} | {_fmt(row['acc'])} | "
            f"{_fmt(row['conf'])} | {int(row['n_questions'])} |"
        )

    lines.extend(["", "## A4 Surface Confounding", ""])
    contrast = outputs["surface_contrast"]
    lines.append("| confidence | mis_vs_sup_coef_controlled | p | n | r2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in contrast.to_dict("records"):
        lines.append(
            f"| {row['confidence']} | {_fmt(row.get('is_mis_coef'))} | {_fmt(row.get('is_mis_p'))} | "
            f"{int(row['n'])} | {_fmt(row['r2'])} |"
        )

    lines.extend(["", "## A6 Rho Subgroups", ""])
    rho = outputs["rho"]
    lines.append("| subgroup | rho_P | rho_E | rho_PE | rho_eps | estimator | n |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for row in rho[rho["confidence"] == primary].to_dict("records"):
        lines.append(
            f"| {row['subgroup']} | {_fmt(row.get('rho_P'))} | {_fmt(row.get('rho_E'))} | "
            f"{_fmt(row.get('rho_PE'))} | {_fmt(row.get('rho_eps'))} | {row.get('estimator')} | "
            f"{int(row['n_questions'])} |"
        )

    lines.extend(["", "## A7 IRR Behavior", ""])
    irr = outputs["irr_behavior"]
    lines.append("| behavior | sample_rate | question_rate | acc | samples | questions |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in irr.to_dict("records"):
        lines.append(
            f"| {row['irr_behavior']} | {_fmt(row['sample_rate'])} | {_fmt(row['question_rate'])} | "
            f"{_fmt(row['acc'])} | {int(row['n_samples'])} | {int(row['n_questions'])} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `a1_closed_strata.csv`",
            "- `a1_flip_diagnostics.csv`",
            "- `a2_mis_source_split.csv`",
            "- `a3_popularity_axis.csv`",
            "- `a4_surface_within_condition.csv`",
            "- `a4_surface_mis_sup_controlled.csv`",
            "- `a6_rho_subgroups.csv`",
            "- `a7_irr_behavior.csv`",
        ]
    )
    (out_dir / "phase25_a_group_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2.5 A-group analyses from the single-source spec.")
    parser.add_argument(
        "--predictions",
        default="runs/phase2_unique_with_irr/predictions_qwen_full.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/phase2_unique_with_irr/analysis_phase25_qwen",
    )
    parser.add_argument(
        "--context-metadata",
        nargs="*",
        default=[
            "runs/phase2_unique/contexts_phase2.jsonl",
            "runs/phase2_unique_irr/contexts_phase2.jsonl",
        ],
        help="Context JSONL files used only to restore source-role metadata missing from merged predictions.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = enrich_prediction_metadata(load_predictions(args.predictions), args.context_metadata)
    conf_cols = _ordered_conf_cols(df)
    if not conf_cols:
        raise ValueError("No confidence columns found.")

    closed_strata, flip = closed_strata_summary(df, conf_cols)
    mis_source = mis_source_summary(df, conf_cols)
    pop = popularity_summary(df, conf_cols)
    surface_within, surface_contrast = surface_regressions(df, conf_cols)
    rho = rho_subgroups(df, conf_cols)
    irr = irr_behavior(df)

    outputs = {
        "closed_strata": closed_strata,
        "flip": flip,
        "mis_source": mis_source,
        "popularity": pop,
        "surface_within": surface_within,
        "surface_contrast": surface_contrast,
        "rho": rho,
        "irr_behavior": irr,
    }
    file_names = {
        "closed_strata": "a1_closed_strata.csv",
        "flip": "a1_flip_diagnostics.csv",
        "mis_source": "a2_mis_source_split.csv",
        "popularity": "a3_popularity_axis.csv",
        "surface_within": "a4_surface_within_condition.csv",
        "surface_contrast": "a4_surface_mis_sup_controlled.csv",
        "rho": "a6_rho_subgroups.csv",
        "irr_behavior": "a7_irr_behavior.csv",
    }
    for key, table in outputs.items():
        table.to_csv(out_dir / file_names[key], index=False)

    metadata = {
        "predictions": str(args.predictions),
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "confidence_columns": conf_cols,
        "primary_confidence": PRIMARY_CONF if PRIMARY_CONF in conf_cols else conf_cols[0],
    }
    (out_dir / "phase25_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, outputs, conf_cols)
    print(json.dumps({"out_dir": str(out_dir), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
