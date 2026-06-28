#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOOL_COLS = ["supports_target", "contradicts_gold", "explicit_answer_anchor", "parse_ok"]
SCORE_COLS = ["naturalness", "template_likeness", "misleading_plausibility"]


def read_jsonl(path: str | Path, judge_name: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    df = pd.DataFrame(rows)
    df["judge"] = judge_name
    return df


def to_bool_num(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return np.nan


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in BOOL_COLS:
        work[col] = work[col].map(to_bool_num)
    for col in SCORE_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    rows = []
    for (judge, condition), group in work.groupby(["judge", "hidden_condition"], sort=False):
        row = {
            "judge": judge,
            "hidden_condition": condition,
            "n": int(len(group)),
        }
        for col in BOOL_COLS:
            row[f"{col}_rate"] = float(group[col].mean())
        for col in SCORE_COLS:
            row[f"{col}_mean"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def agreement(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in BOOL_COLS:
        work[col] = work[col].map(to_bool_num)
    for col in SCORE_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    rows = []
    judges = sorted(work["judge"].dropna().unique())
    if len(judges) != 2:
        return pd.DataFrame(rows)
    left = work[work["judge"] == judges[0]].set_index("audit_id")
    right = work[work["judge"] == judges[1]].set_index("audit_id")
    joined = left.join(right, lsuffix="_left", rsuffix="_right", how="inner")
    for col in BOOL_COLS:
        a = joined[f"{col}_left"]
        b = joined[f"{col}_right"]
        mask = a.notna() & b.notna()
        rows.append(
            {
                "metric": col,
                "n_overlap": int(mask.sum()),
                "agreement": float((a[mask] == b[mask]).mean()) if mask.any() else np.nan,
                "judge_left": judges[0],
                "judge_right": judges[1],
            }
        )
    for col in SCORE_COLS:
        a = joined[f"{col}_left"]
        b = joined[f"{col}_right"]
        mask = a.notna() & b.notna()
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 2 else np.nan
        rows.append(
            {
                "metric": col,
                "n_overlap": int(mask.sum()),
                "agreement": corr,
                "judge_left": judges[0],
                "judge_right": judges[1],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize blind stimulus audit judge outputs.")
    parser.add_argument("--judge", action="append", required=True, help="name=path.jsonl")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    frames = []
    for item in args.judge:
        name, path = item.split("=", 1)
        frames.append(read_jsonl(path, name))
    df = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "part_d3_blind_audit_judgments.csv", index=False)
    summary = summarize(df)
    agree = agreement(df)
    summary.to_csv(out_dir / "part_d3_blind_audit_summary.csv", index=False)
    agree.to_csv(out_dir / "part_d3_blind_audit_agreement.csv", index=False)
    print(json.dumps({"out_dir": str(out_dir), "judgments": len(df)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
