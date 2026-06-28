#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_python() -> str:
    candidates = [sys.executable]
    for name in ("python", "python3"):
        path = shutil.which(name)
        if path and path not in candidates:
            candidates.append(path)
    probe = "import datasets, pandas, huggingface_hub"
    for path in candidates:
        result = subprocess.run(
            [path, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return path
    return sys.executable


def compile_python() -> None:
    for path in list((ROOT / "ragcalib").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def smoke_prepare() -> dict:
    with tempfile.TemporaryDirectory(prefix="ragcalib_validate_") as tmp:
        cmd = [
            find_python(),
            str(ROOT / "scripts" / "prepare_phase2_data.py"),
            "--out-dir",
            tmp,
            "--n-total",
            "9",
            "--conflict-file",
            "conflictQA-popQA-qwen7b.json",
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        diag_path = Path(tmp) / "data_diagnostics.json"
        ctx_path = Path(tmp) / "contexts_phase2.jsonl"
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        n_contexts = sum(1 for _ in ctx_path.open("r", encoding="utf-8"))
        if n_contexts != diag["n_selected"] * 3:
            raise AssertionError(f"Expected 3 contexts per question, got {n_contexts}")
        return {"diagnostics": diag, "n_contexts": n_contexts}


def main() -> None:
    compile_python()
    smoke = smoke_prepare()
    print(json.dumps({"status": "ok", **smoke}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
