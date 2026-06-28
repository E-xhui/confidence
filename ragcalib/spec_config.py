"""Centralized decisions copied from the provided single-source spec."""

from __future__ import annotations

from dataclasses import dataclass


SPEC_TITLE = "RAG Calibration 实验 · 单一事实来源 spec.md"


@dataclass(frozen=True)
class Phase2Spec:
    main_models: tuple[str, str] = (
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    )
    dataset_popqa: str = "akariasai/PopQA"
    dataset_conflictqa: str = "osunlp/ConflictQA"
    phase2_conditions: tuple[str, str, str] = ("closed", "sup", "mis")
    evidence_conditions_for_rho: tuple[str, ...] = ("sup", "mis", "irr")
    n_total: int = 1500
    n_per_popularity_bin: int = 500
    popularity_bins: tuple[str, str, str] = ("low", "mid", "high")
    k_samples: int = 5
    temperature: float = 0.7
    max_new_answer_tokens: int = 32
    max_new_conf_tokens: int = 8
    random_seed: int = 20260618
    ece_bins: int = 10
    logit_eps: float = 1e-4
    abstain_markers: tuple[str, ...] = (
        "i don't know",
        "cannot answer",
        "can't answer",
        "unknown",
        "not enough information",
    )


SPEC = Phase2Spec()

