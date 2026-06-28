from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    s = str(text).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def parse_answer_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        return [stripped]
    return [str(value)]


def answer_in_text(text: str, answers: Iterable[str]) -> bool:
    norm_text = normalize_text(text)
    for answer in answers:
        norm_answer = normalize_text(answer)
        if not norm_answer:
            continue
        if re.search(rf"(?<!\w){re.escape(norm_answer)}(?!\w)", norm_text):
            return True
    return False


def exact_match_any(prediction: str, answers: Iterable[str]) -> bool:
    pred_norm = normalize_text(prediction)
    return any(pred_norm == normalize_text(a) for a in answers if normalize_text(a))


def token_len(text: str) -> int:
    return len(normalize_text(text).split())


def entity_count(text: str) -> int:
    return len(set(_ENTITY_RE.findall(text or "")))


def extract_entities(text: str) -> set[str]:
    return {normalize_text(entity) for entity in _ENTITY_RE.findall(text or "") if normalize_text(entity)}


def safe_ratio(numer: float, denom: float) -> float | None:
    if denom == 0 or math.isnan(denom):
        return None
    return numer / denom


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))
