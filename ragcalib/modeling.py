from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .spec_config import SPEC
from .text_utils import exact_match_any


_NUMBER_RE = re.compile(r"(?<!\d)(100|[0-9]{1,2})(?:\.\d+)?(?!\d)")


@dataclass
class ModelBundle:
    model_name: str
    tokenizer: Any
    model: Any
    device: torch.device


def load_model_bundle(
    model_name: str,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device
    return ModelBundle(model_name=model_name, tokenizer=tokenizer, model=model, device=device)


def _plain_prompt(context: str, question: str) -> str:
    if context.strip():
        return (
            "Use the context to answer the question. Give only the short answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    return f"Answer the question. Give only the short answer.\n\nQuestion: {question}\nAnswer:"


def _chat_prompt(tokenizer: Any, user_content: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You answer factual questions with a concise answer only.",
        },
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_content


def answer_prompt(tokenizer: Any, context: str, question: str) -> str:
    return _chat_prompt(tokenizer, _plain_prompt(context, question))


def confidence_prompt(tokenizer: Any, context: str, question: str, answer: str) -> str:
    context_block = f"Context:\n{context}\n\n" if context.strip() else ""
    user_content = (
        f"{context_block}Question: {question}\n"
        f"Your answer: {answer}\n\n"
        "What is the probability from 0 to 100 that your answer is correct? "
        "Reply with one number only."
    )
    return _chat_prompt(tokenizer, user_content)


def ptrue_prompt(tokenizer: Any, context: str, question: str, answer: str) -> str:
    context_block = f"Context:\n{context}\n\n" if context.strip() else ""
    user_content = (
        f"{context_block}Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        "Is the proposed answer correct? Reply True or False."
    )
    return _chat_prompt(tokenizer, user_content)


def generate_text(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    seed: int | None = None,
) -> tuple[str, list[int], list[int]]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.model.device)
    with torch.no_grad():
        output = bundle.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )
    prompt_ids = inputs["input_ids"][0].tolist()
    new_ids = output[0][len(prompt_ids) :].tolist()
    text = bundle.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, prompt_ids, new_ids


def clean_answer(text: str) -> str:
    text = text.strip()
    text = text.split("\n", 1)[0].strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text, flags=re.I).strip()
    return text.strip(" .;:")


def parse_confidence(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    value = float(match.group(0))
    if value > 1.0:
        return max(0.0, min(1.0, value / 100.0))
    return max(0.0, min(1.0, value))


def sequence_mean_logprob(bundle: ModelBundle, prompt_ids: list[int], answer_ids: list[int]) -> float | None:
    if not answer_ids:
        return None
    ids = torch.tensor([prompt_ids + answer_ids], device=bundle.model.device)
    with torch.no_grad():
        logits = bundle.model(ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = len(prompt_ids)
    vals: list[float] = []
    for pos in range(prompt_len, prompt_len + len(answer_ids)):
        vals.append(float(log_probs[pos - 1, ids[0, pos]].detach().cpu()))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def token_probability(bundle: ModelBundle, prompt: str, variants: list[str]) -> float:
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.model.device)
    with torch.no_grad():
        logits = bundle.model(**inputs).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    total = 0.0
    for variant in variants:
        ids = bundle.tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            total += float(probs[ids[0]].detach().cpu())
    return total


def ptrue_score(bundle: ModelBundle, prompt: str) -> float | None:
    p_true = token_probability(bundle, prompt, [" True", "True", " true", "true"])
    p_false = token_probability(bundle, prompt, [" False", "False", " false", "false"])
    denom = p_true + p_false
    if denom <= 0:
        return None
    return p_true / denom


def is_abstention(answer: str) -> bool:
    norm = answer.strip().lower()
    return any(marker in norm for marker in SPEC.abstain_markers)


def score_answer(answer: str, gold_answers: list[str]) -> int:
    if is_abstention(answer):
        return 0
    return int(exact_match_any(answer, gold_answers))


def seqlik_to_unit_interval(mean_logprob: float | None) -> float | None:
    if mean_logprob is None or math.isnan(mean_logprob):
        return None
    return float(max(0.0, min(1.0, math.exp(mean_logprob))))

