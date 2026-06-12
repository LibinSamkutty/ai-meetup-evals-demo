# models.py — Operation Blackout
from __future__ import annotations

import os
import time

import streamlit as st  # type: ignore
from openai import OpenAI  # type: ignore

import evaluator
from config import JUDGE_MODEL_ID, PERSONA_MODEL_ID

# ---------------------------------------------------------------------------
# OpenAI initialisation (call once at startup)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def init_openai() -> None:
    """Verify OPENAI_API_KEY is configured and prime a shared client."""
    global _client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    _client = OpenAI(api_key=api_key)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        init_openai()
    return _client  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Prompt template (shared by persona + benchmark calls)
# ---------------------------------------------------------------------------

PERSONA_PROMPT_TEMPLATE = """\
You are investigating the Nilgiri Taj painting theft at MIFA Museum — case NTT-2026-001.
The painting was stolen on the night of 15–16 March 2026. Today's date is June 20, 2026.

CASE EVIDENCE (retrieved from investigation documents):
{context}

INVESTIGATION QUESTION:
{query}

Respond as your investigator persona. Base your analysis on the evidence provided above.\
"""


# ---------------------------------------------------------------------------
# Persona model call
# ---------------------------------------------------------------------------

def call_persona(
    system_prompt: str,
    query: str,
    context: str,
    *,
    persona_key: str = None,
    question_id: str = None,
) -> dict:
    """Call OpenAI for a persona response.
    Returns: {"text": str, "latency_ms": int, "error": str|None}
    """
    return _call_openai_persona(system_prompt, query, context)


def _call_openai_persona(system_prompt: str, query: str, context: str) -> dict:
    start = time.time()
    try:
        _mp = st.session_state.get("model_params", {})
        client = _get_client()
        prompt = PERSONA_PROMPT_TEMPLATE.format(context=context, query=query)
        response = client.chat.completions.create(
            model=PERSONA_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=_mp.get("temperature", 0.7),
            top_p=_mp.get("top_p", 0.95),
            max_tokens=512,
        )
        latency = int((time.time() - start) * 1000)
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else ""
        truncated = (choice.finish_reason == "length") if choice else False
        return {
            "text": text or "",
            "latency_ms": latency,
            "error": None,
            "truncated": truncated,
        }
    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        return {"text": "", "latency_ms": latency, "error": str(exc)}


# ---------------------------------------------------------------------------
# Judge eval calls — NOVA (no gold answer) and VERA (gold answer provided)
# ---------------------------------------------------------------------------

def _call_llm_judge(
    query: str,
    response: str,
    context: str,
    prior_responses: list | None = None,
) -> dict:
    """NOVA judge call — evaluates against retrieved evidence only."""
    prompt = evaluator.JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
        response=response,
        prior_runs_section=evaluator._build_prior_runs_section(prior_responses),
    )
    raw = call_judge(prompt)
    if raw["error"]:
        return {"error": raw["error"], "raw": ""}
    return evaluator.parse_judge_json(raw["text"])


def _call_vera_llm_judge(
    query: str,
    response: str,
    context: str,
    gold_answer: str,
    prior_responses: list | None = None,
) -> dict:
    """VERA judge call — evaluates against gold answer + retrieved evidence."""
    prompt = evaluator.VERA_JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
        gold_answer=gold_answer,
        response=response,
        prior_runs_section=evaluator._build_prior_runs_section(prior_responses),
    )
    raw = call_judge(prompt)
    if raw["error"]:
        return {"error": raw["error"], "raw": ""}
    return evaluator.parse_judge_json(raw["text"])


def call_judge_eval(
    query: str,
    response: str,
    context: str,
    *,
    persona_key: str = None,
    question_id: str = None,
    chunks: list[dict] | None = None,
    prior_responses: list | None = None,
) -> dict:
    """NOVA judge evaluation (LLM, no gold answer). Calls OpenAI."""
    return _call_llm_judge(query, response, context, prior_responses)


def call_vera_eval(
    query: str,
    response: str,
    context: str,
    *,
    gold_answer: str = "",
    persona_key: str = None,
    question_id: str = None,
    chunks: list[dict] | None = None,
    prior_responses: list | None = None,
) -> dict:
    """VERA judge evaluation (LLM, gold answer provided).
    Uses gold answer to verify factual correctness; evidence for other dimensions.
    """
    return _call_vera_llm_judge(
        query, response, context, gold_answer, prior_responses
    )


def call_judge(judge_prompt: str) -> dict:
    """Raw OpenAI judge call. Returns {"text": str, "error": str|None}."""
    try:
        client = _get_client()
        message = client.chat.completions.create(
            model=JUDGE_MODEL_ID,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        return {"text": message.choices[0].message.content or "", "error": None}
    except Exception as exc:
        return {"text": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Consistency judge call
# ---------------------------------------------------------------------------

def call_consistency_judge(
    query: str,
    runs: list[dict],
    context: str,
) -> dict:
    """Evaluate consistency across 3 self-consistency runs of the same persona."""
    texts = [r.get("text", "") for r in runs]
    if not all(texts):
        return {"error": "One or more runs returned an empty response."}

    mode_intro = (
        "The same investigation question was sent to the same Athena "
        "configuration three times. Your task: determine whether the three "
        "responses are consistent with each other."
    )
    prompt = evaluator.CONSISTENCY_JUDGE_PROMPT_TEMPLATE.format(
        mode_intro=mode_intro,
        phrasings_block="",
        query=query,
        context=context,
        response_1=texts[0],
        response_2=texts[1],
        response_3=texts[2],
    )
    raw = call_judge(prompt)
    if raw["error"]:
        return {"error": raw["error"]}
    return evaluator.parse_consistency_judge_json(raw["text"])


# ---------------------------------------------------------------------------
# Optional OpenAI availability check
# ---------------------------------------------------------------------------

def openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Benchmark model call — used by Golden Dataset Lab batch eval
# ---------------------------------------------------------------------------

def call_benchmark_model(
    model_id: str,
    provider: str,
    query: str,
    context: str,
    system_prompt: str,
    max_output_tokens: int = 1024,
    thinking_budget: int | None = None,
) -> dict:
    """
    Fire a single benchmark call against a specific OpenAI model.
    provider: "openai" (only supported provider after the OpenAI-only migration)
    Returns: {"text": str, "latency_ms": int, "error": str|None}
    """
    start = time.time()
    try:
        if provider != "openai":
            raise ValueError(
                f"Unsupported provider: {provider}. Only 'openai' is supported."
            )

        client = _get_client()
        user_content = PERSONA_PROMPT_TEMPLATE.format(context=context, query=query)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_output_tokens,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""

        latency = int((time.time() - start) * 1000)
        return {"text": text, "latency_ms": latency, "error": None}

    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        return {"text": "", "latency_ms": latency, "error": str(exc)}
