from __future__ import annotations

import json
import re
from typing import Any

from walmart_genai.core.llm import get_client, get_model
from walmart_genai.core.schema import GenAIResponse

# -------------------------
# Prompt A: Base (weak constraints)
# -------------------------
BASE_SYSTEM = """You are a helpful retail assistant for Walmart.
Answer the user's question as best as you can.
"""

# -------------------------
# Prompt B: Structured (engineering contract)
# RAG-aware: takes CONTEXT, but for now we pass NO_CONTEXT
# -------------------------
STRUCTURED_SYSTEM = """You are a Walmart retail assistant.

You must follow these rules:
1) Treat the provided CONTEXT as the only source of truth for factual claims.
2) If the CONTEXT does not contain enough evidence to answer, do NOT guess.
   Instead set next_action="ask_clarification" and explain what information you need.
3) Return ONLY valid JSON that matches the schema below. No markdown, no extra text.
4) Every factual claim must be supported by at least one citation quote from CONTEXT.
5) If user asks for policy that isn't in CONTEXT, refuse to invent it.

Output JSON schema:
{
  "answer": string,
  "citations": [{"source": string, "quote": string}],
  "intent": "product_info|delivery|return|store_info|other",
  "entities": {"product_id": string|null, "store_id": string|null},
  "confidence": number between 0 and 1,
  "next_action": "respond|ask_clarification|handoff_human"
}

Uncertainty handling:
- If uncertain due to missing evidence: next_action="ask_clarification" and confidence <= 0.4
- If request requires human handling (e.g. payment/account access): next_action="handoff_human"
"""

USER_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}
"""

# One retry keeps reliability high without blowing up costs.
MAX_RETRIES = 1

# Extract the first JSON object from output (in case model adds extra text).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    m = _JSON_OBJECT_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0).strip()


def _parse_json(text: str) -> dict[str, Any]:
    candidate = _extract_json_object(text)
    return json.loads(candidate)


def _call_llm(system: str, user_text: str) -> str:
    client = get_client()
    model = get_model()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    )
    return resp.output_text


def _call_llm_retry_json(system: str, user_text: str, reason: str) -> str:
    """
    One-shot repair call: explicitly instruct model to output JSON only.
    This is used when JSON parse or schema validation fails.
    """
    client = get_client()
    model = get_model()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
            {
                "role": "user",
                "content": (
                    f"Your previous output failed because: {reason}\n"
                    "Now output ONLY a single valid JSON object that matches the schema exactly. "
                    "No markdown, no explanation, no extra text."
                ),
            },
        ],
    )
    return resp.output_text


def _base_mode_fallback(reason: str) -> GenAIResponse:
    """
    Base prompt is intentionally NOT a strict contract.
    If it violates schema, we return a controlled response so demos/scripts don't crash.
    """
    return GenAIResponse(
        answer=f"Base prompt produced non-contract-compliant output. ({reason})",
        citations=[],
        intent="other",
        entities={"product_id": None, "store_id": None},
        confidence=0.0,
        next_action="ask_clarification",
    )


def answer_question(question: str, mode: str = "structured") -> GenAIResponse:
    """
    Day 2 Step 1+2:
    - base vs structured prompt (mode)
    - RAG-aware prompt interface (CONTEXT placeholder)
    - robustness: JSON extraction + retry
    - strict schema for structured mode; controlled fallback for base mode
    """
    # Day 2: No RAG yet, but keep the interface ready.
    context = "NO_CONTEXT"
    user_text = USER_TEMPLATE.format(question=question, context=context)

    system_prompt = BASE_SYSTEM if mode == "base" else STRUCTURED_SYSTEM

    raw = _call_llm(system_prompt, user_text)

    # 1) JSON parse (+ optional retry)
    data: dict[str, Any]
    last_err: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = _parse_json(raw)
            break
        except Exception as e:
            last_err = f"JSON parsing error: {type(e).__name__}: {e}"
            if attempt >= MAX_RETRIES:
                if mode == "base":
                    return _base_mode_fallback(last_err)
                raise
            raw = _call_llm_retry_json(system_prompt, user_text, reason=last_err)
    else:
        # Should never hit
        if mode == "base":
            return _base_mode_fallback(last_err or "Unknown JSON parsing error")
        raise RuntimeError(last_err or "Unknown JSON parsing error")

    # 2) Schema validation (+ optional retry for structured mode)
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return GenAIResponse.model_validate(data)
        except Exception as e:
            last_err = f"Schema validation error: {type(e).__name__}: {e}"
            if mode == "base":
                return _base_mode_fallback(last_err)
            if attempt >= MAX_RETRIES:
                raise
            raw = _call_llm_retry_json(system_prompt, user_text, reason=last_err)
            data = _parse_json(raw)

    # Should never hit
    if mode == "base":
        return _base_mode_fallback(last_err or "Unknown schema error")
    raise RuntimeError(last_err or "Unknown schema error")
