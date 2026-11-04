"""LLM step 2 — turn the already-made decision into a readable explanation.

The explanation is prose ONLY. The decision, reasons and numbers are fixed by
the deterministic engine before this runs. The prompt forbids the model from
changing the verdict or introducing facts not present in the supplied context —
this is the project's main anti-hallucination guardrail.

If no API key is configured, a deterministic template explanation is produced so
the system is always fully functional without an LLM.
"""

from __future__ import annotations

from .gemini_client import generate_text, is_available
from .schemas import Decision, DecisionType, Reason

# Final prompt (see docs/prompt-engineering.md for V1 -> V2 evolution).
EXPLANATION_PROMPT = """You are the explanation component of a loan-screening \
system. A deterministic rule engine has ALREADY made the decision below. Your \
only job is to explain it clearly to a loan officer in 2-4 sentences.

ABSOLUTE RULES:
- Do NOT change, question, or second-guess the decision.
- Use ONLY the facts and rule results provided. Do not introduce any number, \
policy, or fact that is not listed here.
- Do not give financial advice or predict approval odds.
- Write plainly and neutrally.

DECISION: {decision}

RULE RESULTS:
{reasons}

MISSING INFORMATION: {missing}

Write the explanation now."""


def _format_reasons(reasons: list[Reason]) -> str:
    return "\n".join(
        f"- [{r.rule_id}] {r.description}: {'PASS' if r.passed else 'FAIL'} — {r.detail}"
        for r in reasons
    )


def _template_explanation(decision: Decision) -> str:
    """Deterministic fallback used when the LLM is unavailable."""
    failed = [r for r in decision.reasons if not r.passed]
    passed = [r for r in decision.reasons if r.passed]
    verb = {
        DecisionType.APPROVE: "meets all policy requirements and is recommended for approval",
        DecisionType.REJECT: "fails one or more mandatory policy requirements and is recommended for rejection",
        DecisionType.REQUIRES_REVIEW: "requires manual review before a final decision",
        DecisionType.INCOMPLETE: "cannot be assessed because required information is missing",
    }[decision.decision]
    parts = [f"This application {verb}."]
    if failed:
        parts.append("Issues: " + "; ".join(r.detail for r in failed) + ".")
    if decision.missing_information:
        parts.append("Missing: " + ", ".join(decision.missing_information) + ".")
    if decision.decision == DecisionType.APPROVE and passed:
        parts.append(f"All {len(passed)} policy checks passed.")
    return " ".join(parts)


def explain(decision: Decision) -> str:
    """Return a natural-language explanation of the fixed decision."""
    if not is_available():
        return _template_explanation(decision)

    prompt = EXPLANATION_PROMPT.format(
        decision=decision.decision.value,
        reasons=_format_reasons(decision.reasons),
        missing=", ".join(decision.missing_information) or "none",
    )
    text = generate_text(prompt)
    return text if text else _template_explanation(decision)
