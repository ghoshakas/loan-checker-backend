"""The end-to-end screening pipeline, assembled from its stages.

    LoanApplication
        -> extract_facts        (LLM, only for free text; code validates)
        -> compute_derived_facts (code)
        -> evaluate             (code: the decision)
        -> explain              (LLM, prose only)
        -> Decision             (validated structured output)

Kept separate from the FastAPI layer so the eval harness and unit tests can call
the exact same code path the API uses.
"""

from __future__ import annotations

from .config import get_policy
from .explainer import explain
from .fact_extractor import extract_facts
from .rule_engine import compute_derived_facts, evaluate
from .schemas import Decision, LoanApplication


def screen(application: LoanApplication, with_explanation: bool = True) -> Decision:
    policy = get_policy()

    # 1. Facts: structured input + LLM extraction from free text (code-validated).
    facts = extract_facts(application)

    # 2. Derived facts: EMI, DTI, LTV, monthly income (pure math).
    facts = compute_derived_facts(facts, policy)

    # 3. Decision: deterministic rule engine.
    result = evaluate(facts, policy)

    decision = Decision(
        decision=result["decision"],
        reasons=result["reasons"],
        missing_information=result["missing_information"],
        policy_rules_used=result["policy_rules_used"],
        facts_used=facts,
        explanation="",
    )

    # 4. Explanation: LLM prose over the fixed decision (optional for eval speed).
    if with_explanation:
        decision.explanation = explain(decision)

    return decision
