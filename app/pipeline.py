"""The end-to-end screening pipeline, assembled from its stages.

    LoanApplication (+ uploaded documents)
        -> build_structured_facts   (code: the form)
        -> analyze_documents        (LLM reads files; code normalises, merges, cross-checks)
        -> extract_from_free_text   (LLM, only for notes; code validates)
        -> compute_derived_facts    (code: EMI, DTI, LTV)
        -> evaluate                 (code: the RECOMMENDATION)
        -> explain                  (LLM prose over the fixed recommendation)
        -> generate_insights        (LLM advisory risk read for the officer)
        -> Decision                 (validated structured output)

The FINAL approve/reject is made by a human officer in the admin panel; this
pipeline only produces the recommendation and the evidence for it.

Kept separate from the FastAPI layer so the eval harness and unit tests call
the exact same code path the API uses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from .config import get_policy
from .document_analyzer import UploadedDocument, analyze_documents
from .explainer import explain
from .fact_extractor import build_structured_facts, extract_from_free_text
from .risk_analyst import generate_insights
from .rule_engine import compute_derived_facts, evaluate
from .schemas import Decision, LoanApplication


def screen(
    application: LoanApplication,
    documents: Sequence[UploadedDocument] = (),
    with_explanation: bool = True,
    with_insights: Optional[bool] = None,
) -> Decision:
    policy = get_policy()
    if with_insights is None:
        with_insights = with_explanation

    # 1. Facts, in fixed precedence: form > uploaded documents > free-text notes.
    facts = build_structured_facts(application)
    findings = analyze_documents(facts, list(documents), policy)
    extract_from_free_text(facts, application.free_text)

    # 2. Derived facts: EMI, DTI, LTV, monthly income (pure math).
    facts = compute_derived_facts(facts, policy)

    # 3. Recommendation: deterministic rule engine.
    result = evaluate(facts, policy)

    decision = Decision(
        decision=result["decision"],
        reasons=result["reasons"],
        missing_information=result["missing_information"],
        policy_rules_used=result["policy_rules_used"],
        facts_used=facts,
        explanation="",
        documents=findings,
    )

    # 4. LLM prose + advisory insights over the FIXED recommendation. Both only
    #    read the decision, so run them at the same time (saves one LLM round trip).
    with ThreadPoolExecutor(max_workers=2) as pool:
        explanation = pool.submit(explain, decision) if with_explanation else None
        insights = pool.submit(generate_insights, decision) if with_insights else None
        if explanation:
            decision.explanation = explanation.result()
        if insights:
            decision.insights = insights.result()

    return decision
