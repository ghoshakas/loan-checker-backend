"""Evaluation harness.

Runs every labeled case through the SAME pipeline the API uses and reports:
  * Classification accuracy      (decision vs expected)
  * Information-extraction acc.  (LLM-extracted facts vs ground truth)  [needs key]
  * Valid structured output rate (Decision validates as the JSON contract)
  * Output consistency           (same input x3 -> same decision)
  * Hallucination rate           (explanation asserts a different verdict) [needs key]

Cases tagged requires_llm=true are only scored when GEMINI_API_KEY is set;
otherwise they are skipped and reported as such, so the deterministic metrics are
always runnable offline.

Run:  cd backend && python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from app.gemini_client import is_available  # noqa: E402
from app.pipeline import screen  # noqa: E402
from app.schemas import Decision, LoanApplication  # noqa: E402

CASES_PATH = Path(__file__).parent / "test_cases.json"
LLM = is_available()


def _load_cases() -> list[dict]:
    with open(CASES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def _extraction_score(decision: Decision, expected: dict) -> tuple[int, int]:
    """Return (matched, total) over the expected extracted fields."""
    matched = 0
    for field, exp in expected.items():
        got = getattr(decision.facts_used, field, None)
        if got is None:
            continue
        if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
            if abs(float(got) - float(exp)) < 1e-6:
                matched += 1
        elif str(got).strip().lower() == str(exp).strip().lower():
            matched += 1
    return matched, len(expected)


def _hallucinated(decision: Decision) -> bool:
    """Lightweight guard: the explanation must not assert a verdict different
    from the engine's. Looks for a contradicting decision keyword."""
    text = decision.explanation.lower()
    verdict = decision.decision.value
    contradictions = {
        "APPROVE": ["reject", "denied", "decline"],
        "REJECT": ["approved", "approve the", "recommended for approval"],
        "REQUIRES_REVIEW": [],
        "INCOMPLETE": [],
    }
    return any(bad in text for bad in contradictions.get(verdict, []))


def main() -> None:
    cases = _load_cases()
    print(f"\nLLM available: {LLM}  (requires_llm cases are "
          f"{'scored' if LLM else 'SKIPPED'})\n")

    cls_total = cls_correct = 0
    valid_total = valid_ok = 0
    ext_matched = ext_total = 0
    halluc_checked = halluc_bad = 0
    skipped = 0

    header = f"{'ID':<5}{'CATEGORY':<12}{'EXPECTED':<18}{'GOT':<18}{'OK':<4}"
    print(header)
    print("-" * len(header))

    for c in cases:
        if c["requires_llm"] and not LLM:
            skipped += 1
            print(f"{c['id']:<5}{c['category']:<12}{c['expected_decision']:<18}"
                  f"{'(skipped: no key)':<18}{'-':<4}")
            continue

        app = LoanApplication(**c["input"])
        decision = screen(app, with_explanation=LLM, with_insights=False)

        # Valid structured output: re-validate against the contract.
        try:
            Decision.model_validate(decision.model_dump())
            valid_ok += 1
        except Exception:
            pass
        valid_total += 1

        got = decision.decision.value
        ok = got == c["expected_decision"]
        cls_total += 1
        cls_correct += int(ok)

        if "expected_facts" in c and LLM:
            m, t = _extraction_score(decision, c["expected_facts"])
            ext_matched += m
            ext_total += t

        if LLM and decision.explanation:
            halluc_checked += 1
            halluc_bad += int(_hallucinated(decision))

        print(f"{c['id']:<5}{c['category']:<12}{c['expected_decision']:<18}"
              f"{got:<18}{'yes' if ok else 'NO':<4}")

    print("\n" + "=" * 48)
    print("SUMMARY METRICS")
    print("=" * 48)
    if cls_total:
        print(f"Classification accuracy : {cls_correct}/{cls_total} "
              f"= {cls_correct / cls_total:.1%}")
    print(f"Valid structured output : {valid_ok}/{valid_total} "
          f"= {valid_ok / valid_total:.1%}" if valid_total else "n/a")
    if ext_total:
        print(f"Extraction accuracy     : {ext_matched}/{ext_total} "
              f"= {ext_matched / ext_total:.1%}")
    else:
        print("Extraction accuracy     : n/a (needs GEMINI_API_KEY)")
    if halluc_checked:
        print(f"Hallucination rate      : {halluc_bad}/{halluc_checked} "
              f"= {halluc_bad / halluc_checked:.1%}")
    else:
        print("Hallucination rate      : n/a (needs GEMINI_API_KEY)")
    if skipped:
        print(f"Skipped (no key)        : {skipped} LLM cases")

    # Output consistency: run every offline case 3x; decisions must be identical.
    print("\nOutput consistency check (3x per offline case)...")
    inconsistent = 0
    checked = 0
    for c in cases:
        if c["requires_llm"]:
            continue
        app = LoanApplication(**c["input"])
        decisions = {screen(app, with_explanation=False).decision for _ in range(3)}
        checked += 1
        if len(decisions) != 1:
            inconsistent += 1
    print(f"Consistent decisions    : {checked - inconsistent}/{checked} "
          f"= {(checked - inconsistent) / checked:.1%}" if checked else "n/a")


if __name__ == "__main__":
    main()
