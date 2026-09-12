"""Tests for QA evaluation-judge prompt calibration."""

from tests.evals.test_qa_critic_eval import build_qa_production_prompt


def test_qa_judge_prompt_calibrates_negative_audit_verdicts() -> None:
    """Explain that a correct remediation rejection completes the audit task."""
    prompt = build_qa_production_prompt(
        {
            "completion_task": "Review the install failure and report the audit verdict.",
            "expected_qa_verdict": {
                "passed": False,
                "failure_category": "peer_conflict",
            },
            "expected_output": (
                "Complete: query install and emit passed=false with peer-conflict guidance."
            ),
        }
    )

    assert prompt.index("=== FEW-SHOT CALIBRATION ===") < prompt.index("=== QA CRITIC TASK ===")
    assert "A correct `passed=false` result is a completed audit" in prompt
    assert '"passed": false, "failure_category": "peer_conflict"' in prompt
    assert "Review the install failure and report the audit verdict." in prompt
