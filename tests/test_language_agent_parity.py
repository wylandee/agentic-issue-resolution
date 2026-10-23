"""Tests for language-selected agent evidence and package-manager surfaces."""

from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.qa_evaluator import _qa_log_records_for_phase
from remediation_engine.orchestration.qa_types import _QAExecutionResults
from remediation_engine.orchestration.workaround_subagent import (
    _build_workaround_system_prompt,
    _default_no_fix_package_manager,
)


def test_python_no_fix_package_manager_is_inferred_from_supported_manifests() -> None:
    assert (
        _default_no_fix_package_manager(ProjectLanguage.PYTHON, ["requirements-dev.txt"]) == "pip"
    )
    assert _default_no_fix_package_manager(ProjectLanguage.PYTHON, ["Pipfile"]) == "pipenv"
    assert (
        _default_no_fix_package_manager(ProjectLanguage.PYTHON, ["requirements.txt", "Pipfile"])
        == ""
    )
    assert _default_no_fix_package_manager(ProjectLanguage.NODEJS, []) == "npm"


def test_qa_fallback_log_labels_do_not_assume_npm() -> None:
    results = _QAExecutionResults(
        install=(False, "pip install failed"),
        tests=(False, "pytest failed"),
    )

    install = _qa_log_records_for_phase(results, "install")
    tests = _qa_log_records_for_phase(results, "tests")

    assert install[0].label == "dependency installation"
    assert tests[0].label == "unit tests"


def test_python_workaround_prompt_reuses_the_shared_safety_contract() -> None:
    prompt = " ".join(_build_workaround_system_prompt(ProjectLanguage.PYTHON).split())

    assert "read_repository_map is paged by default" in prompt
    assert "Follow Investigate -> Plan -> Execute -> Validate" in prompt
    assert "CODE_FAILURE rolls back the whole pending set" in prompt
    assert "validate_python_workaround" in prompt
    assert "remove_no_fix_python_dependency" in prompt
    assert "validate_workaround" not in prompt
    assert "remove_no_fix_dependency" not in prompt
    assert "pytest" in prompt
