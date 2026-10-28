"""Offline contracts for fixture-backed end-to-end golden evaluations."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from remediation_engine.contracts.schemas import VulnerabilityIssue
from remediation_engine.evals import end_to_end as e2e
from remediation_engine.evals.end_to_end_goldens import (
    DEFAULT_CLONES_ROOT,
    DEFAULT_GOLDEN_PATH,
    SUPPRESSION_CONTAINER_PATH,
    EndToEndGolden,
    build_golden_request,
    generate_suppressions_xml,
    golden_settings,
    isolated_golden_workspace,
    load_end_to_end_goldens,
)
from remediation_engine.evals.run_evidence import RunEvidence, assess_contracts, judge_payload
from remediation_engine.language import ProjectLanguage
from remediation_engine.settings import AppSettings

_CASE_TARGETS = {
    "e2e-npm-single-update": ["@tootallnate/once"],
    "e2e-npm-batch-update": ["@angular/common", "@angular/compiler", "@angular/core"],
    "e2e-npm-transitive-parent-update": ["crypto-js"],
    "e2e-npm-code-migration": ["express-jwt"],
    "e2e-npm-no-fix-removal": ["notevil"],
    "e2e-npm-mixed-strategies": [
        "@angular/common",
        "@angular/compiler",
        "@angular/core",
        "express-jwt",
        "notevil",
    ],
    "e2e-python-single-update": ["pyyaml"],
    "e2e-python-batch-update": ["certifi", "sqlparse"],
    "e2e-java-single-update": ["org.postgresql:postgresql"],
    "e2e-java-batch-update": [
        "ch.qos.logback:logback-classic",
        "ch.qos.logback:logback-core",
    ],
}
_ANGULAR_CLOSURE = {
    "@angular/animations",
    "@angular/common",
    "@angular/compiler",
    "@angular/compiler-cli",
    "@angular/core",
    "@angular/forms",
    "@angular/platform-browser",
    "@angular/platform-browser-dynamic",
    "@angular/router",
}


def _test_golden(tmp_path: Path, ecosystem: str, package: str) -> EndToEndGolden:
    repo = tmp_path / "clones" / "test-repo"
    repo.mkdir(parents=True)
    language = {
        "npm": ProjectLanguage.NODEJS,
        "python": ProjectLanguage.PYTHON,
        "java": ProjectLanguage.JAVA,
    }[ecosystem]
    return EndToEndGolden(
        case={
            "case_id": "test-golden",
            "input": "Remediate the target.",
            "context": [],
            "expected_output": "Expected behavior.",
            "expected_tools": [],
            "live": {"ecosystem": ecosystem, "target_packages": [package]},
        },
        repo_root=repo,
        fixture_path=tmp_path / "fixture.jsonl",
        issues=(
            VulnerabilityIssue(
                source="odc",
                issue_type="sca",
                package_name=package,
                package_version="1.0",
                ecosystem=ecosystem,
            ),
        ),
        request_target_packages=(package,),
        requested_packages=(package,),
        language=language,
    )


def test_dataset_has_exact_cases_targets_versions_and_angular_closure() -> None:
    goldens = load_end_to_end_goldens()

    assert len(goldens) == 10
    assert {golden.case_id for golden in goldens} == set(_CASE_TARGETS)
    assert {golden.case_id: list(golden.requested_packages) for golden in goldens} == _CASE_TARGETS
    for golden in goldens:
        assert golden.fixture_path.is_file()
        assert golden.repo_root == DEFAULT_CLONES_ROOT / golden.live["clone_dir"]
        assert golden.issues
        assert {issue.package_name.casefold() for issue in golden.issues} == {
            package.casefold() for package in golden.requested_packages
        }
        assert {issue.ecosystem for issue in golden.issues} == {
            {"npm": "npm", "python": "pypi", "java": "maven"}[golden.live["ecosystem"]]
        }

    cases = {golden.case_id: golden.case for golden in goldens}
    assert set(cases["e2e-npm-batch-update"]["expected_closure"]) == _ANGULAR_CLOSURE
    assert set(cases["e2e-npm-mixed-strategies"]["expected_closure"]) == _ANGULAR_CLOSURE
    for case_id, expected_fragments in {
        "e2e-npm-single-update": ["@tootallnate/once", "2.0.1"],
        "e2e-npm-batch-update": ["21.2.17"],
        "e2e-npm-transitive-parent-update": ["crypto-js", "pdfkit", "0.12.2"],
        "e2e-npm-code-migration": ["express-jwt", "6.0.0", "lib/insecurity.ts"],
        "e2e-npm-no-fix-removal": ["notevil", "routes/b2bOrder.ts"],
        "e2e-npm-mixed-strategies": ["21.2.17", "6.0.0", "notevil", "lib/insecurity.ts"],
        "e2e-python-single-update": ["PyYAML", "5.4"],
        "e2e-python-batch-update": ["certifi", "2024.7.4", "sqlparse", "0.4.4"],
        "e2e-java-single-update": ["org.postgresql:postgresql", "42.7.11"],
        "e2e-java-batch-update": ["logback-classic", "logback-core", "1.2.13"],
    }.items():
        expected = cases[case_id]["expected_output"]
        assert all(fragment in expected for fragment in expected_fragments)
    assert "crypto-js" not in cases["e2e-npm-mixed-strategies"]["expected_output"]


def test_case_selection_is_stable_and_rejects_unknown_ids() -> None:
    selected = load_end_to_end_goldens(case_ids=["e2e-java-single-update"])
    assert [golden.case_id for golden in selected] == ["e2e-java-single-update"]
    with pytest.raises(ValueError, match="Unknown end-to-end golden"):
        load_end_to_end_goldens(case_ids=["unknown-case"])


def test_loader_rejects_target_missing_from_its_fixture(tmp_path: Path) -> None:
    project_root = tmp_path
    golden_path = project_root / "tests" / "evals" / "golden" / "end_to_end_cases.json"
    golden_path.parent.mkdir(parents=True)
    cases = json.loads(DEFAULT_GOLDEN_PATH.read_text(encoding="utf-8"))
    for case in cases:
        relative = case["live"]["fixture_path"]
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_GOLDEN_PATH.parents[3] / relative, destination)
    cases[0]["live"]["target_packages"] = ["missing-package"]
    golden_path.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match="targets are missing"):
        load_end_to_end_goldens(
            case_ids=[cases[0]["case_id"]], golden_path=golden_path, clones_root=tmp_path / "clones"
        )


@pytest.mark.parametrize(
    ("ecosystem", "package", "target_purl", "other_purl"),
    [
        (
            "npm",
            "@angular/common",
            "pkg:npm/%40angular%2Fcommon@21.2.17",
            "pkg:npm/%40angular%2Fanimations@21.2.17",
        ),
        (
            "python",
            "PyYAML",
            "pkg:pypi/pyyaml@5.4",
            "pkg:pypi/a-package-not-in-the-baseline@9.0",
        ),
        (
            "java",
            "org.postgresql:postgresql",
            "pkg:maven/org.postgresql/postgresql@42.7.11",
            "pkg:maven/org.example/newly-detected@9.0",
        ),
    ],
)
def test_suppression_rules_are_package_scoped_and_version_independent(
    tmp_path: Path, ecosystem: str, package: str, target_purl: str, other_purl: str
) -> None:
    golden = _test_golden(tmp_path, ecosystem, package)
    root = ET.fromstring(generate_suppressions_xml(golden))
    namespace = "{https://jeremylong.github.io/DependencyCheck/dependency-suppression.1.4.xsd}"
    matcher = re.compile(root.find(f".//{namespace}packageUrl").text, re.IGNORECASE)

    assert matcher.fullmatch(target_purl) is None
    assert matcher.fullmatch(other_purl)
    changed_target_version = target_purl.rsplit("@", maxsplit=1)[0] + "@999.0"
    assert matcher.fullmatch(changed_target_version) is None


def test_generated_suppression_is_installed_and_original_bytes_are_restored(tmp_path: Path) -> None:
    golden = _test_golden(tmp_path, "npm", "@tootallnate/once")
    suppression = golden.repo_root / "suppressions.xml"
    original = b"<original suppression bytes>\r\n"
    suppression.write_bytes(original)

    with isolated_golden_workspace(golden):
        assert SUPPRESSION_CONTAINER_PATH in golden_settings(AppSettings()).odc_extra_args
        assert suppression.read_text(encoding="utf-8").startswith('<?xml version="1.0"')

    assert suppression.read_bytes() == original


def test_suppression_file_is_removed_after_failure_or_interruption(tmp_path: Path) -> None:
    golden = _test_golden(tmp_path, "python", "pyyaml")
    suppression = golden.repo_root / "suppressions.xml"

    with pytest.raises(RuntimeError, match="engine failure"), isolated_golden_workspace(golden):
        raise RuntimeError("engine failure")
    assert not suppression.exists()

    with pytest.raises(KeyboardInterrupt), isolated_golden_workspace(golden):
        raise KeyboardInterrupt
    assert not suppression.exists()


def test_golden_run_restores_suppressions_after_judge_failure(monkeypatch, tmp_path: Path) -> None:
    golden = _test_golden(tmp_path, "java", "org.postgresql:postgresql")
    suppression = golden.repo_root / "suppressions.xml"
    original = b"preserve me"
    suppression.write_bytes(original)
    monkeypatch.setattr(
        e2e,
        "run_fresh",
        lambda request, *, settings: RunEvidence(
            source_run_id="source-run", request={}, final_state={}
        ),
    )
    monkeypatch.setattr(
        e2e,
        "evaluate_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("judge failed")),
    )

    with pytest.raises(RuntimeError, match="judge failed"):
        e2e.evaluate_golden(golden, settings=AppSettings())
    assert suppression.read_bytes() == original


def test_golden_run_restores_suppressions_after_engine_failure(monkeypatch, tmp_path: Path) -> None:
    golden = _test_golden(tmp_path, "npm", "express-jwt")
    suppression = golden.repo_root / "suppressions.xml"
    original = b"preserve after engine failure"
    suppression.write_bytes(original)
    monkeypatch.setattr(
        e2e,
        "run_fresh",
        lambda request, *, settings: (_ for _ in ()).throw(RuntimeError("engine failed")),
    )

    with pytest.raises(RuntimeError, match="engine failed"):
        e2e.evaluate_golden(golden, settings=AppSettings())
    assert suppression.read_bytes() == original


def test_golden_settings_preserve_models_credentials_and_scanner_options() -> None:
    configured = AppSettings(
        openai_api_key="configured-key",
        serper_api_key="serper-key",
        github_token="github-key",
        langsmith_api_key="langsmith-key",
        remedy_llm_model="base-model",
        triage_llm_model="triage-model",
        update_llm_model="update-model",
        workaround_llm_model="workaround-model",
        qa_llm_model="qa-model",
        odc_extra_args='--foo "two words" --suppression old.xml --suppression-file=other.xml --bar',
    )

    effective = golden_settings(configured)

    assert effective.openai_api_key == "configured-key"
    assert effective.serper_api_key == "serper-key"
    assert effective.github_token == "github-key"
    assert effective.langsmith_api_key == "langsmith-key"
    assert effective.remedy_llm_model == "base-model"
    assert effective.triage_llm_model == "triage-model"
    assert effective.update_llm_model == "update-model"
    assert effective.workaround_llm_model == "workaround-model"
    assert effective.qa_llm_model == "qa-model"
    assert effective.remedy_disable_post_qa_triage
    assert shlex.split(effective.odc_extra_args) == [
        "--foo",
        "two words",
        "--bar",
        "--suppression",
        SUPPRESSION_CONTAINER_PATH,
    ]
    assert "old.xml" not in effective.odc_extra_args
    assert "other.xml" not in effective.odc_extra_args
    assert "--foo" in effective.odc_extra_args and "--bar" in effective.odc_extra_args


def test_request_contains_only_fixture_targets_not_golden_expected_versions(tmp_path: Path) -> None:
    golden = _test_golden(tmp_path, "npm", "express-jwt")

    request = build_golden_request(golden)

    assert request.repo_root == golden.repo_root
    assert request.target_packages == ["express-jwt"]
    assert request.issues == list(golden.issues)
    assert request.system_context.environment == "development"
    assert "expected_output" not in request.model_dump()
    assert "6.0.0" not in json.dumps(request.model_dump(mode="json"))


def test_new_findings_stay_in_evidence_but_outside_completion_goal() -> None:
    evidence = RunEvidence(
        source_run_id="source-run",
        request={"issues": [{"cve_id": "CVE-test"}]},
        final_state={"new_vulnerability_identifiers": ["CVE-2099-1234"]},
    )

    goal, outcome, _tools, _omissions = judge_payload(evidence, e2e.assess_contracts(evidence))

    assert "CVE-test" in goal
    assert "CVE-2099-1234" not in goal
    assert "CVE-2099-1234" in outcome
    assert "New vulnerabilities discovered during remediation" in goal


def test_golden_compaction_keeps_failed_update_and_workaround_lineage() -> None:
    snapshots = {
        "initial-update": {
            "task_id": "express-task",
            "task_revision": 1,
            "attempt_number": 1,
            "strategy": "version_bump",
            "strategy_stage": "update",
        },
        "workaround": {
            "task_id": "express-task",
            "parent_task_id": "express-task",
            "task_revision": 2,
            "attempt_number": 2,
            "strategy": "workaround",
            "strategy_stage": "migration",
        },
    }
    qa_results = {
        "initial-update": {
            "attempt_id": "initial-update",
            "task_id": "express-task",
            "evaluation": {"passed": False, "failure_category": "breaking_change"},
        },
        "workaround": {
            "attempt_id": "workaround",
            "task_id": "express-task",
            "evaluation": {"passed": True},
        },
    }
    for index in range(70):
        attempt_id = f"unrelated-{index}"
        snapshots[attempt_id] = {
            "task_id": f"other-task-{index}",
            "task_revision": 1,
            "attempt_number": 1,
            "strategy": "version_bump",
        }
        qa_results[attempt_id] = {
            "attempt_id": attempt_id,
            "task_id": f"other-task-{index}",
            "evaluation": {"passed": True},
        }
    evidence = RunEvidence(
        source_run_id="large-run",
        request={},
        final_state={
            "task_queue": {
                "express-task": {
                    "task_id": "express-task",
                    "target_package_name": "express-jwt",
                    "current_attempt_id": "workaround",
                    "status": "qa_passed",
                }
            },
            "attempt_snapshots_by_id": snapshots,
            "worker_results_by_attempt": {},
            "qa_results_by_attempt": qa_results,
            "diff": "update express-jwt and apply the source migration",
        },
    )

    _goal, outcome_text, _tools, _omissions = judge_payload(
        evidence,
        assess_contracts(evidence),
        golden={
            "case_id": "e2e-npm-code-migration",
            "live": {"target_packages": ["express-jwt"]},
            "expected_output": "Update then pivot after QA failure.",
        },
    )
    outcome = json.loads(outcome_text)["evidence"]

    assert {"initial-update", "workaround"} <= set(outcome["attempt_snapshots_by_id"])
    assert outcome["attempt_snapshots_by_id"]["initial-update"]["strategy_stage"] == "update"
    assert outcome["attempt_snapshots_by_id"]["workaround"]["strategy"] == "workaround"
    assert outcome["qa_results_by_attempt"]["initial-update"]["evaluation"]["passed"] is False
    assert outcome["qa_results_by_attempt"]["workaround"]["evaluation"]["passed"] is True


def test_task_completion_templates_repeat_new_finding_rule_in_verdict_prompts() -> None:
    template = e2e._task_completion_template()

    extraction = template.extract_goal_and_outcome(
        input="original request", actual_output="new finding remains", tools_called_formatted=""
    )
    trace_extraction = template.extract_task_and_outcome_from_trace(trace_json="{}")
    verdict = template.generate_verdict(task="original request", actual_outcome="new finding")

    assert "must not lower Task Completion" in extraction
    assert "must not lower Task Completion" in trace_extraction
    assert "must not lower Task Completion" in verdict
    assert "Unresolved original target findings" in verdict
