"""Tests for bounded live-fixture evidence and isolated patch verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from examples.juice_shop import fixture_acceptance as acceptance


def test_result_binding_rejects_stale_evidence_with_identical_patch() -> None:
    """Trajectory identity distinguishes identical output patches from separate runs."""
    patch_text = "--- a/package.json\n+++ b/package.json\n"
    result = {
        "status": "completed",
        "changed_files": ["package.json"],
        "trajectory_path": "/tmp/run-b.md",
    }
    evidence: dict[str, Any] = {
        "result_binding": {
            "status": "completed",
            "changed_files": ["package.json"],
            "diff_sha256": acceptance.hashlib.sha256(patch_text.encode()).hexdigest(),
            "trajectory_path": "/tmp/run-a.md",
        }
    }

    with pytest.raises(ValueError, match="trajectory paths differ"):
        acceptance._assert_result_binding(result, evidence, patch_text)


def test_task_lineage_uses_initial_root_and_effective_pivot_child() -> None:
    """A pivoted package root remains the initial task while its child is effective."""
    evidence = {
        "task_queue": {
            "root": {
                "task_id": "root",
                "parent_group_id": "group-1",
                "parent_task_id": None,
                "target_package_name": "got",
                "status": "pivoted",
            },
            "child": {
                "task_id": "child",
                "parent_group_id": "group-1",
                "parent_task_id": "root",
                "target_package_name": "got",
                "status": "qa_passed",
            },
        }
    }

    roots = acceptance._task_ids_by_package(evidence, {"got"})
    lineages = acceptance._lineage_task_ids_by_package(evidence, roots)
    statuses = acceptance._effective_package_statuses(evidence, roots)
    accepted = {
        "child": {
            "snapshot": {"task_id": "child", "attempt_number": 1},
        }
    }

    assert roots == {"got": ["root"]}
    assert lineages == {"got": [["root", "child"]]}
    assert statuses == {"got": ["qa_passed"]}
    assert (
        acceptance._accepted_attempt_for_lineage(accepted, lineages["got"][0], "got")
        is accepted["child"]
    )


def test_singleton_attempts_accept_absent_batch_metadata() -> None:
    """Unclustered singleton tasks legitimately have no dispatch batch IDs."""
    evidence: dict[str, Any] = {"portfolio_solver_plan": {"batches": []}}
    tasks = {"got": ["task-got"], "esbuild": ["task-esbuild"]}
    first = {
        package: [{"cluster_id": None, "dispatch_batch_id": None}] for package in ("got", "esbuild")
    }

    acceptance._assert_singleton_attempts(evidence, tasks, first, "atomic-pair-batch")


def test_all_singleton_roots_for_a_package_are_checked() -> None:
    """Multiple esbuild roots each have to be singleton attempts."""
    evidence: dict[str, Any] = {
        "task_queue": {
            task_id: {
                "task_id": task_id,
                "target_package_name": package,
                "parent_task_id": None,
            }
            for package, task_ids in {
                "got": ["task-got"],
                "esbuild": ["task-esbuild-a", "task-esbuild-b"],
            }.items()
            for task_id in task_ids
        },
        "attempt_snapshots_by_id": {
            f"attempt-{task_id}": {
                "task_id": task_id,
                "attempt_number": 1,
                "cluster_id": None,
                "dispatch_batch_id": None,
            }
            for task_id in ("task-got", "task-esbuild-a", "task-esbuild-b")
        },
        "portfolio_solver_plan": {"batches": []},
    }

    tasks = acceptance._task_ids_by_package(evidence, {"got", "esbuild"})
    first = acceptance._first_attempts_by_package(evidence, tasks)

    assert tasks == {
        "got": ["task-got"],
        "esbuild": ["task-esbuild-a", "task-esbuild-b"],
    }
    acceptance._assert_singleton_attempts(evidence, tasks, first, "atomic-pair-batch")


def test_empty_runner_patch_has_distinct_no_accepted_patch_error(tmp_path: Path) -> None:
    """Empty output reports missing accepted patch instead of a file mismatch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    result_path = tmp_path / "result.json"
    evidence_path = tmp_path / "evidence.json"
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text("", encoding="utf-8")
    result = {
        "status": "completed_with_errors",
        "changed_files": [],
        "diff": "",
        "trajectory_path": "/tmp/run.md",
    }
    binding = {
        "status": result["status"],
        "changed_files": [],
        "diff_sha256": acceptance.hashlib.sha256(b"").hexdigest(),
        "trajectory_path": result["trajectory_path"],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    evidence_path.write_text(json.dumps({"result_binding": binding}), encoding="utf-8")

    with pytest.raises(ValueError, match="no accepted patch"):
        acceptance._load_acceptance_inputs(repo, result_path, evidence_path, patch_path)


def test_patch_replay_keeps_unchanged_lockfiles_and_handles_add_delete(tmp_path: Path) -> None:
    """Replay applies add/delete/modify sections and copies unchanged lockfiles."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text("before\n", encoding="utf-8")
    (repo / "removed.txt").write_text("remove me\n", encoding="utf-8")
    (repo / "package-lock.json").write_text(
        json.dumps({"packages": {"node_modules/foo": {"version": "1.0.0"}}}),
        encoding="utf-8",
    )
    nested_lock = repo / "packages" / "app" / "package-lock.json"
    nested_lock.parent.mkdir(parents=True)
    nested_lock.write_text(
        json.dumps({"packages": {"node_modules/jws": {"version": "3.2.2"}}}),
        encoding="utf-8",
    )
    patch_text = """--- a/package.json
+++ b/package.json
@@ -1 +1 @@
-before
+after
--- a/removed.txt
+++ /dev/null
@@ -1 +0,0 @@
-remove me
--- /dev/null
+++ b/added.txt
@@ -0,0 +1 @@
+new file
"""
    patch_path = tmp_path / "change.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    result = {
        "changed_files": ["package.json", "removed.txt", "added.txt"],
    }

    replay_temporary, changed_paths = acceptance._replay_patch(repo, patch_path, result)
    try:
        replay = Path(replay_temporary.name)
        assert changed_paths == set(result["changed_files"])
        assert (replay / "package.json").read_text(encoding="utf-8") == "after\n"
        assert not (replay / "removed.txt").exists()
        assert (replay / "added.txt").read_text(encoding="utf-8") == "new file\n"
        assert (replay / "package-lock.json").is_file()
        assert (replay / "packages/app/package-lock.json").read_bytes() == nested_lock.read_bytes()
        assert acceptance._package_occurrences(replay, "jws") == {
            ("packages/app/package-lock.json", "node_modules/jws"): "3.2.2"
        }
    finally:
        replay_temporary.cleanup()


def test_scanned_occurrences_are_compared_with_only_covered_lockfiles(tmp_path: Path) -> None:
    """A package in another project lockfile is outside this task's scan scope."""
    repo = tmp_path / "repo"
    frontend_lock = repo / "frontend" / "package-lock.json"
    root_lock = repo / "package-lock.json"
    frontend_lock.parent.mkdir(parents=True)
    root_lock.parent.mkdir(parents=True, exist_ok=True)
    frontend_lock.write_text(
        json.dumps({"packages": {"node_modules/esbuild": {"version": "0.28.1"}}}),
        encoding="utf-8",
    )
    root_lock.write_text(
        json.dumps({"packages": {"node_modules/esbuild": {"version": "0.28.1"}}}),
        encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "latest_successful_scan_by_task": {
            "task-frontend": {
                "task_id": "task-frontend",
                "attempt_id": "attempt-frontend",
                "complete": True,
                "covered_input_paths": ["frontend/package-lock.json"],
                "covered_occurrences": [
                    {
                        "source_input_path": "frontend/package-lock.json",
                        "package_name": "esbuild",
                        "version": "0.28.1",
                        "lockfile_key": "node_modules/esbuild",
                        "found_identifiers": [],
                    }
                ],
            }
        }
    }

    acceptance._assert_scanned_target_occurrences(
        evidence, "esbuild", "task-frontend", "attempt-frontend", repo
    )


def test_scanned_occurrence_outside_covered_input_is_rejected(tmp_path: Path) -> None:
    """Evidence cannot claim a target occurrence from an unscanned lockfile."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package-lock.json").write_text(
        json.dumps({"packages": {"node_modules/esbuild": {"version": "0.28.1"}}}),
        encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "latest_successful_scan_by_task": {
            "task-root": {
                "task_id": "task-root",
                "attempt_id": "attempt-root",
                "complete": True,
                "covered_input_paths": ["other/package-lock.json"],
                "covered_occurrences": [
                    {
                        "source_input_path": "package-lock.json",
                        "package_name": "esbuild",
                        "version": "0.28.1",
                        "lockfile_key": "node_modules/esbuild",
                        "found_identifiers": [],
                    }
                ],
            }
        }
    }

    with pytest.raises(ValueError, match="outside its covered inputs"):
        acceptance._assert_scanned_target_occurrences(
            evidence, "esbuild", "task-root", "attempt-root", repo
        )


def test_shared_name_verifier_cleans_replay_after_assertion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifier owns and removes its replay directory when validation fails."""
    replay_temporary = tempfile.TemporaryDirectory(prefix="acceptance-cleanup-test-")
    replay_path = Path(replay_temporary.name)
    monkeypatch.setattr(
        acceptance,
        "_load_acceptance_inputs",
        lambda *_: ({}, {}, replay_temporary, set()),
    )

    def fail_validation(*_: Any) -> dict[str, Any]:
        raise ValueError("intentional invalid fixture evidence")

    monkeypatch.setattr(acceptance, "_verify_shared_name_fixture_loaded", fail_validation)
    with pytest.raises(ValueError, match="intentional invalid fixture evidence"):
        acceptance.verify_shared_name_fixture(
            Path("repo"), Path("result"), Path("evidence"), Path("patch")
        )

    assert not replay_path.exists()
