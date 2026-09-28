"""Offline behavior tests for authorized Python manifest transactions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from remediation_engine.contracts import NoFixMitigationStage
from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt
from remediation_engine.orchestration.tools_manifest import (
    _normalize_python_manifest_targets,
    _package_checkpoint_paths,
)
from remediation_engine.orchestration.tools_manifest_python import (
    _make_modify_and_validate_python_dependency_tool,
)


def _result(exit_code: int = 0, stdout: str = "ok", stderr: str = "") -> CommandResult:
    return CommandResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.1,
    )


def _python_update_tool(
    sandbox,
    *,
    manifest: str = "requirements.txt",
    allowed_versions=None,
    allowed_types=None,
    execution_state=None,
    package_checkpoints=None,
    touched_files=None,
):
    tool = _make_modify_and_validate_python_dependency_tool(
        sandbox,
        touched_files if touched_files is not None else set(),
        {"requests": [manifest]},
        allowed_target_versions_by_package=(
            allowed_versions if allowed_versions is not None else {"requests": ["2.32.0"]}
        ),
        allowed_dependency_types_by_package=(
            allowed_types if allowed_types is not None else {"requests": ["requirements"]}
        ),
        execution_state=execution_state,
        package_checkpoints=package_checkpoints,
    )
    return tool


def _files_sandbox(files: dict[str, str], *, run_handler=None, write_handler=None):
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: files.get(path)

    def write(path, content):
        if write_handler is not None:
            write_handler(path, content)
        files[path] = content

    sandbox.write_file.side_effect = write
    sandbox.run.side_effect = run_handler or (lambda *_args, **_kwargs: _result())
    return sandbox


def _projection(result: str) -> list[str]:
    marker = "JSON: "
    assert marker in result
    return json.loads(result.split(marker, 1)[1])["changed_files"]


def test_python_manifest_targets_and_checkpoints_exclude_npm_locks():
    assert _normalize_python_manifest_targets(
        ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.cfg", "Pipfile"]
    ) == ["Pipfile", "pyproject.toml", "requirements-dev.txt", "requirements.txt", "setup.cfg"]
    assert _package_checkpoint_paths(["requirements-dev.txt"], package_ecosystem="pypi") == [
        "requirements-dev.txt"
    ]
    assert _package_checkpoint_paths(["Pipfile"], package_ecosystem="pypi") == [
        "Pipfile",
        "Pipfile.lock",
    ]
    with pytest.raises(ValueError):
        _normalize_python_manifest_targets(["setup.py"])


def test_authorized_requirements_update_validates_and_reports_changed_files():
    files = {"requirements.txt": "# keep\nrequests>=2.31.0\n"}
    sandbox = _files_sandbox(files)
    touched: set[str] = set()
    tool = _python_update_tool(sandbox, touched_files=touched)

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "requirements",
            "manifest_path": "requirements.txt",
        }
    )

    assert result.startswith("SUCCESS:")
    assert files["requirements.txt"] == "# keep\nrequests==2.32.0\n"
    assert "pip install --dry-run -r /workspace/requirements.txt" in sandbox.run.call_args.args[0]
    assert touched == {"requirements.txt"}
    assert _projection(result) == ["requirements.txt"]


def test_pep621_and_setup_cfg_transactions_edit_only_the_approved_declaration():
    cases = (
        (
            "pyproject.toml",
            '[project]\nname = "sample"\ndependencies = ["requests>=2.31.0"]\n',
            "dependencies",
        ),
        (
            "pyproject.toml",
            '[project]\nname = "sample"\n[project.optional-dependencies]\ntest = ["requests>=2.31.0"]\n',
            "optional-dependencies",
        ),
        (
            "setup.cfg",
            "[metadata]\nname = sample\n[options]\ninstall_requires =\n    requests>=2.31.0\n",
            "install_requires",
        ),
    )

    for manifest, original, declaration_type in cases:
        files = {manifest: original}
        sandbox = _files_sandbox(files)
        tool = _python_update_tool(
            sandbox,
            manifest=manifest,
            allowed_types={"requests": [declaration_type]},
        )

        result = tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.32.0",
                "dependency_type": declaration_type,
                "manifest_path": manifest,
            }
        )

        assert result.startswith("SUCCESS:")
        assert "requests==2.32.0" in files[manifest]
        assert _projection(result) == [manifest]


def test_unapproved_manifest_candidate_and_declaration_type_do_not_mutate():
    files = {"requirements.txt": "requests==2.31.0\n", "other.txt": "requests==2.31.0\n"}
    sandbox = _files_sandbox(files)
    tool = _python_update_tool(sandbox)

    rejected = [
        tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.33.0",
                "dependency_type": "requirements",
                "manifest_path": "requirements.txt",
            }
        ),
        tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.32.0",
                "dependency_type": "dependencies",
                "manifest_path": "requirements.txt",
            }
        ),
        tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.32.0",
                "dependency_type": "requirements",
                "manifest_path": "other.txt",
            }
        ),
        tool.invoke(
            {
                "package_name": "Requests",
                "target_version": "2.32.0",
                "dependency_type": "requirements",
                "manifest_path": "requirements.txt",
            }
        ),
    ]

    assert all(result.startswith("ERROR_CODE:") for result in rejected)
    assert files == {"requirements.txt": "requests==2.31.0\n", "other.txt": "requests==2.31.0\n"}
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_not_called()


def test_failed_python_validation_restores_manifest_and_touched_projection():
    baseline = "requests==2.31.0\n"
    files = {"requirements.txt": baseline}
    failure = _result(1, "", "resolution failed")
    sandbox = _files_sandbox(files, run_handler=lambda *_args, **_kwargs: failure)
    touched = {"src/module.py"}
    tool = _python_update_tool(sandbox, touched_files=touched)

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "requirements",
            "manifest_path": "requirements.txt",
        }
    )

    assert result.startswith("ERROR_CODE: PYTHON_TRANSACTION_FAILED:")
    assert "resolution failed" in result
    assert "Rollback: package checkpoint restored." in result
    assert files["requirements.txt"] == baseline
    assert touched == {"src/module.py"}


def test_pipfile_update_locks_then_syncs_and_attributes_both_files():
    files = {
        "Pipfile": '[packages]\nrequests = "*"\n',
        "Pipfile.lock": '{"default": {"requests": {"version": "==2.31.0"}}}\n',
    }
    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command)
        if "pipenv lock" in command:
            files["Pipfile.lock"] = '{"default": {"requests": {"version": "==2.32.0"}}}\n'
        return _result()

    sandbox = _files_sandbox(files, run_handler=run)
    touched: set[str] = set()
    tool = _python_update_tool(
        sandbox,
        manifest="Pipfile",
        allowed_types={"requests": ["packages"]},
        touched_files=touched,
    )

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )

    assert result.startswith("SUCCESS:")
    assert files["Pipfile"] == '[packages]\nrequests = "==2.32.0"\n'
    assert "pipenv lock" in commands[0]
    assert "PIPENV_VENV_IN_PROJECT=1" in commands[1]
    assert "pipenv sync --dev" in commands[1]
    assert touched == {"Pipfile", "Pipfile.lock"}
    assert _projection(result) == ["Pipfile", "Pipfile.lock"]


def test_pipfile_sync_failure_restores_manifest_lock_and_resyncs_saved_lock():
    baseline_pipfile = '[packages]\nrequests = "*"\n'
    baseline_lock = '{"default": {"requests": {"version": "==2.31.0"}}}\n'
    files = {"Pipfile": baseline_pipfile, "Pipfile.lock": baseline_lock}
    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command)
        if "pipenv lock" in command:
            files["Pipfile.lock"] = "generated lock\n"
        if "pipenv sync --dev" in command and len(commands) == 2:
            return _result(1, "partial sync", "sync failure")
        return _result()

    sandbox = _files_sandbox(files, run_handler=run)
    touched = {"src/app.py"}
    tool = _python_update_tool(
        sandbox,
        manifest="Pipfile",
        allowed_types={"requests": ["packages"]},
        touched_files=touched,
    )

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )

    assert "sync failure" in result
    assert "Rollback: package checkpoint restored." in result
    assert files == {"Pipfile": baseline_pipfile, "Pipfile.lock": baseline_lock}
    assert "pipenv sync --dev" in commands[-1]
    assert touched == {"src/app.py"}


def test_pipfile_rollback_restore_failure_is_reported_explicitly():
    baseline_lock = '{"default": {"requests": {"version": "==2.31.0"}}}\n'
    files = {"Pipfile": '[packages]\nrequests = "*"\n', "Pipfile.lock": baseline_lock}

    def write(path, _content):
        if path == "Pipfile.lock":
            raise OSError("lock restore denied")

    def run(command, **_kwargs):
        if "pipenv lock" in command:
            files["Pipfile.lock"] = "changed lock\n"
        if "pipenv sync --dev" in command and files["Pipfile.lock"] != baseline_lock:
            return _result(1, "", "rollback sync failed")
        if "pipenv sync --dev" in command:
            return _result(1, "", "update sync failed")
        return _result()

    sandbox = _files_sandbox(files, run_handler=run, write_handler=write)
    tool = _python_update_tool(
        sandbox,
        manifest="Pipfile",
        allowed_types={"requests": ["packages"]},
    )

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )

    assert "Rollback failed" in result
    assert "lock restore denied" in result
    assert "Pipenv rollback sync failed" in result


def test_pipfile_manifest_removal_is_direct_only_and_runs_lock_sync():
    files = {
        "Pipfile": '[packages]\nrequests = "*"\n[dev-packages]\npytest = "*"\n',
        "Pipfile.lock": '{"default": {}, "develop": {}}\n',
    }
    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command)
        if "pipenv lock" in command:
            files["Pipfile.lock"] = '{"default": {}, "develop": {}}\\n'
        return _result()

    sandbox = _files_sandbox(files, run_handler=run)
    plan_state = {
        "recorded": True,
        "package_removal_planned": True,
        "no_fix_stage": NoFixMitigationStage.PACKAGE_REMOVAL.value,
    }
    toolbelt = build_workaround_toolbelt(
        sandbox,
        set(),
        Path("/tmp/repo"),
        plan_state=plan_state,
        no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
        no_fix_package_name="requests",
        no_fix_manifest_paths=["Pipfile"],
        no_fix_package_manager="pipenv",
        language=ProjectLanguage.PYTHON,
    )
    tools = {item.name: item for item in toolbelt}
    assert "remove_no_fix_python_dependency" in tools
    assert "remove_no_fix_dependency" not in tools

    result = tools["remove_no_fix_python_dependency"].invoke(
        {"package_name": "requests", "manifest_path": "Pipfile"}
    )

    assert result.startswith("SUCCESS:")
    assert "requests" not in files["Pipfile"]
    assert "pipenv lock" in commands[0]
    assert "pipenv sync --dev" in commands[1]
    assert _projection(result) == ["Pipfile", "Pipfile.lock"]


def test_python_removal_rejects_undeclared_package_without_editing():
    files = {"requirements.txt": "flask==3.0\n"}
    sandbox = _files_sandbox(files)
    plan_state = {
        "recorded": True,
        "package_removal_planned": True,
        "no_fix_stage": NoFixMitigationStage.PACKAGE_REMOVAL.value,
    }
    toolbelt = build_workaround_toolbelt(
        sandbox,
        set(),
        Path("/tmp/repo"),
        plan_state=plan_state,
        no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
        no_fix_package_name="requests",
        no_fix_manifest_paths=["requirements.txt"],
        no_fix_package_manager="pip",
        language=ProjectLanguage.PYTHON,
    )
    tool = next(item for item in toolbelt if item.name == "remove_no_fix_python_dependency")

    result = tool.invoke({"package_name": "requests", "manifest_path": "requirements.txt"})

    assert result.startswith("NOT_APPLICABLE:")
    assert files["requirements.txt"] == "flask==3.0\n"
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_not_called()
