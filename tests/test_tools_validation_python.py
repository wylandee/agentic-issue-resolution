from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt
from remediation_engine.orchestration.tools_validation import (
    _make_run_targeted_python_test_tool,
    _make_validate_python_syntax_tool,
    _make_validate_python_workaround_tool,
)


def _tool_map(tools):
    return {tool.name: tool for tool in tools}


def _signature_contract(tool):
    signature = inspect.signature(tool.func)
    return (
        tuple((parameter.name, parameter.default) for parameter in signature.parameters.values()),
        signature.return_annotation,
    )


def test_python_workaround_tools_have_exact_names_signatures_and_language_scope():
    python_tools = _tool_map(
        build_workaround_toolbelt(
            sandbox=object(),
            touched_files=set(),
            host_repo_root=Path("/repo"),
            language=ProjectLanguage.PYTHON,
        )
    )
    node_tools = _tool_map(
        build_workaround_toolbelt(
            sandbox=object(),
            touched_files=set(),
            host_repo_root=Path("/repo"),
        )
    )

    expected_signatures = {
        "validate_python_syntax": (("file_path", inspect.Parameter.empty),),
        "run_targeted_python_test": (
            ("test_file", inspect.Parameter.empty),
            ("test_name", None),
        ),
        "validate_python_workaround": (
            ("modified_files", inspect.Parameter.empty),
            ("runtime_smoke_file", None),
            ("targeted_test_file", None),
            ("targeted_test_name", None),
        ),
    }
    for name, expected_parameters in expected_signatures.items():
        assert name in python_tools
        parameters, return_annotation = _signature_contract(python_tools[name])
        assert parameters == expected_parameters
        assert return_annotation in (str, "str")
        assert name not in node_tools

    assert "run_targeted_test" not in python_tools
    assert "validate_workaround" not in python_tools
    assert "modify_and_validate_npm_dependency" not in python_tools
    assert "remove_no_fix_dependency" not in python_tools
    assert "validate_workaround" in node_tools
    assert "validate_python_workaround" not in node_tools


def test_python_syntax_validation_is_read_only_and_fails_closed():
    sandbox = MagicMock()
    sandbox.run.return_value = CommandResult(
        exit_code=1,
        stdout="",
        stderr="SyntaxError: invalid syntax",
        duration_seconds=0.01,
    )
    validator = _make_validate_python_syntax_tool(sandbox)

    result = validator.invoke({"file_path": "src/service.py"})

    assert result.startswith("FAILURE: Syntax validation failed for src/service.py")
    command = sandbox.run.call_args.args[0]
    assert ".venv/bin/python -c" in command
    assert "ast.parse" in command
    assert "node" not in command
    assert sandbox.write_file.call_count == 0


def test_python_syntax_validation_rejects_non_python_source_without_running():
    sandbox = MagicMock()
    validator = _make_validate_python_syntax_tool(sandbox)

    result = validator.invoke({"file_path": "src/service.js"})

    assert result.startswith("ERROR:")
    sandbox.run.assert_not_called()


def test_python_targeted_test_runs_only_the_selected_pytest_node():
    sandbox = MagicMock()
    sandbox.read_file.return_value = "def test_validates_request():\n    assert True\n"
    sandbox.run.return_value = CommandResult(
        exit_code=0,
        stdout="1 passed",
        stderr="",
        duration_seconds=0.01,
    )
    targeted_test = _make_run_targeted_python_test_tool(sandbox)

    result = targeted_test.invoke(
        {"test_file": "tests/test_service.py", "test_name": "test_validates_request"}
    )

    assert result.startswith("SUCCESS: Targeted test passed (pytest)")
    command = sandbox.run.call_args.args[0]
    assert ".venv/bin/python -m pytest -q" in command
    assert "tests/test_service.py::test_validates_request" in command
    assert "npm" not in command


def test_python_runtime_smoke_imports_module_with_source_root_pythonpath():

    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: (
        "VALUE = 1\n" if path == "src/demo/module.py" else None
    )
    sandbox.run.return_value = CommandResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.01,
    )
    validator = _make_validate_python_workaround_tool(
        sandbox,
        {"src/demo/module.py"},
        {"targeted_test_required": False},
    )

    result = validator.invoke(
        {
            "modified_files": ["src/demo/module.py"],
            "runtime_smoke_file": "src/demo/module.py",
        }
    )

    assert result.startswith("SUCCESS: Workaround validation gate passed")
    commands = [call.args[0] for call in sandbox.run.call_args_list]
    smoke_command = next(command for command in commands if "importlib.import_module" in command)
    assert "PYTHONPATH=src" in smoke_command
    assert 'import_module("demo.module")' in smoke_command
    assert "python src/demo/module.py" not in smoke_command


def test_python_runtime_smoke_rejects_invalid_module_components_and_bootstraps():

    invalid_module_sandbox = MagicMock()
    invalid_module_sandbox.read_file.side_effect = lambda path: (
        "VALUE = 1\n" if path == "src/demo-package/module.py" else None
    )
    invalid_module_sandbox.run.return_value = CommandResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.01,
    )
    invalid_module = _make_validate_python_workaround_tool(
        invalid_module_sandbox,
        {"src/demo-package/module.py"},
        {"targeted_test_required": False},
    ).invoke(
        {
            "modified_files": ["src/demo-package/module.py"],
            "runtime_smoke_file": "src/demo-package/module.py",
        }
    )
    assert "does not map to a valid dotted module name" in invalid_module
    assert not any(
        "importlib.import_module" in call.args[0]
        for call in invalid_module_sandbox.run.call_args_list
    )

    bootstrap_sandbox = MagicMock()
    bootstrap_sandbox.read_file.side_effect = lambda path: (
        'if __name__ == "__main__":\n    app.run()\n' if path == "src/manage.py" else None
    )
    bootstrap = _make_validate_python_workaround_tool(
        bootstrap_sandbox,
        {"src/manage.py"},
        {"targeted_test_required": False},
    ).invoke({"modified_files": ["src/manage.py"], "runtime_smoke_file": "src/manage.py"})
    assert "unsafe" in bootstrap
    bootstrap_sandbox.run.assert_not_called()
