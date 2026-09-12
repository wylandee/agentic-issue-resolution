"""Deterministic boundaries and production-node adapters for Phase 2 evals.

The live replay suites call the real production node and model, but replace
only side effects that would make an evaluation nondeterministic or unsafe:
Docker execution, repository commands, scanner execution, and HTTP advisory
fetches.  The helper is intentionally test-only and does not change the
public remediation API.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from remediation_engine.contracts.schemas import (
    CommandResult,
    CVEEnrichment,
    IssueSource,
    IssueType,
    Severity,
    SystemContext,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.runtime.path_policy import normalize_workspace_path


@dataclasses.dataclass
class ReplayCapture:
    """Observable result produced by one production replay.

    Attributes:
        case_id: Golden case identifier.
        component: Production component that was executed.
        actual_output: JSON or text representation of the live result.
        actual_tools: Serialized tool events in execution order.
        typed_result: Raw production result, when one is available.
        changed_files: Files reported by the production component.
        errors: Production and replay-boundary errors.
        attempt_id: Attempt identity returned by a worker, if applicable.
        task_revision: Task revision returned by a worker, if applicable.
        external_calls: Recorded non-model calls made through replay doubles.
        final_files: Final contents of the adapter-owned in-memory workspace
            after the production node completes, including rollback decisions.
            Triage captures use an empty mapping because they have no workspace.
        input_tokens: Prompt/input tokens observed from the production model,
            or ``None`` when the provider did not return usage metadata.
        output_tokens: Completion/output tokens observed from the production
            model, or ``None`` when usage metadata was unavailable.
        total_tokens: Sum of observed input and output tokens, or ``None`` when
            usage metadata was unavailable.
        cached_input_tokens: Provider-reported cached prompt tokens, or
            ``None`` when the provider omitted cache metadata.
        token_cost: Provider-reported token cost in USD, when available. The
            replay layer never guesses pricing.
    """

    case_id: str
    component: str
    actual_output: str = ""
    actual_tools: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    typed_result: Any = None
    changed_files: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    attempt_id: str | None = None
    task_revision: int | None = None
    external_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    final_files: dict[str, str] = dataclasses.field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    token_cost: float | None = None


def _serialize_invocation_messages(
    messages: Sequence[BaseMessage],
) -> list[dict[str, Any]]:
    """Serialize model invocation messages for structural replay assertions."""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "type": str(getattr(message, "type", message.__class__.__name__)),
                "content": serialize_result(getattr(message, "content", "")),
                "additional_kwargs": serialize_result(
                    getattr(message, "additional_kwargs", {}) or {}
                ),
                "name": getattr(message, "name", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
                "tool_calls": serialize_result(getattr(message, "tool_calls", []) or []),
            }
        )
    return serialized


class ScriptedReplayModel:
    """Deterministic test-only LangChain chat-model substitute.

    The model supplies controlled assistant messages at the LLM boundary while
    production nodes, bounded-loop recovery, tools, and state transitions
    remain real.  Fixture tool outputs are deliberately ignored.
    """

    def __init__(
        self,
        messages: Sequence[AIMessage] = (),
        *,
        structured_result: BaseModel | None = None,
        enforce_bound_tools: bool = True,
    ) -> None:
        """Initialize a scripted model.

        Args:
            messages: Assistant messages returned by successive ``invoke`` calls.
            structured_result: Typed result returned by the structured-output
                wrapper, when configured.
            enforce_bound_tools: Whether assistant calls must be present in the
                currently bound tool set. Strict by default; historical
                workaround replay may opt out so production can record a
                phase violation.
        """
        self._messages = list(messages)
        self._enforce_bound_tools = enforce_bound_tools
        self._structured_result = structured_result
        self._next_message = 0
        self._bound_tool_names: set[str] = set()
        self.bound_tool_names: list[str] = []
        self.invocation_count = 0
        self.consumed_tool_calls: list[dict[str, Any]] = []
        self.invocation_messages: list[list[dict[str, Any]]] = []
        self.structured_invocation_count = 0
        self.structured_invocations: list[Any] = []

    @classmethod
    def from_tool_trace(
        cls,
        trace: Sequence[Mapping[str, Any]],
        *,
        final_text: str | None = None,
        enforce_bound_tools: bool = True,
    ) -> ScriptedReplayModel:
        """Build assistant tool-call messages from a canonical trace.

        Args:
            trace: Ordered records containing ``name`` and optional ``args``.
                Any recorded ``output`` values are ignored.
            final_text: Optional no-tool assistant message appended after the
                scripted calls.
            enforce_bound_tools: Whether to preserve strict bound-tool
                assertions for this replay model. Defaults to ``True``.

        Returns:
            A model that emits one assistant message per trace record.

        Raises:
            ValueError: If a trace record has an empty tool name or invalid
                arguments.
        """
        messages: list[AIMessage] = []
        for index, item in enumerate(trace, start=1):
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"Scripted replay trace item {index} has no tool name.")
            raw_args = item.get("args", {})
            if not isinstance(raw_args, Mapping):
                raise ValueError(f"Scripted replay trace item {index} has invalid args.")
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": dict(raw_args),
                            "id": f"call-{index}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
        if final_text is not None:
            messages.append(AIMessage(content=final_text))
        return cls(messages, enforce_bound_tools=enforce_bound_tools)

    @classmethod
    def from_structured_result(cls, result: BaseModel) -> ScriptedReplayModel:
        """Build a model for one structured-output invocation.

        Args:
            result: Typed result returned to the production structured-output
                path.

        Returns:
            A model configured with the supplied typed result.
        """
        return cls(structured_result=result)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedReplayModel:
        """Record the production tool binding and return this model.

        Args:
            tools: LangChain tools exposed by the production node.
            **kwargs: Provider binding options, including
                ``parallel_tool_calls=False``.

        Returns:
            This model instance, matching LangChain's binding contract.
        """
        del kwargs
        self.bound_tool_names = [str(getattr(tool, "name", "")) for tool in tools]
        self._bound_tool_names = set(self.bound_tool_names)
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Return the next scripted assistant message.

        Args:
            messages: Current production conversation.  The conversation is
                accepted to match the chat-model protocol and is not rewritten.

        Returns:
            The next scripted assistant message.

        Raises:
            AssertionError: If the production loop consumes too many messages,
                or, when strict bound checking is enabled, emits a tool not
                exposed by the production toolbelt.
        """
        self.invocation_messages.append(_serialize_invocation_messages(messages))
        if self._next_message >= len(self._messages):
            raise AssertionError(
                "Production consumed more scripted replay model responses than provided."
            )
        response = self._messages[self._next_message]
        self._next_message += 1
        self.invocation_count += 1
        for tool_call in response.tool_calls:
            name = str(tool_call.get("name", ""))
            if self._enforce_bound_tools and name not in self._bound_tool_names:
                raise AssertionError(
                    f"Scripted replay tool {name!r} was not bound by the production toolbelt."
                )
            self.consumed_tool_calls.append(
                {
                    "name": name,
                    "args": dict(tool_call.get("args", {}) or {}),
                    "id": str(tool_call.get("id", "")),
                }
            )
        return response

    def with_structured_output(self, schema: type[BaseModel]) -> ScriptedStructuredOutput:
        """Return a one-shot typed-output wrapper for production triage.

        Args:
            schema: Pydantic schema requested by the production node.

        Returns:
            A wrapper implementing the structured-output ``invoke`` contract.

        Raises:
            AssertionError: If this model was not configured with a typed
                result or the configured result does not match ``schema``.
        """
        if self._structured_result is None:
            raise AssertionError("Scripted replay model has no structured result.")
        if not isinstance(self._structured_result, schema):
            raise AssertionError(
                f"Scripted structured result is not an instance of {schema.__name__}."
            )
        self.structured_schema = schema
        return ScriptedStructuredOutput(self, schema, self._structured_result)


class ScriptedStructuredOutput:
    """One-shot structured-output wrapper owned by ``ScriptedReplayModel``."""

    def __init__(
        self,
        owner: ScriptedReplayModel,
        schema: type[BaseModel],
        result: BaseModel,
    ) -> None:
        """Initialize the structured-output wrapper."""
        self._owner = owner
        self._schema = schema
        self._result = result
        self._invoked = False

    def invoke(self, prompt: Any) -> BaseModel:
        """Return the configured typed result exactly once."""
        if self._invoked:
            raise AssertionError("Production invoked scripted structured output more than once.")
        self._invoked = True
        self._owner.structured_invocation_count += 1
        self._owner.structured_invocations.append(prompt)
        return self._result


class ReplaySandbox:
    """In-memory substitute for ``DockerSandbox`` used by eval replays.

    Args:
        files: Initial repository-relative file contents.
        command_responses: Optional exact or substring command responses. A
            value may be a ``CommandResult``, a mapping accepted by
            ``CommandResult``, or a callable receiving the command string.
        default_response: Response for commands without a configured route.
            The default is a deterministic failure so new production commands
            cannot silently escape the replay contract.

    The sandbox never writes the host repository, starts Docker, or invokes a
    subprocess.  ``files`` is copied on construction and can be inspected via
    ``files`` after the context exits.
    """

    def __init__(
        self,
        files: Mapping[str, str] | None = None,
        *,
        command_responses: Mapping[str, Any] | None = None,
        default_response: CommandResult | Mapping[str, Any] | None = None,
    ) -> None:
        self.files: dict[str, str] = {}
        for path, content in dict(files or {}).items():
            self.files[self._path(path)] = str(content)
        self.baseline_files = dict(self.files)
        self.command_responses = dict(command_responses or {})
        self.default_response = self._coerce_result(
            default_response
            or {
                "exit_code": 127,
                "stdout": "",
                "stderr": "ReplaySandbox: command is not registered.",
                "duration_seconds": 0.0,
            }
        )
        self.commands: list[str] = []
        self.writes: list[str] = []
        self.external_calls: list[dict[str, Any]] = []
        self._alive = False

    @staticmethod
    def _coerce_result(value: Any) -> CommandResult:
        """Convert a configured route response to ``CommandResult``."""
        if isinstance(value, CommandResult):
            return value
        if isinstance(value, Mapping):
            return CommandResult(
                exit_code=int(value.get("exit_code", 0)),
                stdout=str(value.get("stdout", "")),
                stderr=str(value.get("stderr", "")),
                duration_seconds=float(value.get("duration_seconds", 0.0)),
            )
        raise TypeError(f"Unsupported replay command response: {value!r}")

    @staticmethod
    def _path(path: str) -> str:
        """Normalize a workspace-relative path using production policy."""
        return normalize_workspace_path(str(path))

    def __enter__(self) -> ReplaySandbox:
        """Start the in-memory sandbox context."""
        self._alive = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the sandbox without affecting the host filesystem."""
        del exc_type, exc, traceback
        self._alive = False

    def restore_baseline(self, paths: Iterable[str] | None = None) -> None:
        """Rollback selected or all in-memory files to the initial snapshot.

        Args:
            paths: Optional repository-relative paths to restore.  When omitted,
                files created during the replay are removed as well.

        Raises:
            WorkspacePathError: If a selected path is not repository-relative.
        """
        if paths is None:
            self.files = dict(self.baseline_files)
        else:
            for file_path in paths:
                normalized = self._path(file_path)
                if normalized in self.baseline_files:
                    self.files[normalized] = self.baseline_files[normalized]
                else:
                    self.files.pop(normalized, None)
        self.writes.clear()

    def read_file(self, file_path: str) -> str | None:
        """Read a normalized workspace file from memory."""
        if not self._alive:
            return None
        return self.files.get(self._path(file_path))

    def write_file(self, file_path: str, content: str) -> None:
        """Write a normalized workspace file to memory."""
        if not self._alive:
            raise RuntimeError("ReplaySandbox is not running.")
        path = self._path(file_path)
        self.files[path] = str(content)
        if path not in self.writes:
            self.writes.append(path)

    def run(self, command: str, timeout: int = 300) -> CommandResult:
        """Return a deterministic response for a registered command.

        ``timeout`` is accepted to match ``DockerSandbox.run`` and recorded in
        no output because it is not part of the production command result.
        """
        del timeout
        if not self._alive:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr="ReplaySandbox: sandbox is not running.",
                duration_seconds=0.0,
            )
        self.commands.append(command)

        for matcher, configured in self.command_responses.items():
            if matcher == "__default__" or matcher in command:
                response = configured(command) if callable(configured) else configured
                return self._coerce_result(response)

        if "npm pkg set" in command:
            return self._apply_npm_pkg_set(command)
        if "find ." in command:
            return self._find_files()
        if "grep -RInE" in command:
            return self._grep_files(command)
        if command.lstrip().startswith(("rm -f --", "rm -rf --")):
            return self._remove_file(command)
        return self.default_response

    def _workspace_path_from_command(self, command: str, filename: str = "package.json") -> str:
        """Resolve a command's ``cd /workspace`` directory to a file path."""
        match = re.search(r"cd\s+/workspace(?:/([^\s&]+))?\s*&&", command)
        directory = match.group(1) if match else ""
        return self._path(f"{directory}/{filename}" if directory else filename)

    def _apply_npm_pkg_set(self, command: str) -> CommandResult:
        """Emulate the narrow ``npm pkg set`` operation used by update tools."""
        try:
            tokens = shlex.split(command)
            start = next(
                index
                for index in range(len(tokens) - 2)
                if tokens[index : index + 3] == ["npm", "pkg", "set"]
            )
            expression = tokens[start + 3]
        except (StopIteration, IndexError, ValueError) as exc:
            return CommandResult(
                exit_code=2,
                stdout="",
                stderr=f"ReplaySandbox: malformed npm pkg set command: {exc}",
                duration_seconds=0.0,
            )
        if "=" not in expression:
            return CommandResult(
                exit_code=2,
                stdout="",
                stderr="ReplaySandbox: npm pkg set expression is missing '='.",
                duration_seconds=0.0,
            )
        key, value = expression.rsplit("=", 1)
        key_match = re.fullmatch(r"([A-Za-z0-9_.-]+)\[(.+)\]", key)
        if not key_match:
            return CommandResult(
                exit_code=2,
                stdout="",
                stderr=f"ReplaySandbox: unsupported npm key {key!r}.",
                duration_seconds=0.0,
            )
        section, package_name = key_match.groups()
        package_name = package_name.strip("'\"")
        manifest_path = self._workspace_path_from_command(command)
        content = self.files.get(manifest_path)
        if content is None:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"ReplaySandbox: {manifest_path} does not exist.",
                duration_seconds=0.0,
            )
        try:
            package_json = json.loads(content)
        except json.JSONDecodeError as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"ReplaySandbox: invalid JSON: {exc}",
                duration_seconds=0.0,
            )
        section_data: dict[str, Any] = package_json
        for section_part in section.split("."):
            nested = section_data.setdefault(section_part, {})
            if not isinstance(nested, dict):
                return CommandResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"ReplaySandbox: manifest section {section!r} is not an object.",
                    duration_seconds=0.0,
                )
            section_data = nested
        section_data[package_name] = value
        self.files[manifest_path] = json.dumps(package_json, indent=2) + "\n"
        if manifest_path not in self.writes:
            self.writes.append(manifest_path)
        return CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    def _find_files(self) -> CommandResult:
        """Return a deterministic repository map for the current files."""
        return CommandResult(
            exit_code=0,
            stdout="\n".join(sorted(self.files)) + ("\n" if self.files else ""),
            stderr="",
            duration_seconds=0.0,
        )

    def _grep_files(self, command: str) -> CommandResult:
        """Emulate the source-only grep used by investigation tools."""
        pattern_match = re.search(r"--\s+'(.+?)'\s+'([^']+)'\s*\|", command)
        pattern = pattern_match.group(1) if pattern_match else ".*"
        target = pattern_match.group(2).strip("./") if pattern_match else ""
        try:
            compiled = re.compile(pattern)
        except re.error:
            compiled = re.compile(re.escape(pattern))
        lines: list[str] = []
        for path, content in sorted(self.files.items()):
            if target and target != "." and not (path == target or path.startswith(target + "/")):
                continue
            if Path(path).suffix.lower() not in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if compiled.search(line):
                    lines.append(f"{path}:{line_number}:{line}")
        return CommandResult(
            exit_code=0 if lines else 1,
            stdout="\n".join(lines) + ("\n" if lines else ""),
            stderr="",
            duration_seconds=0.0,
        )

    def _remove_file(self, command: str) -> CommandResult:
        """Emulate the narrowly scoped file removal used by stage resets."""
        try:
            tokens = shlex.split(command)
            path = tokens[-1]
            normalized = self._path(path)
        except (ValueError, IndexError) as exc:
            return CommandResult(
                exit_code=2,
                stdout="",
                stderr=f"ReplaySandbox: invalid rm command: {exc}",
                duration_seconds=0.0,
            )
        recursive = "rm -rf --" in command
        if recursive:
            prefix = normalized.rstrip("/") + "/"
            targets = [
                candidate
                for candidate in self.files
                if candidate == normalized or candidate.startswith(prefix)
            ]
        else:
            targets = [normalized]
        for target in targets:
            existed = target in self.files
            self.files.pop(target, None)
            if existed and target not in self.writes:
                self.writes.append(target)
        return CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)


def serialize_tool_events(events: Iterable[Any]) -> list[dict[str, Any]]:
    """Serialize runtime tool events into canonical golden records.

    Args:
        events: Objects exposing ``name``, ``args``, and ``content`` fields.

    Returns:
        JSON-compatible ordered records with ``name``, ``args``, and ``output``.
    """
    serialized: list[dict[str, Any]] = []
    for event in events:
        args = getattr(event, "args", {}) or {}
        serialized.append(
            {
                "name": str(getattr(event, "name", "")),
                "args": dict(args) if isinstance(args, Mapping) else {},
                "output": str(getattr(event, "content", "") or ""),
            }
        )
    return serialized


def serialize_result(value: Any) -> Any:
    """Convert a Pydantic/dataclass/dict result into JSON-compatible data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): serialize_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize_result(item) for item in value]
    return value


def format_tool_trace(tool_calls: Iterable[Mapping[str, Any]], label: str = "Observed") -> str:
    """Render serialized tool calls for a task-completion judge."""
    calls = list(tool_calls)
    if not calls:
        return f"{label} tool trace: none"
    lines = [f"{label} tool trace:"]
    for index, call in enumerate(calls, start=1):
        args = json.dumps(call.get("args", {}) or {}, sort_keys=True, default=str)
        output = str(call.get("output", "") or "")
        lines.append(f"{index}. {call.get('name', '')}({args}) -> {output}")
    return "\n".join(lines)


def cached_replay(
    cache: dict[tuple[str, str], ReplayCapture],
    component: str,
    case_id: str,
    factory: Callable[[], ReplayCapture],
) -> ReplayCapture:
    """Run one replay once per component/case within a pytest process."""
    key = (component, case_id)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _first_match(pattern: str, text: str, default: str | None = None) -> str | None:
    """Return the first regex group from text."""
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else default


def build_system_context(case: Mapping[str, Any]) -> SystemContext:
    """Build ``SystemContext`` from structured replay data and scenario text."""
    replay = case.get("replay", {}) if isinstance(case.get("replay"), Mapping) else {}
    raw = replay.get("input", {}).get("system_context", {}) if isinstance(replay, Mapping) else {}
    raw = raw if isinstance(raw, Mapping) else {}
    scenario = str(case.get("scenario", case.get("input", "")))
    scenario_lower = scenario.lower()
    completion = str(case.get("completion_task", ""))
    combined_lower = f"{scenario}\n{completion}".lower()

    public_facing: bool | None
    if "public_facing" in raw:
        public_facing = raw.get("public_facing")
    elif re.search(r"\bpublic[_ -]?facing\s*=\s*false\b", scenario_lower):
        public_facing = False
    elif re.search(r"\bpublic[- ]facing\b", scenario_lower) or re.search(
        r"\bpublic\s+(?:production\s+)?(?:api|service|application)\b", scenario_lower
    ):
        public_facing = True
    elif re.search(r"\ban internal (?:microservice|daemon|service)\b", scenario_lower):
        public_facing = False
    else:
        public_facing = None

    if raw.get("environment") is not None:
        environment = str(raw["environment"])
    elif re.search(r"\bproduction\b", scenario_lower):
        environment = "production"
    elif re.search(r"\b(?:dev|development|test|ci)\b", scenario_lower):
        environment = "dev"
    elif re.search(r"\bproduction\b", combined_lower):
        # A few synthetic grouped cases put the deployment context in the
        # completion contract rather than repeating it in the scenario.
        environment = "production"
    else:
        environment = "dev"

    deployment_os = (
        str(raw["deployment_os"])
        if raw.get("deployment_os") is not None
        else ("windows" if re.search(r"\bwindows\b", scenario_lower) else "linux")
    )
    sensitivity = _first_match(r"data sensitivity\s*(?:is|=)\s*([A-Za-z]+)", scenario)
    if sensitivity is None:
        sensitivity = "high" if "high data sensitivity" in scenario_lower else "low"
    raw_tags = raw.get("tags", {})
    tags = (
        {str(key): str(value) for key, value in raw_tags.items()}
        if isinstance(raw_tags, Mapping)
        else {}
    )
    return SystemContext(
        repo_url=raw.get("repo_url"),
        base_ref=raw.get("base_ref"),
        scanned_at=raw.get("scanned_at", datetime(2026, 1, 1, tzinfo=UTC)),
        environment=environment,
        deployment_os=deployment_os,
        public_facing=raw.get("public_facing", public_facing),
        primary_language=raw.get("primary_language", "javascript"),
        deployment_architecture=raw.get("deployment_architecture", "service"),
        data_sensitivity=raw.get("data_sensitivity", sensitivity),
        tags=tags,
    )


def build_vulnerability_group(case: Mapping[str, Any]) -> VulnerabilityGroup:
    """Build a valid ``VulnerabilityGroup`` for triage replay.

    The replay payload may provide a complete serialized group.  Otherwise the
    helper derives a conservative typed group from the existing scenario and
    domain metadata, which keeps historical goldens backwards compatible while
    the canonical migration is applied.
    """
    replay = case.get("replay", {}) if isinstance(case.get("replay"), Mapping) else {}
    replay_input = replay.get("input", {}) if isinstance(replay, Mapping) else {}
    raw_group = replay_input.get("group") if isinstance(replay_input, Mapping) else None
    if isinstance(raw_group, Mapping) and raw_group.get("group_id"):
        return VulnerabilityGroup.model_validate(raw_group)

    scenario = str(case.get("scenario", case.get("input", "")))
    vulnerability = case.get("vulnerability_context", {})
    vulnerability = vulnerability if isinstance(vulnerability, Mapping) else {}
    group_id = str(vulnerability.get("group_id") or case.get("case_id") or "replay-group")
    package = str(
        vulnerability.get("vulnerable_component")
        or _first_match(r"^([@\w./-]+)\s+is declared in the dependency manifest", scenario)
        or _first_match(
            r"\b(?:uses|affects|depends on)\s+(?:direct dependency\s+)?"
            r"([@\w./-]+?)(?=@\d|\s|[,.;]|$)",
            scenario,
        )
        or _first_match(r"\bpackage\s+named\s+([@\w./-]+)", scenario)
        or _first_match(r"\b(?:dependency|package)\s+([@\w./-]+)@\d", scenario)
        or _first_match(r"\buses\s+([@\w./-]+)", scenario)
        or "replay-package"
    ).rstrip(".,;:")
    cves = [str(cve).upper() for cve in list(vulnerability.get("cve_ids", []) or [])]
    if not cves:
        cves = [
            cve.upper() for cve in re.findall(r"CVE-\d{4}-\d{4,}", scenario, flags=re.IGNORECASE)
        ]
    if not cves and "cve" in scenario.lower():
        raw_cves = re.findall(r"CVE-[0-9A-Za-z_-]+", scenario, flags=re.IGNORECASE)
        valid_cves = []
        for cve in raw_cves:
            long_match = re.match(r"^CVE-\d{4}-\d{4,}$", cve, flags=re.IGNORECASE)
            short_match = re.match(r"^CVE-(\d+)$", cve, flags=re.IGNORECASE)
            if long_match:
                valid_cves.append(cve.upper())
            elif short_match:
                valid_cves.append(f"CVE-2022-{short_match.group(1).zfill(4)}")
        cves = list(dict.fromkeys(valid_cves))
    ghsas = [str(ghsa).upper() for ghsa in list(vulnerability.get("ghsa_ids", []) or [])]
    if not ghsas:
        ghsas = [
            ghsa.upper() for ghsa in re.findall(r"GHSA-[0-9A-Z-]+", scenario, flags=re.IGNORECASE)
        ]
    issue_type = (
        IssueType.SAST
        if re.search(r"\b(?:semgrep|sast)\b", scenario, flags=re.IGNORECASE)
        else IssueType.SCA
    )
    issue_source = IssueSource.SEMGREP if issue_type == IssueType.SAST else IssueSource.ODC
    version = _first_match(
        rf"{re.escape(package)}@(\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.-]+)?)", scenario
    )
    file_path = str(
        vulnerability.get("file_path")
        or _first_match(r"(test(?:/|\\)[^\s,]+|lib(?:/|\\)[^\s,]+\.\w+)", scenario)
        or "package.json"
    ).rstrip(".,")
    severity_name = (
        _first_match(r"original\s+(?:scanner\s+)?severity\s*(?:is|=)?\s*([A-Z]+)", scenario)
        or "MEDIUM"
    )
    try:
        severity = Severity(str(severity_name).upper())
    except ValueError:
        severity = Severity.MEDIUM
    issue_id = f"{group_id}:issue"
    issue = VulnerabilityIssue(
        id=uuid5(NAMESPACE_URL, issue_id),
        finding_id=issue_id,
        source=issue_source,
        issue_type=issue_type,
        cve_id=cves[0] if cves else None,
        ghsa_id=ghsas[0] if ghsas else None,
        severity=severity,
        file_path=file_path,
        package_name=package,
        package_version=version or "0.0.0",
        ecosystem="npm",
        message=scenario,
    )
    epss_matches = [
        float(value)
        for value in re.findall(r"\bEPSS\b[^\d]*(0?\.\d+)", scenario, flags=re.IGNORECASE)
    ]
    epss = max(epss_matches) if epss_matches else 0.0
    in_kev = bool(re.search(r"\b(?:CISA\s+)?KEV\b", scenario, flags=re.IGNORECASE))
    enrichment = (
        CVEEnrichment(
            cve_id=(cves[-1] if in_kev else cves[0]) if cves else "CVE-2022-0001",
            epss=epss,
            epss_percentile=epss,
            in_kev=in_kev,
            enrichment_source="eval_replay",
        )
        if (cves or in_kev or epss > 0.0)
        else None
    )
    reachable = not bool(
        re.search(r"\bis_reachable\s*=\s*false\b|\bunreachable\b", scenario.lower())
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=issue_type,
        vulnerable_component=package,
        file_path=file_path,
        file_paths=[file_path],
        cve_ids=cves,
        ghsa_ids=ghsas,
        versions=[version] if version else ["0.0.0"],
        dependency_ancestry=[],
        dependency_versions={package: version or "0.0.0"},
        parent_contexts=[],
        sources=[issue_source],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[],
        enrichment=enrichment,
        is_reachable=reachable,
    )


def build_command_responses(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic command routes from replay fixture metadata."""
    replay = case.get("replay", {}) if isinstance(case.get("replay"), Mapping) else {}
    raw = replay.get("input", {}) if isinstance(replay, Mapping) else {}
    configured = raw.get("command_responses", {}) if isinstance(raw, Mapping) else {}
    responses = dict(configured) if isinstance(configured, Mapping) else {}
    default = raw.get("default_command_response") if isinstance(raw, Mapping) else None
    if default is not None:
        responses["__default__"] = default
    return responses


def seed_replay_files(case: Mapping[str, Any]) -> dict[str, str]:
    """Return workspace files declared by a case, with a safe default repo."""
    replay = case.get("replay", {}) if isinstance(case.get("replay"), Mapping) else {}
    raw = replay.get("input", {}) if isinstance(replay, Mapping) else {}
    files = raw.get("workspace_files", {}) if isinstance(raw, Mapping) else {}
    if isinstance(files, Mapping) and files:
        return {str(path): str(content) for path, content in files.items()}
    return {
        "package.json": json.dumps(
            {"name": "eval-replay", "version": "1.0.0", "dependencies": {}},
            indent=2,
        )
        + "\n",
        "package-lock.json": '{"name":"eval-replay","lockfileVersion":3,"packages":{}}\n',
        "src/replay.js": "module.exports = function replay() { return true; };\n",
        "test/replay.test.js": "describe('replay', () => { it('passes', () => {}); });\n",
    }


def workspace_temp_root() -> Path:
    """Return the ignored workspace-local root used by replay temp repos."""
    root = Path(__file__).resolve().parents[2] / ".pytest-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_temp_repo(repo_root: Path, files: Mapping[str, str]) -> None:
    """Seed a workspace-local host baseline for worker revert/map helpers."""
    for path, content in files.items():
        target = repo_root / normalize_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")


def patch_runtime_loop(module: Any, captures: list[Any]) -> Any:
    """Return a patch context that records a worker/QA bounded-loop result."""
    original = module.run_bounded_subagent_loop

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        captures.append(result)
        return result

    return patch.object(module, "run_bounded_subagent_loop", side_effect=wrapped)


def output_with_trace(
    value: Any, tool_calls: list[dict[str, Any]], *, label: str = "Observed"
) -> str:
    """Serialize a production result and append its captured tool trace."""
    payload = json.dumps(serialize_result(value), indent=2, sort_keys=True, default=str)
    return f"{payload}\n\n{format_tool_trace(tool_calls, label=label)}"
