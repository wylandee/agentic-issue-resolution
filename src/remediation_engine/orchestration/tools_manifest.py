"""Manifest mutation, package checkpoint, and rollback tools."""

from __future__ import annotations

from remediation_engine.contracts.schemas import MultiPackageAction

from ._tool_support import (
    _MANIFEST_SYNC_TIMEOUT_SECONDS,
    Any,
    DockerSandbox,
    Iterable,
    Mapping,
    NoFixMitigationStage,
    Path,
    Sequence,
    WorkaroundExecutionPhase,
    _normalize_manifest_targets,
    _normalize_package_manifest_targets,
    _validate_workspace_path,
    _workspace_dir_for_manifest,
    dataclass,
    json,
    logger,
    re,
    shlex,
    tool,
)


@dataclass
class _PackageCheckpoint:
    """Capture one package's pre-edit workspace state for an update task."""

    files: dict[str, str | None]
    touched_files_before: set[str]


def _package_checkpoint_paths(manifest_paths: Iterable[str]) -> list[str]:
    """Return manifests and npm lockfiles that may change for a package update."""
    paths: set[str] = set()
    for manifest_path in manifest_paths:
        normalized_manifest = _validate_workspace_path(manifest_path)
        paths.add(normalized_manifest)
        parent = Path(normalized_manifest).parent
        ancestor_dirs = [parent, *parent.parents]
        for directory in ancestor_dirs:
            for lockfile_name in ("package-lock.json", "npm-shrinkwrap.json"):
                paths.add(_validate_workspace_path((directory / lockfile_name).as_posix()))
    return sorted(paths)


def _capture_package_checkpoint(
    sandbox: DockerSandbox,
    manifest_paths: Iterable[str],
    touched_files: set[str],
) -> _PackageCheckpoint:
    """Capture the current package manifests and related lockfiles before editing."""
    files: dict[str, str | None] = {}
    for path in _package_checkpoint_paths(manifest_paths):
        content = sandbox.read_file(path)
        files[path] = content if isinstance(content, str) else None
    return _PackageCheckpoint(files=files, touched_files_before=set(touched_files))


def _restore_package_checkpoint(
    sandbox: DockerSandbox,
    checkpoint: _PackageCheckpoint,
    touched_files: set[str],
) -> str | None:
    """Restore one package checkpoint and return a rollback error, if any."""
    rollback_errors: list[str] = []
    for path, content in checkpoint.files.items():
        try:
            if content is None:
                result = sandbox.run(f"rm -f -- {shlex.quote(path)}")
                if result.exit_code != 0:
                    rollback_errors.append(
                        f"{path}: delete failed with exit {result.exit_code}: {result.stderr}"
                    )
            else:
                sandbox.write_file(path, content)
        except Exception as exc:  # noqa: BLE001
            rollback_errors.append(f"{path}: {exc}")

    touched_files.difference_update(checkpoint.files)
    touched_files.update(checkpoint.touched_files_before)
    if rollback_errors:
        return "Package checkpoint rollback failed: " + " | ".join(rollback_errors)
    return None


def rollback_pending_package_updates(
    sandbox: DockerSandbox,
    package_checkpoints: dict[str, _PackageCheckpoint],
    touched_files: set[str],
) -> list[str]:
    """Restore package edits that were not successfully validated in this run."""
    errors: list[str] = []
    for package_name, checkpoint in list(package_checkpoints.items()):
        rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
        if rollback_error:
            errors.append(f"Package '{package_name}': {rollback_error}")
        package_checkpoints.pop(package_name, None)
    return errors


def _tool_error(code: str, message: str) -> str:
    """Return a stable machine-readable update-tool error."""
    return f"ERROR_CODE: {code}: {message}"


def _bounded_command_output(value: Any, limit: int = 4_000) -> str:
    """Convert command output to a bounded diagnostic string."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated)"


def _lockfile_target_versions(payload: Mapping[str, Any], package_name: str) -> set[str]:
    """Return versions recorded for a package in npm lockfile layouts.

    npm v2/v3 lockfiles use a ``packages`` map while npm v1 uses nested
    ``dependencies`` records.  Workspace lockfiles can contain either a
    ``name`` field or a ``node_modules/<package>`` path, so both forms are
    checked before accepting the requested mutation.
    """
    versions: set[str] = set()
    packages = payload.get("packages")
    if isinstance(packages, Mapping):
        suffix = f"node_modules/{package_name}"
        for path, record in packages.items():
            if not isinstance(record, Mapping):
                continue
            normalized_path = str(path).replace("\\", "/").rstrip("/")
            if record.get("name") == package_name or normalized_path.endswith(suffix):
                version = record.get("version")
                if isinstance(version, str) and version.strip():
                    versions.add(version.strip().lstrip("vV"))

    def visit(dependencies: Any) -> None:
        """Walk npm v1 nested dependency records without trusting keys alone."""
        if not isinstance(dependencies, Mapping):
            return
        for name, record in dependencies.items():
            if not isinstance(record, Mapping):
                continue
            package = str(name)
            if package == package_name or record.get("name") == package_name:
                version = record.get("version")
                if isinstance(version, str) and version.strip():
                    versions.add(version.strip().lstrip("vV"))
            visit(record.get("dependencies"))

    visit(payload.get("dependencies"))
    return versions


def _verify_lockfile_mutations(
    sandbox: DockerSandbox,
    checkpoint: _PackageCheckpoint,
    mutations: Sequence[Any],
) -> None:
    """Verify every existing/generated npm lockfile contains requested targets."""
    mutation_by_lockfile: dict[str, list[Any]] = {}
    for path in checkpoint.files:
        if Path(path).name not in {"package-lock.json", "npm-shrinkwrap.json"}:
            continue
        lockfile_parent = Path(path).parent
        for mutation in mutations:
            manifest_parent = Path(mutation.manifest_path).parent
            # Ancestor lockfiles are included in the checkpoint union so an
            # unexpected npm side effect can still be rolled back, but a
            # nested manifest is validated against the lockfile in its own
            # package directory. A repository root lockfile is not evidence
            # for an independent ``frontend/package.json`` installation.
            if lockfile_parent == manifest_parent:
                mutation_by_lockfile.setdefault(path, []).append(mutation)

    for path, path_mutations in sorted(mutation_by_lockfile.items()):
        before = checkpoint.files.get(path)
        content = sandbox.read_file(path)
        if not isinstance(content, str):
            # A lockfile that did not exist before the action is optional: npm
            # may be configured with package-lock=false. Existing lockfiles,
            # however, must survive and prove every requested target.
            if before is not None:
                raise RuntimeError(f"lockfile disappeared after synchronization: {path}")
            continue
        try:
            lockfile = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"lockfile is not valid JSON after synchronization: {path}") from exc
        if not isinstance(lockfile, Mapping):
            raise RuntimeError(f"lockfile must contain a JSON object: {path}")
        for mutation in path_mutations:
            versions = _lockfile_target_versions(lockfile, mutation.package_name)
            expected = mutation.target_version.strip().lstrip("vV")
            if expected not in versions:
                raise RuntimeError(
                    f"lockfile verification failed for {mutation.package_name} in {path}: "
                    f"expected {mutation.target_version}, found {sorted(versions) or 'no package entry'}"
                )


def _sync_package_manifests(
    sandbox: DockerSandbox,
    manifest_paths: Sequence[str],
    package_specs_by_manifest: Mapping[str, Iterable[str]] | None = None,
) -> tuple[bool, str]:
    """Validate and synchronize package manifests.

    Args:
        sandbox: Running Docker workspace sandbox.
        manifest_paths: Repository-relative manifests to synchronize.
        package_specs_by_manifest: Optional legacy parameter retained for compatibility.

    Returns:
        ``(True, "")`` when every synchronization succeeds, otherwise a
        stable tool error and ``False``.
    """
    del package_specs_by_manifest

    def build_command(manifest_path: str, *, dry_run: bool) -> str:
        """Build one deterministic npm manifest validation or sync command."""
        npm_args = [
            "npm",
            "install",
            "--package-lock-only",
            "--ignore-scripts",
            "--legacy-peer-deps",
            "--no-audit",
            "--no-fund",
        ]
        if dry_run:
            npm_args.append("--dry-run")
        workspace_dir = _workspace_dir_for_manifest(manifest_path)
        return f"cd {shlex.quote(workspace_dir)} && {shlex.join(npm_args)}"

    for manifest_path in manifest_paths:
        for dry_run in (True, False):
            cmd = build_command(manifest_path, dry_run=dry_run)
            phase = "preflight validation" if dry_run else "synchronization"
            try:
                result = sandbox.run(cmd, timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001
                return False, _tool_error(
                    "MANIFEST_SYNC_FAILED",
                    f"Manifest {phase} failed for {manifest_path}: {exc}",
                )
            if result.exit_code != 0:
                return False, _tool_error(
                    "MANIFEST_SYNC_FAILED",
                    (
                        f"Manifest {phase} failed for {manifest_path} "
                        f"(exit {result.exit_code}). "
                        f"stdout: {_bounded_command_output(result.stdout)}; "
                        f"stderr: {_bounded_command_output(result.stderr)}"
                    ),
                )
    return True, ""


def apply_multi_package_action(
    sandbox: DockerSandbox,
    action: MultiPackageAction,
    touched_files: set[str],
    checkpoint_store: dict[str, _PackageCheckpoint] | None = None,
) -> tuple[bool, str]:
    """Apply one Supervisor-committed multi-package npm action atomically.

    All manifest edits are staged before any lockfile synchronization. A
    failure in an edit, synchronization, or manifest verification restores the
    union checkpoint for every mutation.

    Args:
        sandbox: Running Docker workspace sandbox.
        action: Immutable action authorized by the Supervisor.
        touched_files: Mutable changed-file projection for the worker result.
        checkpoint_store: Optional mutable store that retains the union
            checkpoint until the caller explicitly accepts the successful
            worker attempt or rolls it back.

    Returns:
        ``(True, "")`` after every mutation validates, otherwise ``(False,
        diagnostic)`` after best-effort rollback.
    """
    manifest_paths = sorted({mutation.manifest_path for mutation in action.package_mutations})
    checkpoint = _capture_package_checkpoint(sandbox, manifest_paths, touched_files)
    checkpoint_key = "__multi_package_action__"
    if checkpoint_store is not None:
        checkpoint_store[checkpoint_key] = checkpoint
    try:
        for mutation in sorted(
            action.package_mutations,
            key=lambda item: (item.manifest_path, item.package_name, item.task_id),
        ):
            rel_manifest = _validate_workspace_path(mutation.manifest_path)
            dependency_path = (
                "pnpm.overrides"
                if mutation.dependency_type == "pnpm_overrides"
                else mutation.dependency_type
            )
            package_expr = f"{dependency_path}[{mutation.package_name}]={mutation.target_version}"
            command = shlex.join(["npm", "pkg", "set", package_expr])
            workspace_dir = _workspace_dir_for_manifest(rel_manifest)
            if workspace_dir != "/workspace":
                command = f"cd {shlex.quote(workspace_dir)} && {command}"
            result = sandbox.run(command)
            if result.exit_code != 0:
                raise RuntimeError(
                    f"manifest edit failed for {mutation.package_name} in {rel_manifest} "
                    f"(exit {result.exit_code}): {_bounded_command_output(result.stderr or result.stdout)}"
                )
            touched_files.add(rel_manifest)

        for manifest_path in manifest_paths:
            synchronized, error = _sync_package_manifests(
                sandbox,
                [manifest_path],
            )
            if not synchronized:
                raise RuntimeError(error)

        for mutation in action.package_mutations:
            content = sandbox.read_file(mutation.manifest_path)
            if not isinstance(content, str):
                raise RuntimeError(f"could not read {mutation.manifest_path} after synchronization")
            payload = json.loads(content)
            dependency_path = (
                "pnpm.overrides"
                if mutation.dependency_type == "pnpm_overrides"
                else mutation.dependency_type
            )
            current: Any = payload
            for segment in dependency_path.split("."):
                current = current.get(segment, {}) if isinstance(current, dict) else {}
            if (
                not isinstance(current, dict)
                or current.get(mutation.package_name) != mutation.target_version
            ):
                raise RuntimeError(
                    f"manifest verification failed for {mutation.package_name} in {mutation.manifest_path}"
                )
        _verify_lockfile_mutations(sandbox, checkpoint, action.package_mutations)
        for path, before in checkpoint.files.items():
            after_value = sandbox.read_file(path)
            after = after_value if isinstance(after_value, str) else None
            if after != before:
                touched_files.add(path)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - atomic boundary owns rollback
        rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
        if checkpoint_store is not None:
            checkpoint_store.pop(checkpoint_key, None)
        suffix = f" Rollback: {rollback_error}" if rollback_error else " Rollback complete."
        return False, f"Multi-package action failed: {exc}.{suffix}"


def _make_modify_batch_npm_dependencies_tool(
    sandbox: DockerSandbox,
    action: MultiPackageAction,
    touched_files: set[str],
    execution_state: dict[str, Any] | None = None,
    package_checkpoints: dict[str, _PackageCheckpoint] | None = None,
) -> Any:
    """Build the sole LLM tool for a Supervisor-committed dependency batch.

    The action is captured in the tool closure. The model can request its
    execution, but cannot provide package names, versions, dependency types,
    or manifest paths. ``apply_multi_package_action`` remains the deterministic
    atomic boundary and owns checkpointing and rollback.

    Args:
        sandbox: Running Docker workspace sandbox.
        action: Immutable action already validated by the Supervisor boundary.
        touched_files: Mutable changed-file projection for the worker result.
        execution_state: Mutable per-worker state used to prevent a second
            execution of the same committed action.
        package_checkpoints: Optional store retaining the successful union
            checkpoint until the worker result is accepted.

    Returns:
        A LangChain tool that executes the captured action exactly once.
    """
    if execution_state is None:
        execution_state = {}

    @tool
    def modify_batch_npm_dependencies() -> str:
        """Modify and synchronize the complete committed npm batch once."""
        if execution_state.get("multi_package_action_executed"):
            return _tool_error(
                "MULTI_PACKAGE_ACTION_ALREADY_EXECUTED",
                "The committed cluster action has already been executed; return control to the Supervisor.",
            )

        # Set the barrier before entering the atomic executor. Even a failed
        # action is returned to the Supervisor for a new committed attempt;
        # the model must not replay a failed batch in the same worker turn.
        execution_state["multi_package_action_executed"] = True
        try:
            if package_checkpoints is None:
                succeeded, error = apply_multi_package_action(sandbox, action, touched_files)
            else:
                succeeded, error = apply_multi_package_action(
                    sandbox,
                    action,
                    touched_files,
                    checkpoint_store=package_checkpoints,
                )
        except Exception as exc:  # noqa: BLE001 - tool boundary returns typed failure
            succeeded = False
            error = f"Multi-package action raised an unexpected error: {exc}"

        execution_state["multi_package_action_succeeded"] = succeeded
        execution_state["multi_package_action_error"] = error or None
        if succeeded:
            cluster_label = action.cluster_id or "unclustered"
            return (
                "SUCCESS: Applied the Supervisor-committed multi-package action "
                f"for cluster {cluster_label}; every mutation and lockfile check passed."
            )
        return _tool_error(
            "MULTI_PACKAGE_ACTION_FAILED",
            error or "The committed multi-package action failed and was rolled back.",
        )

    return modify_batch_npm_dependencies


def _make_modify_and_validate_npm_dependency_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    allowed_target_versions_by_package: Mapping[str, Iterable[str]] | None = None,
    override_required_packages: Iterable[str] | None = None,
    allowed_dependency_types_by_package: Mapping[str, Iterable[str]] | None = None,
    execution_state: dict[str, Any] | None = None,
    package_checkpoints: dict[str, _PackageCheckpoint] | None = None,
):
    """Build the atomic update transaction tool.

    The returned tool performs the manifest edit and synchronization in one
    invocation. A failed edit or synchronization restores the package
    checkpoint before returning an ``ERROR_CODE`` result.
    """
    allowed_manifest_paths_by_package = {
        package_name: set(manifest_paths)
        for package_name, manifest_paths in _normalize_package_manifest_targets(
            package_manifest_paths
        ).items()
    }
    if execution_state is None:
        execution_state = {}
    if package_checkpoints is None:
        package_checkpoints = {}
    override_required_package_names = {
        package_name.strip()
        for package_name in (override_required_packages or [])
        if package_name and package_name.strip()
    }
    allowed_dependency_types = {
        str(package_name).strip(): {
            str(value).strip() for value in dependency_types if str(value).strip()
        }
        for package_name, dependency_types in (allowed_dependency_types_by_package or {}).items()
        if str(package_name).strip()
    }
    allowed_target_versions = {
        str(package_name).strip(): {
            str(version).strip().lstrip("vV") for version in versions if str(version).strip()
        }
        for package_name, versions in (allowed_target_versions_by_package or {}).items()
        if str(package_name).strip()
    }
    supported_dependency_types = (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
        "overrides",
        "resolutions",
        "pnpm_overrides",
    )
    override_types = {"overrides", "resolutions", "pnpm_overrides"}
    safe_pattern = re.compile(r"^[a-zA-Z0-9.\-/@~^*]+$")

    def record_attempt(package_name: str, signature: str) -> str | None:
        """Track one package call and reject duplicate or exhausted attempts."""
        attempts_by_package = execution_state.setdefault(
            "manifest_transaction_attempts_by_package", {}
        )
        signatures_by_package = execution_state.setdefault(
            "manifest_transaction_signatures_by_package", {}
        )
        attempts = int(attempts_by_package.get(package_name, 0))
        signatures = set(signatures_by_package.get(package_name, []))
        if signature in signatures:
            return _tool_error(
                "RETRY_PARAMETERS_UNCHANGED",
                "Retry the package with a different allowed target_version or dependency_type.",
            )
        if attempts >= 3:
            return _tool_error(
                "RETRY_LIMIT_REACHED",
                "At most three combined update attempts are allowed for this package.",
            )
        signatures.add(signature)
        signatures_by_package[package_name] = sorted(signatures)
        attempts_by_package[package_name] = attempts + 1
        execution_state["manifest_transaction_attempts"] = (
            int(execution_state.get("manifest_transaction_attempts", 0)) + 1
        )
        return None

    @tool
    def modify_and_validate_npm_dependency(
        package_name: str,
        target_version: str,
        dependency_type: str,
        manifest_path: str = "package.json",
    ) -> str:
        """Atomically edit and synchronize one npm dependency transaction."""
        package_key = str(package_name or "").strip()
        version = str(target_version or "").strip()
        dependency_key = str(dependency_type or "").strip()

        if not safe_pattern.fullmatch(package_key):
            return _tool_error(
                "INVALID_ARGUMENT",
                f"Invalid package_name {package_key!r}; only safe npm package characters are allowed.",
            )
        if not safe_pattern.fullmatch(version):
            return _tool_error(
                "INVALID_ARGUMENT",
                f"Invalid target_version {version!r}; only safe npm version characters are allowed.",
            )
        if dependency_key not in supported_dependency_types:
            return _tool_error(
                "INVALID_ARGUMENT",
                "dependency_type must be one of: " + ", ".join(supported_dependency_types) + ".",
            )

        package_allowed_versions = allowed_target_versions.get(package_key)
        normalized_version = version.lstrip("vV")
        if (
            package_key in allowed_target_versions
            and normalized_version not in package_allowed_versions
        ):
            allowed = ", ".join(sorted(package_allowed_versions)) or "none"
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"target_version {version!r} is not in the Supervisor-approved candidates: {allowed}.",
            )

        if package_key in override_required_package_names and dependency_key not in override_types:
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"Package {package_key!r} requires a native override dependency type.",
            )
        package_allowed_types = allowed_dependency_types.get(package_key)
        if package_key in allowed_dependency_types and dependency_key not in package_allowed_types:
            allowed = ", ".join(sorted(package_allowed_types)) or "none"
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"dependency_type {dependency_key!r} is not in the Supervisor-approved candidates: {allowed}.",
            )

        try:
            rel_manifest = _validate_workspace_path(manifest_path)
        except ValueError as exc:
            return _tool_error("INVALID_ARGUMENT", str(exc))
        if Path(rel_manifest).name != "package.json":
            return _tool_error("INVALID_ARGUMENT", "manifest_path must point to package.json.")

        allowed_manifest_paths = allowed_manifest_paths_by_package.get(package_key)
        if not allowed_manifest_paths:
            known_packages = ", ".join(sorted(allowed_manifest_paths_by_package))
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"package_name {package_key!r} is not an allowed target; allowed packages: {known_packages}.",
            )
        if rel_manifest not in allowed_manifest_paths:
            allowed = ", ".join(sorted(allowed_manifest_paths))
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"manifest_path {rel_manifest!r} is not allowed for {package_key!r}; allowed paths: {allowed}.",
            )

        if execution_state is not None:
            pending_package = str(execution_state.get("pending_validation_package", "") or "")
            if pending_package and pending_package != package_key:
                return _tool_error(
                    "TARGET_NOT_ALLOWED",
                    f"Package {pending_package!r} has an unfinished update transaction.",
                )

        # Count only a fully validated transaction candidate. Invalid input
        # and disallowed targets must not consume the package's retry budget or
        # leave retry bookkeeping behind.
        signature = "|".join((package_key, normalized_version, dependency_key, rel_manifest))
        attempt_error = record_attempt(package_key, signature)
        if attempt_error:
            return attempt_error

        checkpoint = None
        try:
            if package_checkpoints is not None and package_key not in package_checkpoints:
                package_checkpoints[package_key] = _capture_package_checkpoint(
                    sandbox,
                    allowed_manifest_paths,
                    touched_files,
                )
            checkpoint = package_checkpoints.get(package_key) if package_checkpoints else None

            package_path = (
                "pnpm.overrides" if dependency_key == "pnpm_overrides" else dependency_key
            )
            package_expr = f"{package_path}[{package_key}]={version}"
            npm_cmd = shlex.join(["npm", "pkg", "set", package_expr])
            workspace_dir = _workspace_dir_for_manifest(rel_manifest)
            cmd_str = (
                npm_cmd
                if workspace_dir == "/workspace"
                else (f"cd {shlex.quote(workspace_dir)} && {npm_cmd}")
            )

            logger.info("remedy_tools: modifying npm dependency in sandbox: %s", cmd_str)
            edit_result = sandbox.run(cmd_str)
            if edit_result.exit_code != 0:
                failure = _tool_error(
                    "EDIT_FAILED",
                    (
                        f"Failed to modify npm dependency in {rel_manifest} (exit {edit_result.exit_code}). "
                        f"stdout: {_bounded_command_output(edit_result.stdout)}; "
                        f"stderr: {_bounded_command_output(edit_result.stderr)}"
                    ),
                )
                rollback_error = (
                    _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                    if checkpoint is not None
                    else None
                )
                if package_checkpoints is not None:
                    package_checkpoints.pop(package_key, None)
                if execution_state is not None:
                    execution_state["pending_validation_package"] = None
                return f"{failure} {rollback_error}" if rollback_error else failure

            touched_files.add(rel_manifest)
            if execution_state is not None:
                execution_state["edits_started"] = True
                execution_state["pending_validation_package"] = package_key
                execution_state["validation_calls"] = (
                    int(execution_state.get("validation_calls", 0)) + 1
                )

            manifest_sync_succeeded, sync_error = _sync_package_manifests(
                sandbox,
                sorted(allowed_manifest_paths),
            )
            if not manifest_sync_succeeded:
                rollback_error = (
                    _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                    if checkpoint is not None
                    else None
                )
                if package_checkpoints is not None:
                    package_checkpoints.pop(package_key, None)
                if execution_state is not None:
                    execution_state["pending_validation_package"] = None
                rollback_note = (
                    f" Rollback: {rollback_error}"
                    if rollback_error
                    else " Rollback: package checkpoint restored."
                )
                return f"{sync_error}{rollback_note}"

            if checkpoint is not None:
                for path, before in checkpoint.files.items():
                    after_value = sandbox.read_file(path)
                    after = after_value if isinstance(after_value, str) else None
                    if after != before:
                        touched_files.add(path)
                if package_checkpoints is not None:
                    package_checkpoints.pop(package_key, None)

            if execution_state is not None:
                execution_state["pending_validation_package"] = None
                validated_packages = list(execution_state.get("validated_packages", []) or [])
                if package_key not in validated_packages:
                    validated_packages.append(package_key)
                execution_state["validated_packages"] = validated_packages

            return (
                "SUCCESS: Natively updated and synchronized "
                f"{dependency_key}.{package_key} to {version} in {rel_manifest}; "
                f"validated manifests: {', '.join(sorted(allowed_manifest_paths))}."
            )
        except Exception as exc:  # noqa: BLE001
            rollback_error = (
                _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                if checkpoint is not None
                else None
            )
            if package_checkpoints is not None:
                package_checkpoints.pop(package_key, None)
            if execution_state is not None:
                execution_state["pending_validation_package"] = None
            rollback_note = f" Rollback: {rollback_error}" if rollback_error else ""
            return f"{_tool_error('EDIT_FAILED', str(exc))}{rollback_note}"

    return modify_and_validate_npm_dependency


def _is_prohibited_target(rel_path: str) -> bool:
    """Check if file is a manifest or test file that workaround subagent must not modify."""
    norm = rel_path.replace("\\", "/").lstrip("/")
    basename = Path(norm).name.lower()

    if basename in (
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pom.xml",
        "build.gradle",
        "requirements.txt",
        "pyproject.toml",
    ):
        return True

    parts = norm.lower().split("/")
    if any(p in ("test", "tests", "__tests__", "spec", "specs") for p in parts):
        return True

    return any(
        basename.endswith(ext)
        for ext in (
            ".test.js",
            ".spec.js",
            ".test.ts",
            ".spec.ts",
            ".test.jsx",
            ".spec.jsx",
            ".test.tsx",
            ".spec.tsx",
            "_test.py",
            "_spec.py",
        )
    )


def _is_allowlisted_no_fix_package_file(
    rel_path: str,
    plan_state: Mapping[str, Any] | None,
) -> bool:
    """Return whether a manifest/lockfile is allowed by scoped NO_FIX removal."""
    if (
        not plan_state
        or plan_state.get("no_fix_stage") != NoFixMitigationStage.PACKAGE_REMOVAL.value
    ):
        return False
    if not plan_state.get("package_removal_planned", False):
        return False
    normalized = rel_path.replace("\\", "/").lstrip("/")
    allowlisted = {
        str(path).replace("\\", "/").lstrip("/")
        for path in plan_state.get("no_fix_package_files", [])
    }
    # The validation contract names the complete manifest/lockfile set even
    # when npm leaves one of those files byte-for-byte unchanged.
    return normalized in allowlisted


def _remember_stage_baseline(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any] | None,
    paths: Iterable[str],
) -> None:
    """Capture first-seen file contents for a later NO_FIX stage reset."""
    if plan_state is None:
        return
    baseline = plan_state.setdefault("stage_baseline_snapshots", {})
    absent = set(plan_state.setdefault("stage_baseline_absent_paths", []))
    for path in paths:
        normalized = _validate_workspace_path(path)
        if normalized in baseline or normalized in absent:
            continue
        content = sandbox.read_file(normalized)
        if isinstance(content, str):
            baseline[normalized] = content
        else:
            absent.add(normalized)
    plan_state["stage_baseline_absent_paths"] = sorted(absent)


def _make_remove_no_fix_dependency_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any],
    package_name: str,
    manifest_paths: Sequence[str],
    package_manager: str,
):
    """Build the allowlisted package-removal operation for a NO_FIX attempt.

    The operation intentionally supports only npm at present. Unsupported
    managers fail closed instead of falling back to direct lockfile edits.
    """
    normalized_package = str(package_name or "").strip()
    normalized_manifests = _normalize_manifest_targets(manifest_paths)
    manager = str(package_manager or "").strip().lower()
    checkpoint: _PackageCheckpoint | None = None

    @tool
    def remove_no_fix_dependency(
        requested_package: str,
        manifest_path: str,
    ) -> str:
        """Remove only the configured vulnerable direct dependency safely."""
        nonlocal checkpoint

        if plan_state.get("no_fix_stage") != NoFixMitigationStage.PACKAGE_REMOVAL.value:
            return "NOT_APPLICABLE: package removal is available only during PACKAGE_REMOVAL."
        if not plan_state.get("recorded") or not plan_state.get("package_removal_planned"):
            return (
                "ERROR: [PLAN_VIOLATION] Record an explicit package-removal plan before "
                "calling remove_no_fix_dependency."
            )
        if requested_package.strip() != normalized_package:
            return (
                "ERROR: [ALLOWLIST] remove_no_fix_dependency accepts only the configured "
                f"package '{normalized_package}'."
            )
        try:
            normalized_manifest = _validate_workspace_path(manifest_path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if normalized_manifest not in normalized_manifests:
            return (
                "ERROR: [ALLOWLIST] manifest_path is not an authorized target. "
                f"Allowed paths: {', '.join(normalized_manifests)}."
            )
        if manager != "npm":
            return (
                "NOT_APPLICABLE: package-manager-aware removal is not implemented for "
                f"'{manager or 'unknown'}'; no manifest or lockfile was changed."
            )

        manifest_content = sandbox.read_file(normalized_manifest)
        if not isinstance(manifest_content, str):
            return f"NOT_APPLICABLE: manifest '{normalized_manifest}' is missing or unreadable."
        try:
            manifest_data = json.loads(manifest_content)
        except json.JSONDecodeError as exc:
            return f"NOT_APPLICABLE: manifest '{normalized_manifest}' is invalid JSON: {exc}."
        if not isinstance(manifest_data, dict):
            return f"NOT_APPLICABLE: manifest '{normalized_manifest}' is not a JSON object."

        direct_locations = [
            dep_type
            for dep_type in ("dependencies", "devDependencies", "optionalDependencies")
            if isinstance(manifest_data.get(dep_type), dict)
            and normalized_package in manifest_data[dep_type]
        ]
        if not direct_locations:
            return (
                "NOT_APPLICABLE: the vulnerable package has no removable direct declaration "
                f"in '{normalized_manifest}'. A transitive package must not be removed by editing a lockfile."
            )

        if checkpoint is None:
            checkpoint = _capture_package_checkpoint(
                sandbox,
                normalized_manifests,
                touched_files,
            )
            # Keep an attempt-local rollback anchor as well as the stage
            # baseline.  A failed package-removal attempt must be undone
            # without restoring edits accepted for another task in the same
            # cumulative workspace.
            attempt_snapshots = plan_state.setdefault("attempt_file_snapshots", {})
            attempt_absent = set(plan_state.setdefault("attempt_absent_paths", []))
            for path, content in checkpoint.files.items():
                if isinstance(content, str):
                    attempt_snapshots.setdefault(path, content)
                else:
                    attempt_absent.add(path)
            plan_state["attempt_absent_paths"] = sorted(attempt_absent)
            baseline = plan_state.setdefault("stage_baseline_snapshots", {})
            absent = set(plan_state.setdefault("stage_baseline_absent_paths", []))
            for path, content in checkpoint.files.items():
                if isinstance(content, str):
                    baseline.setdefault(path, content)
                else:
                    absent.add(path)
            plan_state["stage_baseline_absent_paths"] = sorted(absent)

        for dep_type in direct_locations:
            del manifest_data[dep_type][normalized_package]
            if not manifest_data[dep_type]:
                del manifest_data[dep_type]

        try:
            sandbox.write_file(
                normalized_manifest,
                json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n",
            )
        except Exception as exc:  # noqa: BLE001
            rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
            checkpoint = None
            suffix = f" {rollback_error}" if rollback_error else ""
            return f"FAILURE: package manifest write failed: {exc}.{suffix}"

        workspace_dir = _workspace_dir_for_manifest(normalized_manifest)
        command = "npm install --package-lock-only --ignore-scripts"
        command = (
            f"cd {shlex.quote(workspace_dir)} && {command}"
            if workspace_dir != "/workspace"
            else command
        )
        try:
            result = sandbox.run(command, timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            result = None
            sync_failure = f"package-manager synchronization raised {exc}"
        else:
            sync_failure = (
                f"package-manager synchronization failed (exit {result.exit_code}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
                if result.exit_code != 0
                else ""
            )
        if sync_failure:
            rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
            checkpoint = None
            suffix = f" {rollback_error}" if rollback_error else ""
            return f"FAILURE: [ROLLBACK] {sync_failure}{suffix}"

        changed_files: list[str] = []
        for path, before in checkpoint.files.items():
            after = sandbox.read_file(path)
            if after != before:
                touched_files.add(path)
                changed_files.append(path)
        plan_state["package_removal_planned"] = True
        plan_state["package_removal_completed"] = True
        plan_state["package_removal_files"] = sorted(
            set(plan_state.get("package_removal_files", [])) | set(changed_files)
        )
        plan_state["no_fix_package_removed"] = True
        # Package-only plans have no later source edit to advance the
        # lifecycle, so expose validation immediately. Plans containing
        # source replacements remain in EXECUTE until those replacements are
        # applied before the single cumulative validation call.
        plan_state["phase"] = (
            WorkaroundExecutionPhase.VALIDATE.value
            if not plan_state.get("planned_replacements")
            else WorkaroundExecutionPhase.EXECUTE.value
        )
        return (
            "SUCCESS: Removed the configured direct dependency through npm and synchronized "
            f"the lockfile without lifecycle scripts. Changed files: {', '.join(changed_files)}."
        )

    return remove_no_fix_dependency


__all__ = [name for name in globals() if not name.startswith("__")]
