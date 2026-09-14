"""Deterministic unified-diff and package-change extraction helpers."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .report_context import PackageChange, _inline_code_text, _items, _text, _unique_texts, _value

_PACKAGE_FILE_RE = re.compile(
    r"(?:^|/)(?:package\.json|package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml)$",
    re.IGNORECASE,
)

_PACKAGE_LINE_RE = re.compile(r'^\s*"(?P<name>(?:@[^" ]+/)?[^" ]+)"\s*:\s*"(?P<version>[^"\n]+)"')

_LOCKFILE_PACKAGE_RE = re.compile(r'^\s*"(?P<name>(?:node_modules/)?(?:@[^" ]+/)?[^" ]+)"\s*:\s*\{')

_LOCKFILE_RESOLVED_RE = re.compile(r'^\s*"resolved"\s*:\s*"(?P<url>[^"\n]+)"')

_MANIFEST_SECTION_RE = re.compile(
    r'^\s*"(?P<section>dependencies|devDependencies|optionalDependencies|'
    r'peerDependencies|overrides|resolutions)"\s*:\s*\{'
)

_NON_PACKAGE_KEYS = {
    "author",
    "bin",
    "browser",
    "bundleDependencies",
    "bundled",
    "contributors",
    "cpu",
    "description",
    "name",
    "deprecated",
    "directories",
    "engines",
    "engineStrict",
    "files",
    "funding",
    "hasInstallScript",
    "homepage",
    "keywords",
    "license",
    "main",
    "man",
    "module",
    "node",
    "optional",
    "os",
    "peer",
    "peerDependenciesMeta",
    "publishConfig",
    "readme",
    "repository",
    "version",
    "lockfileVersion",
    "requires",
    "integrity",
    "resolved",
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
    "packages",
    "snapshots",
    "scripts",
    "sideEffects",
    "type",
    "types",
    "workspaces",
}


def _normalise_replay_text(value: str) -> str:
    """Normalize line endings before replaying an attempt replacement."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _diff_file_paths(diff: str) -> list[str]:
    """Return repository paths represented by unified-diff file headers."""
    return _unique_texts(
        line[6:].strip() for line in diff.splitlines() if line.startswith("+++ b/")
    )


def _normalized_path(value: Any) -> str:
    """Normalize a repository path for comparisons while preserving display text elsewhere."""
    return _text(value).strip().replace("\\", "/").removeprefix("./")


def _path_matches_any(path: Any, candidates: Sequence[Any]) -> bool:
    """Return whether a path matches one of several relative path candidates."""
    normalized = _normalized_path(path)
    if not normalized:
        return False
    for candidate in candidates:
        candidate_normalized = _normalized_path(candidate)
        if not candidate_normalized:
            continue
        if normalized == candidate_normalized:
            return True
        if normalized.endswith(f"/{candidate_normalized}") or candidate_normalized.endswith(
            f"/{normalized}"
        ):
            return True
    return False


def _package_name_from_resolved_url(value: Any) -> str:
    """Extract an npm package name from a registry tarball URL."""
    match = re.search(
        r"(?:https?://[^/]+/)(?P<package>(?:@[^/]+/)?[^/]+)/-/",
        _text(value),
        re.IGNORECASE,
    )
    return match.group("package") if match else ""


def _diff_line_changes(diff: str) -> dict[str, list[tuple[str, str]]]:
    """Collect added and removed lines by file from a unified diff."""
    changes: dict[str, list[tuple[str, str]]] = {}
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        if line.startswith("--- a/"):
            continue
        if not current_file or line.startswith(("+++", "---")):
            continue
        if line[:1] in {"+", "-"}:
            content = _inline_code_text(_diff_content(line).strip())
            changes.setdefault(current_file, []).append((line[:1], content))
    return changes


def _diff_code_change_details(diff: str, files: Sequence[Any] = ()) -> list[str]:
    """Summarize source-line additions and removals for selected files."""
    details: list[str] = []
    for path, line_changes in _diff_line_changes(diff).items():
        if _PACKAGE_FILE_RE.search(path) or (files and not _path_matches_any(path, files)):
            continue
        index = 0
        while index < len(line_changes):
            prefix, content = line_changes[index]
            if (
                prefix == "-"
                and index + 1 < len(line_changes)
                and line_changes[index + 1][0] == "+"
            ):
                details.append(
                    f"{path}: removed {content or 'blank line'}; added {line_changes[index + 1][1] or 'blank line'}"
                )
                index += 2
                continue
            verb = "added" if prefix == "+" else "removed"
            details.append(f"{path}: {verb} {content or 'blank line'}")
            index += 1
    return details


def _unified_diff_blocks(
    diff: str,
    files: Sequence[Any] | None = None,
    *,
    include_package_files: bool = False,
) -> list[str]:
    """Return complete unified-diff file blocks for selected source files.

    The report uses these blocks for code workarounds so a reviewer can inspect
    the exact source edit. Package manifests are normally excluded because
    their compact version transition is already rendered as prose. Package
    removal attempts opt in so their manifest edits are visible beside the
    source changes.
    """
    blocks: list[tuple[str, list[str]]] = []
    current_path = ""
    current_lines: list[str] = []

    def finish() -> None:
        if current_path and current_lines:
            blocks.append((current_path, list(current_lines)))

    for line in diff.splitlines():
        if line.startswith("--- a/"):
            finish()
            current_path = ""
            current_lines = [line]
            continue
        if current_lines:
            current_lines.append(line)
            if line.startswith("+++ b/"):
                current_path = line[6:].strip()
    finish()

    selected = (
        [_normalized_path(path) for path in files if _normalized_path(path)]
        if files is not None
        else None
    )
    result: list[str] = []
    for path, lines in blocks:
        if not include_package_files and _PACKAGE_FILE_RE.search(path):
            continue
        if selected is not None and (not selected or not _path_matches_any(path, selected)):
            continue
        result.append("\n".join(lines).strip())
    return result


def _replay_diff_blocks(
    metadata: Any,
    files: Sequence[Any] | None = None,
    *,
    include_package_files: bool = False,
) -> list[str]:
    """Build per-attempt unified-diff blocks from committed replacements.

    A final workspace diff can contain edits from several workaround tasks that
    touched the same source file. Replay replacements are attempt-scoped, so
    they are the authoritative source for isolating those edits in a report.
    Multiple replacements for one file are kept in one code block.
    """
    replay_plan = _value(metadata, "replay_plan")
    if replay_plan is None:
        return []

    selected = (
        [_normalized_path(path) for path in files if _normalized_path(path)]
        if files is not None
        else None
    )
    replacements_by_path: dict[str, list[Any]] = {}
    for edit_set in _items(_value(replay_plan, "successful_edit_sets")):
        for replacement in _items(_value(edit_set, "replacements")):
            path = _normalized_path(_value(replacement, "file_path"))
            if not path or (not include_package_files and _PACKAGE_FILE_RE.search(path)):
                continue
            if selected is not None and (not selected or not _path_matches_any(path, selected)):
                continue
            replacements_by_path.setdefault(path, []).append(replacement)

    snapshots = _value(replay_plan, "pre_attempt_snapshots")
    snapshot_by_path = (
        {
            _normalized_path(path): content
            for path, content in snapshots.items()
            if _normalized_path(path)
        }
        if isinstance(snapshots, Mapping)
        else {}
    )
    blocks: list[str] = []
    for path, replacements in replacements_by_path.items():
        baseline = snapshot_by_path.get(path)
        if isinstance(baseline, str):
            current = baseline
            replay_valid = True
            for replacement in replacements:
                old_text = _normalise_replay_text(_text(_value(replacement, "old_text")))
                new_text = _normalise_replay_text(_text(_value(replacement, "new_text")))
                expected = _value(replacement, "expected_occurrences", 1) or 1
                count = _normalise_replay_text(current).count(old_text)
                if count == expected:
                    current = _normalise_replay_text(current).replace(old_text, new_text, expected)
                elif count == 0 and _normalise_replay_text(current).count(new_text) >= expected:
                    # The edit is already present in a cumulative retry
                    # workspace; it contributes no new net hunk.
                    continue
                else:
                    replay_valid = False
                    break

            if replay_valid and current != baseline:
                before_lines = baseline.splitlines()
                after_lines = current.splitlines()
                hunks: list[str] = []
                matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
                for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
                    if tag == "equal":
                        continue
                    hunks.extend(
                        [
                            "@@",
                            *(f"-{line}" for line in before_lines[before_start:before_end]),
                            *(f"+{line}" for line in after_lines[after_start:after_end]),
                        ]
                    )
                if hunks:
                    blocks.append("\n".join([f"--- a/{path}", f"+++ b/{path}", *hunks]))
                    continue

        # Legacy/sparse replay plans may not carry a baseline snapshot. Keep
        # their exact replacement evidence, but group all replacements for a
        # file into one code block.
        hunks = []
        for replacement in replacements:
            old_lines = _text(_value(replacement, "old_text")).splitlines() or [""]
            new_lines = _text(_value(replacement, "new_text")).splitlines() or [""]
            hunks.extend(
                [
                    "@@",
                    *(f"-{line}" for line in old_lines),
                    *(f"+{line}" for line in new_lines),
                ]
            )
        if hunks:
            blocks.append("\n".join([f"--- a/{path}", f"+++ b/{path}", *hunks]))
    return blocks


def _diff_block_paths(blocks: Sequence[str]) -> set[str]:
    """Return normalized file paths represented by unified-diff blocks."""
    paths: set[str] = set()
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("+++ b/"):
                paths.add(_normalized_path(line[6:].strip()))
                break
    return paths


def _diff_change_counts(block: str) -> tuple[Counter[str], Counter[str]]:
    """Return removed and added diff lines from one unified-diff block."""
    removed: Counter[str] = Counter()
    added: Counter[str] = Counter()
    for line in block.splitlines():
        if line.startswith(("--- a/", "+++ b/", "@@")):
            continue
        if line.startswith("-"):
            removed[line[1:]] += 1
        elif line.startswith("+"):
            added[line[1:]] += 1
    return removed, added


def _replay_blocks_are_backed_by_diff(
    final_diff: str,
    replay_blocks: Sequence[str],
) -> bool:
    """Return whether each replay change is present in the emitted final diff."""
    if not replay_blocks:
        return False

    final_blocks_by_path = {
        path: block
        for block in _unified_diff_blocks(final_diff, None, include_package_files=True)
        for path in _diff_block_paths([block])
    }
    for replay_block in replay_blocks:
        replay_paths = _diff_block_paths([replay_block])
        if not replay_paths:
            return False
        expected_removed, expected_added = _diff_change_counts(replay_block)
        for path in replay_paths:
            final_block = final_blocks_by_path.get(path)
            if final_block is None:
                return False
            actual_removed, actual_added = _diff_change_counts(final_block)
            if expected_removed - actual_removed or expected_added - actual_added:
                return False
    return True


def _diff_content(line: str) -> str:
    """Remove the unified-diff prefix from one line."""
    return line[1:] if line[:1] in {"+", "-", " "} else line


def _line_indent(line: str) -> int:
    """Return the leading-space count for diff content."""
    return len(line) - len(line.lstrip())


def _record_package_change(
    records: dict[str, dict[str, str]],
    name: str,
    version: str,
    prefix: str,
    file_path: str,
    section: str = "",
) -> None:
    """Record one added or removed package version from a diff line."""
    if name in _NON_PACKAGE_KEYS:
        return
    record = records.setdefault(
        name,
        {"old": "", "new": "", "file": file_path, "section": section},
    )
    recorded_files = [item.strip() for item in record.get("file", "").split(";")]
    if file_path and file_path not in recorded_files:
        record["file"] = "; ".join([*recorded_files, file_path])
    if section and not record.get("section"):
        record["section"] = section
    if prefix == "+":
        record["new"] = version
    elif prefix == "-":
        record["old"] = version


def _package_changes(diff: str) -> list[PackageChange]:
    """Extract direct manifest changes and classified lockfile changes.

    Only dependency-section entries from ``package.json`` are considered
    direct changes. Lockfiles contribute only package ``version`` fields under
    package entries; metadata such as ``engines.node`` or ``deprecated`` is
    deliberately excluded.
    """
    manifest_records: dict[str, dict[str, str]] = {}
    lockfile_records: dict[str, dict[str, str]] = {}
    current_file = ""
    manifest_section: tuple[str, int] | None = None
    lockfile_package: tuple[str, int] | None = None
    lockfile_version_candidates: list[tuple[int, str, str, str, str]] = []

    for line_number, line in enumerate(diff.splitlines()):
        if line.startswith("@@"):
            manifest_section = None
            lockfile_package = None
            lockfile_version_candidates.clear()
            continue
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            manifest_section = None
            lockfile_package = None
            lockfile_version_candidates.clear()
            continue
        if line.startswith("--- a/") or not _PACKAGE_FILE_RE.search(current_file):
            continue
        prefix = line[:1]
        content = _diff_content(line)
        indent = _line_indent(content)

        if current_file.lower().endswith("package.json") and not current_file.lower().endswith(
            "package-lock.json"
        ):
            section_match = _MANIFEST_SECTION_RE.match(content)
            if section_match:
                manifest_section = (section_match.group("section"), indent)
                continue
            if manifest_section is not None:
                section_name, section_indent = manifest_section
                if content.strip() and indent <= section_indent:
                    manifest_section = None
                elif prefix in {"+", "-"}:
                    package_match = _PACKAGE_LINE_RE.match(content)
                    if package_match:
                        _record_package_change(
                            manifest_records,
                            package_match.group("name"),
                            package_match.group("version"),
                            prefix,
                            current_file,
                            section_name,
                        )
            elif prefix in {"+", "-"}:
                # Sparse diffs may omit the surrounding dependency-section
                # context. The metadata deny-list keeps this fallback bounded.
                package_match = _PACKAGE_LINE_RE.match(content)
                if package_match:
                    _record_package_change(
                        manifest_records,
                        package_match.group("name"),
                        package_match.group("version"),
                        prefix,
                        current_file,
                        "",
                    )
            continue

        if "lock" in current_file.lower():
            package_match = _LOCKFILE_PACKAGE_RE.match(content)
            if package_match:
                raw_name = package_match.group("name")
                name = raw_name.removeprefix("node_modules/")
                if name and name not in _NON_PACKAGE_KEYS:
                    lockfile_package = (name, indent)
                    lockfile_version_candidates.clear()
                elif lockfile_package and indent <= lockfile_package[1]:
                    lockfile_package = None
            elif (
                lockfile_package
                and content.strip().startswith("}")
                and indent <= lockfile_package[1]
            ):
                lockfile_package = None

            if prefix in {"+", "-"} and lockfile_package:
                version_match = _PACKAGE_LINE_RE.match(content)
                if (
                    version_match
                    and version_match.group("name") == "version"
                    and indent > lockfile_package[1]
                ):
                    version = version_match.group("version")
                    _record_package_change(
                        lockfile_records,
                        lockfile_package[0],
                        version,
                        prefix,
                        current_file,
                        "lockfile",
                    )
                    lockfile_version_candidates.append(
                        (line_number, prefix, current_file, lockfile_package[0], version)
                    )

            resolved_match = _LOCKFILE_RESOLVED_RE.match(content)
            if prefix in {"+", "-"} and resolved_match:
                resolved_name = _package_name_from_resolved_url(resolved_match.group("url"))
                candidate = next(
                    (
                        item
                        for item in reversed(lockfile_version_candidates)
                        if item[1] == prefix
                        and item[2] == current_file
                        and 0 < line_number - item[0] <= 4
                    ),
                    None,
                )
                if resolved_name and candidate is not None and candidate[3] != resolved_name:
                    _, _, _, previous_name, version = candidate
                    previous_record = lockfile_records.get(previous_name)
                    version_key = "new" if prefix == "+" else "old"
                    if previous_record is not None and previous_record.get(version_key) == version:
                        previous_record[version_key] = ""
                    _record_package_change(
                        lockfile_records,
                        resolved_name,
                        version,
                        prefix,
                        current_file,
                        "lockfile",
                    )

    direct_names = set(manifest_records)
    changes: list[PackageChange] = []
    for name in sorted(manifest_records):
        record = manifest_records[name]
        if record["old"] or record["new"]:
            evidence_file = record["file"]
            if name in lockfile_records:
                evidence_file += f"; {lockfile_records[name]['file']}"
            changes.append(
                PackageChange(
                    name,
                    record["old"],
                    record["new"],
                    evidence_file,
                    "direct",
                    record.get("section", ""),
                )
            )
    for name in sorted(lockfile_records):
        if name in direct_names:
            continue
        record = lockfile_records[name]
        if record["old"] or record["new"]:
            changes.append(
                PackageChange(
                    name,
                    record["old"],
                    record["new"],
                    record["file"],
                    "transitive",
                    record.get("section", "lockfile"),
                )
            )
    return sorted(changes, key=lambda change: (change.scope != "direct", change.name))


__all__ = [
    "PackageChange",
]
