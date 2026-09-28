"""Project language detection and execution profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType


class ProjectLanguage(str, Enum):  # noqa: UP042
    """Supported project languages for remediation execution."""

    NODEJS = "javascript/nodejs"
    PYTHON = "python"


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    """Execution settings selected for one project language."""

    docker_image: str
    root_manifest_markers: tuple[str, ...]
    source_suffixes: frozenset[str]


LANGUAGE_CONFIGS: Mapping[ProjectLanguage, LanguageConfig] = MappingProxyType(
    {
        ProjectLanguage.NODEJS: LanguageConfig(
            docker_image="node:22",
            root_manifest_markers=("package.json",),
            source_suffixes=frozenset(
                {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
            ),
        ),
        ProjectLanguage.PYTHON: LanguageConfig(
            docker_image="python:3.11-slim",
            root_manifest_markers=(
                "requirements.txt",
                "setup.py",
                "setup.cfg",
                "Pipfile",
                "pyproject.toml",
            ),
            source_suffixes=frozenset({".py"}),
        ),
    }
)

_LANGUAGE_ALIASES: Mapping[str, ProjectLanguage] = MappingProxyType(
    {
        "python": ProjectLanguage.PYTHON,
        "python3": ProjectLanguage.PYTHON,
        "nodejs": ProjectLanguage.NODEJS,
        "node": ProjectLanguage.NODEJS,
        "javascript": ProjectLanguage.NODEJS,
        "javascript/nodejs": ProjectLanguage.NODEJS,
    }
)


def detect_project_language(repo_root: Path) -> ProjectLanguage:
    """Detect the root project's language from root-level manifests only.

    Node.js takes precedence in mixed roots to preserve the existing runtime
    behavior. Projects without a recognized marker retain the Node.js default.

    Args:
        repo_root: Root directory of the project being remediated.

    Returns:
        The detected execution language.
    """
    root = Path(repo_root)
    if (root / "package.json").is_file():
        return ProjectLanguage.NODEJS
    if any(
        (root / marker).is_file()
        for marker in LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].root_manifest_markers
    ):
        return ProjectLanguage.PYTHON
    return ProjectLanguage.NODEJS


def resolve_project_language(
    repo_root: Path,
    explicit_language: str | None = None,
) -> ProjectLanguage:
    """Resolve an alias or free-form context value to an execution profile.

    Recognized aliases override root detection. Unknown free-form language
    values remain available as triage metadata but do not select an execution
    profile, so this function falls back to root detection for them.

    Args:
        repo_root: Root directory of the project being remediated.
        explicit_language: Optional recognized language alias or context value.

    Returns:
        The selected execution language.
    """
    if explicit_language is not None:
        language = _LANGUAGE_ALIASES.get(explicit_language.strip().casefold())
        if language is not None:
            return language
    return detect_project_language(repo_root)
