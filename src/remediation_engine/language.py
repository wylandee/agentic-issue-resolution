"""Project language detection and execution profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


class ProjectLanguage(StrEnum):
    """Supported project languages for remediation execution."""

    NODEJS = "javascript/nodejs"
    JAVA = "java"
    PYTHON = "python"


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    """Execution settings selected for one project language."""

    language: ProjectLanguage
    docker_image: str
    manifest_names: tuple[str, ...]
    source_suffixes: frozenset[str]
    install_command: str
    compile_command: str | None
    test_command: str
    test_include_patterns: tuple[str, ...]
    excluded_dirs: frozenset[str]

    @property
    def root_manifest_markers(self) -> tuple[str, ...]:
        """Return the root-level manifest markers used by language detection."""
        return self.manifest_names


_NODE_SOURCE_SUFFIXES = frozenset({".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})
_NODE_TEST_INCLUDE_PATTERNS = (
    "**/test/**",
    "**/tests/**",
    "**/__tests__/**",
    "**/spec/**",
    "**/specs/**",
    "**/*.test.js",
    "**/*.spec.js",
    "**/*.test.mjs",
    "**/*.spec.mjs",
    "**/*.test.cjs",
    "**/*.spec.cjs",
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/*.test.tsx",
    "**/*.spec.tsx",
    "**/*.test.jsx",
    "**/*.spec.jsx",
)

LANGUAGE_CONFIGS: Mapping[ProjectLanguage, LanguageConfig] = MappingProxyType(
    {
        ProjectLanguage.NODEJS: LanguageConfig(
            language=ProjectLanguage.NODEJS,
            docker_image="node:22",
            manifest_names=("package.json",),
            source_suffixes=_NODE_SOURCE_SUFFIXES,
            install_command="npm install --package-lock=true",
            compile_command=None,
            test_command="npm test",
            test_include_patterns=_NODE_TEST_INCLUDE_PATTERNS,
            excluded_dirs=frozenset(),
        ),
        ProjectLanguage.JAVA: LanguageConfig(
            language=ProjectLanguage.JAVA,
            docker_image="maven:3.9-eclipse-temurin-17",
            manifest_names=("pom.xml",),
            source_suffixes=frozenset({".java"}),
            install_command="mvn -B -q -DskipTests clean package",
            compile_command="mvn -B -q -DskipTests compile",
            test_command="mvn -B test -Dsurefire.useFile=false",
            test_include_patterns=(
                "**/Test*.java",
                "**/*Test.java",
                "**/*Tests.java",
                "**/*TestCase.java",
            ),
            excluded_dirs=frozenset({"target"}),
        ),
        ProjectLanguage.PYTHON: LanguageConfig(
            language=ProjectLanguage.PYTHON,
            docker_image="remediation-engine-python:3.11",
            manifest_names=(
                "requirements.txt",
                "setup.py",
                "setup.cfg",
                "Pipfile",
                "pyproject.toml",
            ),
            source_suffixes=frozenset({".py"}),
            install_command="python -m pip install -e .",
            compile_command=None,
            test_command=".venv/bin/python -m pytest",
            test_include_patterns=("**/test_*.py", "**/*_test.py", "**/tests/**"),
            excluded_dirs=frozenset({"__pycache__", ".venv", "venv", ".remedy-pipenv"}),
        ),
    }
)

_LANGUAGE_ALIASES: Mapping[str, ProjectLanguage] = MappingProxyType(
    {
        "node": ProjectLanguage.NODEJS,
        "nodejs": ProjectLanguage.NODEJS,
        "javascript": ProjectLanguage.NODEJS,
        ProjectLanguage.NODEJS.value: ProjectLanguage.NODEJS,
        "java": ProjectLanguage.JAVA,
        ProjectLanguage.JAVA.value: ProjectLanguage.JAVA,
        "python": ProjectLanguage.PYTHON,
        "python3": ProjectLanguage.PYTHON,
        ProjectLanguage.PYTHON.value: ProjectLanguage.PYTHON,
    }
)


def detect_project_language(repo_root: Path) -> ProjectLanguage:
    """Detect language from root manifests, keeping JavaScript precedence.

    Args:
        repo_root: Root directory of the project being remediated.

    Returns:
        The detected language. JavaScript remains the default when no supported
        root manifest is present and takes precedence in mixed-language roots.
    """
    root = Path(repo_root)
    if (root / "package.json").is_file():
        return ProjectLanguage.NODEJS
    if (root / "pom.xml").is_file():
        return ProjectLanguage.JAVA
    python_markers = LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].root_manifest_markers
    if any((root / marker).is_file() for marker in python_markers):
        return ProjectLanguage.PYTHON
    return ProjectLanguage.NODEJS


def resolve_project_language(
    repo_root: Path,
    explicit_language: str | None = None,
) -> ProjectLanguage:
    """Resolve a recognized alias or fall back to root-manifest detection.

    Args:
        repo_root: Root directory of the project being remediated.
        explicit_language: Optional language alias or free-form context value.

    Returns:
        A supported project language. Unknown context values do not select an
        execution profile and therefore retain the detected/default language.
    """
    if explicit_language is not None:
        language = _LANGUAGE_ALIASES.get(explicit_language.strip().casefold())
        if language is not None:
            return language
    return detect_project_language(repo_root)
