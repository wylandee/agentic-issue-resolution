"""Project-language detection and immutable runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ProjectLanguage(StrEnum):
    """A supported application language for one remediation run."""

    NODEJS = "javascript/nodejs"
    JAVA = "java"


@dataclass(frozen=True)
class LanguageConfig:
    """Language-specific project metadata consumed by later runtime phases."""

    language: ProjectLanguage
    docker_image: str
    manifest_names: tuple[str, ...]
    source_suffixes: frozenset[str]
    install_command: str
    compile_command: str | None
    test_command: str
    test_include_patterns: tuple[str, ...]
    excluded_dirs: frozenset[str]


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

LANGUAGE_CONFIGS: dict[ProjectLanguage, LanguageConfig] = {
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
        install_command="mvn -B -q dependency:resolve",
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
}

_LANGUAGE_ALIASES = {
    "node": ProjectLanguage.NODEJS,
    "nodejs": ProjectLanguage.NODEJS,
    "javascript": ProjectLanguage.NODEJS,
    ProjectLanguage.NODEJS.value: ProjectLanguage.NODEJS,
    ProjectLanguage.JAVA.value: ProjectLanguage.JAVA,
}


def detect_project_language(repo_root: Path) -> ProjectLanguage:
    """Detect a language from manifests at the repository root only.

    Maven takes precedence when both supported manifests exist. A project
    without either manifest keeps the established Node.js default.
    """
    root = Path(repo_root)
    if (root / "pom.xml").is_file():
        return ProjectLanguage.JAVA
    if (root / "package.json").is_file():
        return ProjectLanguage.NODEJS
    return ProjectLanguage.NODEJS


def resolve_project_language(
    repo_root: Path,
    explicit_language: str | None = None,
) -> ProjectLanguage:
    """Resolve an explicit language alias or detect one from root manifests.

    ``None`` and ``auto`` use root-only manifest detection. Explicit values
    outside the registered Node.js and Java aliases fail closed.
    """
    if explicit_language is None:
        return detect_project_language(repo_root)

    normalized = explicit_language.strip().lower()
    if normalized == "auto":
        return detect_project_language(repo_root)
    language = _LANGUAGE_ALIASES.get(normalized)
    if language is None or language not in LANGUAGE_CONFIGS:
        raise ValueError(f"Unsupported project language: {explicit_language!r}")
    return language
