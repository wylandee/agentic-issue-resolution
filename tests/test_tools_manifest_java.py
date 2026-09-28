from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from remediation_engine.contracts import (
    IssueSource,
    IssueType,
    NoFixMitigationStage,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.state import normalize_group_paths
from remediation_engine.orchestration.tools_manifest_java import (
    _make_modify_and_validate_maven_dependency_tool,
    _MavenPackageCheckpoint,
    _remove_maven_no_fix_dependency_transaction,
)
from remediation_engine.tools.maven_manifest_locator import (
    add_dependency_management_entry,
    locate_maven_from_issue,
    remove_direct_dependency_entry,
    update_pom_dependency_version,
    update_pom_property_value,
)

GAV = "org.example:widget"
_NS = 'xmlns="http://maven.apache.org/POM/4.0.0"'


def _issue(**updates) -> VulnerabilityIssue:
    values = {
        "source": IssueSource.ODC,
        "issue_type": IssueType.SCA,
        "severity": Severity.HIGH,
        "ecosystem": "maven",
        "package_name": GAV,
        "package_version": "1.0.0",
        "purl": "pkg:maven/org.example/widget@1.0.0",
        "raw_payload": {"filePath": "/scan/pom.xml"},
    }
    values.update(updates)
    return VulnerabilityIssue(**values)


def _dependency(version: str | None = "1.0.0", extra: str = "") -> str:
    declared = f"<version>{version}</version>" if version is not None else ""
    return (
        "<dependency><groupId>org.example</groupId><artifactId>widget</artifactId>"
        f"{declared}{extra}</dependency>"
    )


def _pom(body: str = "", *, artifact: str = "app") -> str:
    return (
        f"<project {_NS}>\n"
        "  <modelVersion>4.0.0</modelVersion>\n"
        "  <groupId>com.acme</groupId>\n"
        f"  <artifactId>{artifact}</artifactId>\n"
        "  <version>1.0.0</version>\n"
        f"{body}"
        "</project>\n"
    )


class _Sandbox:
    def __init__(
        self, files: dict[str, str], *, sync_exit: int = 0, raise_sync: Exception | None = None
    ):
        self.files = dict(files)
        self.sync_exit = sync_exit
        self.raise_sync = raise_sync
        self.commands: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def read_file(self, path: str) -> str | None:
        return self.files.get(path)

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content
        self.writes.append((path, content))

    def run(self, command: str, timeout: int = 0):
        self.commands.append(command)
        if command.startswith("rm -f -- "):
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        if self.raise_sync is not None:
            raise self.raise_sync
        return SimpleNamespace(
            exit_code=self.sync_exit,
            stdout="simulated stdout",
            stderr="simulated stderr" if self.sync_exit else "",
        )


def _updater(
    sandbox: _Sandbox,
    touched: set[str],
    *,
    paths: tuple[str, ...] = ("pom.xml",),
    version: str = "2.0.0",
    dependency_type: str = "dependencies",
    state: dict | None = None,
):
    return _make_modify_and_validate_maven_dependency_tool(
        sandbox,
        touched,
        {GAV: paths},
        {GAV: {version}},
        {GAV: {dependency_type}},
        execution_state=state,
    )


def _invoke(updater, *, version="2.0.0", kind="dependencies", path="pom.xml", package=GAV):
    return updater.invoke(
        {
            "package_name": package,
            "target_version": version,
            "dependency_type": kind,
            "manifest_path": path,
        }
    )


def _module_pom(name: str, body: str, *, parent: str = "../pom.xml") -> str:
    return (
        f"<project {_NS}>\n"
        "  <modelVersion>4.0.0</modelVersion>\n"
        "  <parent><groupId>com.acme</groupId><artifactId>reactor</artifactId>"
        f"<version>1.0.0</version><relativePath>{parent}</relativePath></parent>\n"
        f"  <artifactId>{name}</artifactId>\n"
        f"{body}"
        "</project>\n"
    )


def test_direct_explicit_version_is_updated_in_place_and_synced_with_quoted_pom():
    path = "service with space/pom.xml"
    original = _pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({path: original})
    touched = {"src/App.java"}
    updater = _updater(sandbox, touched, paths=(path,))

    result = _invoke(updater, path=path)

    assert result.startswith("SUCCESS:")
    assert "<version>2.0.0</version>" in sandbox.files[path]
    assert "<artifactId>app</artifactId>" in sandbox.files[path]
    assert touched == {"src/App.java", path}
    assert sandbox.commands == ["mvn -B -q -f 'service with space/pom.xml' dependency:resolve"]
    assert updater.name == "modify_and_validate_maven_dependency"
    assert "manifest_path" in updater.args


def test_direct_managed_and_transitive_override_use_dependency_management():
    managed_pom = _pom(
        "  <dependencyManagement><dependencies>"
        f"{_dependency('1.0.0')}"
        "</dependencies></dependencyManagement>\n"
        f"  <dependencies>{_dependency(None)}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": managed_pom})
    result = _invoke(
        _updater(sandbox, set(), dependency_type="dependencyManagement"),
        kind="dependencyManagement",
    )

    assert result.startswith("SUCCESS:")
    assert sandbox.files["pom.xml"].count("<version>2.0.0</version>") == 1
    assert "<dependencies>" in sandbox.files["pom.xml"]

    transitive = _pom(
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>other</artifactId><version>1.0.0</version></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": transitive})
    result = _invoke(
        _updater(sandbox, set(), dependency_type="dependencyManagement"),
        kind="dependencyManagement",
    )

    assert result.startswith("SUCCESS:")
    assert "<dependencyManagement>" in sandbox.files["pom.xml"]
    assert (
        "<groupId>org.example</groupId><artifactId>widget</artifactId>" in sandbox.files["pom.xml"]
    )
    assert "<version>2.0.0</version>" in sandbox.files["pom.xml"]


def test_external_bom_is_overridden_by_new_local_managed_entry_only():
    bom_import = (
        "<dependency><groupId>org.example</groupId><artifactId>widgets-bom</artifactId>"
        "<version>5.0.0</version><type>pom</type><scope>import</scope></dependency>"
    )
    root = _pom(
        "  <dependencyManagement><dependencies>"
        f"{bom_import}"
        "</dependencies></dependencyManagement>\n"
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>widget</artifactId></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": root})
    result = _invoke(
        _updater(sandbox, set(), dependency_type="dependencyManagement"),
        kind="dependencyManagement",
    )

    assert result.startswith("SUCCESS:")
    assert "widgets-bom" in sandbox.files["pom.xml"]
    assert sandbox.files["pom.xml"].count("<artifactId>widget</artifactId>") == 2
    assert sandbox.files["pom.xml"].count("<version>2.0.0</version>") == 1


def test_unsafe_transitive_insertion_fails_without_rewriting_inline_pom():
    source = (
        f"<project {_NS}><modelVersion>4.0.0</modelVersion><groupId>com.acme</groupId>"
        "<artifactId>app</artifactId><version>1.0.0</version></project>"
    )
    sandbox = _Sandbox({"pom.xml": source})
    result = _invoke(
        _updater(sandbox, set(), dependency_type="dependencyManagement"),
        kind="dependencyManagement",
    )

    assert "ERROR_CODE: POM_INSERTION_UNSAFE:" in result
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


def test_parent_property_owner_and_multi_module_different_properties_are_updated():
    root = _pom(
        "  <modules><module>one</module><module>two</module></modules>\n",
        artifact="reactor",
    )
    one = _module_pom(
        "one",
        "  <properties><one.version>1.0.0</one.version></properties>\n"
        f"  <dependencies>{_dependency('${one.version}')}</dependencies>\n",
    )
    two = _module_pom(
        "two",
        "  <properties><two.version>1.0.0</two.version></properties>\n"
        f"  <dependencies>{_dependency('${two.version}')}</dependencies>\n",
    )
    paths = ("pom.xml", "one/pom.xml", "two/pom.xml")
    sandbox = _Sandbox({"pom.xml": root, "one/pom.xml": one, "two/pom.xml": two})
    updater = _updater(sandbox, set(), paths=paths)

    result = _invoke(updater, path="one/pom.xml")

    assert result.startswith("SUCCESS:")
    assert "<one.version>2.0.0</one.version>" in sandbox.files["one/pom.xml"]
    assert "<two.version>2.0.0</two.version>" in sandbox.files["two/pom.xml"]


def test_parent_property_owner_is_updated_only_in_authorized_parent_pom():
    root = _pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        "  <modules><module>module</module></modules>\n",
        artifact="reactor",
    )
    child = _module_pom(
        "module",
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n",
    )
    sandbox = _Sandbox({"pom.xml": root, "module/pom.xml": child})
    touched: set[str] = set()
    updater = _updater(sandbox, touched, paths=("pom.xml", "module/pom.xml"))

    result = _invoke(updater, path="module/pom.xml")

    assert result.startswith("SUCCESS:")
    assert "<widget.version>2.0.0</widget.version>" in sandbox.files["pom.xml"]
    assert sandbox.files["module/pom.xml"] == child
    assert touched == {"pom.xml"}


def test_parent_owned_property_is_localized_and_normalized_as_repo_relative(tmp_path: Path):
    root = tmp_path / "pom.xml"
    root.write_text(
        _pom(
            "  <properties><widget.version>1.0.0</widget.version></properties>\n"
            "  <modules><module>module</module></modules>\n",
            artifact="reactor",
        ),
        encoding="utf-8",
    )
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "pom.xml").write_text(
        _module_pom(
            "module",
            f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n",
        ),
        encoding="utf-8",
    )
    localized = locate_maven_from_issue(
        _issue(raw_payload={"filePath": "/scan/module/src/main/java/Widget.java"}), tmp_path
    )
    group = VulnerabilityGroup(
        group_id="sca:maven:org.example:widget",
        issue_type=IssueType.SCA,
        vulnerable_component=GAV,
        file_path=localized.manifest_file,
        file_paths=[localized.manifest_file],
        representative_issue_id=localized.issue.id,
        issues=[localized.issue],
        localized_issues=[localized.model_copy(update={"version_property_file": str(root)})],
    )

    normalized = normalize_group_paths([group], str(tmp_path))[0]

    assert localized.version_property_name == "widget.version"
    assert localized.version_property_file == "pom.xml"
    assert normalized.localized_issues[0].version_property_file == "pom.xml"


def test_property_shared_and_unresolved_fail_closed_without_writes():
    shared = _pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
        "  <name>${widget.version}</name>\n"
    )
    sandbox = _Sandbox({"pom.xml": shared})
    touched = {"src/Main.java"}
    result = _invoke(_updater(sandbox, touched))

    assert "ERROR_CODE: POM_PROPERTY_SHARED:" in result
    assert sandbox.files["pom.xml"] == shared
    assert touched == {"src/Main.java"}
    assert sandbox.commands == []

    unresolved = _pom(f"  <dependencies>{_dependency('${external.version}')}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": unresolved})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_PROPERTY_UNRESOLVED:" in result
    assert sandbox.files["pom.xml"] == unresolved

    cyclic = _pom(
        "  <properties><widget.version>${widget.version}</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": cyclic})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_PROPERTY_UNRESOLVED:" in result
    assert sandbox.files["pom.xml"] == cyclic


@pytest.mark.parametrize(
    "dependency, extra_body",
    [
        (_dependency() + _dependency(), ""),
        (
            _dependency(),
            "  <profiles><profile><id>active</id><dependencies>"
            + _dependency()
            + "</dependencies></profile></profiles>\n",
        ),
        (_dependency(extra="<classifier>tests</classifier>"), ""),
    ],
)
def test_duplicate_profile_and_classifier_targets_are_ambiguous(dependency, extra_body):
    source = _pom(f"  <dependencies>{dependency}</dependencies>\n{extra_body}")
    sandbox = _Sandbox({"pom.xml": source})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_TARGET_AMBIGUOUS:" in result
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


@pytest.mark.parametrize(
    "source",
    [
        '<!DOCTYPE project [<!ENTITY x "boom">]><project><modelVersion>4.0.0</modelVersion></project>',
        "<project><modelVersion>4.0.0</project>",
    ],
)
def test_dtd_and_malformed_pom_fail_with_invalid_xml(source):
    sandbox = _Sandbox({"pom.xml": source})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_INVALID_XML:" in result
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


def test_version_type_gav_and_path_authorization_fail_closed():
    source = _pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source})
    updater = _updater(sandbox, set())

    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, package="org.example:widget;id")
    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, version="2.0.0;touch /tmp/x")
    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, version="2.1.0-rc1")
    assert "ERROR_CODE: TARGET_NOT_ALLOWED:" in _invoke(updater, path="not-authorized/pom.xml")
    assert "ERROR_CODE: TARGET_NOT_ALLOWED:" in _invoke(updater, version="3.0.0")
    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, path="../pom.xml")
    assert "ERROR_CODE: TARGET_NOT_ALLOWED:" in _invoke(updater, kind="dependencyManagement")
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


def test_failed_maven_sync_restores_every_authorized_pom_and_touched_files():
    root = _pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        "  <modules><module>module</module></modules>\n",
        artifact="reactor",
    )
    child = _module_pom(
        "module",
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n",
    )
    sandbox = _Sandbox(
        {"pom.xml": root, "module/pom.xml": child},
        sync_exit=1,
    )
    touched = {"src/AlreadyChanged.java", "module/pom.xml"}
    result = _invoke(
        _updater(sandbox, touched, paths=("pom.xml", "module/pom.xml")),
        path="module/pom.xml",
    )

    assert "ERROR_CODE: MANIFEST_SYNC_FAILED:" in result
    assert sandbox.files == {"pom.xml": root, "module/pom.xml": child}
    assert touched == {"src/AlreadyChanged.java", "module/pom.xml"}
    assert len(sandbox.commands) == 1
    assert "dependency:resolve" in sandbox.commands[0]


def test_maven_sync_timeout_restores_the_pom_checkpoint():
    source = _pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source}, raise_sync=TimeoutError("timed out"))
    touched = {"src/Existing.java"}

    result = _invoke(_updater(sandbox, touched))

    assert "ERROR_CODE: MANIFEST_SYNC_FAILED:" in result
    assert sandbox.files["pom.xml"] == source
    assert touched == {"src/Existing.java"}


def test_success_marks_only_changed_authorized_poms_touched():
    source = _pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source})
    touched = {"src/Existing.java"}
    result = _invoke(_updater(sandbox, touched))

    assert result.startswith("SUCCESS:")
    assert touched == {"src/Existing.java", "pom.xml"}
    assert len(sandbox.commands) == 1


def test_direct_only_no_fix_removal_and_sync_rollback():
    source = _pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    plan = {
        "no_fix_stage": NoFixMitigationStage.PACKAGE_REMOVAL.value,
        "recorded": True,
        "package_removal_planned": True,
    }
    sandbox = _Sandbox({"pom.xml": source})
    touched: set[str] = set()
    result = _remove_maven_no_fix_dependency_transaction(
        sandbox,
        touched,
        plan,
        GAV,
        ["pom.xml"],
        GAV,
        "pom.xml",
    )

    assert result.startswith("SUCCESS:")
    assert "<artifactId>widget</artifactId>" not in sandbox.files["pom.xml"]
    assert touched == {"pom.xml"}
    assert plan["package_removal_completed"] is True
    assert plan["phase"] == "VALIDATE"

    transitive = _pom(
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>other</artifactId><version>1.0.0</version></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": transitive})
    result = _remove_maven_no_fix_dependency_transaction(
        sandbox, set(), plan, GAV, ["pom.xml"], GAV, "pom.xml"
    )
    assert "ERROR_CODE: EDIT_FAILED:" in result
    assert sandbox.files["pom.xml"] == transitive
    assert sandbox.commands == []

    sandbox = _Sandbox({"pom.xml": source}, sync_exit=1)
    touched = {"src/Main.java"}
    result = _remove_maven_no_fix_dependency_transaction(
        sandbox, touched, plan, GAV, ["pom.xml"], GAV, "pom.xml"
    )
    assert "ERROR_CODE: MANIFEST_SYNC_FAILED:" in result
    assert sandbox.files["pom.xml"] == source
    assert touched == {"src/Main.java"}


def test_byte_span_helpers_escape_and_preserve_unrelated_xml():
    source = _pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
    )
    updated = update_pom_property_value(source, "widget.version", "2.0.0")
    assert "<widget.version>2.0.0</widget.version>" in updated
    assert "<artifactId>app</artifactId>" in updated

    escaped = update_pom_dependency_version(
        _pom(f"  <dependencies>{_dependency()}</dependencies>\n"),
        "org.example",
        "widget",
        "2<&0",
    )
    assert "<version>2&lt;&amp;0</version>" in escaped
    managed = add_dependency_management_entry(_pom(), "org.example", "widget", "2.0.0")
    assert "<dependencyManagement>" in managed
    removed = remove_direct_dependency_entry(
        _pom(f"  <dependencies>{_dependency()}</dependencies>\n"),
        "org.example",
        "widget",
    )
    assert "<artifactId>widget</artifactId>" not in removed


def test_update_retry_signature_and_three_attempt_limit_are_per_gav():
    source = _pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source})
    touched: set[str] = set()
    updater = _make_modify_and_validate_maven_dependency_tool(
        sandbox,
        touched,
        {GAV: ["pom.xml"]},
        {GAV: {"2.0.0", "2.1.0", "2.2.0", "2.3.0"}},
        {GAV: {"dependencies"}},
        execution_state={},
    )

    assert _invoke(updater, version="2.0.0").startswith("SUCCESS:")
    assert "ERROR_CODE: RETRY_PARAMETERS_UNCHANGED:" in _invoke(updater, version="2.0.0")
    assert _invoke(updater, version="2.1.0").startswith("SUCCESS:")
    assert _invoke(updater, version="2.2.0").startswith("SUCCESS:")
    assert "ERROR_CODE: RETRY_LIMIT_REACHED:" in _invoke(updater, version="2.3.0")
    assert len(sandbox.commands) == 3


def test_checkpoint_is_frozen_and_captures_pretransaction_touched_set():
    checkpoint = _MavenPackageCheckpoint({"pom.xml": "<project/>"}, {"A.java"})

    assert checkpoint.files == {"pom.xml": "<project/>"}
    assert checkpoint.touched_files_before == {"A.java"}


def test_repeated_property_references_are_resolved_without_false_cycle() -> None:
    source = _pom(
        "  <properties><base.version>1.0.0</base.version>"
        "<widget.version>${base.version}${base.version}</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": source})

    result = _invoke(_updater(sandbox, set()))

    assert result.startswith("SUCCESS:")
    assert "<widget.version>2.0.0</widget.version>" in sandbox.files["pom.xml"]
    assert "<base.version>1.0.0</base.version>" in sandbox.files["pom.xml"]


def test_local_parent_relative_path_must_match_parent_coordinates() -> None:
    root = _pom(
        "  <modules><module>module</module></modules>\n",
        artifact="unrelated",
    )
    child = _module_pom(
        "module",
        f"  <dependencies>{_dependency()}</dependencies>\n",
    )
    sandbox = _Sandbox({"pom.xml": root, "module/pom.xml": child})
    updater = _updater(sandbox, set(), paths=("pom.xml", "module/pom.xml"))

    result = _invoke(updater, path="module/pom.xml")

    assert "ERROR_CODE: POM_TARGET_AMBIGUOUS:" in result
    assert sandbox.files == {"pom.xml": root, "module/pom.xml": child}
    assert sandbox.commands == []


def test_single_character_artifact_coordinate_is_authorized() -> None:
    coordinate = "org.example:x"
    source = _pom(
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>x</artifactId><version>1.0.0</version></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": source})
    updater = _make_modify_and_validate_maven_dependency_tool(
        sandbox,
        set(),
        {coordinate: ["pom.xml"]},
        {coordinate: {"2.0.0"}},
        {coordinate: {"dependencies"}},
    )

    result = _invoke(updater, package=coordinate)

    assert result.startswith("SUCCESS:")
    assert "<version>2.0.0</version>" in sandbox.files["pom.xml"]
