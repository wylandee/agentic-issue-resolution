"""Contract tests for typed resolver-learned solver conflicts."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from remediation_engine.contracts.solver_models import (
    CertificationStatistics,
    PackageResolutionCertificate,
    PackageResolutionStatus,
    SolverCandidateConflict,
    SolverCandidateCutKind,
    SolverCandidateLiteral,
    SolverCandidateRejectionReason,
)
from remediation_engine.orchestration.portfolio_certifier import _candidate_conflict


def _digest(value: str = "evidence") -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _conflict(**overrides: object) -> SolverCandidateConflict:
    values: dict[str, object] = {
        "assignment_digest": _digest("assignment"),
        "reason_code": "dependency_range",
        "cut_kind": "pair",
        "literals": [
            {"variable_id": "source", "version": "v2.0.0"},
            {"variable_id": "child", "version": "1.0.0"},
        ],
        "evidence_digest": _digest(),
        "summary": "resolved dependency misses the published range",
    }
    values.update(overrides)
    return SolverCandidateConflict.model_validate(values)


def test_candidate_conflict_canonicalizes_literals_and_versions() -> None:
    conflict = _conflict()

    assert [(literal.variable_id, literal.version) for literal in conflict.literals] == [
        ("child", "1.0.0"),
        ("source", "2.0.0"),
    ]
    assert SolverCandidateConflict.model_validate_json(conflict.model_dump_json()) == conflict


def test_candidate_literal_is_frozen_and_normalizes_identifier() -> None:
    literal = SolverCandidateLiteral(variable_id=" occurrence-1 ", version="V3.2.1")

    assert literal.variable_id == "occurrence-1"
    assert literal.version == "3.2.1"
    with pytest.raises(ValidationError):
        literal.version = "4.0.0"  # type: ignore[misc]


def test_candidate_conflict_rejects_duplicate_variables_and_wrong_cut_shape() -> None:
    duplicate = [
        {"variable_id": "same", "version": "1.0.0"},
        {"variable_id": "same", "version": "2.0.0"},
    ]
    with pytest.raises(ValidationError, match="unique variable IDs"):
        _conflict(literals=duplicate)
    with pytest.raises(ValidationError, match="unary cuts require"):
        _conflict(reason_code="dependency_range", cut_kind="unary", literals=[duplicate[0]])


def test_candidate_conflict_requires_sha256_digests_and_bounded_summary() -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        _conflict(evidence_digest="not-a-digest")
    with pytest.raises(ValidationError):
        _conflict(summary="x" * 501)


def test_candidate_conflict_digests_are_stable_for_canonical_evidence() -> None:
    first = _candidate_conflict(
        {"z": "v2.0.0", "a": "1.0.0"},
        SolverCandidateRejectionReason.DEPENDENCY_RANGE,
        cut_kind=SolverCandidateCutKind.PAIR,
        literal_ids=("z", "a"),
        evidence={"package_name": "child", "required_range": "^1.0.0"},
        summary="range mismatch",
    )
    second = _candidate_conflict(
        {"a": "1.0.0", "z": "2.0.0"},
        SolverCandidateRejectionReason.DEPENDENCY_RANGE,
        cut_kind=SolverCandidateCutKind.PAIR,
        literal_ids=("a", "z"),
        evidence={"required_range": "^1.0.0", "package_name": "child"},
        summary="range mismatch",
    )
    changed_assignment = _candidate_conflict(
        {"a": "1.0.0", "z": "2.0.0", "extra": "3.0.0"},
        SolverCandidateRejectionReason.DEPENDENCY_RANGE,
        cut_kind=SolverCandidateCutKind.PAIR,
        literal_ids=("a", "z"),
        evidence={"package_name": "child", "required_range": "^1.0.0"},
        summary="range mismatch",
    )

    assert first.assignment_digest == second.assignment_digest
    assert first.evidence_digest == second.evidence_digest
    assert first.assignment_digest != changed_assignment.assignment_digest


def test_certification_statistics_have_empty_compatible_defaults() -> None:
    statistics = CertificationStatistics.model_validate_json("{}")

    assert statistics.solver_calls == 0
    assert statistics.rejection_counts_by_reason == {}
    assert CertificationStatistics.model_validate_json(statistics.model_dump_json()) == statistics

    with pytest.raises(ValidationError):
        CertificationStatistics(metadata_bytes_read=-1)
    with pytest.raises(ValidationError):
        CertificationStatistics(rejection_counts_by_reason={"unknown": 1})


def test_old_certificate_payload_defaults_to_empty_conflict_evidence() -> None:
    payload = {
        "status": PackageResolutionStatus.UNKNOWN.value,
        "portfolio_plan_id": "plan-1",
        "solver_input_digest": "solver",
        "repository_fingerprint": "repository",
        "workspace_graph_digest": "workspace",
        "candidate_catalog_digest": "catalog",
        "candidate_assignment_digest": "assignment",
        "resolved_graph_digest": "resolved",
        "runtime_fingerprint": {
            "node_version": "22.0.0",
            "npm_version": "10.0.0",
            "platform": "darwin",
            "architecture": "arm64",
        },
    }

    certificate = PackageResolutionCertificate.model_validate_json(json.dumps(payload))

    assert certificate.rejection_conflicts == []
    assert certificate.candidate_plan_id is None
    assert certificate.certification_statistics == CertificationStatistics()
    assert (
        PackageResolutionCertificate.model_validate_json(certificate.model_dump_json())
        == certificate
    )
