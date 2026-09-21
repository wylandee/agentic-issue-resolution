"""Bounded, deterministic cache primitives for npm registry packuments.

The cache stores raw registry metadata rather than a filtered candidate list.  It
is deliberately independent from :mod:`registry_tools`: callers inject the
network fetcher, which keeps this module useful to both the portfolio solver and
existing registry tools without introducing an import cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, TypeAlias

Packument: TypeAlias = dict[str, Any]
PackumentFetcher: TypeAlias = Callable[[str], Mapping[str, Any]]

CACHE_SCHEMA_VERSION: Final[int] = 1
DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 5 * 1024 * 1024
DEFAULT_DIAGNOSTICS_LIMIT: Final[int] = 32
_MAX_DIAGNOSTIC_LENGTH: Final[int] = 512
_PACKAGE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:@[^/@\\\s]+/[^/@\\\s]+|[^@/\\\s]+)$"
)


class RegistryCacheError(ValueError):
    """Raised when a fetched packument cannot be validated for caching."""


def validate_package_name(package_name: str) -> str:
    """Validate and return an npm package name.

    The validator intentionally checks the identity and path-safety properties
    needed by the cache.  It accepts ordinary and scoped npm names, but rejects
    path separators, control/whitespace characters, empty segments, and dot
    traversal names.  The input is not lower-cased: the package identity stored
    by npm is compared exactly on cache reads.

    Args:
        package_name: Package name supplied by a registry caller.

    Returns:
        The stripped package name.

    Raises:
        ValueError: If ``package_name`` is not a valid cache package identity.
    """
    if not isinstance(package_name, str):
        raise ValueError("package_name must be a string")
    value = package_name.strip()
    if not value or value in {".", ".."} or not _PACKAGE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid npm package name: {package_name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"Invalid npm package name: {package_name!r}")
    segments = value.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError(f"Invalid npm package name: {package_name!r}")
    return value


def validate_packument(package_name: str, packument: Mapping[str, Any]) -> Packument:
    """Validate and copy a raw npm packument.

    A packument must be a mapping with a mapping-valued ``versions`` member.
    When npm supplies a ``name`` member it must match the requested package;
    accepting a missing name keeps injected test/offline fetchers compatible
    with minimal packument fixtures while still enforcing stored identity in the
    cache envelope.

    Args:
        package_name: Expected npm package name.
        packument: Raw decoded registry JSON.

    Returns:
        A detached, mutable ``dict`` suitable for deterministic serialization.

    Raises:
        ValueError: If the package identity or packument shape is invalid.
    """
    expected_name = validate_package_name(package_name)
    if not isinstance(packument, Mapping):
        raise RegistryCacheError("registry packument must be a JSON object")
    supplied_name = packument.get("name")
    if supplied_name is not None and supplied_name != expected_name:
        raise RegistryCacheError(
            f"registry packument name {supplied_name!r} does not match {expected_name!r}"
        )
    versions = packument.get("versions")
    if not isinstance(versions, Mapping):
        raise RegistryCacheError("registry packument must contain an object-valued versions field")
    # JSON round-tripping is avoided here: callers may pass harmless mapping
    # subclasses, and cache serialization below performs the actual deep copy.
    return dict(packument)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a cache envelope using stable, compact UTF-8 JSON."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


class RegistryPackumentCache:
    """Atomic on-disk cache for validated raw npm packuments.

    The cache is disabled when ``cache_dir`` is ``None`` or ``enabled`` is
    false.  Cache failures are non-fatal: reads return misses and writes return
    false while a bounded diagnostic is retained for observability.  Network
    fetch failures are not swallowed by :func:`load_or_fetch_packument`.

    Args:
        cache_dir: Directory containing cache entries, or ``None`` to disable.
        enabled: Explicitly disable cache reads and writes when false.
        max_payload_bytes: Maximum encoded envelope size for one entry.
        diagnostics_limit: Maximum number of retained diagnostic messages.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        enabled: bool = True,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_bytes: int | None = None,
        diagnostics_limit: int = DEFAULT_DIAGNOSTICS_LIMIT,
    ) -> None:
        """Initialize a bounded cache without performing filesystem I/O."""
        if max_bytes is not None:
            max_payload_bytes = max_bytes
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be greater than zero")
        if diagnostics_limit < 0:
            raise ValueError("diagnostics_limit must not be negative")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.enabled = bool(enabled) and self.cache_dir is not None
        self.max_payload_bytes = int(max_payload_bytes)
        self.diagnostics_limit = int(diagnostics_limit)
        self._diagnostics: list[str] = []

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return bounded diagnostics collected by cache reads and writes."""
        return tuple(self._diagnostics)

    def clear_diagnostics(self) -> None:
        """Discard retained cache diagnostics for a new planning operation."""
        self._diagnostics.clear()

    @staticmethod
    def cache_key(package_name: str) -> str:
        """Return a collision-resistant filename for a package identity."""
        name = validate_package_name(package_name)
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return f"packument-v{CACHE_SCHEMA_VERSION}-{digest}.json"

    def path_for(self, package_name: str) -> Path:
        """Return the cache path for ``package_name`` without touching disk."""
        if self.cache_dir is None:
            raise ValueError("cache is disabled")
        return self.cache_dir / self.cache_key(package_name)

    def get(self, package_name: str) -> Packument | None:
        """Load a validated packument, returning ``None`` on a cache miss.

        Malformed, oversized, identity-mismatched, or partially written files
        are treated as corrupt entries.  They are removed on a best-effort basis
        and never escape as exceptions to the caller.
        """
        name = validate_package_name(package_name)
        if not self.enabled or self.cache_dir is None:
            return None
        path = self.path_for(name)
        try:
            size = path.stat().st_size
            if size > self.max_payload_bytes:
                raise RegistryCacheError("cache entry exceeds max_payload_bytes")
            with path.open("rb") as handle:
                raw = handle.read(self.max_payload_bytes + 1)
            if len(raw) > self.max_payload_bytes:
                raise RegistryCacheError("cache entry exceeds max_payload_bytes")
            envelope = json.loads(raw.decode("utf-8"))
            if not isinstance(envelope, Mapping):
                raise RegistryCacheError("cache entry is not a JSON object")
            if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise RegistryCacheError("cache entry has an unsupported schema version")
            if envelope.get("package_name") != name:
                raise RegistryCacheError("cache entry package identity mismatch")
            cached = validate_packument(name, envelope.get("packument"))
            return cached
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._diagnose(f"read failed for {name!r}: {exc}")
            self._remove_corrupt_entry(path, name)
            return None

    def put(self, package_name: str, packument: Mapping[str, Any]) -> bool:
        """Atomically store a validated packument and report write success.

        Temporary files are created in the destination directory and are always
        cleaned up.  ``os.replace`` makes a completed write visible in one step;
        interrupted writes cannot leave a partial target entry.
        """
        name = validate_package_name(package_name)
        if not self.enabled or self.cache_dir is None:
            return False
        try:
            validated = validate_packument(name, packument)
            envelope: dict[str, Any] = {
                "package_name": name,
                "packument": validated,
                "schema_version": CACHE_SCHEMA_VERSION,
            }
            payload = _json_bytes(envelope)
            if len(payload) > self.max_payload_bytes:
                raise RegistryCacheError("packument exceeds max_payload_bytes")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            destination = self.path_for(name)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=self.cache_dir,
                    delete=False,
                ) as handle:
                    temporary_name = handle.name
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
                temporary_name = None
                return True
            finally:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        self._diagnose(f"cleanup failed for {name!r}: {exc}")
        except (OSError, TypeError, ValueError) as exc:
            self._diagnose(f"write failed for {name!r}: {exc}")
            return False

    def load_or_fetch(self, package_name: str, fetcher: PackumentFetcher) -> Packument:
        """Return a cached packument or fetch, validate, and cache it."""
        return load_or_fetch_packument(package_name, fetcher, cache=self)

    def _diagnose(self, message: str) -> None:
        """Append one bounded diagnostic without allowing diagnostics to fail work."""
        if self.diagnostics_limit <= 0:
            return
        self._diagnostics.append(str(message)[:_MAX_DIAGNOSTIC_LENGTH])
        del self._diagnostics[self.diagnostics_limit :]

    def _remove_corrupt_entry(self, path: Path, package_name: str) -> None:
        """Remove one corrupt entry without masking the original read failure."""
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._diagnose(f"cleanup failed for corrupt {package_name!r}: {exc}")


def load_or_fetch_packument(
    package_name: str,
    fetcher: PackumentFetcher,
    *,
    cache: RegistryPackumentCache | None = None,
) -> Packument:
    """Load a packument from cache or an injected fetcher.

    Args:
        package_name: npm package identity to fetch.
        fetcher: Callable receiving the package name and returning raw JSON.
        cache: Optional cache.  ``None`` and disabled caches perform no disk I/O.

    Returns:
        A validated detached packument mapping.

    Raises:
        ValueError: If the package name or fetched packument is invalid.
        Exception: Any exception raised by the injected fetcher is propagated.
    """
    name = validate_package_name(package_name)
    if cache is not None:
        cached = cache.get(name)
        if cached is not None:
            return cached
    fetched = validate_packument(name, fetcher(name))
    if cache is not None:
        cache.put(name, fetched)
    return fetched


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_DIAGNOSTICS_LIMIT",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "Packument",
    "PackumentFetcher",
    "RegistryCacheError",
    "RegistryPackumentCache",
    "load_or_fetch_packument",
    "validate_package_name",
    "validate_packument",
]
