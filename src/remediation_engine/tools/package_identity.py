"""Canonical package and PURL identity helpers.

Scanner payloads do not consistently encode npm scopes.  This module owns
the small amount of PURL parsing needed by ingestion, locating, and planning
so a package cannot change identity merely because one fallback parser ran.
"""

from __future__ import annotations

from urllib.parse import unquote

try:
    from packageurl import PackageURL
except ImportError:  # pragma: no cover - optional dependency fallback
    PackageURL = None  # type: ignore[assignment,misc]


def package_name_from_purl(purl: str | None) -> str | None:
    """Return the canonical package name, including npm scopes.

    Args:
        purl: Package URL such as ``pkg:npm/@scope/pkg@1.2.3``.

    Returns:
        A decoded package name, or ``None`` for an invalid/empty PURL.
    """
    if not purl or not str(purl).strip().startswith("pkg:"):
        return None
    value = str(purl).strip()

    if PackageURL is not None:
        try:
            parsed = PackageURL.from_string(value)
            name = parsed.name or ""
            namespace = parsed.namespace or ""
            if name:
                if parsed.type == "npm":
                    if namespace:
                        return f"{unquote(namespace)}/{unquote(name)}"
                    return unquote(name)
                return f"{namespace}:{name}" if namespace else unquote(name)
        except Exception:  # noqa: BLE001 - malformed scanner PURLs use fallback
            pass

    try:
        purl_body = value[len("pkg:") :]
        ecosystem, _, remainder = purl_body.partition("/")
        if not ecosystem or not remainder:
            return None
        remainder = unquote(remainder.split("?", 1)[0].split("#", 1)[0])

        if ecosystem.lower() == "npm":
            if remainder.startswith("@"):
                slash = remainder.find("/")
                if slash < 0:
                    return None
                version_at = remainder.find("@", slash + 1)
                return (remainder if version_at < 0 else remainder[:version_at]) or None
            return remainder.split("@", 1)[0] or None

        name = remainder.rsplit("/", 1)[-1]
        namespace = remainder.rsplit("/", 1)[0]
        name = name.split("@", 1)[0]
        return f"{namespace}:{name}" if ecosystem.lower() == "maven" and namespace else name
    except (AttributeError, IndexError, ValueError):
        return None
