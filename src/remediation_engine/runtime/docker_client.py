"""Small, testable boundary for Docker client acquisition and closure."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_docker_client() -> Any:
    """Create and ping a Docker SDK client.

    Returns:
        A connected Docker SDK client.

    Raises:
        ImportError: If the optional Docker SDK is not installed.
        RuntimeError: If the Docker daemon cannot be reached.
    """
    try:
        import docker  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError(
            "The 'docker' package is required for DockerSandbox. "
            "Install it with: pip install 'docker>=7.0.0'"
        ) from exc

    logger.info("Docker: connecting to Docker daemon.")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # noqa: BLE001 - normalize SDK boundary failures
        logger.error("Docker daemon unreachable: %s", exc)
        raise RuntimeError(f"Docker daemon unreachable: {exc}") from exc
    return client


def close_docker_client(client: Any) -> None:
    """Close a Docker client when it exposes the SDK close method.

    Args:
        client: Docker SDK client or a test double. ``None`` is accepted.

    Side effects:
        Closes the client connection. Exceptions are intentionally propagated
        so callers can classify cleanup failures at their own boundary.
    """
    close = getattr(client, "close", None)
    if callable(close):
        close()


__all__ = ["close_docker_client", "get_docker_client"]
