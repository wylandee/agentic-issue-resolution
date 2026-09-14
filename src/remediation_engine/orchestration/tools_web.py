"""Web search and readable-page fetch tools."""

from __future__ import annotations

from ._tool_support import (
    _GITHUB_API_URL_PREFIX,
    _JINA_READER_URL_PREFIX,
    _READ_WEB_PAGE_MAX_CHARS,
    _READ_WEB_PAGE_TIMEOUT,
    _SEARCH_WEB_MAX_CALLS,
    _SERPER_MAX_RESULTS,
    _SERPER_REQUEST_TIMEOUT,
    _SERPER_SEARCH_URL,
    Any,
    WorkaroundExecutionPhase,
    _is_authoritative_evidence_source,
    b64decode,
    get_runtime_settings,
    logger,
    quote,
    requests,
    tool,
    urlparse,
)


def _make_search_web_tool(
    mandatory_search_terms: dict[str, str] | None = None,
    plan_state: dict[str, Any] | None = None,
):
    """Create a web search tool backed by Serper.dev."""
    _calls_remaining = [3]

    @tool
    def search_web(query: str) -> str:
        """Search the web for vulnerability fixes, migration guides, or documentation.

        Pass your own targeted query. Do not wrap queries in quotes.
        """
        if plan_state is not None and not plan_state.get("local_investigation_complete", False):
            return (
                "ERROR: [INVESTIGATION_REQUIRED] Local codebase investigation must complete "
                "using search_codebase_pattern, read_workspace_file, or inspect_ast_symbol "
                "before calling search_web."
            )
        if (
            plan_state is not None
            and plan_state.get("phase") == WorkaroundExecutionPhase.VALIDATE.value
        ):
            return (
                "ERROR: [PHASE_VIOLATION] Web research is not a validation action. "
                "Call validate_workaround, or register an evidence-backed alternative test "
                "after an infrastructure-only targeted-test failure."
            )

        if _calls_remaining[0] <= 0:
            return f"ERROR: search_web call limit reached (max {_SEARCH_WEB_MAX_CALLS} per session). Use the results you already have."

        api_key = get_runtime_settings().serper_api_key
        if not api_key:
            return "ERROR: SERPER_API_KEY is not set. Cannot perform web search."

        _calls_remaining[0] -= 1

        effective_query = (query or "").replace('"', "").replace("'", "").strip()
        if not effective_query:
            return (
                "ERROR: search_web requires a worker-selected query. Classify the workaround "
                "type first and include the relevant package, advisory, migration, scanner, "
                "or test-regression terms."
            )

        try:
            resp = requests.post(
                _SERPER_SEARCH_URL,
                json={"q": effective_query},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=_SERPER_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic") or []

            results = []
            for item in organic[:_SERPER_MAX_RESULTS]:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                results.append(f"**{title}**\n{snippet}\nURL: {link}")

            calls_left = _calls_remaining[0]
            if plan_state is not None:
                plan_state["web_search_performed"] = True

            if not results:
                return f"Effective Query: {effective_query}\nNo results found for this query."

            header = f"Effective Query: {effective_query}\nFound {len(results)} results ({calls_left} searches remaining):\n\n"
            return header + "\n\n---\n\n".join(results)

        except Exception as exc:
            logger.warning("search_web failed: %s", exc)
            return f"Effective Query: {effective_query}\nERROR: Web search failed - {exc}."

    return search_web


def _github_api_url(target_url: str) -> str | None:
    """Translate a public GitHub URL into its corresponding GitHub API URL."""
    parsed = urlparse(target_url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"github.com", "www.github.com"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None

    if parts[0].lower() == "advisories" and len(parts) >= 2:
        return f"{_GITHUB_API_URL_PREFIX}advisories/{quote(parts[1], safe='')}"
    if len(parts) < 2:
        return None

    owner, repository = parts[0], parts[1]
    if repository.endswith(".git"):
        repository = repository[:-4]
    base = f"{_GITHUB_API_URL_PREFIX}repos/{quote(owner, safe='')}/{quote(repository, safe='')}"
    if len(parts) == 2:
        return base

    resource = parts[2].lower()
    if resource in {"issues", "pulls", "commits"} and len(parts) >= 4:
        return f"{base}/{resource}/{quote(parts[3], safe='')}"
    if resource == "releases" and len(parts) >= 5 and parts[3].lower() == "tag":
        return f"{base}/releases/tags/{quote('/'.join(parts[4:]), safe='')}"
    if resource in {"blob", "raw", "tree"} and len(parts) >= 4:
        ref = quote(parts[3], safe="")
        content_path = quote("/".join(parts[4:]), safe="/")
        endpoint = f"{base}/contents/{content_path}" if content_path else f"{base}/contents"
        return f"{endpoint}?ref={ref}"
    return base


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers without exposing an optional token in logs."""
    headers = {
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = get_runtime_settings().github_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _decode_github_response(resp: Any, target_url: str) -> str:
    """Decode raw GitHub content or a JSON API response into readable text."""
    text = getattr(resp, "text", "") or ""
    if text.strip():
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            return text
        if isinstance(payload, dict):
            if payload.get("encoding") == "base64" and payload.get("content"):
                try:
                    return b64decode(str(payload["content"]).replace("\n", "")).decode("utf-8")
                except Exception:  # noqa: BLE001
                    return text
            title = (
                payload.get("title")
                or payload.get("name")
                or payload.get("login")
                or "GitHub Content"
            )
            body = payload.get("body") or payload.get("description") or payload.get("content") or ""
            if title or body:
                return f"Title: {title}\nURL: {target_url}\n\n{body}".strip()
            import json

            return json.dumps(payload, indent=2, ensure_ascii=False)
        if isinstance(payload, list):
            import json

            return json.dumps(payload, indent=2, ensure_ascii=False)
        return text
    return f"No readable content extracted from {target_url}."


def _make_read_web_page_tool(plan_state: dict[str, Any] | None = None):
    """Create a tool to fetch web pages, using GitHub API, raw content, Jina, and npm fallbacks."""

    @tool
    def read_web_page(url: str) -> str:
        """
        Fetch the full readable markdown text of a web page given its URL.

        Use this tool after search_web to read complete migration guides, breaking change lists,
        or documentation pages found in search results.
        """
        if plan_state is not None:
            if not plan_state.get("local_investigation_complete", False):
                return (
                    "ERROR: [INVESTIGATION_REQUIRED] Local codebase investigation must complete "
                    "before calling read_web_page."
                )
            if not plan_state.get("web_search_performed", False):
                return "ERROR: [SEARCH_REQUIRED] You must execute search_web before calling read_web_page."
            if plan_state.get("phase") == WorkaroundExecutionPhase.VALIDATE.value:
                return (
                    "ERROR: [PHASE_VIOLATION] Web research is not a validation action. "
                    "Call validate_workaround instead."
                )

        target_url = (url or "").strip()
        if not target_url:
            return "ERROR: url is required."

        # 1. GitHub API & Raw GitHub fallback
        if "github.com" in target_url and not target_url.startswith(
            "https://raw.githubusercontent.com"
        ):
            github_api_url = _github_api_url(target_url)
            if github_api_url:
                try:
                    resp = requests.get(
                        github_api_url,
                        headers=_github_headers(),
                        timeout=_READ_WEB_PAGE_TIMEOUT,
                    )
                    resp.raise_for_status()
                    text = _decode_github_response(resp, target_url)
                    if text and text.strip():
                        if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                            text = (
                                text[:_READ_WEB_PAGE_MAX_CHARS]
                                + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                            )
                        if plan_state is not None and _is_authoritative_evidence_source(target_url):
                            plan_state["has_authoritative_evidence"] = True
                            plan_state["evidence_source"] = target_url
                        return f"--- Markdown content of {target_url} ---\n\n{text}"
                except Exception as exc:  # noqa: BLE001
                    logger.debug("GitHub API fetch failed for %s: %s", target_url, exc)

            raw_url = (
                target_url.replace("github.com", "raw.githubusercontent.com")
                .replace("/blob/", "/")
                .replace("/tree/", "/")
            )
            try:
                resp = requests.get(raw_url, timeout=_READ_WEB_PAGE_TIMEOUT)
                resp.raise_for_status()
                text = resp.text or ""
                if text and text.strip():
                    if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                        text = (
                            text[:_READ_WEB_PAGE_MAX_CHARS]
                            + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                        )
                    if plan_state is not None and _is_authoritative_evidence_source(target_url):
                        plan_state["has_authoritative_evidence"] = True
                        plan_state["evidence_source"] = target_url
                    return f"--- Markdown content of {target_url} ---\n\n{text}"
            except Exception as exc:  # noqa: BLE001
                logger.debug("Raw GitHub fetch failed for %s: %s", raw_url, exc)

        # 2. Jina Reader fallback
        jina_url = f"{_JINA_READER_URL_PREFIX}{target_url}"
        try:
            resp = requests.get(
                jina_url,
                headers={"Accept": "text/plain"},
                timeout=_READ_WEB_PAGE_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.text or ""
            if text and text.strip():
                if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                    text = (
                        text[:_READ_WEB_PAGE_MAX_CHARS]
                        + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                    )
                if plan_state is not None and _is_authoritative_evidence_source(target_url):
                    plan_state["has_authoritative_evidence"] = True
                    plan_state["evidence_source"] = target_url
                return f"--- Markdown content of {target_url} ---\n\n{text}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Jina Reader fetch failed for %s: %s", target_url, exc)

        # 3. Direct page fetch fallback
        try:
            resp = requests.get(target_url, timeout=_READ_WEB_PAGE_TIMEOUT)
            resp.raise_for_status()
            text = resp.text or ""
            if text and text.strip():
                if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                    text = (
                        text[:_READ_WEB_PAGE_MAX_CHARS]
                        + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                    )
                if plan_state is not None and _is_authoritative_evidence_source(target_url):
                    plan_state["has_authoritative_evidence"] = True
                    plan_state["evidence_source"] = target_url
                return f"--- Markdown content of {target_url} ---\n\n{text}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Direct page fetch failed for %s: %s", target_url, exc)

        # 4. npm registry fallback if URL relates to npm package
        if (
            "npmjs.com" in target_url
            or "registry.npmjs.org" in target_url
            or not target_url.startswith("http")
        ):
            pkg_name = (
                target_url.split("package/")[-1].split("/")[0].strip()
                if "package/" in target_url
                else target_url.strip()
            )
            if pkg_name:
                try:
                    resp = requests.get(
                        f"https://registry.npmjs.org/{pkg_name}",
                        timeout=_READ_WEB_PAGE_TIMEOUT,
                    )
                    resp.raise_for_status()
                    text = resp.text or ""
                    if text and text.strip():
                        if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                            text = (
                                text[:_READ_WEB_PAGE_MAX_CHARS]
                                + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                            )
                        if plan_state is not None and _is_authoritative_evidence_source(target_url):
                            plan_state["has_authoritative_evidence"] = True
                            plan_state["evidence_source"] = target_url
                        return f"--- Markdown content of {target_url} ---\n\n{text}"
                except Exception:  # noqa: BLE001
                    pass

        return f"ERROR: Failed to read web page {target_url} - No readable content extracted from any fallback source."

    return read_web_page


__all__ = [name for name in globals() if not name.startswith("__")]
