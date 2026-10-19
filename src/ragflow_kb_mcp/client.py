"""Thin async client for the RAGFlow HTTP API (``/api/v1``)."""

from __future__ import annotations

from typing import Any

import httpx


class RagflowError(RuntimeError):
    """RAGFlow returned an error code or a response this client cannot read."""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not params:
        return None
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        out[key] = ("true" if value else "false") if isinstance(value, bool) else value
    return out


class RagflowClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            base += "/api/v1"
        self._http = httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        # IDs from tool arguments are interpolated into the path; refuse anything that
        # would leave the intended endpoint.
        if "?" in path or "#" in path or any(seg in ("", ".", "..") for seg in path.split("/")[1:]):
            raise RagflowError(f"refusing request path {path!r}: IDs must not be empty or contain / ? # or ..")
        try:
            return await self._http.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise RagflowError(
                f"{method} {path}: no answer within the client timeout (RAGFLOW_TIMEOUT); "
                "RAGFlow may still be working on it"
            ) from None
        except httpx.HTTPError as exc:
            raise RagflowError(
                f"{method} {path}: request failed: {type(exc).__name__}: {exc}"
            ) from None

    async def call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
        full: bool = False,
    ) -> Any:
        """Send a request and return its ``data`` field (or the whole body with ``full``)."""
        resp = await self._send(
            method, path, params=_clean_params(params), json=json, data=data, files=files
        )
        body = self._json_body(resp, method, path)
        code = body.get("code", 0 if resp.is_success else resp.status_code)
        if code != 0:
            message = body.get("message") or body.get("error") or "unknown error"
            raise RagflowError(
                f"{method} {path}: RAGFlow code {code}: {message}",
                code=code,
                data=body.get("data"),
            )
        return body if full else body.get("data")

    async def raw(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
    ) -> tuple[bytes, str]:
        """Send a request whose success answer is a file. RAGFlow reports errors as JSON."""
        resp = await self._send(
            method, path, params=_clean_params(params), json=json, data=data, files=files
        )
        content_type = resp.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            body = self._json_body(resp, method, path)
            if body.get("code", 0) != 0:
                message = body.get("message") or "unknown error"
                raise RagflowError(
                    f"{method} {path}: RAGFlow code {body.get('code')}: {message}",
                    code=body.get("code"),
                    data=body.get("data"),
                )
        if not resp.is_success:
            raise RagflowError(f"{method} {path}: HTTP {resp.status_code}")
        return resp.content, content_type

    async def download(self, path: str, params: dict[str, Any] | None = None) -> tuple[bytes, str]:
        return await self.raw("GET", path, params=params)

    async def json_with_status(self, method: str, path: str) -> tuple[int, dict[str, Any]]:
        """Return the HTTP status and JSON body without treating error statuses as failures."""
        resp = await self._send(method, path)
        return resp.status_code, self._json_body(resp, method, path)

    @staticmethod
    def _json_body(resp: httpx.Response, method: str, path: str) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError:
            raise RagflowError(
                f"{method} {path}: HTTP {resp.status_code} with a non-JSON response"
            ) from None
        if not isinstance(body, dict):
            raise RagflowError(f"{method} {path}: unexpected response shape")
        return body
