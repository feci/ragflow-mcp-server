"""Shared pieces for the tool modules: context, annotations, argument types and byte helpers."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from .client import RagflowClient

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
UPLOAD = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
DELETE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)

DEFAULT_MAX_CHARS = 20_000
MAX_BASE64_BYTES = 5 * 1024 * 1024

# IDs are interpolated into request paths, so they may not contain / ? # or spaces.
ID_PATTERN = r"^[A-Za-z0-9_.:@+=-]+$"

DatasetId = Annotated[str, Field(pattern=ID_PATTERN, description="Dataset (knowledge base) ID")]
DocumentId = Annotated[str, Field(pattern=ID_PATTERN, description="Document ID")]
ChatId = Annotated[str, Field(pattern=ID_PATTERN, description="Chat assistant ID")]
AgentId = Annotated[str, Field(pattern=ID_PATTERN, description="Agent ID")]
SessionId = Annotated[str, Field(pattern=ID_PATTERN, description="Session ID")]
MemoryId = Annotated[str, Field(pattern=ID_PATTERN, description="Memory ID")]
FileId = Annotated[str, Field(pattern=ID_PATTERN, description="File or folder ID in the RAGFlow file manager")]
SearchId = Annotated[str, Field(pattern=ID_PATTERN, description="Search app ID")]
WorkspaceId = Annotated[str, Field(pattern=ID_PATTERN, description="Workspace ID")]
Page = Annotated[int, Field(ge=1, description="Page number, starting at 1")]
PageSize = Annotated[int, Field(ge=1, le=1000, description="Items per page")]
MaxChars = Annotated[
    int, Field(ge=1, le=500_000, description="Return at most this many characters of text")
]


@dataclass
class Context:
    """What every tool module receives from build_server."""

    client: RagflowClient
    tool: Callable[[ToolAnnotations], Callable]
    upload_roots: tuple[Path, ...] = ()
    default_embedding_model: str | None = None
    poll_interval: float = 3.0
    extra: dict[str, Any] = field(default_factory=dict)


def pick(src: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: src[k] for k in keys if k in src}


def non_empty(name: str, values: list[str]) -> list[str]:
    cleaned = [v for v in values if v]
    if not cleaned:
        raise ValueError(f"{name} must list at least one ID")
    return cleaned


def drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def parse_upload_roots(raw: str | None) -> tuple[Path, ...]:
    if not raw:
        return ()
    return tuple(Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip())


def read_local_file(ctx: Context, file_path: str) -> tuple[bytes, str]:
    """Read a file from an allowed upload root; returns (bytes, file name)."""
    if not ctx.upload_roots:
        raise ValueError(
            "file_path uploads are disabled; the operator can allow directories "
            "with RAGFLOW_UPLOAD_ROOTS"
        )
    path = Path(file_path).expanduser().resolve()
    if not any(path.is_relative_to(root) for root in ctx.upload_roots):
        raise ValueError("file_path is outside the allowed upload roots")
    if not path.is_file():
        raise ValueError("file_path does not point to a file")
    return path.read_bytes(), path.name


def decode_base64(value: str, name: str = "content_base64") -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name} is not valid base64: {exc}") from None


def read_bytes_source(
    ctx: Context,
    *,
    text: str | None = None,
    content_base64: str | None = None,
    file_path: str | None = None,
    filename: str | None = None,
) -> tuple[bytes, str | None]:
    """Turn exactly one of text, content_base64 or file_path into bytes and a file name."""
    given = [
        k
        for k, v in (("text", text), ("content_base64", content_base64), ("file_path", file_path))
        if v is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "pass exactly one of text, content_base64 or file_path"
            + (f" (got {', '.join(given)})" if given else "")
        )
    if text is not None:
        return text.encode("utf-8"), filename
    if content_base64 is not None:
        return decode_base64(content_base64), filename
    blob, name = read_local_file(ctx, file_path)
    return blob, filename or name


def describe_bytes(
    blob: bytes,
    content_type: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_base64: bool = False,
) -> dict[str, Any]:
    """Return text for UTF-8 content; for binary content the size, and base64 on request."""
    result: dict[str, Any] = {"content_type": content_type, "size": len(blob)}
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        result["text"] = None
        if include_base64:
            if len(blob) > MAX_BASE64_BYTES:
                result["note"] = f"binary content larger than {MAX_BASE64_BYTES} bytes; not returned"
            else:
                result["content_base64"] = base64.b64encode(blob).decode("ascii")
        else:
            result["note"] = "binary content; pass include_base64=true to receive the bytes"
        return result
    result["text"] = text[:max_chars]
    result["truncated"] = len(text) > max_chars
    return result
