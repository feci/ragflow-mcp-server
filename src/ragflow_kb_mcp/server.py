"""MCP server that exposes the RAGFlow HTTP API as tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .client import RagflowClient
from .common import Context, parse_upload_roots
from .tools import agents, chats, files, knowledge, memories, searches

INSTRUCTIONS = """\
Tools for the RAGFlow HTTP API: datasets, documents, parsing, chunks, metadata, retrieval,
chat assistants and sessions, agents, memories, the file manager, workspaces and search apps.
Typical document flow: ragflow_list_datasets (or ragflow_create_dataset) ->
ragflow_upload_document (parses automatically; pass wait_seconds to wait for chunks) ->
ragflow_list_chunks or ragflow_retrieval. Uploads accept text, content_base64 (keep it for
small files, the bytes pass through the model), a url that RAGFlow downloads, or file_path
inside directories the operator allowed. Conversation tools call the configured LLM and
answer without streaming. Delete tools remove data permanently.
"""

MODULES = (knowledge, chats, agents, memories, files, searches)


def build_server(
    client: RagflowClient,
    *,
    disabled_tools: frozenset[str] = frozenset(),
    upload_roots: tuple[Path, ...] = (),
    default_embedding_model: str | None = None,
    poll_interval: float = 3.0,
) -> FastMCP:
    mcp = FastMCP("ragflow-kb", instructions=INSTRUCTIONS)

    def tool(annotations: ToolAnnotations):
        def register(fn):
            if fn.__name__ not in disabled_tools:
                mcp.tool(annotations=annotations)(fn)
            return fn

        return register

    ctx = Context(
        client=client,
        tool=tool,
        upload_roots=upload_roots,
        default_embedding_model=default_embedding_model,
        poll_interval=poll_interval,
    )
    for module in MODULES:
        module.register(ctx)
    return mcp


def main() -> None:
    base_url = os.environ.get("RAGFLOW_BASE_URL", "").strip()
    api_key = os.environ.get("RAGFLOW_API_KEY", "").strip()
    if not base_url or not api_key:
        sys.exit("ragflow-kb-mcp: set RAGFLOW_BASE_URL and RAGFLOW_API_KEY")
    client = RagflowClient(
        base_url, api_key, timeout=float(os.environ.get("RAGFLOW_TIMEOUT", "120"))
    )
    disabled = frozenset(
        t.strip() for t in os.environ.get("RAGFLOW_DISABLED_TOOLS", "").split(",") if t.strip()
    )
    server = build_server(
        client,
        disabled_tools=disabled,
        upload_roots=parse_upload_roots(os.environ.get("RAGFLOW_UPLOAD_ROOTS")),
        default_embedding_model=os.environ.get("RAGFLOW_DEFAULT_EMBEDDING_MODEL") or None,
    )
    server.run()


if __name__ == "__main__":
    main()
