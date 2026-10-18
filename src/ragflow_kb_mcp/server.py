"""MCP tools for RAGFlow datasets, documents, parsing, chunks, metadata and retrieval."""

from __future__ import annotations

import asyncio
import base64
import binascii
import mimetypes
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from .client import RagflowClient, RagflowError

INSTRUCTIONS = """\
Manage RAGFlow knowledge bases.
Typical flow: ragflow_list_datasets (or ragflow_create_dataset) -> ragflow_upload_document
(parses automatically; pass wait_seconds to wait for chunks) -> ragflow_list_chunks or
ragflow_retrieval. Upload accepts exactly one source: text, content_base64, url (RAGFlow
fetches it) or file_path (only inside the directories the operator allowed). Base64 passes
through the model, so keep it for small files and prefer url or file_path for large ones.
Delete tools remove data permanently.
"""

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
UPLOAD = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
DELETE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "STOPPED", "CANCELED", "CANCELLED", "DONE", "FAIL", "CANCEL"}
MAX_WAIT_SECONDS = 600
POLL_INTERVAL = 3.0

DatasetId = Annotated[str, Field(description="Dataset (knowledge base) ID")]
DocumentId = Annotated[str, Field(description="Document ID")]
Page = Annotated[int, Field(ge=1, description="Page number, starting at 1")]
PageSize = Annotated[int, Field(ge=1, le=1000, description="Items per page")]
WaitSeconds = Annotated[
    int,
    Field(
        ge=0,
        le=MAX_WAIT_SECONDS,
        description="Seconds to wait for parsing to finish; 0 returns immediately",
    ),
]


def _pick(src: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: src[k] for k in keys if k in src}


def _dataset(d: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        d,
        "id",
        "name",
        "description",
        "document_count",
        "chunk_count",
        "token_num",
        "embedding_model",
        "parser_id",
        "language",
        "permission",
        "create_time",
        "update_time",
    )


def _document(d: dict[str, Any]) -> dict[str, Any]:
    out = _pick(
        d,
        "id",
        "name",
        "suffix",
        "size",
        "source_type",
        "ingestion_status",
        "progress",
        "chunk_count",
        "token_count",
        "meta_fields",
        "parser_id",
        "create_time",
        "update_time",
    )
    if "status" in d:
        out["enabled"] = str(d["status"]) == "1"
    status = str(d.get("ingestion_status") or "").upper()
    progress_msg = d.get("progress_msg")
    if (status in {"FAILED", "FAIL"} or (d.get("progress") or 0) < 0) and progress_msg:
        text = progress_msg if isinstance(progress_msg, str) else "\n".join(map(str, progress_msg))
        out["progress_msg"] = text[-1500:]
    return out


def _chunk(c: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": c.get("chunk_id") or c.get("id"),
        "content": c.get("content_with_weight", c.get("content")),
    }
    for public, *raw in (
        ("important_keywords", "important_kwd", "important_keywords"),
        ("questions", "question_kwd", "questions"),
        ("tags", "tag_kwd"),
    ):
        value = next((c[k] for k in raw if c.get(k)), None)
        if value:
            out[public] = value
    for key in ("available", "positions"):
        if c.get(key) not in (None, []):
            out[key] = c[key]
    return out


def _retrieved(c: dict[str, Any]) -> dict[str, Any]:
    out = _pick(c, "id", "content", "document_id", "dataset_id", "similarity")
    out["document_name"] = c.get("document_keyword") or c.get("docnm_kwd")
    if c.get("important_keywords"):
        out["important_keywords"] = c["important_keywords"]
    return out


def _non_empty(name: str, values: list[str]) -> list[str]:
    cleaned = [v for v in values if v]
    if not cleaned:
        raise ValueError(f"{name} must list at least one ID")
    return cleaned


def _is_terminal(doc: dict[str, Any]) -> bool:
    status = str(doc.get("ingestion_status") or "").upper()
    progress = doc.get("progress") or 0
    return status in TERMINAL_STATUSES or progress >= 1 or progress < 0


def parse_upload_roots(raw: str | None) -> tuple[Path, ...]:
    if not raw:
        return ()
    return tuple(Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip())


def build_server(
    client: RagflowClient,
    *,
    disabled_tools: frozenset[str] = frozenset(),
    upload_roots: tuple[Path, ...] = (),
    default_embedding_model: str | None = None,
    poll_interval: float = POLL_INTERVAL,
) -> FastMCP:
    mcp = FastMCP("ragflow-kb", instructions=INSTRUCTIONS)

    def tool(annotations: ToolAnnotations):
        def register(fn):
            if fn.__name__ not in disabled_tools:
                mcp.tool(annotations=annotations)(fn)
            return fn

        return register

    async def list_docs(dataset_id: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            data = await client.call("GET", f"/datasets/{dataset_id}/documents", params=params)
        except RagflowError as exc:
            # An id or name filter without a match comes back as code 102 with an
            # empty list ("you don't own the document ..."); report it as no match.
            if exc.code == 102 and isinstance(exc.data, dict) and exc.data.get("docs") == []:
                return {"total": 0, "docs": []}
            raise
        return data or {}

    async def get_document(dataset_id: str, document_id: str) -> dict[str, Any]:
        docs = (await list_docs(dataset_id, {"id": document_id})).get("docs") or []
        if not docs:
            raise RagflowError(f"document {document_id} not found in dataset {dataset_id}")
        return docs[0]

    async def find_by_name(dataset_id: str, name: str) -> list[dict[str, Any]]:
        data = await list_docs(dataset_id, {"name": name, "page": 1, "page_size": 100})
        return [d for d in data.get("docs") or [] if d.get("name") == name]

    async def wait_for(dataset_id: str, document_ids: list[str], wait_seconds: int) -> dict[str, Any]:
        deadline = time.monotonic() + min(wait_seconds, MAX_WAIT_SECONDS)
        docs: dict[str, dict[str, Any]] = {}
        while True:
            for doc_id in document_ids:
                if doc_id not in docs or not _is_terminal(docs[doc_id]):
                    docs[doc_id] = await get_document(dataset_id, doc_id)
            pending = [i for i in document_ids if not _is_terminal(docs[i])]
            if not pending or time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_interval)
        return {
            "documents": [_document(docs[i]) for i in document_ids],
            "still_parsing": pending,
        }

    # Datasets

    @tool(READ)
    async def ragflow_list_datasets(
        name: Annotated[str | None, Field(description="Exact dataset name filter")] = None,
        page: Page = 1,
        page_size: PageSize = 30,
    ) -> dict[str, Any]:
        """List datasets (knowledge bases) with IDs, descriptions and document/chunk counts."""
        body = await client.call(
            "GET",
            "/datasets",
            params={"name": name, "page": page, "page_size": page_size},
            full=True,
        )
        items = body.get("data") or []
        total = body.get("total_datasets", body.get("total", len(items)))
        return {"total": total, "datasets": [_dataset(d) for d in items]}

    @tool(READ)
    async def ragflow_get_dataset(dataset_id: DatasetId) -> dict[str, Any]:
        """Get one dataset, including its parser configuration."""
        data = await client.call("GET", f"/datasets/{dataset_id}")
        return {**_dataset(data), "parser_config": data.get("parser_config")}

    @tool(WRITE)
    async def ragflow_create_dataset(
        name: Annotated[str, Field(description="Dataset name (unique, max 128 characters)")],
        description: Annotated[str | None, Field(description="Short description")] = None,
        embedding_model: Annotated[
            str | None,
            Field(
                description="Embedding model as model_name@factory; defaults to the "
                "server's configured default"
            ),
        ] = None,
        parser_id: Annotated[
            str,
            Field(description="Built-in chunking method; 'general' fits most documents"),
        ] = "general",
        language: Annotated[str | None, Field(description="Dataset language, e.g. English")] = None,
        permission: Annotated[
            Literal["me", "team"] | None, Field(description="Who may manage the dataset")
        ] = None,
    ) -> dict[str, Any]:
        """Create a dataset (knowledge base) that uses a built-in chunking method."""
        body: dict[str, Any] = {"name": name, "parser_id": parser_id, "parse_type": 1}
        model = embedding_model or default_embedding_model
        if model:
            body["embedding_model"] = model
        if language:
            body["language"] = language
        if permission:
            body["permission"] = permission
        data = await client.call("POST", "/datasets", json=body)
        if description:
            # Dataset creation does not accept a description; set it right after.
            data = await client.call(
                "PUT", f"/datasets/{data['id']}", json={"description": description}
            )
        return _dataset(data)

    @tool(WRITE)
    async def ragflow_update_dataset(
        dataset_id: DatasetId,
        name: str | None = None,
        description: str | None = None,
        language: str | None = None,
        permission: Literal["me", "team"] | None = None,
        pagerank: Annotated[int | None, Field(ge=0, le=100)] = None,
        embedding_model: Annotated[
            str | None,
            Field(description="Only allowed while the dataset has no chunks"),
        ] = None,
        parser_id: Annotated[
            str | None,
            Field(description="Default chunking method for newly parsed documents"),
        ] = None,
    ) -> dict[str, Any]:
        """Update dataset settings. Only the fields you pass change."""
        body = {
            k: v
            for k, v in {
                "name": name,
                "description": description,
                "language": language,
                "permission": permission,
                "pagerank": pagerank,
                "embedding_model": embedding_model,
                "parser_id": parser_id,
            }.items()
            if v is not None
        }
        if not body:
            raise ValueError("pass at least one field to update")
        if parser_id is not None:
            body["parse_type"] = 1
        data = await client.call("PUT", f"/datasets/{dataset_id}", json=body)
        return _dataset(data)

    @tool(DELETE)
    async def ragflow_delete_datasets(
        dataset_ids: Annotated[list[str], Field(description="IDs of the datasets to delete")],
    ) -> dict[str, Any]:
        """Permanently delete datasets together with all their documents and chunks."""
        ids = _non_empty("dataset_ids", dataset_ids)
        await client.call("DELETE", "/datasets", json={"ids": ids})
        return {"deleted": ids}

    # Documents

    @tool(READ)
    async def ragflow_list_documents(
        dataset_id: DatasetId,
        keywords: Annotated[str | None, Field(description="Match document names")] = None,
        name: Annotated[str | None, Field(description="Exact document name")] = None,
        document_id: Annotated[str | None, Field(description="Return only this document")] = None,
        page: Page = 1,
        page_size: PageSize = 30,
        orderby: Literal["create_time", "update_time"] = "create_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List documents in a dataset with their parsing status and chunk counts."""
        data = await list_docs(
            dataset_id,
            {
                "keywords": keywords,
                "name": name,
                "id": document_id,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
            },
        )
        docs = data.get("docs") or []
        return {
            "total": data.get("total", len(docs)),
            "documents": [_document(d) for d in docs],
        }

    @tool(UPLOAD)
    async def ragflow_upload_document(
        dataset_id: DatasetId,
        filename: Annotated[
            str | None,
            Field(
                description="Document name with an extension that sets the parser "
                "(.md, .txt, .pdf, .docx, ...). Required for text and content_base64; "
                "defaults to the file name for file_path and url."
            ),
        ] = None,
        text: Annotated[
            str | None, Field(description="UTF-8 text content, e.g. Markdown")
        ] = None,
        content_base64: Annotated[
            str | None, Field(description="File bytes as base64 (small files only)")
        ] = None,
        url: Annotated[
            str | None,
            Field(description="http(s) URL that RAGFlow downloads (a file or a web page)"),
        ] = None,
        file_path: Annotated[
            str | None,
            Field(description="Path on the MCP server host, inside an allowed upload root"),
        ] = None,
        meta_fields: Annotated[
            dict[str, Any] | None,
            Field(description="Document metadata, used for filtering in retrieval"),
        ] = None,
        replace_existing: Annotated[
            bool,
            Field(
                description="If a document with the same name exists, replace it. "
                "Without this flag a duplicate name is an error."
            ),
        ] = False,
        parse: Annotated[bool, Field(description="Start parsing after the upload")] = True,
        wait_seconds: WaitSeconds = 0,
    ) -> dict[str, Any]:
        """Add a document to a dataset from text, base64 bytes, a URL or an allowed local file."""
        sources = {
            "text": text,
            "content_base64": content_base64,
            "url": url,
            "file_path": file_path,
        }
        given = [k for k, v in sources.items() if v is not None]
        if len(given) != 1:
            raise ValueError(
                "pass exactly one of text, content_base64, url or file_path"
                + (f" (got {', '.join(given)})" if given else "")
            )
        source = given[0]

        blob: bytes | None = None
        if source == "text":
            blob = text.encode("utf-8")
        elif source == "content_base64":
            try:
                blob = base64.b64decode(content_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"content_base64 is not valid base64: {exc}") from None
        elif source == "file_path":
            if not upload_roots:
                raise ValueError(
                    "file_path uploads are disabled; the operator can allow directories "
                    "with RAGFLOW_UPLOAD_ROOTS"
                )
            path = Path(file_path).expanduser().resolve()
            if not any(path.is_relative_to(root) for root in upload_roots):
                raise ValueError("file_path is outside the allowed upload roots")
            if not path.is_file():
                raise ValueError("file_path does not point to a file")
            blob = path.read_bytes()
            filename = filename or path.name
        else:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("url must be an http or https URL")
            filename = filename or PurePosixPath(parsed.path).name or parsed.netloc

        if not filename:
            raise ValueError("filename is required for text and content_base64 uploads")
        if source != "url" and not PurePosixPath(filename).suffix:
            raise ValueError(
                "filename needs an extension such as .md, .txt or .pdf; RAGFlow picks "
                "the parser from it"
            )

        existing = await find_by_name(dataset_id, filename)
        if existing and not replace_existing:
            raise ValueError(
                f"a document named {filename!r} already exists "
                f"(id {', '.join(d['id'] for d in existing)}); pass replace_existing=true "
                "to replace it"
            )

        if source == "url":
            doc = await client.call(
                "POST",
                f"/datasets/{dataset_id}/documents",
                params={"type": "web"},
                data={"name": filename, "url": url},
            )
        else:
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            data = await client.call(
                "POST",
                f"/datasets/{dataset_id}/documents",
                files={"file": (filename, blob, mime)},
            )
            doc = data[0]
        doc_id = doc["id"]

        replaced = [d["id"] for d in existing]
        if replaced:
            await client.call(
                "DELETE", f"/datasets/{dataset_id}/documents", json={"ids": replaced}
            )
            if doc.get("name") != filename:
                # RAGFlow renamed the upload to avoid the clash; restore the name.
                await client.call(
                    "PATCH", f"/datasets/{dataset_id}/documents/{doc_id}", json={"name": filename}
                )
        if meta_fields:
            await client.call(
                "PATCH",
                f"/datasets/{dataset_id}/documents/{doc_id}",
                json={"meta_fields": meta_fields},
            )

        result: dict[str, Any] = {"replaced": replaced, "parse_started": False}
        if parse:
            await client.call(
                "POST", f"/datasets/{dataset_id}/chunks", json={"document_ids": [doc_id]}
            )
            result["parse_started"] = True
        if parse and wait_seconds:
            waited = await wait_for(dataset_id, [doc_id], wait_seconds)
            result["document"] = waited["documents"][0]
            result["still_parsing"] = bool(waited["still_parsing"])
        else:
            result["document"] = _document(await get_document(dataset_id, doc_id))
        return result

    @tool(WRITE)
    async def ragflow_update_document(
        dataset_id: DatasetId,
        document_id: DocumentId,
        name: Annotated[str | None, Field(description="New document name")] = None,
        meta_fields: Annotated[
            dict[str, Any] | None,
            Field(description="Replaces the document's metadata"),
        ] = None,
        enabled: Annotated[
            bool | None, Field(description="Whether retrieval may use this document")
        ] = None,
        parser_id: Annotated[
            str | None,
            Field(description="Chunking method; takes effect when the document is parsed again"),
        ] = None,
    ) -> dict[str, Any]:
        """Rename a document, replace its metadata, enable or disable it, or change its parser."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if meta_fields is not None:
            body["meta_fields"] = meta_fields
        if enabled is not None:
            body["enabled"] = 1 if enabled else 0
        if parser_id is not None:
            body["parser_id"] = parser_id
        if not body:
            raise ValueError("pass at least one field to update")
        await client.call("PATCH", f"/datasets/{dataset_id}/documents/{document_id}", json=body)
        return _document(await get_document(dataset_id, document_id))

    @tool(READ)
    async def ragflow_download_document(
        dataset_id: DatasetId,
        document_id: DocumentId,
        max_chars: Annotated[int, Field(ge=1, le=500_000)] = 20_000,
    ) -> dict[str, Any]:
        """Read a document's original file. Text files come back as text; binary files only as size and type."""
        blob, content_type = await client.download(
            f"/datasets/{dataset_id}/documents/{document_id}"
        )
        result: dict[str, Any] = {"content_type": content_type, "size": len(blob)}
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            result["text"] = None
            result["note"] = "binary file; use ragflow_list_chunks to read its parsed text"
            return result
        result["text"] = text[:max_chars]
        result["truncated"] = len(text) > max_chars
        return result

    @tool(DELETE)
    async def ragflow_delete_documents(
        dataset_id: DatasetId,
        document_ids: Annotated[list[str], Field(description="IDs of the documents to delete")],
    ) -> dict[str, Any]:
        """Permanently delete documents and their chunks from a dataset."""
        ids = _non_empty("document_ids", document_ids)
        await client.call("DELETE", f"/datasets/{dataset_id}/documents", json={"ids": ids})
        return {"deleted": ids}

    @tool(WRITE)
    async def ragflow_parse_documents(
        dataset_id: DatasetId,
        document_ids: Annotated[list[str], Field(description="IDs of the documents to parse")],
        wait_seconds: WaitSeconds = 0,
    ) -> dict[str, Any]:
        """Start (or restart) parsing documents into chunks; optionally wait until it finishes."""
        ids = _non_empty("document_ids", document_ids)
        await client.call("POST", f"/datasets/{dataset_id}/chunks", json={"document_ids": ids})
        if not wait_seconds:
            return {"parse_started": ids}
        return {"parse_started": ids, **await wait_for(dataset_id, ids, wait_seconds)}

    @tool(WRITE)
    async def ragflow_stop_parsing(
        dataset_id: DatasetId,
        document_ids: Annotated[list[str], Field(description="IDs of the documents to stop")],
    ) -> dict[str, Any]:
        """Stop parsing documents that are queued or running."""
        ids = _non_empty("document_ids", document_ids)
        await client.call("DELETE", f"/datasets/{dataset_id}/chunks", json={"document_ids": ids})
        return {"stopped": ids}

    # Chunks

    @tool(READ)
    async def ragflow_list_chunks(
        dataset_id: DatasetId,
        document_id: DocumentId,
        keywords: Annotated[str | None, Field(description="Match chunk content")] = None,
        page: Page = 1,
        page_size: PageSize = 30,
    ) -> dict[str, Any]:
        """List the parsed chunks of a document."""
        data = await client.call(
            "GET",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            params={"keywords": keywords, "page": page, "page_size": page_size},
        )
        data = data or {}
        chunks = data.get("chunks") or []
        result: dict[str, Any] = {
            "total": data.get("total", len(chunks)),
            "chunks": [_chunk(c) for c in chunks],
        }
        if isinstance(data.get("doc"), dict):
            result["document"] = _document(data["doc"])
        return result

    @tool(WRITE)
    async def ragflow_add_chunk(
        dataset_id: DatasetId,
        document_id: DocumentId,
        content: Annotated[str, Field(description="Chunk text")],
        important_keywords: list[str] | None = None,
        questions: Annotated[
            list[str] | None, Field(description="Questions this chunk answers")
        ] = None,
    ) -> dict[str, Any]:
        """Add a hand-written chunk to a document."""
        body: dict[str, Any] = {"content": content}
        if important_keywords:
            body["important_keywords"] = important_keywords
        if questions:
            body["questions"] = questions
        data = await client.call(
            "POST", f"/datasets/{dataset_id}/documents/{document_id}/chunks", json=body
        )
        chunk = (data or {}).get("chunk", data) or {}
        return _chunk(chunk)

    @tool(WRITE)
    async def ragflow_update_chunk(
        dataset_id: DatasetId,
        document_id: DocumentId,
        chunk_id: Annotated[str, Field(description="Chunk ID")],
        content: str | None = None,
        important_keywords: list[str] | None = None,
        questions: list[str] | None = None,
        available: Annotated[
            bool | None, Field(description="Whether retrieval may return this chunk")
        ] = None,
    ) -> dict[str, Any]:
        """Edit a chunk's text, keywords or questions, or switch it on or off for retrieval."""
        body = {
            k: v
            for k, v in {
                "content": content,
                "important_keywords": important_keywords,
                "questions": questions,
                "available": available,
            }.items()
            if v is not None
        }
        if not body:
            raise ValueError("pass at least one field to update")
        await client.call(
            "PATCH",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}",
            json=body,
        )
        return {"updated": chunk_id, "fields": sorted(body)}

    @tool(DELETE)
    async def ragflow_delete_chunks(
        dataset_id: DatasetId,
        document_id: DocumentId,
        chunk_ids: Annotated[list[str], Field(description="IDs of the chunks to delete")],
    ) -> dict[str, Any]:
        """Permanently delete chunks from a document."""
        ids = _non_empty("chunk_ids", chunk_ids)
        await client.call(
            "DELETE",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            json={"chunk_ids": ids},
        )
        return {"deleted": ids}

    # Metadata

    @tool(READ)
    async def ragflow_metadata_summary(dataset_id: DatasetId) -> dict[str, Any]:
        """Show which metadata keys and values the documents of a dataset use, with counts."""
        data = await client.call("GET", f"/datasets/{dataset_id}/metadata/summary")
        return (data or {}).get("summary", data) or {}

    @tool(WRITE)
    async def ragflow_update_metadata(
        dataset_id: DatasetId,
        document_ids: Annotated[
            list[str] | None, Field(description="Limit the change to these documents")
        ] = None,
        metadata_condition: Annotated[
            dict[str, Any] | None,
            Field(
                description='Select documents by metadata, e.g. {"logic": "and", '
                '"conditions": [{"name": "author", "comparison_operator": "is", '
                '"value": "alice"}]}'
            ),
        ] = None,
        updates: Annotated[
            list[dict[str, str]] | None,
            Field(
                description='Values to set: [{"key": ..., "value": ..., "match": '
                "optional current value}]"
            ),
        ] = None,
        deletes: Annotated[
            list[dict[str, str]] | None,
            Field(description='Keys to remove: [{"key": ..., "value": optional}]'),
        ] = None,
        all_documents: Annotated[
            bool,
            Field(description="Required to change every document when no selector is given"),
        ] = False,
    ) -> dict[str, Any]:
        """Set or remove metadata on many documents of a dataset at once."""
        if not updates and not deletes:
            raise ValueError("pass updates or deletes")
        selector: dict[str, Any] = {}
        if document_ids:
            selector["document_ids"] = document_ids
        if metadata_condition:
            selector["metadata_condition"] = metadata_condition
        if not selector and not all_documents:
            raise ValueError(
                "without document_ids or metadata_condition this changes every document; "
                "pass all_documents=true to confirm"
            )
        body: dict[str, Any] = {"updates": updates or [], "deletes": deletes or []}
        if selector:
            body["selector"] = selector
        return await client.call("POST", f"/datasets/{dataset_id}/metadata/update", json=body)

    # Retrieval

    @tool(READ)
    async def ragflow_retrieval(
        question: Annotated[str, Field(description="Query, ideally in the documents' language")],
        dataset_ids: Annotated[list[str], Field(description="Datasets to search")],
        document_ids: Annotated[
            list[str] | None, Field(description="Limit the search to these documents")
        ] = None,
        page: Page = 1,
        page_size: Annotated[int, Field(ge=1, le=100)] = 8,
        similarity_threshold: Annotated[float | None, Field(ge=0, le=1)] = None,
        vector_similarity_weight: Annotated[float | None, Field(ge=0, le=1)] = None,
        top_k: Annotated[
            int | None, Field(ge=1, description="Candidates considered before reranking")
        ] = None,
        rerank_id: Annotated[
            str | None, Field(description="Rerank model as model_name@factory")
        ] = None,
        keyword: Annotated[
            bool | None, Field(description="Expand the query with LLM keywords")
        ] = None,
        metadata_condition: Annotated[
            dict[str, Any] | None, Field(description="Metadata filter, as in ragflow_update_metadata")
        ] = None,
    ) -> dict[str, Any]:
        """Search datasets and return the matching chunks with their exact text and source document."""
        body: dict[str, Any] = {
            "question": question,
            "dataset_ids": _non_empty("dataset_ids", dataset_ids),
            "page": page,
            "page_size": page_size,
        }
        optional = {
            "document_ids": document_ids,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "rerank_id": rerank_id,
            "keyword": keyword,
            "metadata_condition": metadata_condition,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        data = await client.call("POST", "/retrieval", json=body) or {}
        chunks = data.get("chunks") or []
        return {
            "total": data.get("total", len(chunks)),
            "chunks": [_retrieved(c) for c in chunks],
            "documents": [
                _pick(a, "doc_id", "doc_name", "count") for a in data.get("doc_aggs") or []
            ],
        }

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
