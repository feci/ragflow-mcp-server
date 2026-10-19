"""Search apps and system health."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import Field

from ..client import RagflowError
from ..common import DELETE, READ, WRITE, Context, Page, PageSize, SearchId, drop_none, pick

DatasetIds = Annotated[
    list[str] | None, Field(description="Datasets the search app searches (kb_ids)")
]
DocumentIds = Annotated[
    list[str] | None, Field(description="Limit the search to these documents (doc_ids)")
]
SimilarityThreshold = Annotated[
    float | None, Field(ge=0, le=1, description="Minimum similarity for a chunk to count")
]
VectorWeight = Annotated[
    float | None,
    Field(ge=0, le=1, description="Weight of vector similarity; keyword weight is 1 minus this"),
]
TopK = Annotated[int | None, Field(ge=1, description="Candidates considered before reranking")]
RerankId = Annotated[str | None, Field(description="Rerank model as model_name@factory")]
Keyword = Annotated[bool | None, Field(description="Expand the question with LLM keywords")]
UseKg = Annotated[bool | None, Field(description="Also search the knowledge graph")]
CrossLanguages = Annotated[
    list[str] | None, Field(description="Also translate the question into these languages")
]
MetaDataFilter = Annotated[
    dict[str, Any] | None, Field(description="Metadata filter applied to the documents")
]
ChatModel = Annotated[
    str | None,
    Field(description="Chat model for answers as model_name@factory; default is the tenant's"),
]
ExtraConfig = Annotated[
    dict[str, Any] | None,
    Field(description="Other search_config keys to set as-is, e.g. llm_setting"),
]


def _search_app(s: dict[str, Any]) -> dict[str, Any]:
    out = pick(s, "id", "name", "description", "create_time", "update_time")
    if isinstance(s.get("search_config"), dict):
        out["search_config"] = s["search_config"]
    return out


def _config(
    dataset_ids: list[str] | None,
    document_ids: list[str] | None,
    similarity_threshold: float | None,
    vector_similarity_weight: float | None,
    top_k: int | None,
    rerank_id: str | None,
    keyword: bool | None,
    use_kg: bool | None,
    cross_languages: list[str] | None,
    meta_data_filter: dict[str, Any] | None,
    chat_model: str | None,
    search_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = dict(search_config or {})
    config.update(
        drop_none(
            {
                "kb_ids": dataset_ids,
                "doc_ids": document_ids,
                "similarity_threshold": similarity_threshold,
                "vector_similarity_weight": vector_similarity_weight,
                "top_k": top_k,
                "rerank_id": rerank_id,
                "keyword": keyword,
                "use_kg": use_kg,
                "cross_languages": cross_languages,
                "meta_data_filter": meta_data_filter,
                "chat_id": chat_model,
            }
        )
    )
    return config


def parse_sse(blob: bytes) -> list[dict[str, Any]]:
    """Decode the `data:` lines of a Server-Sent Events body into JSON objects."""
    events = []
    for line in blob.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def register(ctx: Context) -> None:
    client = ctx.client
    tool = ctx.tool

    async def put_search(search_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await client.call("PUT", f"/searches/{search_id}", json=body)

    @tool(READ)
    async def ragflow_list_search_apps(
        keywords: Annotated[str | None, Field(description="Match search app names")] = None,
        page: Page = 1,
        page_size: PageSize = 30,
        orderby: Literal["create_time", "update_time"] = "create_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List search apps (saved search configurations that answer questions from datasets)."""
        data = await client.call(
            "GET",
            "/searches",
            params={
                "keywords": keywords,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
            },
        )
        data = data or {}
        apps = data.get("search_apps") or []
        return {
            "total": data.get("total", len(apps)),
            "search_apps": [_search_app(a) for a in apps],
        }

    @tool(READ)
    async def ragflow_get_search_app(search_id: SearchId) -> dict[str, Any]:
        """Get one search app with its full search configuration."""
        return _search_app(await client.call("GET", f"/searches/{search_id}"))

    @tool(WRITE)
    async def ragflow_create_search_app(
        name: Annotated[str, Field(description="Search app name (unique, max 255 bytes)")],
        description: str | None = None,
        dataset_ids: DatasetIds = None,
        document_ids: DocumentIds = None,
        similarity_threshold: SimilarityThreshold = None,
        vector_similarity_weight: VectorWeight = None,
        top_k: TopK = None,
        rerank_id: RerankId = None,
        keyword: Keyword = None,
        use_kg: UseKg = None,
        cross_languages: CrossLanguages = None,
        meta_data_filter: MetaDataFilter = None,
        chat_model: ChatModel = None,
        search_config: ExtraConfig = None,
    ) -> dict[str, Any]:
        """Create a search app; pass dataset_ids so ragflow_search_app_completion can answer from them."""
        body = drop_none({"name": name, "description": description})
        created = await client.call("POST", "/searches", json=body)
        search_id = created["search_id"]
        config = _config(
            dataset_ids,
            document_ids,
            similarity_threshold,
            vector_similarity_weight,
            top_k,
            rerank_id,
            keyword,
            use_kg,
            cross_languages,
            meta_data_filter,
            chat_model,
            search_config,
        )
        if config:
            # Creation takes only name and description; the configuration needs an update.
            return _search_app(await put_search(search_id, {"name": name, "search_config": config}))
        return _search_app(await client.call("GET", f"/searches/{search_id}"))

    @tool(WRITE)
    async def ragflow_update_search_app(
        search_id: SearchId,
        name: Annotated[str | None, Field(description="New name")] = None,
        description: str | None = None,
        dataset_ids: DatasetIds = None,
        document_ids: DocumentIds = None,
        similarity_threshold: SimilarityThreshold = None,
        vector_similarity_weight: VectorWeight = None,
        top_k: TopK = None,
        rerank_id: RerankId = None,
        keyword: Keyword = None,
        use_kg: UseKg = None,
        cross_languages: CrossLanguages = None,
        meta_data_filter: MetaDataFilter = None,
        chat_model: ChatModel = None,
        search_config: ExtraConfig = None,
    ) -> dict[str, Any]:
        """Change a search app. Configuration keys you pass are merged into the saved ones."""
        config = _config(
            dataset_ids,
            document_ids,
            similarity_threshold,
            vector_similarity_weight,
            top_k,
            rerank_id,
            keyword,
            use_kg,
            cross_languages,
            meta_data_filter,
            chat_model,
            search_config,
        )
        if name is None and description is None and not config:
            raise ValueError("pass at least one field to update")
        if name is None:
            # The update endpoint always requires the name.
            name = (await client.call("GET", f"/searches/{search_id}"))["name"]
        body = {"name": name, "search_config": config}
        if description is not None:
            body["description"] = description
        return _search_app(await put_search(search_id, body))

    @tool(DELETE)
    async def ragflow_delete_search_app(search_id: SearchId) -> dict[str, Any]:
        """Permanently delete a search app (its datasets stay)."""
        await client.call("DELETE", f"/searches/{search_id}")
        return {"deleted": search_id}

    @tool(READ)
    async def ragflow_search_app_completion(
        search_id: SearchId,
        question: Annotated[str, Field(description="Question, ideally in the documents' language")],
        dataset_ids: Annotated[
            list[str] | None,
            Field(description="Datasets to use only when the search app has none configured"),
        ] = None,
        max_references: Annotated[
            int, Field(ge=0, le=100, description="Return at most this many source chunks")
        ] = 10,
    ) -> dict[str, Any]:
        """Answer a question with a search app's datasets and LLM; returns the answer and its source chunks."""
        body = drop_none({"question": question, "kb_ids": dataset_ids})
        blob, _ = await client.raw("POST", f"/searches/{search_id}/completions", json=body)
        events = parse_sse(blob)
        errors = [
            e.get("message") or "unknown error" for e in events if e.get("code") not in (0, None)
        ]
        final = next(
            (
                e["data"]
                for e in events
                if isinstance(e.get("data"), dict) and e["data"].get("final")
            ),
            None,
        )
        if final is None:
            if errors:
                raise RagflowError(f"search completion failed: {'; '.join(errors)}")
            # No final event: rebuild the answer from the streamed pieces.
            answer = "".join(
                e["data"].get("answer") or "" for e in events if isinstance(e.get("data"), dict)
            )
            return {"answer": answer, "references": [], "documents": []}
        reference = final.get("reference") or {}
        chunks = reference.get("chunks") or []
        result: dict[str, Any] = {
            "answer": final.get("answer") or "",
            "references": [
                {
                    "index": i,
                    **pick(
                        c,
                        "id",
                        "content",
                        "document_id",
                        "document_name",
                        "dataset_id",
                        "similarity",
                    ),
                }
                for i, c in enumerate(chunks[:max_references])
            ],
            "documents": [
                pick(a, "doc_id", "doc_name", "count") for a in reference.get("doc_aggs") or []
            ],
        }
        if len(chunks) > max_references:
            result["references_omitted"] = len(chunks) - max_references
        if errors:
            result["errors"] = errors
        return result

    @tool(READ)
    async def ragflow_health() -> dict[str, Any]:
        """Check whether RAGFlow and its database, cache, document engine, storage and queue are healthy."""
        # healthz answers HTTP 500 with the same per-component body when something is down,
        # so read the body directly instead of treating the status as an error.
        status, body = await client.json_with_status("GET", "/system/healthz")
        return {"healthy": status < 400 and body.get("status") == "ok", "components": body}
