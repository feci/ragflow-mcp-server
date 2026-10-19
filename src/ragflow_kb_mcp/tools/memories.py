"""Memories and their messages."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from ..common import (
    DELETE,
    READ,
    WRITE,
    Context,
    MemoryId,
    Page,
    drop_none,
    non_empty,
    pick,
)

MemoryType = Literal["raw", "semantic", "episodic", "procedural"]
MessageId = Annotated[int, Field(ge=0, description="Message ID (an integer within its memory)")]
MemoryIds = Annotated[list[str], Field(description="IDs of the memories to use")]
MessagePageSize = Annotated[int, Field(ge=1, le=100, description="Items per page (max 100)")]


def _memory(m: dict[str, Any]) -> dict[str, Any]:
    return pick(
        m,
        "id",
        "name",
        "description",
        "memory_type",
        "embd_id",
        "llm_id",
        "permissions",
        "memory_size",
        "forgetting_policy",
        "temperature",
        "storage_type",
        "owner_name",
        "create_time",
        "update_time",
    )


def _message(m: dict[str, Any]) -> dict[str, Any]:
    out = pick(
        m,
        "memory_id",
        "message_id",
        "message_type",
        "content",
        "status",
        "agent_id",
        "agent_name",
        "session_id",
        "user_id",
        "valid_at",
        "forget_at",
        "source_id",
    )
    if out.get("forget_at") == "":
        del out["forget_at"]
    task = m.get("task")
    if isinstance(task, dict) and task:
        out["task"] = pick(task, "progress", "progress_msg")
    if m.get("extract"):
        out["extract"] = [_message(e) for e in m["extract"]]
    return out


def _message_path(memory_id: str, message_id: int) -> str:
    return f"/messages/{memory_id}:{message_id}"


def register(ctx: Context) -> None:
    client = ctx.client
    tool = ctx.tool

    @tool(READ)
    async def ragflow_list_memories(
        keywords: Annotated[str | None, Field(description="Fuzzy match on memory names")] = None,
        memory_types: Annotated[
            list[MemoryType] | None,
            Field(description="Only memories that include one of these types"),
        ] = None,
        tenant_ids: Annotated[
            list[str] | None, Field(description="Only memories owned by these tenants")
        ] = None,
        storage_type: Annotated[
            str | None, Field(description="Storage format filter, e.g. table")
        ] = None,
        page: Page = 1,
        page_size: Annotated[int, Field(ge=1, le=1000, description="Items per page")] = 50,
    ) -> dict[str, Any]:
        """List memories (agent conversation memory stores) with their types and models."""
        data = await client.call(
            "GET",
            "/memories",
            params={
                "keywords": keywords,
                "memory_type": ",".join(memory_types) if memory_types else None,
                "tenant_id": ",".join(tenant_ids) if tenant_ids else None,
                "storage_type": storage_type,
                "page": page,
                "page_size": page_size,
            },
        )
        data = data or {}
        items = data.get("memory_list") or []
        return {
            "total": data.get("total_count", len(items)),
            "memories": [_memory(m) for m in items],
        }

    @tool(READ)
    async def ragflow_get_memory_config(memory_id: MemoryId) -> dict[str, Any]:
        """Get a memory's full configuration, including its extraction prompts."""
        data = await client.call("GET", f"/memories/{memory_id}/config") or {}
        return {
            **_memory(data),
            **pick(data, "system_prompt", "user_prompt", "tenant_id"),
        }

    @tool(WRITE)
    async def ragflow_create_memory(
        name: Annotated[str, Field(description="Memory name (max 128 characters)")],
        llm_id: Annotated[
            str,
            Field(description="Chat model that extracts memories, as model_name@factory"),
        ],
        memory_types: Annotated[
            list[MemoryType],
            Field(
                description="What to keep: raw dialogue, semantic facts, episodic events, "
                "procedural habits"
            ),
        ] = ["raw", "semantic"],
        embedding_model: Annotated[
            str | None,
            Field(
                description="Embedding model as model_name@factory; defaults to the "
                "server's configured default"
            ),
        ] = None,
        description: Annotated[str | None, Field(description="Short description")] = None,
    ) -> dict[str, Any]:
        """Create a memory that stores and extracts agent conversations."""
        model = embedding_model or ctx.default_embedding_model
        if not model:
            raise ValueError(
                "embedding_model is required (no RAGFLOW_DEFAULT_EMBEDDING_MODEL is configured)"
            )
        body = {
            "name": name,
            "memory_type": list(dict.fromkeys(memory_types)),
            "embd_id": model,
            "llm_id": llm_id,
        }
        data = await client.call("POST", "/memories", json=body)
        if description:
            # Memory creation does not accept a description; set it right after.
            data = await client.call(
                "PUT", f"/memories/{data['id']}", json={"description": description}
            )
        return _memory(data)

    @tool(WRITE)
    async def ragflow_update_memory(
        memory_id: MemoryId,
        name: str | None = None,
        description: str | None = None,
        permissions: Annotated[
            Literal["me", "team", "all"] | None, Field(description="Who may manage the memory")
        ] = None,
        llm_id: Annotated[str | None, Field(description="Chat model as model_name@factory")] = None,
        embedding_model: Annotated[
            str | None, Field(description="Only allowed while the memory has no messages")
        ] = None,
        memory_types: Annotated[
            list[MemoryType] | None,
            Field(description="Only allowed while the memory has no messages"),
        ] = None,
        memory_size: Annotated[
            int | None,
            Field(gt=0, le=5_242_880, description="Size limit in bytes (max 5 MiB)"),
        ] = None,
        forgetting_policy: Annotated[
            Literal["FIFO"] | None, Field(description="What to evict when the memory is full")
        ] = None,
        temperature: Annotated[float | None, Field(ge=0, le=1)] = None,
        system_prompt: Annotated[
            str | None,
            Field(description="Extraction prompt; keep its OUTPUT REQUIREMENTS and FORMAT parts"),
        ] = None,
        user_prompt: str | None = None,
        avatar: Annotated[str | None, Field(description="Avatar image as base64")] = None,
    ) -> dict[str, Any]:
        """Update memory settings. Only the fields you pass change."""
        body = drop_none(
            {
                "name": name,
                "description": description,
                "permissions": permissions,
                "llm_id": llm_id,
                "embd_id": embedding_model,
                "memory_type": list(dict.fromkeys(memory_types)) if memory_types else None,
                "memory_size": memory_size,
                "forgetting_policy": forgetting_policy,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "avatar": avatar,
            }
        )
        if not body:
            raise ValueError("pass at least one field to update")
        data = await client.call("PUT", f"/memories/{memory_id}", json=body)
        return _memory(data or {})

    @tool(DELETE)
    async def ragflow_delete_memory(memory_id: MemoryId) -> dict[str, Any]:
        """Permanently delete a memory and all of its messages."""
        await client.call("DELETE", f"/memories/{memory_id}")
        return {"deleted": memory_id}

    @tool(READ)
    async def ragflow_list_memory_messages(
        memory_id: MemoryId,
        agent_ids: Annotated[
            list[str] | None, Field(description="Only messages from these agents")
        ] = None,
        session_id: Annotated[str | None, Field(description="Only messages of this session")] = None,
        page: Page = 1,
        page_size: MessagePageSize = 50,
    ) -> dict[str, Any]:
        """List a memory's stored dialogue messages, the memories extracted from each, and extraction progress."""
        data = await client.call(
            "GET",
            f"/memories/{memory_id}",
            params={
                "agent_id": ",".join(agent_ids) if agent_ids else None,
                # RAGFlow applies the "keywords" parameter to session_id.
                "keywords": session_id,
                "page": page,
                "page_size": page_size,
            },
        )
        messages = (data or {}).get("messages") or {}
        items = messages.get("message_list") or []
        return {
            "total": messages.get("total_count", len(items)),
            "messages": [_message(m) for m in items],
        }

    @tool(WRITE)
    async def ragflow_add_memory_message(
        memory_ids: MemoryIds,
        agent_id: Annotated[str, Field(description="ID of the agent the dialogue came from")],
        session_id: Annotated[str, Field(description="ID of the conversation session")],
        user_id: Annotated[
            str, Field(description="ID of the user in the conversation (required with API keys)")
        ],
        user_input: Annotated[str, Field(description="What the user said")],
        agent_response: Annotated[str, Field(description="What the agent answered")],
    ) -> dict[str, Any]:
        """Save one user/agent exchange to memories; extraction runs in the background."""
        ids = non_empty("memory_ids", memory_ids)
        body = await client.call(
            "POST",
            "/messages",
            json={
                "memory_id": ids,
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "user_input": user_input,
                "agent_response": agent_response,
            },
            full=True,
        )
        return {
            "queued": True,
            "memory_ids": ids,
            "message": body.get("message"),
            "note": "check progress with ragflow_list_memory_messages",
        }

    @tool(READ)
    async def ragflow_search_memory_messages(
        query: Annotated[str, Field(description="Search text or question")],
        memory_ids: MemoryIds,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        similarity_threshold: Annotated[float | None, Field(ge=0, le=1)] = None,
        keywords_similarity_weight: Annotated[
            float | None,
            Field(ge=0, le=1, description="Weight of keyword versus embedding matching"),
        ] = None,
        top_n: Annotated[int, Field(ge=1, le=100, description="Maximum number of matches")] = 5,
    ) -> dict[str, Any]:
        """Search memories for messages relevant to a query."""
        data = await client.call(
            "GET",
            "/messages/search",
            params={
                "query": query,
                "memory_id": ",".join(non_empty("memory_ids", memory_ids)),
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "similarity_threshold": similarity_threshold,
                "keywords_similarity_weight": keywords_similarity_weight,
                "top_n": top_n,
            },
        )
        return {"messages": [_message(m) for m in data or []]}

    @tool(READ)
    async def ragflow_recent_memory_messages(
        memory_ids: MemoryIds,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
    ) -> dict[str, Any]:
        """Get the most recent messages stored in memories."""
        data = await client.call(
            "GET",
            "/messages",
            params={
                "memory_id": ",".join(non_empty("memory_ids", memory_ids)),
                "agent_id": agent_id,
                "session_id": session_id,
                "limit": limit,
            },
        )
        return {"messages": [_message(m) for m in data or []]}

    @tool(READ)
    async def ragflow_get_memory_message(
        memory_id: MemoryId, message_id: MessageId
    ) -> dict[str, Any]:
        """Get one memory message with its full content (without the embedding vector)."""
        data = await client.call("GET", f"{_message_path(memory_id, message_id)}/content") or {}
        return _message(data)

    @tool(WRITE)
    async def ragflow_set_memory_message_status(
        memory_id: MemoryId,
        message_id: MessageId,
        enabled: Annotated[bool, Field(description="False hides the message from agents")],
    ) -> dict[str, Any]:
        """Enable or disable a memory message for agents."""
        await client.call("PUT", _message_path(memory_id, message_id), json={"status": enabled})
        return {"memory_id": memory_id, "message_id": message_id, "enabled": enabled}

    @tool(DELETE)
    async def ragflow_forget_memory_message(
        memory_id: MemoryId, message_id: MessageId
    ) -> dict[str, Any]:
        """Forget a memory message: agents stop retrieving it and it is evicted first when space runs out."""
        await client.call("DELETE", _message_path(memory_id, message_id))
        return {"forgotten": {"memory_id": memory_id, "message_id": message_id}}
