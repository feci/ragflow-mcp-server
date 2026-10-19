"""Agents, agent sessions and agent conversations."""

from __future__ import annotations

import mimetypes
from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

from ..common import (
    DEFAULT_MAX_CHARS,
    DELETE,
    READ,
    UPLOAD,
    WRITE,
    AgentId,
    Context,
    MaxChars,
    Page,
    PageSize,
    describe_bytes,
    drop_none,
    non_empty,
    pick,
    read_bytes_source,
)

# Running an agent calls LLMs and whatever tools its DSL contains (web search, code, ...).
RUN = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)

AGENT_FIELDS = (
    "id",
    "title",
    "description",
    "permission",
    "canvas_category",
    "canvas_type",
    "tags",
    "release",
    "nickname",
    "type",
    "create_time",
    "update_time",
    "last_publish_time",
)
Dsl = Annotated[
    dict[str, Any],
    Field(
        description="Agent graph (DSL) as a JSON object with components such as Begin, Agent "
        "and Message; take one from ragflow_list_agent_templates or ragflow_get_agent"
    ),
]


def _agent(a: dict[str, Any], include_dsl: bool = False) -> dict[str, Any]:
    out = pick(a, *AGENT_FIELDS)
    if include_dsl and "dsl" in a:
        out["dsl"] = a["dsl"]
    return out


def _session(s: dict[str, Any], include_messages: bool) -> dict[str, Any]:
    out = pick(
        s, "id", "name", "source", "round", "tokens", "thumb_up", "errors", "create_time", "update_time"
    )
    out["agent_id"] = s.get("agent_id") or s.get("dialog_id")
    if include_messages:
        out["messages"] = [
            pick(m, "id", "role", "content", "reference")
            for m in s.get("message") or []
            if isinstance(m, dict)
        ]
    return out


def _title(value: Any) -> Any:
    # Templates carry titles and descriptions per UI language.
    if isinstance(value, dict):
        return value.get("en") or next(iter(value.values()), None)
    return value


def register(ctx: Context) -> None:
    client = ctx.client
    tool = ctx.tool

    # Agents

    @tool(READ)
    async def ragflow_list_agents(
        keywords: Annotated[str | None, Field(description="Match agent titles")] = None,
        canvas_category: Annotated[
            Literal["agent_canvas", "dataflow_canvas"] | None,
            Field(description="agent_canvas = chat agents, dataflow_canvas = ingestion pipelines"),
        ] = None,
        tags: Annotated[list[str] | None, Field(description="Only agents with these tags")] = None,
        owner_ids: Annotated[
            list[str] | None, Field(description="Only agents owned by these user IDs")
        ] = None,
        page: Page = 1,
        page_size: PageSize = 30,
        orderby: Literal["create_time", "update_time"] = "update_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List agents (agent canvases and ingestion pipelines) without their graphs."""
        data = await client.call(
            "GET",
            "/agents",
            params={
                "keywords": keywords,
                "canvas_category": canvas_category,
                "tags": ",".join(tags) if tags else None,
                "owner_ids": ",".join(owner_ids) if owner_ids else None,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
            },
        )
        data = data or {}
        agents = data.get("canvas") or []
        return {"total": data.get("total", len(agents)), "agents": [_agent(a) for a in agents]}

    @tool(READ)
    async def ragflow_get_agent(
        agent_id: AgentId,
        include_dsl: Annotated[
            bool, Field(description="Also return the agent graph (DSL), which can be large")
        ] = False,
    ) -> dict[str, Any]:
        """Get one agent's settings, and its graph (DSL) on request."""
        data = await client.call("GET", f"/agents/{agent_id}")
        return _agent(data or {}, include_dsl)

    @tool(READ)
    async def ragflow_list_agent_templates(
        template_id: Annotated[
            str | None, Field(description="Return this template including its DSL")
        ] = None,
    ) -> dict[str, Any]:
        """List the built-in agent templates; pass template_id to get one template's DSL for ragflow_create_agent."""
        templates = await client.call("GET", "/agents/templates") or []
        if template_id is not None:
            match = next((t for t in templates if str(t.get("id")) == str(template_id)), None)
            if match is None:
                raise ValueError(f"no agent template with id {template_id}")
            return {
                "id": match.get("id"),
                "title": _title(match.get("title")),
                "description": _title(match.get("description")),
                "canvas_category": match.get("canvas_category"),
                "dsl": match.get("dsl"),
            }
        return {
            "templates": [
                {
                    "id": t.get("id"),
                    "title": _title(t.get("title")),
                    "description": _title(t.get("description")),
                    "canvas_category": t.get("canvas_category"),
                }
                for t in templates
                if _title(t.get("title"))
            ]
        }

    @tool(WRITE)
    async def ragflow_create_agent(
        title: Annotated[
            str, Field(description="Agent title; RAGFlow appends (1), (2) ... if it is taken")
        ],
        dsl: Dsl,
        description: str | None = None,
        permission: Annotated[
            Literal["me", "team"] | None, Field(description="Who may use and edit the agent")
        ] = None,
        canvas_category: Annotated[
            Literal["agent_canvas", "dataflow_canvas"] | None,
            Field(description="Defaults to agent_canvas"),
        ] = None,
    ) -> dict[str, Any]:
        """Create an agent from a graph (DSL)."""
        body = drop_none(
            {
                "title": title,
                "dsl": dsl,
                "description": description,
                "permission": permission,
                "canvas_category": canvas_category,
            }
        )
        data = await client.call("POST", "/agents", json=body)
        return _agent(data or {})

    @tool(WRITE)
    async def ragflow_update_agent(
        agent_id: AgentId,
        title: str | None = None,
        description: str | None = None,
        dsl: Annotated[
            dict[str, Any] | None,
            Field(description="Replacement graph (DSL); saved as a new version"),
        ] = None,
        permission: Literal["me", "team"] | None = None,
        release: Annotated[
            bool,
            Field(
                description="Publish the agent. RAGFlow writes this flag on every update, "
                "so an update without release=true unpublishes a published agent"
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Change an agent's title, description, permission or graph, and publish it with release=true."""
        changes = drop_none(
            {"title": title, "description": description, "dsl": dsl, "permission": permission}
        )
        if not changes and not release:
            raise ValueError("pass at least one field to update")
        data = await client.call(
            "PUT", f"/agents/{agent_id}", json={**changes, "release": release}
        )
        return {
            "id": agent_id,
            "updated": sorted(changes),
            "release": release,
            "update_time": (data or {}).get("update_time"),
        }

    @tool(DELETE)
    async def ragflow_delete_agent(agent_id: AgentId) -> dict[str, Any]:
        """Permanently delete an agent and its versions."""
        await client.call("DELETE", f"/agents/{agent_id}")
        return {"deleted": agent_id}

    # Sessions

    @tool(WRITE)
    async def ragflow_create_agent_session(
        agent_id: AgentId,
        name: Annotated[str | None, Field(description="Session name")] = None,
    ) -> dict[str, Any]:
        """Start a named conversation with an agent; pass its id as session_id to ragflow_agent_chat."""
        data = await client.call(
            "POST", f"/agents/{agent_id}/sessions", json=drop_none({"name": name})
        )
        return _session(data or {}, include_messages=False)

    @tool(READ)
    async def ragflow_list_agent_sessions(
        agent_id: AgentId,
        session_id: Annotated[str | None, Field(description="Return only this session")] = None,
        keywords: Annotated[str | None, Field(description="Match session names")] = None,
        user_id: Annotated[
            str | None, Field(description="Only sessions of this end-user ID")
        ] = None,
        from_date: Annotated[
            str | None, Field(description="Updated on or after this date, YYYY-MM-DD")
        ] = None,
        to_date: Annotated[
            str | None, Field(description="Updated on or before this date, YYYY-MM-DD")
        ] = None,
        include_messages: Annotated[
            bool, Field(description="Include each session's messages")
        ] = True,
        page: Page = 1,
        page_size: PageSize = 30,
        orderby: Literal["create_time", "update_time"] = "update_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List an agent's conversation sessions with their messages."""
        body = await client.call(
            "GET",
            f"/agents/{agent_id}/sessions",
            params={
                "id": session_id,
                "keywords": keywords,
                "user_id": user_id,
                "from_date": from_date,
                "to_date": to_date,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
                "dsl": False,
                "include_history": include_messages,
            },
            full=True,
        )
        sessions = body.get("data") or []
        return {
            "total": body.get("total", len(sessions)),
            "sessions": [_session(s, include_messages) for s in sessions],
        }

    @tool(DELETE)
    async def ragflow_delete_agent_sessions(
        agent_id: AgentId,
        session_ids: Annotated[list[str], Field(description="IDs of the sessions to delete")],
    ) -> dict[str, Any]:
        """Permanently delete conversation sessions of an agent."""
        ids = non_empty("session_ids", session_ids)
        # RAGFlow reads the IDs from the query string; a JSON body is ignored.
        await client.call("DELETE", f"/agents/{agent_id}/sessions", params={"ids": ",".join(ids)})
        return {"deleted": ids}

    # Conversation

    @tool(RUN)
    async def ragflow_agent_chat(
        agent_id: AgentId,
        question: Annotated[
            str,
            Field(description="The user's message; may be empty when the agent only takes inputs"),
        ] = "",
        session_id: Annotated[
            str | None,
            Field(
                description="Continue this session; without it RAGFlow starts a new session "
                "and returns its id"
            ),
        ] = None,
        inputs: Annotated[
            dict[str, Any] | None,
            Field(
                description="Values for the Begin component's input fields, keyed by field "
                'name: a plain value or {"value": ...}'
            ),
        ] = None,
        files: Annotated[
            list[dict[str, Any]] | None,
            Field(description="File objects returned by ragflow_upload_agent_file"),
        ] = None,
        openai_compatible: Annotated[
            bool,
            Field(
                description="Use RAGFlow's OpenAI-compatible mode, which also reports token "
                "usage; the agent run is the same"
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Send a message to an agent and wait for its complete answer."""
        body: dict[str, Any] = {"agent_id": agent_id, "stream": False}
        if session_id:
            body["session_id"] = session_id
        if files:
            body["files"] = files
        if openai_compatible:
            if not question:
                raise ValueError("openai_compatible mode needs a question")
            body["openai-compatible"] = True
            body["messages"] = [{"role": "user", "content": question}]
            result = await client.call("POST", "/agents/chat/completions", json=body, full=True)
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            return drop_none(
                {
                    "answer": message.get("content"),
                    "session_id": result.get("id"),
                    "finish_reason": choice.get("finish_reason"),
                    "reference": message.get("reference") or None,
                    "usage": result.get("usage"),
                }
            )
        body["question"] = question
        if inputs:
            body["inputs"] = inputs
        data = await client.call("POST", "/agents/chat/completions", json=body) or {}
        answer = data.get("data") or {}
        return drop_none(
            {
                "answer": answer.get("content"),
                "session_id": data.get("session_id"),
                "message_id": data.get("message_id"),
                "event": data.get("event"),
                "reference": answer.get("reference") or None,
                "structured": answer.get("structured") or None,
                "usage": answer.get("usage"),
            }
        )

    # Files

    @tool(UPLOAD)
    async def ragflow_upload_agent_file(
        agent_id: AgentId,
        filename: Annotated[
            str | None,
            Field(description="File name with extension; defaults to the file_path name"),
        ] = None,
        text: Annotated[str | None, Field(description="UTF-8 text content")] = None,
        content_base64: Annotated[
            str | None, Field(description="File bytes as base64 (small files only)")
        ] = None,
        file_path: Annotated[
            str | None,
            Field(description="Path on the MCP server host, inside an allowed upload root"),
        ] = None,
    ) -> dict[str, Any]:
        """Upload a file for an agent run; pass the returned object in ragflow_agent_chat files."""
        blob, name = read_bytes_source(
            ctx, text=text, content_base64=content_base64, file_path=file_path, filename=filename
        )
        if not name:
            raise ValueError("filename is required for text and content_base64 uploads")
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = await client.call(
            "POST", f"/agents/{agent_id}/upload", files={"file": (name, blob, mime)}
        )
        return data or {}

    @tool(READ)
    async def ragflow_download_agent_attachment(
        attachment_id: Annotated[
            str, Field(description="Attachment ID, e.g. a file an agent produced or one uploaded")
        ],
        ext: Annotated[
            str | None,
            Field(description="File extension such as pdf or txt, used for the content type"),
        ] = None,
        max_chars: MaxChars = DEFAULT_MAX_CHARS,
        include_base64: Annotated[
            bool, Field(description="Return binary content as base64 (up to 5 MB)")
        ] = False,
    ) -> dict[str, Any]:
        """Download an agent attachment. Text comes back as text; binary content as base64 on request."""
        blob, content_type = await client.raw(
            "GET", f"/agents/attachments/{attachment_id}/download", params={"ext": ext}
        )
        return describe_bytes(blob, content_type, max_chars, include_base64)
