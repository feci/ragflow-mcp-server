"""Chat assistants, their sessions, conversations, related questions, mind maps, text-to-speech and speech-to-text."""

from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath

from typing import Annotated, Any, Literal

from pydantic import Field

from ..common import (
    ID_PATTERN,
    DELETE,
    READ,
    UPLOAD,
    WRITE,
    ChatId,
    Context,
    Page,
    PageSize,
    SessionId,
    describe_bytes,
    drop_none,
    non_empty,
    pick,
    read_bytes_source,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".opus", ".wma"}

MessageId = Annotated[str, Field(pattern=ID_PATTERN, description="Message ID within the session")]
Question = Annotated[str, Field(description="The user's question")]
DatasetIds = Annotated[
    list[str] | None, Field(description="Datasets the assistant retrieves from")
]
LlmSetting = Annotated[
    dict[str, Any] | None,
    Field(
        description="Model parameters such as temperature, top_p, presence_penalty, "
        "frequency_penalty, max_tokens"
    ),
]
PromptConfig = Annotated[
    dict[str, Any] | None,
    Field(
        description="Prompt settings: system (prompt text, use {knowledge} for retrieved "
        "chunks), prologue, empty_response, quote, refine_multiturn, reasoning, "
        "cross_languages, parameters"
    ),
]


def _chat(c: dict[str, Any]) -> dict[str, Any]:
    out = pick(
        c,
        "id",
        "name",
        "description",
        "dataset_ids",
        "kb_names",
        "llm_id",
        "similarity_threshold",
        "keywords_similarity_weight",
        "top_n",
        "top_k",
        "rerank_id",
        "create_time",
        "update_time",
    )
    if "dataset_ids" not in out and "kb_ids" in c:
        out["dataset_ids"] = c["kb_ids"]
    return out


def _message(m: dict[str, Any]) -> dict[str, Any]:
    return pick(m, "id", "role", "content", "thumbup", "feedback", "created_at")


def _reference(ref: Any) -> Any:
    """Keep the cited chunk text and its source; drop vectors, positions and scores breakdown."""
    if not isinstance(ref, dict):
        return ref
    out: dict[str, Any] = {}
    chunks = ref.get("chunks")
    if isinstance(chunks, list):
        out["chunks"] = [
            {
                **pick(ch, "id", "content", "document_id", "dataset_id", "similarity"),
                "document_name": ch.get("document_name")
                or ch.get("doc_name")
                or ch.get("document_keyword")
                or ch.get("docnm_kwd"),
            }
            for ch in chunks
            if isinstance(ch, dict)
        ]
    if isinstance(ref.get("doc_aggs"), list):
        out["documents"] = [
            pick(a, "doc_id", "doc_name", "count") for a in ref["doc_aggs"] if isinstance(a, dict)
        ]
    if "total" in ref:
        out["total"] = ref["total"]
    return out


def _session(s: dict[str, Any], *, messages: bool = True, references: bool = False) -> dict[str, Any]:
    out = pick(s, "id", "chat_id", "name", "user_id", "create_time", "update_time")
    if "chat_id" not in out and "dialog_id" in s:
        out["chat_id"] = s["dialog_id"]
    msgs = s.get("messages") or s.get("message") or []
    if messages and isinstance(msgs, list):
        out["messages"] = [_message(m) for m in msgs if isinstance(m, dict)]
    if references and isinstance(s.get("reference"), list):
        out["reference"] = [_reference(r) for r in s["reference"]]
    return out


def _chat_body(**fields: Any) -> dict[str, Any]:
    return drop_none(fields)


def register(ctx: Context) -> None:
    client = ctx.client
    tool = ctx.tool

    # Chat assistants

    @tool(READ)
    async def ragflow_list_chats(
        keywords: Annotated[str | None, Field(description="Match chat assistant names")] = None,
        name: Annotated[str | None, Field(description="Exact chat assistant name")] = None,
        chat_id: Annotated[str | None, Field(description="Return only this chat assistant")] = None,
        page: Page = 1,
        page_size: PageSize = 30,
        orderby: Literal["create_time", "update_time", "name"] = "create_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List chat assistants with their datasets and retrieval settings."""
        data = await client.call(
            "GET",
            "/chats",
            params={
                "keywords": keywords,
                "name": name,
                "id": chat_id,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
            },
        )
        data = data or {}
        chats = data.get("chats") or []
        return {"total": data.get("total", len(chats)), "chats": [_chat(c) for c in chats]}

    @tool(READ)
    async def ragflow_get_chat(chat_id: ChatId) -> dict[str, Any]:
        """Get one chat assistant with its full model and prompt configuration."""
        data = await client.call("GET", f"/chats/{chat_id}")
        return {
            **_chat(data),
            **pick(
                data,
                "language",
                "llm_setting",
                "prompt_type",
                "prompt_config",
                "meta_data_filter",
                "rerank_candidates_count",
                "do_refer",
                "status",
            ),
        }

    @tool(WRITE)
    async def ragflow_create_chat(
        name: Annotated[str, Field(description="Chat assistant name; a duplicate gets a suffix")],
        dataset_ids: DatasetIds = None,
        description: str | None = None,
        llm_id: Annotated[
            str | None,
            Field(description="Chat model as model_name@factory; defaults to the tenant's model"),
        ] = None,
        llm_setting: LlmSetting = None,
        prompt_config: PromptConfig = None,
        similarity_threshold: Annotated[float | None, Field(ge=0, le=1)] = None,
        keywords_similarity_weight: Annotated[
            float | None,
            Field(ge=0, le=1, description="Weight of keyword match; vector weight is 1 minus this"),
        ] = None,
        top_n: Annotated[int | None, Field(ge=1, description="Chunks passed to the model")] = None,
        top_k: Annotated[int | None, Field(ge=1, description="Candidates before reranking")] = None,
        rerank_id: Annotated[
            str | None, Field(description="Rerank model as model_name@factory")
        ] = None,
        meta_data_filter: Annotated[
            dict[str, Any] | None, Field(description="Metadata filter applied to retrieval")
        ] = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Create a chat assistant that answers from the given datasets."""
        body = _chat_body(
            name=name,
            dataset_ids=dataset_ids,
            description=description,
            llm_id=llm_id,
            llm_setting=llm_setting,
            prompt_config=prompt_config,
            similarity_threshold=similarity_threshold,
            keywords_similarity_weight=keywords_similarity_weight,
            top_n=top_n,
            top_k=top_k,
            rerank_id=rerank_id,
            meta_data_filter=meta_data_filter,
            language=language,
        )
        return _chat(await client.call("POST", "/chats", json=body))

    async def update_chat(method: str, chat_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if not body:
            raise ValueError("pass at least one field to update")
        return _chat(await client.call(method, f"/chats/{chat_id}", json=body))

    @tool(WRITE)
    async def ragflow_update_chat(
        chat_id: ChatId,
        name: str | None = None,
        dataset_ids: DatasetIds = None,
        description: str | None = None,
        llm_id: str | None = None,
        llm_setting: Annotated[
            dict[str, Any] | None,
            Field(description="Keys to change in the model settings; other keys stay"),
        ] = None,
        prompt_config: Annotated[
            dict[str, Any] | None,
            Field(description="Keys to change in the prompt settings; other keys stay"),
        ] = None,
        similarity_threshold: Annotated[float | None, Field(ge=0, le=1)] = None,
        keywords_similarity_weight: Annotated[float | None, Field(ge=0, le=1)] = None,
        top_n: Annotated[int | None, Field(ge=1)] = None,
        top_k: Annotated[int | None, Field(ge=1)] = None,
        rerank_id: str | None = None,
        meta_data_filter: dict[str, Any] | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Change some settings of a chat assistant; llm_setting and prompt_config are merged into the existing ones."""
        body = _chat_body(
            name=name,
            dataset_ids=dataset_ids,
            description=description,
            llm_id=llm_id,
            llm_setting=llm_setting,
            prompt_config=prompt_config,
            similarity_threshold=similarity_threshold,
            keywords_similarity_weight=keywords_similarity_weight,
            top_n=top_n,
            top_k=top_k,
            rerank_id=rerank_id,
            meta_data_filter=meta_data_filter,
            language=language,
        )
        return await update_chat("PATCH", chat_id, body)

    @tool(WRITE)
    async def ragflow_replace_chat_settings(
        chat_id: ChatId,
        name: str | None = None,
        dataset_ids: DatasetIds = None,
        description: str | None = None,
        llm_id: str | None = None,
        llm_setting: Annotated[
            dict[str, Any] | None,
            Field(description="Replaces the whole model settings object"),
        ] = None,
        prompt_config: Annotated[
            dict[str, Any] | None,
            Field(description="Replaces the whole prompt settings object"),
        ] = None,
        similarity_threshold: Annotated[float | None, Field(ge=0, le=1)] = None,
        keywords_similarity_weight: Annotated[float | None, Field(ge=0, le=1)] = None,
        top_n: Annotated[int | None, Field(ge=1)] = None,
        top_k: Annotated[int | None, Field(ge=1)] = None,
        rerank_id: str | None = None,
        meta_data_filter: dict[str, Any] | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Update a chat assistant with PUT semantics: llm_setting and prompt_config you pass replace the stored objects."""
        body = _chat_body(
            name=name,
            dataset_ids=dataset_ids,
            description=description,
            llm_id=llm_id,
            llm_setting=llm_setting,
            prompt_config=prompt_config,
            similarity_threshold=similarity_threshold,
            keywords_similarity_weight=keywords_similarity_weight,
            top_n=top_n,
            top_k=top_k,
            rerank_id=rerank_id,
            meta_data_filter=meta_data_filter,
            language=language,
        )
        return await update_chat("PUT", chat_id, body)

    @tool(DELETE)
    async def ragflow_delete_chat(chat_id: ChatId) -> dict[str, Any]:
        """Permanently delete one chat assistant and its sessions."""
        await client.call("DELETE", f"/chats/{chat_id}")
        return {"deleted": [chat_id]}

    @tool(DELETE)
    async def ragflow_delete_chats(
        chat_ids: Annotated[list[str], Field(description="IDs of the chat assistants to delete")],
    ) -> dict[str, Any]:
        """Permanently delete several chat assistants."""
        ids = non_empty("chat_ids", chat_ids)
        data = await client.call("DELETE", "/chats", json={"ids": ids})
        result: dict[str, Any] = {"deleted": ids}
        if isinstance(data, dict) and data.get("errors"):
            result = {**pick(data, "success_count", "errors"), "requested": ids}
        return result

    # Sessions

    @tool(READ)
    async def ragflow_list_chat_sessions(
        chat_id: ChatId,
        name: Annotated[str | None, Field(description="Exact session name")] = None,
        session_id: Annotated[str | None, Field(description="Return only this session")] = None,
        page: Page = 1,
        page_size: PageSize = 30,
        orderby: Literal["create_time", "update_time", "name"] = "create_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List the sessions of a chat assistant (without their messages)."""
        data = await client.call(
            "GET",
            f"/chats/{chat_id}/sessions",
            params={
                "name": name,
                "id": session_id,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
                "include_history": False,
            },
        )
        sessions = data or []
        return {"sessions": [_session(s, messages=False) for s in sessions]}

    @tool(READ)
    async def ragflow_get_chat_session(
        chat_id: ChatId,
        session_id: SessionId,
        include_references: Annotated[
            bool, Field(description="Also return the chunks each answer cited")
        ] = False,
    ) -> dict[str, Any]:
        """Get a chat session with its messages."""
        data = await client.call("GET", f"/chats/{chat_id}/sessions/{session_id}")
        return _session(data, references=include_references)

    @tool(WRITE)
    async def ragflow_create_chat_session(
        chat_id: ChatId,
        name: Annotated[str | None, Field(description="Session name; default 'New session'")] = None,
    ) -> dict[str, Any]:
        """Start a new session with a chat assistant; it opens with the assistant's prologue."""
        data = await client.call(
            "POST", f"/chats/{chat_id}/sessions", json=drop_none({"name": name})
        )
        return _session(data)

    @tool(WRITE)
    async def ragflow_rename_chat_session(
        chat_id: ChatId,
        session_id: SessionId,
        name: Annotated[str, Field(description="New session name")],
    ) -> dict[str, Any]:
        """Rename a chat session."""
        data = await client.call(
            "PATCH", f"/chats/{chat_id}/sessions/{session_id}", json={"name": name}
        )
        return _session(data, messages=False)

    @tool(DELETE)
    async def ragflow_delete_chat_sessions(
        chat_id: ChatId,
        session_ids: Annotated[list[str], Field(description="IDs of the sessions to delete")],
    ) -> dict[str, Any]:
        """Permanently delete sessions of a chat assistant."""
        ids = non_empty("session_ids", session_ids)
        data = await client.call("DELETE", f"/chats/{chat_id}/sessions", json={"ids": ids})
        result: dict[str, Any] = {"deleted": ids}
        if isinstance(data, dict) and data.get("errors"):
            result = {**pick(data, "success_count", "errors"), "requested": ids}
        return result

    @tool(DELETE)
    async def ragflow_delete_chat_message(
        chat_id: ChatId, session_id: SessionId, message_id: MessageId
    ) -> dict[str, Any]:
        """Delete a question and its answer (they share one message ID) from a chat session; returns the remaining messages."""
        data = await client.call(
            "DELETE", f"/chats/{chat_id}/sessions/{session_id}/messages/{message_id}"
        )
        return _session(data)

    @tool(WRITE)
    async def ragflow_rate_chat_message(
        chat_id: ChatId,
        session_id: SessionId,
        message_id: Annotated[str, Field(pattern=ID_PATTERN, description="ID of an assistant message")],
        thumbup: Annotated[bool, Field(description="true for a good answer, false for a bad one")],
        feedback: Annotated[
            str | None, Field(description="Why the answer was bad (kept only with thumbup false)")
        ] = None,
    ) -> dict[str, Any]:
        """Give thumbs up or down, with optional feedback, to an assistant answer."""
        body = drop_none({"thumbup": thumbup, "feedback": feedback})
        data = await client.call(
            "PUT",
            f"/chats/{chat_id}/sessions/{session_id}/messages/{message_id}/feedback",
            json=body,
        )
        # A question and its answer share one message ID; the rating sits on the answer.
        rated = next(
            (
                m
                for m in (data or {}).get("messages") or []
                if m.get("id") == message_id and m.get("role") == "assistant"
            ),
            None,
        )
        return {"message": _message(rated) if rated else {"id": message_id, "thumbup": thumbup}}

    # Conversations

    @tool(WRITE)
    async def ragflow_chat(
        question: Annotated[
            str | None,
            Field(description="The user's question (or pass messages instead)"),
        ] = None,
        chat_id: Annotated[
            str | None,
            Field(
                description="Chat assistant to ask; without it the tenant's default chat "
                "model answers with no retrieval"
            ),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(description="Continue this session; without it a new session is created"),
        ] = None,
        messages: Annotated[
            list[dict[str, Any]] | None,
            Field(description="Conversation as [{role, content}], last one from the user"),
        ] = None,
        store_history: Annotated[
            bool,
            Field(description="false answers from messages without storing a session"),
        ] = True,
        llm_id: Annotated[str | None, Field(description="Override the chat model")] = None,
        temperature: Annotated[float | None, Field(ge=0, le=2)] = None,
        max_tokens: Annotated[int | None, Field(ge=1)] = None,
        thinking: Annotated[
            Literal["enabled", "disabled"] | None,
            Field(description="Turn model reasoning on or off where supported"),
        ] = None,
    ) -> dict[str, Any]:
        """Ask a chat assistant a question and get its answer with the cited chunks."""
        if question is None and not messages:
            raise ValueError("pass question or messages")
        if not store_history and not messages:
            raise ValueError("store_history=false needs messages")
        if session_id and not chat_id:
            raise ValueError("session_id needs chat_id")
        body = drop_none(
            {
                "question": question,
                "chat_id": chat_id,
                "session_id": session_id,
                "messages": messages,
                "llm_id": llm_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking": thinking,
            }
        )
        body["stream"] = False
        if not store_history:
            body["store_history_messages"] = False
        data = await client.call("POST", "/chat/completions", json=body) or {}
        out = {k: v for k, v in pick(data, "answer", "session_id", "chat_id").items() if v != ""}
        if data.get("id"):
            out["message_id"] = data["id"]
        if data.get("reference"):
            out["reference"] = _reference(data["reference"])
        return out

    @tool(WRITE)
    async def ragflow_openai_chat_completion(
        chat_id: ChatId,
        messages: Annotated[
            list[dict[str, Any]] | None,
            Field(description="OpenAI-style messages [{role, content}], last one from the user"),
        ] = None,
        question: Annotated[
            str | None, Field(description="Shortcut for a single user message")
        ] = None,
        include_reference: Annotated[
            bool, Field(description="Return the chunks the answer used")
        ] = False,
        metadata_condition: Annotated[
            dict[str, Any] | None, Field(description="Metadata filter for retrieval")
        ] = None,
        temperature: Annotated[float | None, Field(ge=0, le=2)] = None,
        max_tokens: Annotated[int | None, Field(ge=1)] = None,
    ) -> dict[str, Any]:
        """Ask a chat assistant through its OpenAI-compatible endpoint; no session is stored."""
        if question is None and not messages:
            raise ValueError("pass messages or question")
        body = drop_none(
            {
                "model": "model",
                "messages": messages or [],
                "question": question,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        body["stream"] = False
        extra = drop_none({"metadata_condition": metadata_condition})
        if include_reference:
            extra["reference"] = True
        if extra:
            body["extra_body"] = extra
        data = await client.call(
            "POST", f"/openai/{chat_id}/chat/completions", json=body, full=True
        )
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        out: dict[str, Any] = {
            "content": message.get("content"),
            **pick(data, "id", "model"),
            "usage": pick(data.get("usage") or {}, "prompt_tokens", "completion_tokens", "total_tokens"),
        }
        if include_reference and isinstance(message.get("reference"), list):
            out["reference"] = [
                pick(r, "id", "content", "document_id", "document_name", "dataset_id", "similarity")
                for r in message["reference"]
                if isinstance(r, dict)
            ]
        return out

    @tool(WRITE)
    async def ragflow_related_questions(
        question: Question,
        search_id: Annotated[
            str | None, Field(description="Search app whose model settings to use")
        ] = None,
    ) -> dict[str, Any]:
        """Generate related questions that help search for a topic."""
        data = await client.call(
            "POST",
            "/chat/recommendation",
            json=drop_none({"question": question, "search_id": search_id}),
        )
        return {"questions": data or []}

    @tool(WRITE)
    async def ragflow_mind_map(
        question: Question,
        dataset_ids: Annotated[list[str], Field(description="Datasets to build the map from")],
        search_id: Annotated[
            str | None, Field(description="Search app whose datasets and settings to add")
        ] = None,
    ) -> dict[str, Any]:
        """Build a mind map (a tree of name/children nodes) of a topic from datasets; runs several LLM calls and can take minutes."""
        body = drop_none(
            {
                "question": question,
                "kb_ids": non_empty("dataset_ids", dataset_ids),
                "search_id": search_id,
            }
        )
        return await client.call("POST", "/chat/mindmap", json=body) or {}

    @tool(WRITE)
    async def ragflow_text_to_speech(
        text: Annotated[str, Field(description="Text to read aloud")],
    ) -> dict[str, Any]:
        """Turn text into speech with the tenant's TTS model; returns the audio as base64."""
        blob, content_type = await client.raw("POST", "/chat/audio/speech", json={"text": text})
        return describe_bytes(blob, content_type, include_base64=True)

    @tool(UPLOAD)
    async def ragflow_speech_to_text(
        filename: Annotated[
            str | None,
            Field(
                description="Audio file name with its extension (.wav, .mp3, .m4a, .aac, "
                ".flac, .ogg, .webm, .opus, .wma); defaults to the name from file_path"
            ),
        ] = None,
        content_base64: Annotated[
            str | None, Field(description="Audio bytes as base64 (small files only)")
        ] = None,
        file_path: Annotated[
            str | None,
            Field(description="Audio file on the MCP server host, inside an allowed upload root"),
        ] = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file with the tenant's speech-to-text model."""
        blob, name = read_bytes_source(
            ctx, content_base64=content_base64, file_path=file_path, filename=filename
        )
        if not name:
            raise ValueError("filename is required with content_base64")
        if PurePosixPath(name).suffix.lower() not in AUDIO_SUFFIXES:
            raise ValueError(
                "unsupported audio format; use one of " + ", ".join(sorted(AUDIO_SUFFIXES))
            )
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = await client.call(
            "POST", "/chat/audio/transcription", files={"file": (name, blob, mime)}
        )
        return pick(data or {}, "text")
