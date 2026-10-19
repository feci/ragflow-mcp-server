import base64
import json

import httpx
import pytest
from conftest import call, make_server, ok, payload


def chat(chat_id="c1", **extra):
    return {
        "id": chat_id,
        "name": "Helper",
        "description": "d",
        "dataset_ids": ["k1"],
        "kb_names": ["KB"],
        "llm_id": "m@f",
        "llm_setting": {"temperature": 0.1},
        "prompt_config": {"system": "x" * 50},
        "icon": "base64-avatar",
        "top_n": 6,
        **extra,
    }


def session(session_id="s1", messages=None):
    return {
        "id": session_id,
        "chat_id": "c1",
        "name": "New session",
        "messages": messages
        if messages is not None
        else [
            {"role": "assistant", "content": "Hi!"},
            {"id": "m1", "role": "user", "content": "Q?"},
            {"id": "m2", "role": "assistant", "content": "A.", "prompt": "huge"},
        ],
        "reference": [
            {
                "chunks": [
                    {
                        "id": "ch1",
                        "content": "text",
                        "document_id": "d1",
                        "document_name": "a.md",
                        "dataset_id": "k1",
                        "vector": [0.1] * 5,
                    }
                ],
                "doc_aggs": [{"doc_id": "d1", "doc_name": "a.md", "count": 1}],
            }
        ],
    }


def body(request):
    return json.loads(request.content)


async def test_list_chats_filters_and_trims(fake):
    fake.on("GET", "/chats", ok({"total": 1, "chats": [chat()]}))
    result = payload(await call(make_server(fake), "ragflow_list_chats", {"name": "Helper"}))
    assert result["total"] == 1
    assert result["chats"][0]["dataset_ids"] == ["k1"]
    assert "prompt_config" not in result["chats"][0] and "icon" not in result["chats"][0]
    params = fake.sent("GET", "/chats")[0].url.params
    assert params["name"] == "Helper" and params["orderby"] == "create_time"


async def test_get_chat_includes_configuration(fake):
    fake.on("GET", "/chats/c1", ok(chat()))
    result = payload(await call(make_server(fake), "ragflow_get_chat", {"chat_id": "c1"}))
    assert result["prompt_config"]["system"].startswith("x")
    assert result["llm_setting"] == {"temperature": 0.1}
    assert "icon" not in result


async def test_create_chat_sends_only_given_fields(fake):
    fake.on("POST", "/chats", ok(chat()))
    payload(
        await call(
            make_server(fake),
            "ragflow_create_chat",
            {"name": "Helper", "dataset_ids": ["k1"], "keywords_similarity_weight": 0.3},
        )
    )
    assert body(fake.sent("POST", "/chats")[0]) == {
        "name": "Helper",
        "dataset_ids": ["k1"],
        "keywords_similarity_weight": 0.3,
    }


@pytest.mark.parametrize(
    "tool, method", [("ragflow_update_chat", "PATCH"), ("ragflow_replace_chat_settings", "PUT")]
)
async def test_update_chat_methods(fake, tool, method):
    fake.on(method, "/chats/c1", ok(chat(name="New")))
    result = payload(
        await call(make_server(fake), tool, {"chat_id": "c1", "prompt_config": {"prologue": "Hey"}})
    )
    assert result["name"] == "New"
    assert body(fake.sent(method, "/chats/c1")[0]) == {"prompt_config": {"prologue": "Hey"}}
    empty = await call(make_server(fake), tool, {"chat_id": "c1"})
    assert empty.isError and "at least one field" in empty.content[0].text


async def test_delete_chats(fake):
    fake.on("DELETE", "/chats/c1", ok(True))
    fake.on("DELETE", "/chats", ok({"success_count": 2}))
    server = make_server(fake)
    assert payload(await call(server, "ragflow_delete_chat", {"chat_id": "c1"})) == {
        "deleted": ["c1"]
    }
    assert payload(await call(server, "ragflow_delete_chats", {"chat_ids": ["a", "b"]})) == {
        "deleted": ["a", "b"]
    }
    assert body(fake.sent("DELETE", "/chats")[0]) == {"ids": ["a", "b"]}
    refused = await call(server, "ragflow_delete_chats", {"chat_ids": []})
    assert refused.isError


async def test_partial_delete_reports_errors(fake):
    fake.on("DELETE", "/chats", ok({"success_count": 1, "errors": ["b: not found"]}))
    result = payload(await call(make_server(fake), "ragflow_delete_chats", {"chat_ids": ["a", "b"]}))
    assert result == {"success_count": 1, "errors": ["b: not found"], "requested": ["a", "b"]}


async def test_list_sessions_without_history(fake):
    fake.on("GET", "/chats/c1/sessions", ok([session()]))
    result = payload(await call(make_server(fake), "ragflow_list_chat_sessions", {"chat_id": "c1"}))
    assert result["sessions"] == [{"id": "s1", "chat_id": "c1", "name": "New session"}]
    assert fake.sent("GET", "/chats/c1/sessions")[0].url.params["include_history"] == "false"


async def test_get_session_messages_and_references(fake):
    fake.on("GET", "/chats/c1/sessions/s1", ok(session()))
    server = make_server(fake)
    plain = payload(await call(server, "ragflow_get_chat_session", {"chat_id": "c1", "session_id": "s1"}))
    assert plain["messages"][2] == {"id": "m2", "role": "assistant", "content": "A."}
    assert "reference" not in plain
    full = payload(
        await call(
            server,
            "ragflow_get_chat_session",
            {"chat_id": "c1", "session_id": "s1", "include_references": True},
        )
    )
    assert full["reference"][0]["chunks"] == [
        {"id": "ch1", "content": "text", "document_id": "d1", "dataset_id": "k1", "document_name": "a.md"}
    ]
    assert full["reference"][0]["documents"] == [{"doc_id": "d1", "doc_name": "a.md", "count": 1}]


async def test_create_and_rename_session(fake):
    fake.on("POST", "/chats/c1/sessions", ok(session(messages=[{"role": "assistant", "content": "Hi!"}])))
    fake.on("PATCH", "/chats/c1/sessions/s1", ok(session()))
    server = make_server(fake)
    created = payload(await call(server, "ragflow_create_chat_session", {"chat_id": "c1"}))
    assert created["messages"] == [{"role": "assistant", "content": "Hi!"}]
    assert body(fake.sent("POST", "/chats/c1/sessions")[0]) == {}
    payload(
        await call(server, "ragflow_rename_chat_session", {"chat_id": "c1", "session_id": "s1", "name": "N"})
    )
    assert body(fake.sent("PATCH", "/chats/c1/sessions/s1")[0]) == {"name": "N"}


async def test_delete_sessions_and_message(fake):
    fake.on("DELETE", "/chats/c1/sessions", ok({}))
    fake.on("DELETE", "/chats/c1/sessions/s1/messages/m1", ok(session(messages=[])))
    server = make_server(fake)
    assert payload(
        await call(server, "ragflow_delete_chat_sessions", {"chat_id": "c1", "session_ids": ["s1"]})
    ) == {"deleted": ["s1"]}
    assert body(fake.sent("DELETE", "/chats/c1/sessions")[0]) == {"ids": ["s1"]}
    left = payload(
        await call(
            server,
            "ragflow_delete_chat_message",
            {"chat_id": "c1", "session_id": "s1", "message_id": "m1"},
        )
    )
    assert left["messages"] == []


async def test_rate_message(fake):
    rated = session()
    # RAGFlow gives a question and its answer the same message ID.
    rated["messages"][1]["id"] = "m2"
    rated["messages"][2].update({"thumbup": False, "feedback": "wrong"})
    fake.on("PUT", "/chats/c1/sessions/s1/messages/m2/feedback", ok(rated))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_rate_chat_message",
            {"chat_id": "c1", "session_id": "s1", "message_id": "m2", "thumbup": False, "feedback": "wrong"},
        )
    )
    assert result["message"] == {
        "id": "m2",
        "role": "assistant",
        "content": "A.",
        "thumbup": False,
        "feedback": "wrong",
    }
    assert body(fake.sent("PUT", "/chats/c1/sessions/s1/messages/m2/feedback")[0]) == {
        "thumbup": False,
        "feedback": "wrong",
    }


async def test_chat_completion_non_streaming(fake):
    fake.on(
        "POST",
        "/chat/completions",
        ok(
            {
                "answer": "It is 42.",
                "id": "m9",
                "session_id": "s1",
                "prompt": "very long prompt",
                "audio_binary": None,
                "reference": {"chunks": [{"id": "ch1", "content": "42", "document_name": "a.md"}], "total": 1},
            }
        ),
    )
    result = payload(
        await call(make_server(fake), "ragflow_chat", {"question": "What?", "chat_id": "c1", "temperature": 0.2})
    )
    assert result == {
        "answer": "It is 42.",
        "session_id": "s1",
        "message_id": "m9",
        "reference": {"chunks": [{"id": "ch1", "content": "42", "document_name": "a.md"}], "total": 1},
    }
    assert body(fake.sent("POST", "/chat/completions")[0]) == {
        "question": "What?",
        "chat_id": "c1",
        "temperature": 0.2,
        "stream": False,
    }


@pytest.mark.parametrize(
    "args, message",
    [
        ({}, "pass question or messages"),
        ({"question": "q", "store_history": False}, "needs messages"),
        ({"question": "q", "session_id": "s1"}, "needs chat_id"),
    ],
)
async def test_chat_validation(fake, args, message):
    result = await call(make_server(fake), "ragflow_chat", args)
    assert result.isError and message in result.content[0].text
    assert not fake.requests


async def test_chat_without_history(fake):
    fake.on("POST", "/chat/completions", ok({"answer": "ok", "session_id": ""}))
    msgs = [{"role": "user", "content": "hi"}]
    result = payload(
        await call(make_server(fake), "ragflow_chat", {"messages": msgs, "store_history": False})
    )
    assert result == {"answer": "ok"}
    sent = body(fake.sent("POST", "/chat/completions")[0])
    assert sent == {"messages": msgs, "stream": False, "store_history_messages": False}


async def test_openai_completion(fake):
    fake.on(
        "POST",
        "/openai/c1/chat/completions",
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "m",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "completion_tokens_details": {}},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Yes.",
                        "reference": [{"id": "ch1", "content": "x", "document_name": "a.md", "positions": []}],
                    }
                }
            ],
        },
    )
    result = payload(
        await call(
            make_server(fake),
            "ragflow_openai_chat_completion",
            {"chat_id": "c1", "question": "Is it?", "include_reference": True},
        )
    )
    assert result == {
        "content": "Yes.",
        "id": "chatcmpl-1",
        "model": "m",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "reference": [{"id": "ch1", "content": "x", "document_name": "a.md"}],
    }
    assert body(fake.sent("POST", "/openai/c1/chat/completions")[0]) == {
        "model": "model",
        "messages": [],
        "question": "Is it?",
        "stream": False,
        "extra_body": {"reference": True},
    }


async def test_openai_completion_error(fake):
    fake.on("POST", "/openai/c1/chat/completions", {"code": 102, "message": "You don't own the chat c1"})
    result = await call(make_server(fake), "ragflow_openai_chat_completion", {"chat_id": "c1", "question": "q"})
    assert result.isError and "You don't own the chat" in result.content[0].text


async def test_related_questions_and_mind_map(fake):
    fake.on("POST", "/chat/recommendation", ok(["a?", "b?"]))
    fake.on("POST", "/chat/mindmap", ok({"name": "Topic", "children": []}))
    server = make_server(fake)
    assert payload(await call(server, "ragflow_related_questions", {"question": "q"})) == {
        "questions": ["a?", "b?"]
    }
    assert payload(
        await call(server, "ragflow_mind_map", {"question": "q", "dataset_ids": ["k1"]})
    ) == {"name": "Topic", "children": []}
    assert body(fake.sent("POST", "/chat/mindmap")[0]) == {"question": "q", "kb_ids": ["k1"]}


async def test_text_to_speech_returns_base64_audio(fake):
    fake.on(
        "POST",
        "/chat/audio/speech",
        httpx.Response(200, content=b"\xff\xfb\x90audio", headers={"content-type": "audio/mpeg"}),
    )
    result = payload(await call(make_server(fake), "ragflow_text_to_speech", {"text": "Hello"}))
    assert result["content_type"] == "audio/mpeg"
    assert base64.b64decode(result["content_base64"]) == b"\xff\xfb\x90audio"


async def test_text_to_speech_error_is_tool_error(fake):
    fake.on("POST", "/chat/audio/speech", {"code": 102, "message": "no default TTS model"})
    result = await call(make_server(fake), "ragflow_text_to_speech", {"text": "Hello"})
    assert result.isError and "no default TTS model" in result.content[0].text


async def test_speech_to_text(fake):
    fake.on("POST", "/chat/audio/transcription", ok({"text": "hello world"}))
    server = make_server(fake)
    result = payload(
        await call(
            server,
            "ragflow_speech_to_text",
            {"filename": "a.wav", "content_base64": base64.b64encode(b"RIFF").decode()},
        )
    )
    assert result == {"text": "hello world"}
    sent = fake.sent("POST", "/chat/audio/transcription")[0]
    assert b'filename="a.wav"' in sent.content and b"RIFF" in sent.content
    bad = await call(
        server,
        "ragflow_speech_to_text",
        {"filename": "a.txt", "content_base64": base64.b64encode(b"x").decode()},
    )
    assert bad.isError and "unsupported audio format" in bad.content[0].text


async def test_mind_map_timeout_is_explained(fake):
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    fake.on("POST", "/chat/mindmap", slow)
    result = await call(make_server(fake), "ragflow_mind_map", {"question": "q", "dataset_ids": ["k1"]})
    assert result.isError and "client timeout" in result.content[0].text
