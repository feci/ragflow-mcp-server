import base64
import json

import httpx
from conftest import call, make_server, ok, payload

DSL = {"components": {"begin": {"obj": {"component_name": "Begin", "params": {}}}}}


def agent_row(agent_id="a1", **extra):
    return {
        "id": agent_id,
        "title": "Helper",
        "description": "d",
        "permission": "me",
        "canvas_category": "agent_canvas",
        "release": False,
        "dsl": DSL,
        "user_id": "u1",
        "update_time": 2,
        **extra,
    }


async def test_list_agents_query_and_trimming(fake):
    fake.on("GET", "/agents", ok({"canvas": [agent_row()], "total": 1}))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_list_agents",
            {"keywords": "help", "tags": ["a", "b"], "owner_ids": ["u1"], "canvas_category": "agent_canvas"},
        )
    )
    assert result["total"] == 1
    assert "dsl" not in result["agents"][0] and result["agents"][0]["title"] == "Helper"
    params = fake.requests[0].url.params
    assert params["keywords"] == "help" and params["tags"] == "a,b" and params["owner_ids"] == "u1"
    assert params["canvas_category"] == "agent_canvas" and params["desc"] == "true"


async def test_get_agent_dsl_only_on_request(fake):
    fake.on("GET", "/agents/a1", ok(agent_row(tenant_id="u1")))
    server = make_server(fake)
    plain = payload(await call(server, "ragflow_get_agent", {"agent_id": "a1"}))
    assert "dsl" not in plain
    full = payload(await call(server, "ragflow_get_agent", {"agent_id": "a1", "include_dsl": True}))
    assert full["dsl"] == DSL


async def test_templates_list_and_single(fake):
    fake.on(
        "GET",
        "/agents/templates",
        ok(
            [
                {"id": 9, "title": {"en": "Starter", "zh": "x"}, "description": {"en": "d"}, "canvas_category": "agent_canvas", "dsl": DSL},
                {"id": 2, "title": {}, "dsl": {}},
            ]
        ),
    )
    server = make_server(fake)
    listed = payload(await call(server, "ragflow_list_agent_templates", {}))
    assert listed == {"templates": [{"id": 9, "title": "Starter", "description": "d", "canvas_category": "agent_canvas"}]}
    one = payload(await call(server, "ragflow_list_agent_templates", {"template_id": "9"}))
    assert one["dsl"] == DSL
    missing = await call(server, "ragflow_list_agent_templates", {"template_id": "77"})
    assert missing.isError


async def test_create_agent_body(fake):
    fake.on("POST", "/agents", ok(agent_row(title="Helper(1)")))
    result = payload(
        await call(make_server(fake), "ragflow_create_agent", {"title": "Helper", "dsl": DSL, "permission": "team"})
    )
    assert result["title"] == "Helper(1)" and "dsl" not in result
    assert json.loads(fake.requests[0].content) == {"title": "Helper", "dsl": DSL, "permission": "team"}


async def test_update_agent_always_sends_release(fake):
    fake.on("PUT", "/agents/a1", ok({"update_time": 5}))
    server = make_server(fake)
    result = payload(await call(server, "ragflow_update_agent", {"agent_id": "a1", "description": "new"}))
    assert result == {"id": "a1", "updated": ["description"], "release": False, "update_time": 5}
    assert json.loads(fake.requests[0].content) == {"description": "new", "release": False}
    payload(await call(server, "ragflow_update_agent", {"agent_id": "a1", "release": True}))
    assert json.loads(fake.requests[1].content) == {"release": True}
    empty = await call(server, "ragflow_update_agent", {"agent_id": "a1"})
    assert empty.isError and "at least one" in empty.content[0].text


async def test_delete_agent(fake):
    fake.on("DELETE", "/agents/a1", ok(True))
    assert payload(await call(make_server(fake), "ragflow_delete_agent", {"agent_id": "a1"})) == {"deleted": "a1"}


async def test_create_session(fake):
    fake.on("POST", "/agents/a1/sessions", ok({"id": "s1", "name": "n", "dialog_id": "a1", "message": [], "dsl": DSL}))
    result = payload(await call(make_server(fake), "ragflow_create_agent_session", {"agent_id": "a1", "name": "n"}))
    assert result == {"id": "s1", "name": "n", "agent_id": "a1"}
    assert json.loads(fake.requests[0].content) == {"name": "n"}


async def test_list_sessions_hides_dsl_and_reads_top_level_total(fake):
    fake.on(
        "GET",
        "/agents/a1/sessions",
        {
            "code": 0,
            "total": 7,
            "data": [
                {
                    "id": "s1",
                    "agent_id": "a1",
                    "name": "n",
                    "round": 1,
                    "message": [{"id": "m", "role": "user", "content": "hi", "created_at": 1.0}],
                }
            ],
        },
    )
    server = make_server(fake)
    result = payload(await call(server, "ragflow_list_agent_sessions", {"agent_id": "a1", "session_id": "s1"}))
    assert result["total"] == 7
    assert result["sessions"][0]["messages"] == [{"id": "m", "role": "user", "content": "hi"}]
    params = fake.requests[0].url.params
    assert params["dsl"] == "false" and params["id"] == "s1" and params["include_history"] == "true"
    bare = payload(await call(server, "ragflow_list_agent_sessions", {"agent_id": "a1", "include_messages": False}))
    assert "messages" not in bare["sessions"][0]


async def test_delete_sessions_uses_query_ids(fake):
    fake.on("DELETE", "/agents/a1/sessions", ok({"Data": None}))
    server = make_server(fake)
    result = payload(await call(server, "ragflow_delete_agent_sessions", {"agent_id": "a1", "session_ids": ["s1", "s2"]}))
    assert result == {"deleted": ["s1", "s2"]}
    request = fake.requests[0]
    assert request.url.params["ids"] == "s1,s2" and request.content == b""
    refused = await call(server, "ragflow_delete_agent_sessions", {"agent_id": "a1", "session_ids": []})
    assert refused.isError and len(fake.requests) == 1


async def test_agent_chat_plain_mode(fake):
    fake.on(
        "POST",
        "/agents/chat/completions",
        ok(
            {
                "event": "message_end",
                "message_id": "m1",
                "session_id": "s9",
                "task_id": "s9",
                "data": {"content": "Echo: hi", "reference": None, "usage": {"total_tokens": 0}},
            }
        ),
    )
    result = payload(
        await call(
            make_server(fake),
            "ragflow_agent_chat",
            {"agent_id": "a1", "question": "hi", "session_id": "s9", "inputs": {"topic": {"value": "x"}}},
        )
    )
    assert result == {
        "answer": "Echo: hi",
        "session_id": "s9",
        "message_id": "m1",
        "event": "message_end",
        "usage": {"total_tokens": 0},
    }
    body = json.loads(fake.requests[0].content)
    assert body == {"agent_id": "a1", "stream": False, "session_id": "s9", "question": "hi", "inputs": {"topic": {"value": "x"}}}


async def test_agent_chat_openai_mode(fake):
    fake.on(
        "POST",
        "/agents/chat/completions",
        {
            "id": "s7",
            "object": "chat.completion",
            "usage": {"total_tokens": 4},
            "choices": [{"message": {"role": "assistant", "content": "Echo: hi"}, "finish_reason": "stop"}],
        },
    )
    server = make_server(fake)
    result = payload(await call(server, "ragflow_agent_chat", {"agent_id": "a1", "question": "hi", "openai_compatible": True}))
    assert result == {"answer": "Echo: hi", "session_id": "s7", "finish_reason": "stop", "usage": {"total_tokens": 4}}
    body = json.loads(fake.requests[0].content)
    assert body == {"agent_id": "a1", "stream": False, "openai-compatible": True, "messages": [{"role": "user", "content": "hi"}]}
    empty = await call(server, "ragflow_agent_chat", {"agent_id": "a1", "openai_compatible": True})
    assert empty.isError and len(fake.requests) == 1


async def test_agent_chat_openai_error_is_tool_error(fake):
    fake.on(
        "POST",
        "/agents/chat/completions",
        httpx.Response(403, json={"error": {"message": "no permission", "type": "permission_error"}}),
    )
    result = await call(make_server(fake), "ragflow_agent_chat", {"agent_id": "a1", "question": "hi", "openai_compatible": True})
    assert result.isError and "no permission" in result.content[0].text


async def test_upload_agent_file(fake):
    uploaded = {"id": "f1", "name": "a.txt", "mime_type": "text/plain", "size": 2}
    fake.on("POST", "/agents/a1/upload", ok(uploaded))
    server = make_server(fake)
    result = payload(await call(server, "ragflow_upload_agent_file", {"agent_id": "a1", "filename": "a.txt", "text": "hi"}))
    assert result == uploaded
    assert b'filename="a.txt"' in fake.requests[0].content
    nameless = await call(server, "ragflow_upload_agent_file", {"agent_id": "a1", "content_base64": base64.b64encode(b"x").decode()})
    assert nameless.isError and "filename" in nameless.content[0].text


async def test_download_attachment(fake):
    fake.on(
        "GET",
        "/agents/attachments/f1/download",
        httpx.Response(200, content=b"\xff\xfe", headers={"content-type": "application/octet-stream"}),
    )
    result = payload(
        await call(make_server(fake), "ragflow_download_agent_attachment", {"attachment_id": "f1", "ext": "bin", "include_base64": True})
    )
    assert result["content_base64"] == base64.b64encode(b"\xff\xfe").decode() and result["size"] == 2
    assert fake.requests[0].url.params["ext"] == "bin"


async def test_download_attachment_error(fake):
    fake.on("GET", "/agents/attachments/zz/download", {"code": 102, "message": "Attachment not found!"})
    result = await call(make_server(fake), "ragflow_download_agent_attachment", {"attachment_id": "zz"})
    assert result.isError and "Attachment not found" in result.content[0].text
