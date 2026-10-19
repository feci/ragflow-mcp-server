import json

import pytest
from conftest import call, make_server, ok, payload


def memory(memory_id="m1", **extra):
    return {
        "id": memory_id,
        "name": "notes",
        "memory_type": ["raw", "semantic"],
        "embd_id": "emb@f",
        "llm_id": "llm@f",
        "permissions": "me",
        "tenant_id": "t1",
        "tenant_llm_id": "x",
        "system_prompt": "long prompt " * 50,
        **extra,
    }


def message(message_id=1, **extra):
    return {
        "memory_id": "m1",
        "message_id": message_id,
        "message_type": "raw",
        "status": True,
        "agent_id": "a1",
        "session_id": "s1",
        "user_id": "u1",
        "valid_at": "2026-10-19 11:43:05",
        "forget_at": "",
        "invalid_at": "",
        **extra,
    }


async def test_list_memories_joins_filters_and_trims(fake):
    fake.on("GET", "/memories", ok({"memory_list": [memory()], "total_count": 1}))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_list_memories",
            {"keywords": "no", "memory_types": ["raw", "episodic"], "page_size": 5},
        )
    )
    assert result["total"] == 1
    assert "system_prompt" not in result["memories"][0]
    assert "tenant_llm_id" not in result["memories"][0]
    params = fake.sent("GET", "/memories")[0].url.params
    assert params["memory_type"] == "raw,episodic"
    assert params["keywords"] == "no" and params["page_size"] == "5"
    assert "tenant_id" not in params


async def test_get_memory_config_keeps_prompts(fake):
    fake.on("GET", "/memories/m1/config", ok(memory(user_prompt=None)))
    result = payload(await call(make_server(fake), "ragflow_get_memory_config", {"memory_id": "m1"}))
    assert result["system_prompt"].startswith("long prompt")
    assert result["embd_id"] == "emb@f"


async def test_create_memory_uses_default_embedding_and_sets_description(fake):
    fake.on("POST", "/memories", ok(memory()))
    fake.on("PUT", "/memories/m1", ok(memory(description="d")))
    server = make_server(fake, default_embedding_model="emb@f")
    result = payload(
        await call(
            server,
            "ragflow_create_memory",
            {"name": "notes", "llm_id": "llm@f", "description": "d"},
        )
    )
    assert result["description"] == "d"
    assert json.loads(fake.sent("POST", "/memories")[0].content) == {
        "name": "notes",
        "memory_type": ["raw", "semantic"],
        "embd_id": "emb@f",
        "llm_id": "llm@f",
    }
    assert json.loads(fake.sent("PUT", "/memories/m1")[0].content) == {"description": "d"}


async def test_create_memory_needs_an_embedding_model(fake):
    result = await call(make_server(fake), "ragflow_create_memory", {"name": "n", "llm_id": "l@f"})
    assert result.isError and "embedding_model is required" in result.content[0].text
    assert not fake.requests


async def test_update_memory_maps_fields(fake):
    fake.on("PUT", "/memories/m1", ok(memory(temperature=0.3)))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_update_memory",
            {"memory_id": "m1", "temperature": 0.3, "embedding_model": "e@f", "memory_types": ["raw"]},
        )
    )
    assert result["temperature"] == 0.3
    assert json.loads(fake.sent("PUT", "/memories/m1")[0].content) == {
        "embd_id": "e@f",
        "memory_type": ["raw"],
        "temperature": 0.3,
    }
    empty = await call(make_server(fake), "ragflow_update_memory", {"memory_id": "m1"})
    assert empty.isError and "at least one field" in empty.content[0].text


async def test_delete_memory(fake):
    fake.on("DELETE", "/memories/m1", ok(None, message="success"))
    assert payload(await call(make_server(fake), "ragflow_delete_memory", {"memory_id": "m1"})) == {
        "deleted": "m1"
    }


async def test_list_memory_messages_maps_session_to_keywords(fake):
    raw = message(
        task={"progress": 1, "progress_msg": "Message saved successfully.", "digest": "1"},
        extract=[message(2, message_type="semantic", source_id=1)],
    )
    fake.on(
        "GET",
        "/memories/m1",
        ok({"messages": {"message_list": [raw], "total_count": 1}, "storage_type": "table"}),
    )
    result = payload(
        await call(
            make_server(fake),
            "ragflow_list_memory_messages",
            {"memory_id": "m1", "session_id": "s1", "agent_ids": ["a1", "a2"]},
        )
    )
    params = fake.sent("GET", "/memories/m1")[0].url.params
    assert params["keywords"] == "s1" and params["agent_id"] == "a1,a2"
    item = result["messages"][0]
    assert item["task"] == {"progress": 1, "progress_msg": "Message saved successfully."}
    assert item["extract"][0]["message_type"] == "semantic"
    assert "forget_at" not in item and "invalid_at" not in item


async def test_list_memory_messages_page_size_limit(fake):
    result = await call(
        make_server(fake), "ragflow_list_memory_messages", {"memory_id": "m1", "page_size": 101}
    )
    assert result.isError
    assert not fake.requests


async def test_add_memory_message_sends_list_and_user(fake):
    fake.on("POST", "/messages", ok(None, message="All add to task."))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_add_memory_message",
            {
                "memory_ids": ["m1"],
                "agent_id": "a1",
                "session_id": "s1",
                "user_id": "u1",
                "user_input": "hi",
                "agent_response": "hello",
            },
        )
    )
    assert result["queued"] is True and result["message"] == "All add to task."
    assert json.loads(fake.sent("POST", "/messages")[0].content) == {
        "memory_id": ["m1"],
        "agent_id": "a1",
        "session_id": "s1",
        "user_id": "u1",
        "user_input": "hi",
        "agent_response": "hello",
    }


async def test_search_memory_messages(fake):
    fake.on("GET", "/messages/search", ok([message(content="User Input: hi")]))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_search_memory_messages",
            {"query": "hi", "memory_ids": ["m1", "m2"], "top_n": 3},
        )
    )
    assert result["messages"][0]["content"] == "User Input: hi"
    params = fake.sent("GET", "/messages/search")[0].url.params
    assert params["memory_id"] == "m1,m2" and params["top_n"] == "3" and params["query"] == "hi"
    assert "similarity_threshold" not in params


async def test_recent_memory_messages(fake):
    fake.on("GET", "/messages", ok([message(), message(2)]))
    result = payload(
        await call(make_server(fake), "ragflow_recent_memory_messages", {"memory_ids": ["m1"], "limit": 2})
    )
    assert [m["message_id"] for m in result["messages"]] == [1, 2]
    params = fake.sent("GET", "/messages")[0].url.params
    assert params["memory_id"] == "m1" and params["limit"] == "2"


async def test_get_memory_message_drops_vectors(fake):
    content = message(2, content="Senec is a lake", q_4096_vec=[0.1] * 8, valid_at_flt=1.0, zone_id=0)
    fake.on("GET", "/messages/m1:2/content", ok(content))
    result = payload(
        await call(make_server(fake), "ragflow_get_memory_message", {"memory_id": "m1", "message_id": 2})
    )
    assert result["content"] == "Senec is a lake"
    assert not any(k.endswith(("_vec", "_flt")) for k in result)


async def test_set_status_and_forget_use_colon_path(fake):
    fake.on("PUT", "/messages/m1:7", ok(None, message=True))
    fake.on("DELETE", "/messages/m1:7", ok(None, message=True))
    server = make_server(fake)
    status = payload(
        await call(
            server,
            "ragflow_set_memory_message_status",
            {"memory_id": "m1", "message_id": 7, "enabled": False},
        )
    )
    assert status["enabled"] is False
    assert json.loads(fake.sent("PUT", "/messages/m1:7")[0].content) == {"status": False}
    forgot = payload(
        await call(server, "ragflow_forget_memory_message", {"memory_id": "m1", "message_id": 7})
    )
    assert forgot == {"forgotten": {"memory_id": "m1", "message_id": 7}}


@pytest.mark.parametrize(
    "name, args",
    [
        ("ragflow_search_memory_messages", {"query": "q", "memory_ids": []}),
        ("ragflow_recent_memory_messages", {"memory_ids": [""]}),
        ("ragflow_add_memory_message", {"memory_ids": [], "agent_id": "a", "session_id": "s", "user_id": "u", "user_input": "i", "agent_response": "r"}),
    ],
)
async def test_memory_ids_required(fake, name, args):
    result = await call(make_server(fake), name, args)
    assert result.isError and "memory_ids" in result.content[0].text
    assert not fake.requests
