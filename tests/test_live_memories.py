"""End-to-end test of the memory tools against a real RAGFlow over MCP stdio.

Runs only with RAGFLOW_LIVE_TEST=1 plus RAGFLOW_BASE_URL, RAGFLOW_API_KEY,
RAGFLOW_DEFAULT_EMBEDDING_MODEL and RAGFLOW_TEST_LLM_ID (a chat model as model_name@factory).
It creates a temporary memory, adds one exchange (the LLM extracts from it) and deletes the
memory at the end.
"""

import asyncio
import json
import os
import sys
import time

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.skipif(
    os.environ.get("RAGFLOW_LIVE_TEST") != "1" or not os.environ.get("RAGFLOW_TEST_LLM_ID"),
    reason="set RAGFLOW_LIVE_TEST=1 and RAGFLOW_TEST_LLM_ID to run",
)


async def test_memory_lifecycle():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ragflow_kb_mcp.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        async def call(tool, /, **args):
            result = await s.call_tool(tool, args)
            assert not result.isError, f"{tool}: {result.content[0].text}"
            return json.loads(result.content[0].text)

        name = f"ragflow-kb-mcp-test-memory-{int(time.time())}"
        mem = await call(
            "ragflow_create_memory",
            name=name,
            llm_id=os.environ["RAGFLOW_TEST_LLM_ID"],
            memory_types=["raw", "semantic"],
            description="live test, safe to delete",
        )
        memory_id = mem["id"]
        try:
            assert mem["description"] == "live test, safe to delete"
            assert sorted(mem["memory_type"]) == ["raw", "semantic"]

            listed = await call("ragflow_list_memories", keywords=name)
            assert [m["id"] for m in listed["memories"]] == [memory_id]
            config = await call("ragflow_get_memory_config", memory_id=memory_id)
            assert config["system_prompt"]
            updated = await call("ragflow_update_memory", memory_id=memory_id, temperature=0.3)
            assert updated["temperature"] == 0.3

            ids = {"agent_id": f"{name}-agent", "session_id": f"{name}-session"}
            added = await call(
                "ragflow_add_memory_message",
                memory_ids=[memory_id],
                user_id=f"{name}-user",
                user_input="My favourite lake is Senec.",
                agent_response="Noted: your favourite lake is Senec.",
                **ids,
            )
            assert added["queued"] is True

            raw = None
            for _ in range(40):
                page = await call(
                    "ragflow_list_memory_messages",
                    memory_id=memory_id,
                    session_id=ids["session_id"],
                )
                done = [
                    m for m in page["messages"] if abs(m.get("task", {}).get("progress", 0)) >= 1
                ]
                if done:
                    raw = done[0]
                    break
                await asyncio.sleep(3)
            assert raw is not None, "extraction did not finish in time"
            assert raw["task"]["progress"] == 1, raw["task"]

            recent = await call("ragflow_recent_memory_messages", memory_ids=[memory_id])
            assert any("Senec" in (m.get("content") or "") for m in recent["messages"])
            found = await call(
                "ragflow_search_memory_messages", query="favourite lake", memory_ids=[memory_id]
            )
            assert found["messages"]

            content = await call(
                "ragflow_get_memory_message", memory_id=memory_id, message_id=raw["message_id"]
            )
            assert "Senec" in content["content"]

            target = (raw.get("extract") or [raw])[0]["message_id"]
            await call(
                "ragflow_set_memory_message_status",
                memory_id=memory_id,
                message_id=target,
                enabled=False,
            )
            await call(
                "ragflow_forget_memory_message", memory_id=memory_id, message_id=raw["message_id"]
            )
            after = await call("ragflow_list_memory_messages", memory_id=memory_id)
            first = after["messages"][0]
            assert first.get("forget_at")
            statuses = {m["message_id"]: m["status"] for m in [first, *first.get("extract", [])]}
            assert statuses[target] is False

            refused = await s.call_tool(
                "ragflow_update_memory", {"memory_id": memory_id, "memory_types": ["raw"]}
            )
            assert refused.isError and "isn't empty" in refused.content[0].text
        finally:
            await call("ragflow_delete_memory", memory_id=memory_id)
        gone = await call("ragflow_list_memories", keywords=name)
        assert gone["memories"] == []
