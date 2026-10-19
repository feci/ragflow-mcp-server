"""End-to-end agent tests against a real RAGFlow over MCP stdio.

Runs only with RAGFLOW_LIVE_TEST=1 plus RAGFLOW_BASE_URL and RAGFLOW_API_KEY. It creates a
temporary echo agent (Begin -> Message, no LLM call) and deletes it at the end.
"""

import json
import os
import sys
import time

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.skipif(
    os.environ.get("RAGFLOW_LIVE_TEST") != "1", reason="set RAGFLOW_LIVE_TEST=1 to run"
)

ECHO_DSL = {
    "components": {
        "begin": {
            "obj": {
                "component_name": "Begin",
                "params": {"mode": "conversational", "prologue": "Hi, I echo."},
            },
            "downstream": ["Message:Echo"],
            "upstream": [],
        },
        "Message:Echo": {
            "obj": {"component_name": "Message", "params": {"content": ["Echo: {sys.query}"]}},
            "downstream": [],
            "upstream": ["begin"],
        },
    },
    "globals": {
        "sys.conversation_turns": 0,
        "sys.date": "",
        "sys.files": [],
        "sys.history": [],
        "sys.query": "",
        "sys.user_id": "",
    },
    "history": [],
    "path": [],
    "retrieval": [],
    "variables": [],
    "memory": [],
    "messages": [],
}


async def test_agent_lifecycle():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ragflow_kb_mcp.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        async def call(tool, /, **args):
            result = await s.call_tool(tool, args)
            assert not result.isError, f"{tool}: {result.content[0].text}"
            return json.loads(result.content[0].text)

        templates = await call("ragflow_list_agent_templates")
        assert templates["templates"]
        first = await call("ragflow_list_agent_templates", template_id=str(templates["templates"][0]["id"]))
        assert first["dsl"]["components"]

        title = f"ragflow-kb-mcp-test-agent-{int(time.time())}"
        agent = await call(
            "ragflow_create_agent", title=title, dsl=ECHO_DSL, description="live test, safe to delete"
        )
        agent_id = agent["id"]
        try:
            listed = await call("ragflow_list_agents", keywords=title)
            assert [a["id"] for a in listed["agents"]] == [agent_id]
            assert "dsl" not in listed["agents"][0]

            got = await call("ragflow_get_agent", agent_id=agent_id, include_dsl=True)
            assert "Message:Echo" in json.dumps(got["dsl"])

            updated = await call("ragflow_update_agent", agent_id=agent_id, description="updated")
            assert updated["updated"] == ["description"]
            assert (await call("ragflow_get_agent", agent_id=agent_id))["description"] == "updated"

            session = await call(
                "ragflow_create_agent_session", agent_id=agent_id, name="ragflow-kb-mcp-test-session"
            )
            answer = await call(
                "ragflow_agent_chat", agent_id=agent_id, question="hello", session_id=session["id"]
            )
            assert answer["answer"] == "Echo: hello"
            assert answer["session_id"] == session["id"]

            fresh = await call("ragflow_agent_chat", agent_id=agent_id, question="new session")
            assert fresh["answer"] == "Echo: new session" and fresh["session_id"] != session["id"]

            openai = await call(
                "ragflow_agent_chat", agent_id=agent_id, question="openai mode", openai_compatible=True
            )
            assert openai["answer"] == "Echo: openai mode" and openai["usage"]["total_tokens"] > 0

            uploaded = await call(
                "ragflow_upload_agent_file",
                agent_id=agent_id,
                filename="ragflow-kb-mcp-test.txt",
                text="attachment body\n",
            )
            with_file = await call(
                "ragflow_agent_chat", agent_id=agent_id, question="with file", files=[uploaded]
            )
            assert with_file["answer"] == "Echo: with file"
            attachment = await call(
                "ragflow_download_agent_attachment", attachment_id=uploaded["id"], ext="txt"
            )
            assert attachment["text"] == "attachment body\n"
            assert attachment["content_type"].startswith("text/plain")

            one = await call("ragflow_list_agent_sessions", agent_id=agent_id, session_id=session["id"])
            assert one["total"] == 1
            assert [m["content"] for m in one["sessions"][0]["messages"]] == ["hello", "Echo: hello"]

            everything = await call("ragflow_list_agent_sessions", agent_id=agent_id, include_messages=False)
            ids = [x["id"] for x in everything["sessions"]]
            assert session["id"] in ids and len(ids) == everything["total"] >= 4

            await call("ragflow_delete_agent_sessions", agent_id=agent_id, session_ids=[session["id"]])
            left = await call("ragflow_list_agent_sessions", agent_id=agent_id, include_messages=False)
            assert left["total"] == len(ids) - 1
            await call(
                "ragflow_delete_agent_sessions",
                agent_id=agent_id,
                session_ids=[x["id"] for x in left["sessions"]],
            )
        finally:
            await call("ragflow_delete_agent", agent_id=agent_id)
        gone = await call("ragflow_list_agents", keywords=title)
        assert gone["agents"] == []
