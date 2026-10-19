"""End-to-end test of the chat tools against a real RAGFlow over MCP stdio.

Runs only with RAGFLOW_LIVE_TEST=1 plus RAGFLOW_BASE_URL and RAGFLOW_API_KEY, and needs
RAGFLOW_LIVE_DATASET_ID: an existing parsed dataset the scratch chat assistant may read.
The test creates chat assistants and sessions and deletes them at the end. It makes a few
short LLM calls. Mind map is left out: it runs several LLM calls and can take minutes.
"""

import base64
import json
import os
import sys
import time

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.skipif(
    os.environ.get("RAGFLOW_LIVE_TEST") != "1" or not os.environ.get("RAGFLOW_LIVE_DATASET_ID"),
    reason="set RAGFLOW_LIVE_TEST=1 and RAGFLOW_LIVE_DATASET_ID to run",
)


async def test_chat_assistant_sessions_and_conversations():
    dataset_id = os.environ["RAGFLOW_LIVE_DATASET_ID"]
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ragflow_kb_mcp.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        async def call(tool, /, **args):
            result = await s.call_tool(tool, args)
            assert not result.isError, f"{tool}: {result.content[0].text}"
            return json.loads(result.content[0].text)

        stamp = int(time.time())
        name = f"ragflow-kb-mcp-test-chat-{stamp}"
        chat = await call(
            "ragflow_create_chat",
            name=name,
            dataset_ids=[dataset_id],
            description="live test, safe to delete",
            prompt_config={"prologue": "Hello from the test.", "empty_response": ""},
        )
        chat_id = chat["id"]
        spare = await call("ragflow_create_chat", name=f"ragflow-kb-mcp-test-spare-{stamp}")
        try:
            listed = await call("ragflow_list_chats", name=name)
            assert [c["id"] for c in listed["chats"]] == [chat_id]

            await call("ragflow_update_chat", chat_id=chat_id, prompt_config={"prologue": "Patched."})
            got = await call("ragflow_get_chat", chat_id=chat_id)
            assert got["prompt_config"]["prologue"] == "Patched."
            assert got["prompt_config"]["empty_response"] == ""
            replaced = await call(
                "ragflow_replace_chat_settings", chat_id=chat_id, description="replaced"
            )
            assert replaced["description"] == "replaced"

            session = await call(
                "ragflow_create_chat_session", chat_id=chat_id, name="ragflow-kb-mcp-test-session"
            )
            session_id = session["id"]
            assert session["messages"][0]["content"] == "Patched."
            renamed = await call(
                "ragflow_rename_chat_session",
                chat_id=chat_id,
                session_id=session_id,
                name="ragflow-kb-mcp-test-renamed",
            )
            assert renamed["name"] == "ragflow-kb-mcp-test-renamed"
            sessions = await call("ragflow_list_chat_sessions", chat_id=chat_id)
            assert [x["id"] for x in sessions["sessions"]] == [session_id]

            answer = await call(
                "ragflow_chat",
                question="Answer in one short sentence: what is in these documents?",
                chat_id=chat_id,
                session_id=session_id,
                max_tokens=200,
            )
            assert answer["answer"] and answer["session_id"] == session_id
            message_id = answer["message_id"]

            history = await call(
                "ragflow_get_chat_session",
                chat_id=chat_id,
                session_id=session_id,
                include_references=True,
            )
            assert [m["role"] for m in history["messages"]] == ["assistant", "user", "assistant"]
            rated = await call(
                "ragflow_rate_chat_message",
                chat_id=chat_id,
                session_id=session_id,
                message_id=message_id,
                thumbup=False,
                feedback="live test",
            )
            assert rated["message"]["role"] == "assistant"
            assert rated["message"]["thumbup"] is False
            left = await call(
                "ragflow_delete_chat_message",
                chat_id=chat_id,
                session_id=session_id,
                message_id=message_id,
            )
            assert len(left["messages"]) == 1

            openai = await call(
                "ragflow_openai_chat_completion",
                chat_id=chat_id,
                question="Name one thing from the documents in a few words.",
                include_reference=True,
                max_tokens=200,
            )
            assert openai["content"]

            stateless = await call(
                "ragflow_chat",
                messages=[{"role": "user", "content": "Reply with the word OK."}],
                store_history=False,
                max_tokens=20,
            )
            assert stateless["answer"] and "session_id" not in stateless

            related = await call("ragflow_related_questions", question="How are backups done?")
            assert related["questions"]

            for tool, args in (
                ("ragflow_text_to_speech", {"text": "Hello."}),
                (
                    "ragflow_speech_to_text",
                    {"filename": "a.wav", "content_base64": base64.b64encode(b"RIFF").decode()},
                ),
            ):
                # Without a TTS or speech-to-text model RAGFlow answers with an error.
                result = await s.call_tool(tool, args)
                if result.isError:
                    assert "RAGFlow code" in result.content[0].text

            await call("ragflow_delete_chat_sessions", chat_id=chat_id, session_ids=[session_id])
            assert (await call("ragflow_list_chat_sessions", chat_id=chat_id))["sessions"] == []
        finally:
            await call("ragflow_delete_chat", chat_id=chat_id)
            await call("ragflow_delete_chats", chat_ids=[spare["id"]])
        assert (await call("ragflow_list_chats", name=name))["total"] == 0
