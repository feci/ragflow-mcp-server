"""Live tests for search apps, health and the extra knowledge tools.

They run only with RAGFLOW_LIVE_TEST=1 plus RAGFLOW_BASE_URL and RAGFLOW_API_KEY. The search
app test needs a parsed dataset: set RAGFLOW_LIVE_DATASET_ID to one (it is only read). All
created objects are named ragflow-kb-mcp-test-* and deleted at the end. Nothing here uses the
parse queue.
"""

import contextlib
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


@contextlib.asynccontextmanager
async def session():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ragflow_kb_mcp.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        async def call(tool, /, **args):
            result = await s.call_tool(tool, args)
            assert not result.isError, f"{tool}: {result.content[0].text}"
            return json.loads(result.content[0].text)

        yield s, call


async def test_health_and_dataset_filters():
    async with session() as (s, call):
        health = await call("ragflow_health")
        assert health["healthy"] is True, health
        listed = await call("ragflow_list_datasets", page_size=5, orderby="update_time", desc=False)
        assert listed["total"] >= len(listed["datasets"])
        if listed["datasets"]:
            first = listed["datasets"][0]
            one = await call("ragflow_list_datasets", dataset_id=first["id"])
            assert [d["id"] for d in one["datasets"]] == [first["id"]]
        missing = await call("ragflow_list_datasets", name="ragflow-kb-mcp-test-no-such-dataset")
        assert missing == {"total": 0, "datasets": []}


@pytest.mark.skipif(not os.environ.get("RAGFLOW_LIVE_DATASET_ID"), reason="needs a parsed dataset")
async def test_search_app_lifecycle_and_completion():
    dataset_id = os.environ["RAGFLOW_LIVE_DATASET_ID"]
    name = f"ragflow-kb-mcp-test-search-{int(time.time())}"
    async with session() as (s, call):
        app = await call(
            "ragflow_create_search_app",
            name=name,
            description="live test, safe to delete",
            dataset_ids=[dataset_id],
            similarity_threshold=0.2,
            vector_similarity_weight=0.7,
        )
        search_id = app["id"]
        try:
            assert app["search_config"]["kb_ids"] == [dataset_id]
            got = await call("ragflow_get_search_app", search_id=search_id)
            assert got["name"] == name and got["description"] == "live test, safe to delete"
            listed = await call("ragflow_list_search_apps", keywords="ragflow-kb-mcp-test-search")
            assert search_id in [a["id"] for a in listed["search_apps"]]
            updated = await call("ragflow_update_search_app", search_id=search_id, top_k=32)
            assert updated["name"] == name
            assert updated["search_config"]["top_k"] == 32
            assert updated["search_config"]["kb_ids"] == [dataset_id]
            renamed = await call("ragflow_update_search_app", search_id=search_id, name=name + "-b")
            assert renamed["name"] == name + "-b"

            answer = await call(
                "ragflow_search_app_completion",
                search_id=search_id,
                question="Which hypervisor runs the servers?",
                max_references=3,
            )
            assert answer["answer"]
            assert len(answer["references"]) <= 3
        finally:
            await call("ragflow_delete_search_app", search_id=search_id)
        gone = await s.call_tool("ragflow_get_search_app", {"search_id": search_id})
        assert gone.isError


async def test_chunk_get_availability_and_ingest_without_queue():
    async with session() as (s, call):
        ds = await call(
            "ragflow_create_dataset", name=f"ragflow-kb-mcp-test-chunks-{int(time.time())}"
        )
        dataset_id = ds["id"]
        try:
            up = await call(
                "ragflow_upload_document",
                dataset_id=dataset_id,
                filename="chunks.md",
                text="# Chunk test\n",
                parse=False,
            )
            doc_id = up["document"]["id"]
            added = await call(
                "ragflow_add_chunk",
                dataset_id=dataset_id,
                document_id=doc_id,
                content="The heron Zuzana nests at lake Senec.",
                important_keywords=["heron"],
            )
            chunk = await call(
                "ragflow_get_chunk", dataset_id=dataset_id, document_id=doc_id, chunk_id=added["id"]
            )
            assert chunk["content"] == "The heron Zuzana nests at lake Senec."
            assert chunk.get("available", True) is True

            await call(
                "ragflow_set_chunks_availability",
                dataset_id=dataset_id,
                document_id=doc_id,
                chunk_ids=[added["id"]],
                available=False,
            )
            off = await call(
                "ragflow_get_chunk", dataset_id=dataset_id, document_id=doc_id, chunk_id=added["id"]
            )
            assert off["available"] is False
            await call(
                "ragflow_set_chunks_availability",
                dataset_id=dataset_id,
                document_id=doc_id,
                chunk_ids=[added["id"]],
                available=True,
            )
            on = await call(
                "ragflow_get_chunk", dataset_id=dataset_id, document_id=doc_id, chunk_id=added["id"]
            )
            assert on.get("available", True) is True

            # Cancelling a document that is not being parsed is refused by RAGFlow.
            cancel = await s.call_tool(
                "ragflow_ingest_documents", {"document_ids": [doc_id], "action": "cancel"}
            )
            assert cancel.isError and "not in RUNNING status" in cancel.content[0].text
            # Unknown documents are refused before anything is queued.
            unknown = await s.call_tool(
                "ragflow_ingest_documents",
                {"document_ids": ["0" * 32], "action": "start"},
            )
            assert unknown.isError
            # Clearing removes the document's chunks without queueing a parse.
            cleared = await call("ragflow_ingest_documents", document_ids=[doc_id], action="clear")
            assert cleared == {"action": "clear", "document_ids": [doc_id]}
            after = await call("ragflow_list_chunks", dataset_id=dataset_id, document_id=doc_id)
            assert after["total"] == 0
        finally:
            await call("ragflow_delete_datasets", dataset_ids=[dataset_id])
