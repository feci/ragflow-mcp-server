"""End-to-end tests against a real RAGFlow over MCP stdio.

They run only with RAGFLOW_LIVE_TEST=1 plus RAGFLOW_BASE_URL and RAGFLOW_API_KEY. Each test
creates a temporary dataset and deletes it at the end. The parsing test needs a working
RAGFlow ingestion queue and embedding model.
"""

import base64
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

HERE = os.path.dirname(os.path.abspath(__file__))


@contextlib.asynccontextmanager
async def live_dataset(label):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ragflow_kb_mcp.server"],
        env={**os.environ, "RAGFLOW_UPLOAD_ROOTS": HERE},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        async def call(tool, /, **args):
            result = await s.call_tool(tool, args)
            assert not result.isError, f"{tool}: {result.content[0].text}"
            return json.loads(result.content[0].text)

        ds = await call(
            "ragflow_create_dataset",
            name=f"ragflow-kb-mcp-{label}-{int(time.time())}",
            description="live test, safe to delete",
        )
        assert ds["description"] == "live test, safe to delete"
        try:
            yield s, call, ds["id"]
        finally:
            await call("ragflow_delete_datasets", dataset_ids=[ds["id"]])


async def test_documents_without_parsing():
    async with live_dataset("docs") as (s, call, dataset_id):
        text = "# Live test\n\nThe heron Zuzana nests at lake Senec every April.\n"
        up = await call(
            "ragflow_upload_document",
            dataset_id=dataset_id,
            filename="live.md",
            text=text,
            meta_fields={"source": "live-test"},
            parse=False,
        )
        doc = up["document"]
        assert doc["ingestion_status"] == "UNSTART"
        assert doc["meta_fields"] == {"source": "live-test"}
        doc_id = doc["id"]

        duplicate = await s.call_tool(
            "ragflow_upload_document",
            {"dataset_id": dataset_id, "filename": "live.md", "text": "again", "parse": False},
        )
        assert duplicate.isError and "already exists" in duplicate.content[0].text

        small = await call(
            "ragflow_upload_document",
            dataset_id=dataset_id,
            filename="small.txt",
            content_base64=base64.b64encode(b"Plain bytes upload.").decode(),
            parse=False,
        )
        local = await call(
            "ragflow_upload_document",
            dataset_id=dataset_id,
            file_path=__file__,
            filename="local-copy.py",
            parse=False,
        )
        assert local["document"]["name"] == "local-copy.py"
        outside = await s.call_tool(
            "ragflow_upload_document",
            {"dataset_id": dataset_id, "file_path": "/etc/hosts", "filename": "hosts.txt"},
        )
        assert outside.isError and "outside the allowed" in outside.content[0].text

        web = await call(
            "ragflow_upload_document",
            dataset_id=dataset_id,
            url="https://example.com/",
            filename="example-page",
            parse=False,
        )
        assert web["document"]["name"].startswith("example-page")

        summary = await call("ragflow_metadata_summary", dataset_id=dataset_id)
        assert "source" in summary
        changed = await call(
            "ragflow_update_metadata",
            dataset_id=dataset_id,
            document_ids=[doc_id],
            updates=[{"key": "source", "value": "live-test-2"}],
        )
        assert changed.get("updated") == 1
        updated = await call(
            "ragflow_update_document", dataset_id=dataset_id, document_id=doc_id, enabled=False
        )
        assert updated["enabled"] is False
        assert updated["meta_fields"].get("source") == "live-test-2"

        downloaded = await call(
            "ragflow_download_document", dataset_id=dataset_id, document_id=doc_id
        )
        assert downloaded["text"] == text

        replaced = await call(
            "ragflow_upload_document",
            dataset_id=dataset_id,
            filename="live.md",
            text="# Replaced\n",
            replace_existing=True,
            parse=False,
        )
        assert replaced["replaced"] == [doc_id]
        assert replaced["document"]["name"] == "live.md"

        await call(
            "ragflow_delete_documents",
            dataset_id=dataset_id,
            document_ids=[small["document"]["id"]],
        )
        listed = await call("ragflow_list_documents", dataset_id=dataset_id)
        names = sorted(d["name"] for d in listed["documents"])
        assert "live.md" in names and "small.txt" not in names and len(names) == 3
        missing = await call("ragflow_list_documents", dataset_id=dataset_id, name="nope.md")
        assert missing == {"total": 0, "documents": []}


async def test_parsing_chunks_and_retrieval():
    async with live_dataset("parse") as (s, call, dataset_id):
        up = await call(
            "ragflow_upload_document",
            dataset_id=dataset_id,
            filename="live.md",
            text="# Live test\n\nThe heron Zuzana nests at lake Senec every April.\n",
            wait_seconds=300,
        )
        doc = up["document"]
        assert doc["ingestion_status"] == "COMPLETED", repr(doc)[:1500]
        doc_id = doc["id"]

        chunks = await call("ragflow_list_chunks", dataset_id=dataset_id, document_id=doc_id)
        assert any("Zuzana" in c["content"] for c in chunks["chunks"])

        added = await call(
            "ragflow_add_chunk",
            dataset_id=dataset_id,
            document_id=doc_id,
            content="Manual chunk about the heron's winter quarters in Tunisia.",
            important_keywords=["heron"],
        )
        await call(
            "ragflow_update_chunk",
            dataset_id=dataset_id,
            document_id=doc_id,
            chunk_id=added["id"],
            content="Manual chunk: the heron winters in Tunisia.",
        )
        found = await call(
            "ragflow_retrieval",
            question="Where does Zuzana nest?",
            dataset_ids=[dataset_id],
            similarity_threshold=0.1,
        )
        assert any("Senec" in c["content"] for c in found["chunks"])
        await call(
            "ragflow_delete_chunks",
            dataset_id=dataset_id,
            document_id=doc_id,
            chunk_ids=[added["id"]],
        )
