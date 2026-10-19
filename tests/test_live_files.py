"""Live test of the file-manager and workspace tools over MCP stdio.

Runs only with RAGFLOW_LIVE_TEST=1 plus RAGFLOW_BASE_URL and RAGFLOW_API_KEY. Everything
happens inside a scratch folder (and a scratch dataset) that is deleted at the end. The one
runtime attachment it uploads cannot be deleted through the API.
"""

import asyncio
import base64
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
# 1x1 transparent PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


async def test_file_manager_and_workspace_commits():
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

        stamp = int(time.time())
        root_name = f"ragflow-kb-mcp-test-{stamp}"
        folder = await call("ragflow_create_folder", name=root_name)
        assert folder["type"] == "folder"
        folder_id = folder["id"]
        dataset_id = None
        try:
            listed = await call("ragflow_list_files", keywords=root_name)
            assert [f["id"] for f in listed["files"]] == [folder_id]

            sub = await call("ragflow_create_folder", name="sub", parent_id=folder_id)
            note = (
                await call(
                    "ragflow_upload_file", filename="note.md", text="hello v1", parent_id=folder_id
                )
            )["files"][0]
            png = (
                await call(
                    "ragflow_upload_file",
                    filename="pixel.png",
                    content_base64=base64.b64encode(PNG).decode(),
                    parent_id=folder_id,
                )
            )["files"][0]
            local = (
                await call(
                    "ragflow_upload_file", file_path=__file__, parent_id=sub["id"]
                )
            )["files"][0]
            assert local["name"] == os.path.basename(__file__)
            outside = await s.call_tool(
                "ragflow_upload_file", {"file_path": "/etc/hosts", "parent_id": folder_id}
            )
            assert outside.isError and "outside the allowed" in outside.content[0].text

            inside = await call("ragflow_list_files", parent_id=folder_id)
            assert inside["parent_folder"]["id"] == folder["parent_id"]
            assert {f["name"] for f in inside["files"]} == {"sub", "note.md", "pixel.png"}

            parent = await call("ragflow_get_parent_folder", file_id=note["id"])
            assert parent["parent_folder"]["id"] == folder_id
            ancestors = await call("ragflow_get_file_ancestors", file_id=local["id"])
            ancestor_ids = [f["id"] for f in ancestors["parent_folders"]]
            assert sub["id"] in ancestor_ids and folder_id in ancestor_ids

            text = await call("ragflow_download_file", file_id=note["id"])
            assert text["text"] == "hello v1"
            binary = await call("ragflow_download_file", file_id=png["id"], include_base64=True)
            assert base64.b64decode(binary["content_base64"]) == PNG

            await call("ragflow_move_files", file_ids=[local["id"]], dest_folder_id=folder_id)
            moved = await call("ragflow_get_parent_folder", file_id=local["id"])
            assert moved["parent_folder"]["id"] == folder_id
            await call("ragflow_move_files", file_ids=[note["id"]], new_name="renamed.md")
            bad_ext = await s.call_tool(
                "ragflow_move_files", {"file_ids": [note["id"]], "new_name": "renamed.txt"}
            )
            assert bad_ext.isError

            pending = await call("ragflow_get_uncommitted_changes", workspace_id=folder_id)
            assert {c["file_id"] for c in pending["changes"]} >= {note["id"], png["id"]}

            first = await call(
                "ragflow_create_commit",
                workspace_id=folder_id,
                message="first",
                files=[
                    {"file_id": note["id"], "file_name": "renamed.md", "operation": "add", "content": "hello v1"}
                ],
            )
            second = await call(
                "ragflow_create_commit",
                workspace_id=folder_id,
                message="second",
                files=[
                    {"file_id": note["id"], "file_name": "renamed.md", "operation": "modify", "content": "hello v2"}
                ],
            )
            assert second["parent_id"] == first["id"]

            commits = await call("ragflow_list_commits", workspace_id=folder_id)
            assert [c["id"] for c in commits["commits"]] == [second["id"], first["id"]]
            detail = await call("ragflow_get_commit", workspace_id=folder_id, commit_id=second["id"])
            assert detail["files"][0]["operation"] == "modify"
            items = await call(
                "ragflow_list_commit_files", workspace_id=folder_id, commit_id=first["id"]
            )
            assert items["files"][0]["file_id"] == note["id"]
            diff = await call(
                "ragflow_diff_commits",
                workspace_id=folder_id,
                from_commit_id=first["id"],
                to_commit_id=second["id"],
            )
            assert any(c["file_id"] == note["id"] for c in diff["changes"])
            tree = await call("ragflow_get_commit_tree", workspace_id=folder_id, commit_id=second["id"])
            assert tree.get("id") == folder_id
            old = await call(
                "ragflow_get_commit_file_content",
                workspace_id=folder_id,
                commit_id=first["id"],
                file_id=note["id"],
            )
            assert old["content"] == "hello v1"
            versions = await call("ragflow_get_file_versions", file_id=note["id"])
            assert len(versions["versions"]) == 2
            current = await call("ragflow_download_file", file_id=note["id"])
            assert current["text"] == "hello v2"

            ds = await call(
                "ragflow_create_dataset", name=f"ragflow-kb-mcp-test-link-{stamp}"
            )
            dataset_id = ds["id"]
            await call(
                "ragflow_link_files_to_datasets", file_ids=[note["id"]], dataset_ids=[dataset_id]
            )
            docs = {"documents": []}
            for _ in range(20):
                docs = await call("ragflow_list_documents", dataset_id=dataset_id)
                if docs["documents"]:
                    break
                await asyncio.sleep(1)
            assert [d["name"] for d in docs["documents"]] == ["renamed.md"]
            assert docs["documents"][0]["ingestion_status"] == "UNSTART"

            attachment = await call(
                "ragflow_upload_attachment",
                filename="ragflow-kb-mcp-test-attachment.txt",
                text="attachment probe",
            )
            assert attachment["id"] and attachment["size"] == len("attachment probe")
        finally:
            if dataset_id:
                await call("ragflow_delete_datasets", dataset_ids=[dataset_id])
            await call("ragflow_delete_files", file_ids=[folder_id])
        gone = await call("ragflow_list_files", keywords=root_name)
        assert gone["files"] == []
