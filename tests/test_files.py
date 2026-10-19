import base64
import json

import httpx
import pytest
from conftest import call, make_server, ok, payload


def entry(file_id="f1", name="a.md", **extra):
    return {
        "id": file_id,
        "name": name,
        "type": "doc",
        "size": 3,
        "parent_id": "root",
        "location": "a.md",
        "tenant_id": "t",
        "created_by": "u",
        "source_type": "",
        "kbs_info": [],
        **extra,
    }


async def test_list_files_trims_and_passes_query(fake):
    fake.on(
        "GET",
        "/files",
        ok(
            {
                "total": 1,
                "files": [entry(kbs_info=[{"kb_id": "k1", "kb_name": "KB"}])],
                "parent_folder": {"id": "root", "name": "/", "tenant_id": "t"},
            }
        ),
    )
    result = payload(
        await call(make_server(fake), "ragflow_list_files", {"parent_id": "p1", "keywords": "a", "page_size": 5})
    )
    assert result == {
        "total": 1,
        "parent_folder": {"id": "root", "name": "/"},
        "files": [
            {
                "id": "f1",
                "name": "a.md",
                "type": "doc",
                "size": 3,
                "parent_id": "root",
                "source_type": "",
                "datasets": [{"kb_id": "k1", "kb_name": "KB"}],
            }
        ],
    }
    params = fake.requests[0].url.params
    assert params["parent_id"] == "p1" and params["keywords"] == "a" and params["page_size"] == "5"
    assert params["desc"] == "true"


async def test_upload_file_text_multipart(fake):
    fake.on("POST", "/files", ok([entry(name="note.md")]))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_upload_file",
            {"filename": "note.md", "text": "hello", "parent_id": "p1"},
        )
    )
    assert result["files"][0]["name"] == "note.md"
    body = fake.sent("POST", "/files")[0].content
    assert b'name="parent_id"' in body and b"p1" in body
    assert b'filename="note.md"' in body and b"hello" in body


async def test_upload_file_base64_and_requires_name(fake):
    fake.on("POST", "/files", ok([entry(name="x.png")]))
    server = make_server(fake)
    payload(
        await call(
            server,
            "ragflow_upload_file",
            {"filename": "x.png", "content_base64": base64.b64encode(b"\x89PNG").decode()},
        )
    )
    body = fake.sent("POST", "/files")[0].content
    assert b"\x89PNG" in body and b'name="parent_id"' not in body
    missing = await call(server, "ragflow_upload_file", {"text": "x"})
    assert missing.isError and "filename is required" in missing.content[0].text


async def test_upload_file_path_respects_roots(fake, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "in.txt").write_text("inside")
    (tmp_path / "out.txt").write_text("outside")
    fake.on("POST", "/files", ok([entry(name="in.txt")]))
    server = make_server(fake, upload_roots=(allowed.resolve(),))
    payload(await call(server, "ragflow_upload_file", {"file_path": str(allowed / "in.txt")}))
    assert b'filename="in.txt"' in fake.sent("POST", "/files")[0].content
    refused = await call(server, "ragflow_upload_file", {"file_path": str(tmp_path / "out.txt")})
    assert refused.isError and "outside the allowed" in refused.content[0].text
    disabled = await call(make_server(fake), "ragflow_upload_file", {"file_path": str(allowed / "in.txt")})
    assert disabled.isError and "disabled" in disabled.content[0].text
    assert len(fake.sent("POST", "/files")) == 1


async def test_create_folder_sends_json_type(fake):
    fake.on("POST", "/files", ok(entry("d1", "docs", type="folder")))
    result = payload(await call(make_server(fake), "ragflow_create_folder", {"name": "docs"}))
    assert result["type"] == "folder"
    assert json.loads(fake.sent("POST", "/files")[0].content) == {"name": "docs", "type": "folder"}


async def test_parent_folder_sends_file_id_query(fake):
    fake.on("GET", "/files/f1/parent", ok({"parent_folder": {"id": "p", "name": "P", "size": 0}}))
    result = payload(await call(make_server(fake), "ragflow_get_parent_folder", {"file_id": "f1"}))
    assert result == {"parent_folder": {"id": "p", "name": "P"}}
    assert fake.requests[0].url.params["file_id"] == "f1"


async def test_ancestors(fake):
    fake.on(
        "GET",
        "/files/f1/ancestors",
        ok({"parent_folders": [{"id": "a", "name": "A", "tenant_id": "t"}, {"id": "r", "name": "/"}]}),
    )
    result = payload(await call(make_server(fake), "ragflow_get_file_ancestors", {"file_id": "f1"}))
    assert result == {"parent_folders": [{"id": "a", "name": "A"}, {"id": "r", "name": "/"}]}


async def test_download_file_text_binary_and_error(fake):
    fake.on("GET", "/files/t", httpx.Response(200, content=b"hello", headers={"content-type": "text/plain"}))
    fake.on("GET", "/files/b", httpx.Response(200, content=b"\xff\x00", headers={"content-type": "image/png"}))
    fake.on("GET", "/files/x", {"code": 401, "message": "document not found"})
    server = make_server(fake)
    text = payload(await call(server, "ragflow_download_file", {"file_id": "t"}))
    assert text["text"] == "hello" and text["truncated"] is False
    binary = payload(await call(server, "ragflow_download_file", {"file_id": "b", "include_base64": True}))
    assert base64.b64decode(binary["content_base64"]) == b"\xff\x00"
    hidden = payload(await call(server, "ragflow_download_file", {"file_id": "b"}))
    assert "content_base64" not in hidden and hidden["text"] is None
    missing = await call(server, "ragflow_download_file", {"file_id": "x"})
    assert missing.isError and "document not found" in missing.content[0].text


async def test_move_files_validation_and_body(fake):
    fake.on("POST", "/files/move", ok(True))
    server = make_server(fake)
    neither = await call(server, "ragflow_move_files", {"file_ids": ["a"]})
    assert neither.isError and "dest_folder_id" in neither.content[0].text
    many = await call(server, "ragflow_move_files", {"file_ids": ["a", "b"], "new_name": "x.md"})
    assert many.isError and "single file" in many.content[0].text
    payload(await call(server, "ragflow_move_files", {"file_ids": ["a", "b"], "dest_folder_id": "d"}))
    payload(await call(server, "ragflow_move_files", {"file_ids": ["a"], "new_name": "b.md"}))
    bodies = [json.loads(r.content) for r in fake.sent("POST", "/files/move")]
    assert bodies == [
        {"src_file_ids": ["a", "b"], "dest_file_id": "d"},
        {"src_file_ids": ["a"], "new_name": "b.md"},
    ]


async def test_delete_files(fake):
    fake.on("DELETE", "/files", ok(True))
    server = make_server(fake)
    assert payload(await call(server, "ragflow_delete_files", {"file_ids": ["a"]})) == {"deleted": ["a"]}
    assert json.loads(fake.sent("DELETE", "/files")[0].content) == {"ids": ["a"]}
    empty = await call(server, "ragflow_delete_files", {"file_ids": []})
    assert empty.isError and len(fake.sent("DELETE", "/files")) == 1


async def test_link_files_defaults_to_add_mode(fake):
    fake.on("POST", "/files/link-to-datasets", ok(True))
    server = make_server(fake)
    payload(await call(server, "ragflow_link_files_to_datasets", {"file_ids": ["f"], "dataset_ids": ["k"]}))
    request = fake.sent("POST", "/files/link-to-datasets")[0]
    assert request.url.params["mode"] == "add"
    assert json.loads(request.content) == {"file_ids": ["f"], "kb_ids": ["k"]}
    refused = await call(server, "ragflow_link_files_to_datasets", {"file_ids": ["f"], "dataset_ids": []})
    assert refused.isError
    payload(
        await call(
            server,
            "ragflow_link_files_to_datasets",
            {"file_ids": ["f"], "dataset_ids": [], "mode": "replace"},
        )
    )
    assert fake.sent("POST", "/files/link-to-datasets")[1].url.params["mode"] == "replace"


async def test_upload_attachment_file_or_url(fake):
    info = {"id": "att1", "name": "a.txt", "size": 3, "extension": "txt", "mime_type": "text/plain", "created_by": "u", "preview_url": None}
    fake.on("POST", "/documents/upload", ok(info))
    server = make_server(fake)
    result = payload(await call(server, "ragflow_upload_attachment", {"filename": "a.txt", "text": "abc"}))
    assert result == {"id": "att1", "name": "a.txt", "size": 3, "extension": "txt", "mime_type": "text/plain"}
    payload(await call(server, "ragflow_upload_attachment", {"url": "https://example.com/x.pdf"}))
    by_url = fake.sent("POST", "/documents/upload")[1]
    assert by_url.url.params["url"] == "https://example.com/x.pdf" and by_url.content == b""
    both = await call(server, "ragflow_upload_attachment", {"url": "https://e.x/a", "text": "x"})
    assert both.isError and len(fake.sent("POST", "/documents/upload")) == 2


@pytest.mark.parametrize(
    "args, message",
    [
        ({"message": "m", "files": [{"file_id": "f", "operation": "add", "content": "x"}]}, "exactly one"),
        (
            {"workspace_id": "w", "dataset_id": "d", "message": "m", "files": [{"file_id": "f", "operation": "delete"}]},
            "exactly one",
        ),
        ({"workspace_id": "w", "message": "m", "files": []}, "at least one"),
        ({"workspace_id": "w", "message": "m", "files": [{"file_id": "f", "operation": "copy"}]}, "operation"),
        ({"workspace_id": "w", "message": "m", "files": [{"file_id": "f", "operation": "modify"}]}, "content"),
        ({"workspace_id": "w", "message": "m", "files": [{"file_id": "f", "operation": "rename"}]}, "new_name"),
    ],
)
async def test_create_commit_validation(fake, args, message):
    result = await call(make_server(fake), "ragflow_create_commit", args)
    assert result.isError and message in result.content[0].text
    assert not fake.requests


async def test_create_commit_and_dataset_scope(fake):
    commit = {"id": "c1", "folder_id": "w", "message": "m", "author_id": "u", "file_count": 1, "tree_state": "{...}", "create_time": 1}
    fake.on("POST", "/workspaces/w/commits", ok(commit))
    fake.on("GET", "/datasets/k/commits", ok({"total": 1, "page": 1, "page_size": 15, "commits": [commit]}))
    server = make_server(fake)
    files = [{"file_id": "f", "file_name": "a.md", "operation": "add", "content": "x"}]
    created = payload(await call(server, "ragflow_create_commit", {"workspace_id": "w", "message": "m", "files": files}))
    assert "tree_state" not in created and created["id"] == "c1"
    assert json.loads(fake.sent("POST", "/workspaces/w/commits")[0].content) == {"message": "m", "files": files}
    listed = payload(await call(server, "ragflow_list_commits", {"dataset_id": "k"}))
    assert listed["total"] == 1 and listed["commits"][0]["id"] == "c1"


async def test_commit_reads(fake):
    item = {"id": "i", "seq": 3, "commit_id": "c1", "file_id": "f", "operation": "modify", "old_hash": "a", "new_hash": "b", "new_location": ".objects/b", "diff": "x" * 5000}
    fake.on("GET", "/workspaces/w/commits/c1", ok({"id": "c1", "folder_id": "w", "message": "m", "file_count": 1, "files": [item]}))
    fake.on("GET", "/workspaces/w/commits/c1/files", ok([item]))
    fake.on("GET", "/workspaces/w/commits/diff", ok([{"file_id": "f", "file_name": "a.md", "operation": "modify", "old_location": "x"}]))
    fake.on("GET", "/workspaces/w/changes", ok([{"file_id": "g", "file_name": "b.md", "operation": "add"}]))
    fake.on("GET", "/workspaces/w/commits/c1/tree", ok({"id": "w", "type": "folder", "children": []}))
    fake.on("GET", "/workspaces/w/commits/c1/files/f/content", ok({"content": "abcdef"}))
    fake.on("GET", "/workspace-files/f/versions", ok([{"commit_id": "c1", "operation": "add"}]))
    server = make_server(fake)
    detail = payload(await call(server, "ragflow_get_commit", {"workspace_id": "w", "commit_id": "c1"}))
    file_change = detail["files"][0]
    assert file_change["operation"] == "modify" and len(file_change["diff"]) == 4000
    assert file_change["diff_truncated"] is True and "new_location" not in file_change
    files = payload(await call(server, "ragflow_list_commit_files", {"workspace_id": "w", "commit_id": "c1"}))
    assert files["files"][0]["file_id"] == "f"
    diff = payload(
        await call(server, "ragflow_diff_commits", {"workspace_id": "w", "from_commit_id": "c0", "to_commit_id": "c1"})
    )
    assert diff == {"changes": [{"file_id": "f", "file_name": "a.md", "operation": "modify"}]}
    params = fake.sent("GET", "/workspaces/w/commits/diff")[0].url.params
    assert params["from"] == "c0" and params["to"] == "c1"
    changes = payload(await call(server, "ragflow_get_uncommitted_changes", {"workspace_id": "w"}))
    assert changes["changes"][0]["operation"] == "add"
    tree = payload(await call(server, "ragflow_get_commit_tree", {"workspace_id": "w", "commit_id": "c1"}))
    assert tree["id"] == "w"
    content = payload(
        await call(
            server,
            "ragflow_get_commit_file_content",
            {"workspace_id": "w", "commit_id": "c1", "file_id": "f", "max_chars": 3},
        )
    )
    assert content == {"content": "abc", "truncated": True}
    versions = payload(await call(server, "ragflow_get_file_versions", {"file_id": "f"}))
    assert versions == {"versions": [{"commit_id": "c1", "operation": "add"}]}
