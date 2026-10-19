import base64
import json

import httpx
import pytest
from conftest import call, doc, make_server, ok, payload
from mcp.shared.memory import create_connected_server_and_client_session


async def test_tools_listed_and_disabled(fake):
    server = make_server(fake, disabled_tools=frozenset({"ragflow_retrieval"}))
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        names = {t.name for t in (await session.list_tools()).tools}
    assert "ragflow_upload_document" in names
    assert "ragflow_retrieval" not in names


async def test_auth_header_and_base_path(fake):
    fake.on("GET", "/datasets", ok([{"id": "k1", "name": "KB"}], total_datasets=1))
    result = payload(await call(make_server(fake), "ragflow_list_datasets", {}))
    assert result == {"total": 1, "datasets": [{"id": "k1", "name": "KB"}]}
    request = fake.requests[0]
    assert request.headers["authorization"] == "Bearer secret-key"
    assert request.url.path == "/api/v1/datasets"


async def test_create_dataset_sets_parse_type_and_description(fake):
    fake.on("POST", "/datasets", ok({"id": "k1", "name": "KB"}))
    fake.on("PUT", "/datasets/k1", ok({"id": "k1", "name": "KB", "description": "d"}))
    server = make_server(fake, default_embedding_model="emb@factory")
    result = payload(
        await call(server, "ragflow_create_dataset", {"name": "KB", "description": "d"})
    )
    assert result["description"] == "d"
    created = json.loads(fake.sent("POST", "/datasets")[0].content)
    assert created == {
        "name": "KB",
        "parser_id": "general",
        "parse_type": 1,
        "embedding_model": "emb@factory",
    }
    assert json.loads(fake.sent("PUT", "/datasets/k1")[0].content) == {"description": "d"}


async def test_upload_text_parses_and_waits(fake):
    states = iter([doc(status="SCHEDULED"), doc(status="RUNNING"), doc(status="COMPLETED", progress=1)])
    listing = {"n": 0}

    def documents(request):
        if request.url.params.get("name"):
            return ok({"docs": [], "total": 0})
        listing["n"] += 1
        return ok({"docs": [next(states)], "total": 1})

    fake.on("GET", "/datasets/k1/documents", documents)
    fake.on("POST", "/datasets/k1/documents", ok([doc()]))
    fake.on("POST", "/datasets/k1/chunks", ok(None))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_upload_document",
            {"dataset_id": "k1", "filename": "note.md", "text": "# Hi", "wait_seconds": 30},
        )
    )
    upload = fake.sent("POST", "/datasets/k1/documents")[0]
    assert b'filename="note.md"' in upload.content and b"# Hi" in upload.content
    assert json.loads(fake.sent("POST", "/datasets/k1/chunks")[0].content) == {
        "document_ids": ["d1"]
    }
    assert result["document"]["ingestion_status"] == "COMPLETED"
    assert result["still_parsing"] is False
    assert "parser_config" not in result["document"]
    assert listing["n"] == 3


async def test_upload_base64_without_parse(fake):
    fake.on("GET", "/datasets/k1/documents", ok({"docs": [doc(name="a.pdf")], "total": 1}))
    fake.on("POST", "/datasets/k1/documents", ok([doc(name="a.pdf")]))
    payload(
        await call(
            make_server(fake),
            "ragflow_upload_document",
            {
                "dataset_id": "k1",
                "filename": "b.pdf",
                "content_base64": base64.b64encode(b"%PDF-1.7").decode(),
                "parse": False,
            },
        )
    )
    assert b"%PDF-1.7" in fake.sent("POST", "/datasets/k1/documents")[0].content
    assert not fake.sent("POST", "/datasets/k1/chunks")


@pytest.mark.parametrize(
    "args, message",
    [
        ({"filename": "a.md"}, "exactly one"),
        ({"filename": "a.md", "text": "x", "url": "https://e.x/a"}, "exactly one"),
        ({"filename": "note", "text": "x"}, "extension"),
        ({"text": "x"}, "filename is required"),
        ({"filename": "a.bin", "content_base64": "@@@"}, "not valid base64"),
        ({"url": "ftp://e.x/a.pdf"}, "http or https"),
        ({"file_path": "/etc/hosts"}, "disabled"),
    ],
)
async def test_upload_rejects_bad_input(fake, args, message):
    result = await call(make_server(fake), "ragflow_upload_document", {"dataset_id": "k1", **args})
    assert result.isError
    assert message in result.content[0].text
    assert not fake.requests


async def test_upload_duplicate_name_needs_replace(fake):
    fake.on("GET", "/datasets/k1/documents", ok({"docs": [doc("old")], "total": 1}))
    result = await call(
        make_server(fake),
        "ragflow_upload_document",
        {"dataset_id": "k1", "filename": "note.md", "text": "x"},
    )
    assert result.isError and "already exists" in result.content[0].text
    assert not fake.sent("POST", "/datasets/k1/documents")


async def test_upload_replace_deletes_old_and_restores_name(fake):
    def documents(request):
        if request.url.params.get("name"):
            return ok({"docs": [doc("old")], "total": 1})
        return ok({"docs": [doc("new")], "total": 1})

    fake.on("GET", "/datasets/k1/documents", documents)
    fake.on("POST", "/datasets/k1/documents", ok([doc("new", name="note(1).md")]))
    fake.on("DELETE", "/datasets/k1/documents", ok(None))
    fake.on("PATCH", "/datasets/k1/documents/new", ok(doc("new")))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_upload_document",
            {
                "dataset_id": "k1",
                "filename": "note.md",
                "text": "x",
                "replace_existing": True,
                "parse": False,
                "meta_fields": {"source": "test"},
            },
        )
    )
    assert result["replaced"] == ["old"]
    assert json.loads(fake.sent("DELETE", "/datasets/k1/documents")[0].content) == {"ids": ["old"]}
    patches = [json.loads(r.content) for r in fake.sent("PATCH", "/datasets/k1/documents/new")]
    assert patches == [{"name": "note.md"}, {"meta_fields": {"source": "test"}}]


async def test_upload_url_uses_ragflow_web_fetch(fake):
    fake.on("POST", "/datasets/k1/documents", ok(doc(name="report.pdf")))
    fake.on("POST", "/datasets/k1/chunks", ok(None))
    fake.on("GET", "/datasets/k1/documents", lambda r: ok({"docs": [] if r.url.params.get("name") else [doc(name="report.pdf")]}))
    payload(
        await call(
            make_server(fake),
            "ragflow_upload_document",
            {"dataset_id": "k1", "url": "https://example.com/files/report.pdf"},
        )
    )
    upload = fake.sent("POST", "/datasets/k1/documents")[0]
    assert upload.url.params["type"] == "web"
    assert upload.content == b"name=report.pdf&url=https%3A%2F%2Fexample.com%2Ffiles%2Freport.pdf"


async def test_upload_file_path_respects_roots(fake, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "a.txt").write_text("hello")
    (tmp_path / "secret.txt").write_text("nope")
    (allowed / "link.txt").symlink_to(tmp_path / "secret.txt")
    fake.on("GET", "/datasets/k1/documents", lambda r: ok({"docs": [] if r.url.params.get("name") else [doc(name="a.txt")]}))
    fake.on("POST", "/datasets/k1/documents", ok([doc(name="a.txt")]))
    server = make_server(fake, upload_roots=(allowed.resolve(),))

    good = payload(
        await call(server, "ragflow_upload_document", {"dataset_id": "k1", "file_path": str(allowed / "a.txt"), "parse": False})
    )
    assert good["document"]["name"] == "a.txt"
    for path in (tmp_path / "secret.txt", allowed / "link.txt", allowed / ".." / "secret.txt"):
        result = await call(server, "ragflow_upload_document", {"dataset_id": "k1", "file_path": str(path)})
        assert result.isError and "outside the allowed" in result.content[0].text
    assert len(fake.sent("POST", "/datasets/k1/documents")) == 1


async def test_ragflow_error_code_becomes_tool_error(fake):
    fake.on("DELETE", "/datasets", {"code": 102, "message": "lacks permission"})
    result = await call(make_server(fake), "ragflow_delete_datasets", {"dataset_ids": ["k1"]})
    assert result.isError
    assert "RAGFlow code 102: lacks permission" in result.content[0].text
    assert "secret-key" not in result.content[0].text


async def test_update_metadata_requires_selector_or_confirmation(fake):
    fake.on("POST", "/datasets/k1/metadata/update", ok({"updated": 2, "matched_docs": 2}))
    server = make_server(fake)
    refused = await call(
        server, "ragflow_update_metadata", {"dataset_id": "k1", "updates": [{"key": "a", "value": "b"}]}
    )
    assert refused.isError and "all_documents" in refused.content[0].text
    done = payload(
        await call(
            server,
            "ragflow_update_metadata",
            {"dataset_id": "k1", "document_ids": ["d1"], "deletes": [{"key": "old"}]},
        )
    )
    assert done == {"updated": 2, "matched_docs": 2}
    body = json.loads(fake.sent("POST", "/datasets/k1/metadata/update")[0].content)
    assert body == {"updates": [], "deletes": [{"key": "old"}], "selector": {"document_ids": ["d1"]}}


async def test_list_chunks_trims_fields(fake):
    fake.on(
        "GET",
        "/datasets/k1/documents/d1/chunks",
        ok(
            {
                "total": 1,
                "doc": doc(status="COMPLETED", progress=1),
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "content_with_weight": "text",
                        "important_kwd": ["k"],
                        "question_kwd": [],
                        "available": True,
                        "positions": None,
                        "entities_kwd": [],
                    }
                ],
            }
        ),
    )
    result = payload(
        await call(make_server(fake), "ragflow_list_chunks", {"dataset_id": "k1", "document_id": "d1"})
    )
    assert result["chunks"] == [
        {"id": "c1", "content": "text", "important_keywords": ["k"], "available": True}
    ]
    assert result["document"]["ingestion_status"] == "COMPLETED"


async def test_failed_document_shows_progress_message(fake):
    failed = doc(status="FAILED", progress=-1)
    failed["progress_msg"] = "Task failed: unsupported file"
    fake.on("GET", "/datasets/k1/documents", ok({"docs": [failed], "total": 1}))
    result = payload(await call(make_server(fake), "ragflow_list_documents", {"dataset_id": "k1"}))
    assert result["documents"][0]["progress_msg"] == "Task failed: unsupported file"


async def test_download_text_and_binary(fake):
    fake.on(
        "GET",
        "/datasets/k1/documents/d1",
        httpx.Response(200, content=b"abcdef", headers={"content-type": "text/markdown"}),
    )
    fake.on(
        "GET",
        "/datasets/k1/documents/d2",
        httpx.Response(200, content=b"\xff\xfe\x00", headers={"content-type": "application/pdf"}),
    )
    server = make_server(fake)
    text = payload(
        await call(server, "ragflow_download_document", {"dataset_id": "k1", "document_id": "d1", "max_chars": 3})
    )
    assert text == {"content_type": "text/markdown", "size": 6, "text": "abc", "truncated": True}
    binary = payload(await call(server, "ragflow_download_document", {"dataset_id": "k1", "document_id": "d2"}))
    assert binary["text"] is None and binary["size"] == 3


async def test_name_filter_without_match_is_empty(fake):
    fake.on(
        "GET",
        "/datasets/k1/documents",
        {"code": 102, "data": {"total": 0, "docs": []}, "message": "you don't own the document x.md"},
    )
    result = payload(await call(make_server(fake), "ragflow_list_documents", {"dataset_id": "k1", "name": "x.md"}))
    assert result == {"total": 0, "documents": []}
