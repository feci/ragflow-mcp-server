import base64
import json

import httpx
import pytest
from conftest import call, doc, make_server, ok, payload
from ragflow_kb_mcp.client import RagflowClient
from ragflow_kb_mcp.server import build_server
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
    states = iter([doc(status="SCHEDULED"), doc(status="RUNNING", progress=1), doc(status="COMPLETED", progress=1)])
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


async def test_list_datasets_filters(fake):
    fake.on("GET", "/datasets", ok([], total_datasets=0))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_list_datasets",
            {"dataset_id": "k1", "keywords": "infra", "parser_id": "general", "orderby": "update_time", "desc": False},
        )
    )
    assert result == {"total": 0, "datasets": []}
    params = fake.requests[0].url.params
    assert params["id"] == "k1" and params["keywords"] == "infra" and params["parser_id"] == "general"
    assert params["orderby"] == "update_time" and params["desc"] == "false"
    assert "include_parsing_status" not in params


async def test_get_chunk(fake):
    fake.on(
        "GET",
        "/datasets/k1/documents/d1/chunks/c1",
        ok({"id": "c1", "content_with_weight": "text", "important_kwd": ["k"], "available_int": 0}),
    )
    result = payload(
        await call(make_server(fake), "ragflow_get_chunk", {"dataset_id": "k1", "document_id": "d1", "chunk_id": "c1"})
    )
    assert result == {"id": "c1", "content": "text", "important_keywords": ["k"], "available": False}


async def test_set_chunks_availability(fake):
    fake.on("PATCH", "/datasets/k1/documents/d1/chunks", ok(True))
    server = make_server(fake)
    result = payload(
        await call(
            server,
            "ragflow_set_chunks_availability",
            {"dataset_id": "k1", "document_id": "d1", "chunk_ids": ["c1", "c2"], "available": False},
        )
    )
    assert result == {"chunk_ids": ["c1", "c2"], "available": False}
    assert json.loads(fake.sent("PATCH", "/datasets/k1/documents/d1/chunks")[0].content) == {
        "chunk_ids": ["c1", "c2"],
        "available": False,
    }
    empty = await call(
        server,
        "ragflow_set_chunks_availability",
        {"dataset_id": "k1", "document_id": "d1", "chunk_ids": [], "available": True},
    )
    assert empty.isError


@pytest.mark.parametrize(
    "action, run, delete",
    [("start", "1", False), ("rerun", "1", True), ("cancel", "2", False), ("clear", "0", True)],
)
async def test_ingest_documents_actions(fake, action, run, delete):
    fake.on("POST", "/documents/ingest", ok(True))
    result = payload(
        await call(make_server(fake), "ragflow_ingest_documents", {"document_ids": ["d1"], "action": action})
    )
    assert result == {"action": action, "document_ids": ["d1"]}
    assert json.loads(fake.sent("POST", "/documents/ingest")[0].content) == {
        "doc_ids": ["d1"],
        "run": run,
        "delete": delete,
        "apply_kb": False,
    }


async def test_path_ids_cannot_escape_the_endpoint(fake):
    server = make_server(fake)
    for bad in ("../chats", "a/b", "x?y=1", ""):
        result = await call(server, "ragflow_get_dataset", {"dataset_id": bad})
        assert result.isError
        text = result.content[0].text
        assert "refusing request path" in text or "should match pattern" in text, text
    assert not fake.requests


async def test_timeout_becomes_readable_tool_error():
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    client = RagflowClient("https://rag.example", "k", transport=httpx.MockTransport(slow))
    server = build_server(client)
    result = await call(server, "ragflow_get_dataset", {"dataset_id": "k1"})
    assert result.isError and "no answer within the client timeout" in result.content[0].text


async def test_url_upload_checks_the_names_ragflow_stores(fake):
    seen = []

    def documents(request):
        name = request.url.params.get("name")
        if name:
            seen.append(name)
            return ok({"docs": [doc("old", name="mypage.html")] if name == "mypage.html" else [], "total": 0})
        return ok({"docs": [doc("new", name="mypage.html")], "total": 1})

    fake.on("GET", "/datasets/k1/documents", documents)
    result = await call(
        make_server(fake), "ragflow_upload_document", {"dataset_id": "k1", "url": "https://e.x/p", "filename": "my page"}
    )
    assert result.isError and "already exists" in result.content[0].text
    assert seen == ["mypage", "mypage.pdf", "mypage.html"]


async def test_url_upload_of_other_types_explains_the_limit(fake):
    fake.on("GET", "/datasets/k1/documents", ok({"docs": [], "total": 0}))
    fake.on("POST", "/datasets/k1/documents", {"code": 102, "message": "this type of file has not been supported yet"})
    result = await call(
        make_server(fake),
        "ragflow_upload_document",
        {"dataset_id": "k1", "url": "https://e.x/README.md", "filename": "README.md"},
    )
    assert result.isError
    assert "only .pdf or .html" in result.content[0].text
