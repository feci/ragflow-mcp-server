import json

import httpx
from conftest import call, make_server, ok, payload


def sse(*events):
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


APP = {
    "id": "s1",
    "name": "Docs",
    "description": "d",
    "tenant_id": "t",
    "created_by": "u",
    "create_time": 1,
    "update_time": 2,
    "search_config": {"kb_ids": ["k1"], "top_k": 64},
}


async def test_list_search_apps(fake):
    fake.on("GET", "/searches", ok({"total": 1, "search_apps": [APP]}))
    result = payload(
        await call(make_server(fake), "ragflow_list_search_apps", {"keywords": "Do", "desc": False})
    )
    assert result["total"] == 1
    assert result["search_apps"][0] == {
        k: APP[k]
        for k in ("id", "name", "description", "create_time", "update_time", "search_config")
    }
    params = fake.requests[0].url.params
    assert (
        params["keywords"] == "Do"
        and params["desc"] == "false"
        and params["orderby"] == "create_time"
    )


async def test_create_search_app_sets_config_with_update(fake):
    fake.on("POST", "/searches", ok({"search_id": "s1"}))
    fake.on("PUT", "/searches/s1", ok(APP))
    result = payload(
        await call(
            make_server(fake),
            "ragflow_create_search_app",
            {
                "name": "Docs",
                "description": "d",
                "dataset_ids": ["k1"],
                "vector_similarity_weight": 0.7,
                "chat_model": "m@f",
                "search_config": {"llm_setting": {"temperature": 0.1}},
            },
        )
    )
    assert result["id"] == "s1"
    assert json.loads(fake.sent("POST", "/searches")[0].content) == {
        "name": "Docs",
        "description": "d",
    }
    assert json.loads(fake.sent("PUT", "/searches/s1")[0].content) == {
        "name": "Docs",
        "search_config": {
            "llm_setting": {"temperature": 0.1},
            "kb_ids": ["k1"],
            "vector_similarity_weight": 0.7,
            "chat_id": "m@f",
        },
    }


async def test_create_search_app_without_config_reads_back(fake):
    fake.on("POST", "/searches", ok({"search_id": "s1"}))
    fake.on("GET", "/searches/s1", ok(APP))
    payload(await call(make_server(fake), "ragflow_create_search_app", {"name": "Docs"}))
    assert not fake.sent("PUT", "/searches/s1")


async def test_update_search_app_fetches_name_when_missing(fake):
    fake.on("GET", "/searches/s1", ok(APP))
    fake.on("PUT", "/searches/s1", ok(APP))
    server = make_server(fake)
    payload(await call(server, "ragflow_update_search_app", {"search_id": "s1", "top_k": 10}))
    assert json.loads(fake.sent("PUT", "/searches/s1")[0].content) == {
        "name": "Docs",
        "search_config": {"top_k": 10},
    }
    refused = await call(server, "ragflow_update_search_app", {"search_id": "s1"})
    assert refused.isError and "at least one field" in refused.content[0].text


async def test_delete_search_app(fake):
    fake.on("DELETE", "/searches/s1", ok(True))
    assert payload(
        await call(make_server(fake), "ragflow_delete_search_app", {"search_id": "s1"})
    ) == {"deleted": "s1"}


async def test_completion_returns_final_answer_and_references(fake):
    chunk = {
        "id": "c1",
        "content": "Proxmox runs the VMs.",
        "document_id": "d1",
        "document_name": "infra.md",
        "dataset_id": "k1",
        "similarity": 0.8,
        "vector_similarity": 0.8,
    }
    fake.on(
        "POST",
        "/searches/s1/completions",
        sse(
            {"code": 0, "message": "", "data": {"answer": "Prox", "reference": {}, "final": False}},
            {"code": 0, "message": "", "data": {"answer": "mox.", "reference": {}, "final": False}},
            {
                "code": 0,
                "message": "",
                "data": {
                    "answer": "Proxmox.",
                    "reference": {
                        "chunks": [chunk, chunk],
                        "doc_aggs": [{"doc_id": "d1", "doc_name": "infra.md", "count": 2}],
                    },
                    "final": True,
                },
            },
            {"code": 0, "message": "", "data": True},
        ),
    )
    result = payload(
        await call(
            make_server(fake),
            "ragflow_search_app_completion",
            {"search_id": "s1", "question": "Which hypervisor?", "max_references": 1},
        )
    )
    assert result["answer"] == "Proxmox."
    assert result["references"] == [
        {
            "index": 0,
            "id": "c1",
            "content": "Proxmox runs the VMs.",
            "document_id": "d1",
            "document_name": "infra.md",
            "dataset_id": "k1",
            "similarity": 0.8,
        }
    ]
    assert result["references_omitted"] == 1
    assert result["documents"] == [{"doc_id": "d1", "doc_name": "infra.md", "count": 2}]
    assert json.loads(fake.sent("POST", "/searches/s1/completions")[0].content) == {
        "question": "Which hypervisor?"
    }


async def test_completion_stream_error_becomes_tool_error(fake):
    fake.on(
        "POST",
        "/searches/s1/completions",
        sse(
            {
                "code": 500,
                "message": "chat model not configured",
                "data": {"answer": "**ERROR**", "reference": []},
            }
        ),
    )
    result = await call(
        make_server(fake), "ragflow_search_app_completion", {"search_id": "s1", "question": "q"}
    )
    assert result.isError and "chat model not configured" in result.content[0].text


async def test_completion_json_error_before_stream(fake):
    fake.on("POST", "/searches/s1/completions", {"code": 102, "message": "`kb_ids` is required"})
    result = await call(
        make_server(fake), "ragflow_search_app_completion", {"search_id": "s1", "question": "q"}
    )
    assert result.isError and "kb_ids" in result.content[0].text


async def test_health_reports_components_even_on_500(fake):
    body = {"db": "ok", "redis": "down", "status": "nok"}
    fake.on("GET", "/system/healthz", httpx.Response(500, json=body))
    result = payload(await call(make_server(fake), "ragflow_health", {}))
    assert result == {"healthy": False, "components": body}
    fake.on("GET", "/system/healthz", {"db": "ok", "status": "ok"})
    assert payload(await call(make_server(fake), "ragflow_health", {}))["healthy"] is True
