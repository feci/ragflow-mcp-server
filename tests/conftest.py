import json

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from ragflow_kb_mcp.client import RagflowClient
from ragflow_kb_mcp.server import build_server


class FakeRagflow:
    """Records requests and answers them from a route table."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], object] = {}

    def on(self, method, path, response):
        self.routes[(method, path)] = response

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1")
        answer = self.routes.get((request.method, path))
        if answer is None:
            return httpx.Response(404, json={"code": 404, "message": f"no route {path}"})
        if callable(answer):
            answer = answer(request)
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def sent(self, method, path):
        return [
            r
            for r in self.requests
            if r.method == method and r.url.path.removeprefix("/api/v1") == path
        ]


def ok(data=None, **extra):
    return {"code": 0, "data": data, **extra}


def doc(doc_id="d1", name="note.md", status="UNSTART", progress=0):
    return {
        "id": doc_id,
        "name": name,
        "suffix": name.rsplit(".", 1)[-1],
        "size": 10,
        "ingestion_status": status,
        "progress": progress,
        "chunk_count": 0,
        "status": "1",
        "parser_config": {"huge": "x" * 100},
    }


@pytest.fixture
def fake():
    return FakeRagflow()


def make_server(fake, **kwargs):
    client = RagflowClient(
        "https://rag.example", "secret-key", transport=httpx.MockTransport(fake.handler)
    )
    return build_server(client, poll_interval=0, **kwargs)


async def call(server, name, args):
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        return await session.call_tool(name, args)


def payload(result):
    assert not result.isError, result.content
    return json.loads(result.content[0].text)
