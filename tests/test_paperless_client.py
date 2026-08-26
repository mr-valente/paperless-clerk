import json
from pathlib import Path

import httpx
import pytest

from paperless_clerk.clients.paperless import PaperlessClient, PaperlessError
from paperless_clerk.config import Settings


@pytest.mark.asyncio
async def test_paginated_vocabulary_and_document_patch_use_paperless_api_contract() -> None:
    requests: list[tuple[str, str]] = []
    patch_body: dict | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal patch_body
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/paperless/api/tags/":
            if request.url.params.get("page") == "2":
                return httpx.Response(
                    200,
                    json={"results": [{"id": 2, "name": "Insurance"}], "next": None},
                )
            return httpx.Response(
                200,
                json={
                    "results": [{"id": 1, "name": "Medical"}],
                    "next": "http://paperless.local/paperless/api/tags/?page=2",
                },
            )
        if request.method == "PATCH" and request.url.path == "/paperless/api/documents/44/":
            patch_body = json.loads(request.content)
            return httpx.Response(200, json={"id": 44, **patch_body})
        return httpx.Response(404)

    client = PaperlessClient(
        Settings(paperless_url="http://paperless.local/paperless", paperless_token="secret")
    )
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Token secret"},
    )

    tags = await client.list_resource("tags")
    updated = await client.update_document(44, {"content": "complete OCR", "tags": [1, 2]})
    await client.close()

    assert [tag["id"] for tag in tags] == [1, 2]
    assert patch_body == {"content": "complete OCR", "tags": [1, 2]}
    assert updated["id"] == 44
    assert requests == [
        ("GET", "/paperless/api/tags/"),
        ("GET", "/paperless/api/tags/"),
        ("PATCH", "/paperless/api/documents/44/"),
    ]


@pytest.mark.asyncio
async def test_document_discovery_exposes_pagination_state() -> None:
    seen_page = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_page
        seen_page = request.url.params.get("page")
        return httpx.Response(
            200,
            json={
                "results": [{"id": 55, "title": "Older document"}],
                "next": "http://paperless.local/api/documents/?page=3",
            },
        )

    client = PaperlessClient(Settings(paperless_url="http://paperless.local"))
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    documents, has_next = await client.list_documents(page_size=25, page=2)
    await client.close()

    assert seen_page == "2"
    assert documents[0]["id"] == 55
    assert has_next is True


@pytest.mark.asyncio
async def test_version_upload_task_poll_and_explicit_content_patch(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/update_version/"):
            assert b'name="version_label"' in request.content
            assert b"Paperless Clerk OCR" in request.content
            assert b'filename="statement.pdf"' in request.content
            return httpx.Response(200, json="task-uuid")
        if request.method == "GET" and request.url.path.endswith("/tasks/"):
            assert request.headers["accept"] == "application/json; version=10"
            assert request.url.params["task_id"] == "task-uuid"
            return httpx.Response(
                200,
                json={
                    "count": 1,
                    "results": [
                        {
                            "task_id": "task-uuid",
                            "status": "success",
                            "result_data": {"document_id": 144},
                        }
                    ],
                },
            )
        if request.method == "PATCH" and request.url.path.endswith("/documents/44/"):
            assert request.url.params["version"] == "144"
            return httpx.Response(200, json={"id": 44, "content": "Clerk OCR"})
        return httpx.Response(404)

    source = tmp_path / "statement.pdf"
    source.write_bytes(b"%PDF stable source")
    client = PaperlessClient(Settings(paperless_url="http://paperless.local"))
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    task_id = await client.upload_document_version(
        44,
        source,
        filename="statement.pdf",
        version_label="Paperless Clerk OCR",
    )
    task = await client.wait_for_task(task_id, timeout_seconds=1, poll_interval_seconds=0)
    updated = await client.update_document(44, {"content": "Clerk OCR"}, version_id=144)
    await client.close()

    assert task["result_data"]["document_id"] == 144
    assert updated["content"] == "Clerk OCR"
    assert [request.method for request in requests] == ["POST", "GET", "PATCH"]


@pytest.mark.asyncio
async def test_version_upload_network_failure_is_ambiguous_and_not_retried(
    tmp_path: Path,
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("response lost", request=request)

    source = tmp_path / "statement.pdf"
    source.write_bytes(b"%PDF stable source")
    client = PaperlessClient(Settings(paperless_url="http://paperless.local", model_max_retries=3))
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(PaperlessError, match="ambiguous outcome") as raised:
        await client.upload_document_version(
            44,
            source,
            filename="statement.pdf",
            version_label="Paperless Clerk OCR",
        )
    await client.close()

    assert raised.value.ambiguous is True
    assert raised.value.retryable is False
    assert attempts == 1
