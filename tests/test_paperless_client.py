import json

import httpx
import pytest

from paperless_clerk.clients.paperless import PaperlessClient
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
