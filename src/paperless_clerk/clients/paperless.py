from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from paperless_clerk.config import Settings


class PaperlessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        ambiguous: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.ambiguous = ambiguous


class PaperlessClient:
    def __init__(self, settings: Settings):
        base = settings.paperless_url.rstrip("/")
        self.root_url = base[:-4] if base.endswith("/api") else base
        self.api_url = f"{self.root_url}/api"
        self.token = settings.paperless_token.get_secret_value()
        self.retries = settings.model_max_retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            verify=settings.paperless_verify_ssl,
            follow_redirects=True,
            headers={"Authorization": f"Token {self.token}"},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        retry: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        url = (
            path
            if path.startswith(("http://", "https://"))
            else f"{self.api_url}/{path.lstrip('/')}"
        )
        if path.startswith(("http://", "https://")):
            expected, actual = urlparse(self.api_url), urlparse(path)
            if (expected.scheme, expected.netloc) != (actual.scheme, actual.netloc):
                raise PaperlessError("Paperless pagination attempted to leave the configured host")
        attempts = self.retries + 1 if retry else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(method, url, **kwargs)
                if response.status_code < 400:
                    return response
                retryable = response.status_code in {408, 425, 429, 500, 502, 503, 504}
                detail = response.text[:800]
                if not retryable or attempt + 1 >= attempts:
                    raise PaperlessError(
                        f"Paperless {method} {urlparse(url).path} returned {response.status_code}: {detail}",
                        retryable=retryable,
                        status_code=response.status_code,
                    )
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    raise PaperlessError(
                        f"Paperless request failed: {exc}", retryable=True
                    ) from exc
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
        raise PaperlessError(f"Paperless request failed: {last_error}", retryable=True)

    async def test_connection(self) -> dict[str, Any]:
        if not self.token:
            raise PaperlessError("Paperless token is not configured")
        response = await self._request("GET", "documents/", params={"page_size": 1})
        body = response.json()
        return {"ok": True, "documents": body.get("count", 0), "url": self.root_url}

    async def get_document(self, document_id: int) -> dict[str, Any]:
        response = await self._request("GET", f"documents/{document_id}/")
        return response.json()

    async def update_document(
        self,
        document_id: int,
        payload: dict[str, Any],
        *,
        version_id: int | None = None,
    ) -> dict[str, Any]:
        params = {"version": version_id} if version_id is not None else None
        response = await self._request(
            "PATCH", f"documents/{document_id}/", params=params, json=payload
        )
        return response.json()

    async def upload_document_version(
        self,
        document_id: int,
        source: Path,
        *,
        filename: str,
        version_label: str,
    ) -> str:
        # A version upload queues Paperless consumption. Do not retry this POST:
        # a lost response could otherwise create two versions of the same file.
        try:
            with source.open("rb") as stream:
                response = await self._request(
                    "POST",
                    f"documents/{document_id}/update_version/",
                    retry=False,
                    files={"document": (filename, stream, "application/octet-stream")},
                    data={"version_label": version_label},
                )
        except PaperlessError as exc:
            if exc.retryable:
                raise PaperlessError(
                    "Paperless version upload had an ambiguous outcome. Clerk will not upload "
                    "again automatically because Paperless may already be creating the version; "
                    "inspect the document's version history before retrying.",
                    ambiguous=True,
                ) from exc
            raise
        try:
            body = response.json()
        except ValueError as exc:
            raise PaperlessError(
                "Paperless accepted the version upload but returned an invalid task response. "
                "Clerk will not upload again automatically; inspect the document's version "
                "history before retrying.",
                ambiguous=True,
            ) from exc
        task_id = body.get("task_id") if isinstance(body, dict) else body
        if not isinstance(task_id, str) or not task_id.strip():
            raise PaperlessError(
                "Paperless accepted the version upload but did not return a task ID. Clerk "
                "will not upload again automatically; inspect the document's version history "
                "before retrying.",
                ambiguous=True,
            )
        return task_id.strip()

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "tasks/",
            params={"task_id": task_id, "page_size": 1},
            headers={"Accept": "application/json; version=10"},
        )
        body = response.json()
        tasks = body.get("results", []) if isinstance(body, dict) else body
        if not isinstance(tasks, list):
            raise PaperlessError("Paperless returned an invalid task-status response")
        return tasks[0] if tasks and isinstance(tasks[0], dict) else None

    async def wait_for_task(
        self,
        task_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            task = await self.get_task(task_id)
            status = str((task or {}).get("status") or "").casefold()
            if status in {"success", "failure", "revoked"}:
                return task or {}
            if asyncio.get_running_loop().time() >= deadline:
                raise PaperlessError(
                    f"Paperless version task {task_id} did not finish within "
                    f"{timeout_seconds:g} seconds",
                    retryable=True,
                )
            await asyncio.sleep(poll_interval_seconds)

    async def update_version_label(
        self, document_id: int, version_id: int, version_label: str
    ) -> dict[str, Any]:
        response = await self._request(
            "PATCH",
            f"documents/{document_id}/versions/{version_id}/",
            json={"version_label": version_label},
        )
        return response.json()

    async def download_document(self, document_id: int, destination: Path) -> str:
        url = f"{self.api_url}/documents/{document_id}/download/"
        hasher = hashlib.sha256()
        try:
            async with self.client.stream("GET", url) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")[:800]
                    raise PaperlessError(
                        f"Paperless download returned {response.status_code}: {body}",
                        retryable=response.status_code >= 500,
                        status_code=response.status_code,
                    )
                with destination.open("wb") as target:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        hasher.update(chunk)
                        target.write(chunk)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise PaperlessError(f"Paperless download failed: {exc}", retryable=True) from exc
        return hasher.hexdigest()

    async def thumbnail(self, document_id: int) -> tuple[bytes, str]:
        response = await self._request("GET", f"documents/{document_id}/thumb/")
        return response.content, response.headers.get("content-type", "image/webp")

    async def list_resource(
        self, resource: str, *, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        url = f"{self.api_url}/{resource.strip('/')}/"
        query = {"page_size": 100, **(params or {})}
        items: list[dict[str, Any]] = []
        while url:
            response = await self._request("GET", url, params=query)
            query = None
            body = response.json()
            items.extend(body.get("results", []))
            url = body.get("next")
        return items

    async def catalog(self) -> dict[str, list[dict[str, Any]]]:
        tags, correspondents, document_types, custom_fields = await asyncio.gather(
            self.list_resource("tags"),
            self.list_resource("correspondents"),
            self.list_resource("document_types"),
            self.list_resource("custom_fields"),
        )
        return {
            "tags": tags,
            "correspondents": correspondents,
            "document_types": document_types,
            "custom_fields": custom_fields,
        }

    async def create_entity(self, resource: str, name: str, **extra: Any) -> dict[str, Any]:
        payload = {"name": name, **extra}
        response = await self._request("POST", f"{resource.strip('/')}/", json=payload, retry=False)
        return response.json()

    async def ensure_tag(self, name: str) -> dict[str, Any]:
        tags = await self.list_resource("tags", params={"name__iexact": name})
        exact = next(
            (tag for tag in tags if str(tag.get("name", "")).casefold() == name.casefold()), None
        )
        if exact:
            return exact
        try:
            return await self.create_entity("tags", name)
        except PaperlessError as exc:
            if exc.status_code != 400:
                raise
            tags = await self.list_resource("tags", params={"name__iexact": name})
            exact = next(
                (tag for tag in tags if str(tag.get("name", "")).casefold() == name.casefold()),
                None,
            )
            if exact:
                return exact
            raise

    async def ensure_entity(self, resource: str, name: str) -> dict[str, Any]:
        if resource == "tags":
            return await self.ensure_tag(name)
        matches = await self.list_resource(resource, params={"name__iexact": name})
        exact = next(
            (item for item in matches if str(item.get("name", "")).casefold() == name.casefold()),
            None,
        )
        if exact:
            return exact
        try:
            return await self.create_entity(resource, name)
        except PaperlessError as exc:
            if exc.status_code != 400:
                raise
            matches = await self.list_resource(resource, params={"name__iexact": name})
            exact = next(
                (
                    item
                    for item in matches
                    if str(item.get("name", "")).casefold() == name.casefold()
                ),
                None,
            )
            if exact:
                return exact
            raise

    async def list_documents(
        self, *, page_size: int, page: int = 1, tag_name: str = ""
    ) -> tuple[list[dict[str, Any]], bool]:
        params: dict[str, Any] = {
            "page_size": page_size,
            "page": page,
            "ordering": "-modified",
            "fields": "id,title,modified,added,tags",
        }
        if tag_name:
            tags = await self.list_resource("tags", params={"name__iexact": tag_name})
            match = next(
                (tag for tag in tags if tag.get("name", "").casefold() == tag_name.casefold()), None
            )
            if not match:
                return [], False
            params["tags__id__all"] = match["id"]
        response = await self._request("GET", "documents/", params=params)
        body = response.json()
        return body.get("results", []), bool(body.get("next"))

    async def create_custom_field(self, name: str, data_type: str) -> dict[str, Any]:
        return await self.create_entity("custom_fields", name, data_type=data_type, extra_data={})
