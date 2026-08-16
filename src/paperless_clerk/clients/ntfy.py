from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from paperless_clerk.config import Settings


class NotificationError(RuntimeError):
    """A bounded ntfy delivery failure."""


class NtfyClient:
    """Publish small operational notifications through ntfy's JSON API."""

    def __init__(self, settings: Settings):
        self.url = settings.ntfy_url.rstrip("/")
        self.topic = settings.ntfy_topic
        token = settings.secret_value("ntfy_token")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10), headers=headers)

    async def close(self) -> None:
        await self.client.aclose()

    async def publish(
        self,
        *,
        title: str,
        message: str,
        priority: int = 4,
        tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not self.topic:
            raise NotificationError("An ntfy topic is required")
        payload = {
            "topic": self.topic,
            "title": title[:200],
            # ntfy.sh limits message bodies to 4096 bytes. At most 1000 Unicode
            # code points stays within that boundary even for four-byte text.
            "message": message[:1000],
            "priority": priority,
            "tags": list(tags),
        }
        try:
            response = await self.client.post(self.url, json=payload)
        except httpx.RequestError as exc:
            raise NotificationError(f"ntfy request failed: {exc}") from exc
        if response.status_code >= 400:
            raise NotificationError(f"ntfy returned {response.status_code}: {response.text[:500]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise NotificationError("ntfy returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise NotificationError("ntfy returned an invalid response")
        return body

    async def test_connection(self) -> dict[str, Any]:
        await self.publish(
            title="Paperless Clerk test",
            message="Notification delivery is configured correctly.",
            priority=3,
            tags=("white_check_mark",),
        )
        return {"message": "Test notification sent", "topic": self.topic}
