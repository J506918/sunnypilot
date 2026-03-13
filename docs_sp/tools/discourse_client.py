"""Minimal Discourse API client using only urllib (zero external deps).

Provides the 5 CRUD operations needed by the docs sync orchestrator:
1. get_category_id(slug)
2. find_topic_by_sync_id(sync_id)
3. create_topic(title, raw, category_id, tags)
4. update_post(post_id, raw, edit_reason)
5. first_post_id(topic_id)

Configuration via environment variables:
  DISCOURSE_URL      - Base URL (e.g. https://community.sunnypilot.ai)
  DISCOURSE_API_KEY  - API key with topic create/update permissions
  DISCOURSE_API_USER - Username for API requests (default: "system")
  DISCOURSE_CATEGORY - Category slug for documentation (default: "documentation")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiscourseConfig:
    """Immutable configuration for the Discourse API client."""

    base_url: str
    api_key: str
    api_user: str = "system"
    category_slug: str = "documentation"

    @classmethod
    def from_env(cls) -> DiscourseConfig:
        """Build config from environment variables.

        Raises:
            ValueError: If required env vars are missing.
        """
        base_url = os.environ.get("DISCOURSE_URL", "")
        api_key = os.environ.get("DISCOURSE_API_KEY", "")

        if not base_url:
            raise ValueError("DISCOURSE_URL environment variable is required")
        if not api_key:
            raise ValueError("DISCOURSE_API_KEY environment variable is required")

        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            api_user=os.environ.get("DISCOURSE_API_USER", "system"),
            category_slug=os.environ.get("DISCOURSE_CATEGORY", "documentation"),
        )


class DiscourseClient:
    """Discourse API client for docs sync operations."""

    def __init__(self, config: DiscourseConfig) -> None:
        self._config = config

    @property
    def config(self) -> DiscourseConfig:
        return self._config

    # ----- Public API -----

    def get_category_id(self, slug: str | None = None) -> int | None:
        """Look up a category ID by slug.

        Args:
            slug: Category slug. Defaults to config.category_slug.

        Returns:
            Category ID, or None if not found.
        """
        slug = slug or self._config.category_slug
        data = self._get(f"/c/{slug}/show.json")
        if data is None:
            return None
        return data.get("category", {}).get("id")

    def find_topic_by_sync_id(self, sync_id: str) -> dict[str, Any] | None:
        """Find an existing topic by its embedded sync ID comment.

        Searches for topics containing <!-- docs-sync-id: {sync_id} -->.

        Args:
            sync_id: The doc path used as sync identifier.

        Returns:
            Topic dict with at least 'id' key, or None if not found.
        """
        query = f"<!-- docs-sync-id: {sync_id} -->"
        encoded = urllib.parse.urlencode({"q": query})
        data = self._get(f"/search.json?{encoded}")
        if data is None:
            return None
        topics = data.get("topics", [])
        return topics[0] if topics else None

    def create_topic(
        self,
        title: str,
        raw: str,
        category_id: int,
        tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Create a new topic in the specified category.

        Args:
            title: Topic title.
            raw: Markdown body content.
            category_id: Discourse category ID.
            tags: Optional list of tags.

        Returns:
            Response dict with 'topic_id', 'id' (post ID), etc., or None on failure.
        """
        payload: dict[str, Any] = {
            "title": title,
            "raw": raw,
            "category": category_id,
        }
        if tags:
            payload["tags"] = tags
        return self._post("/posts.json", payload)

    def update_post(
        self,
        post_id: int,
        raw: str,
        edit_reason: str = "Documentation sync",
    ) -> dict[str, Any] | None:
        """Update an existing post's content.

        Args:
            post_id: The Discourse post ID to update.
            raw: New markdown body content.
            edit_reason: Reason shown in edit history.

        Returns:
            Response dict, or None on failure.
        """
        payload = {
            "post": {
                "raw": raw,
                "edit_reason": edit_reason,
            },
        }
        return self._put(f"/posts/{post_id}.json", payload)

    def first_post_id(self, topic_id: int) -> int | None:
        """Get the first post ID of a topic.

        Args:
            topic_id: The Discourse topic ID.

        Returns:
            Post ID of the first post, or None if not found.
        """
        data = self._get(f"/t/{topic_id}.json")
        if data is None:
            return None
        posts = data.get("post_stream", {}).get("posts", [])
        if not posts:
            return None
        return posts[0].get("id")

    # ----- HTTP helpers -----

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Api-Key": self._config.api_key,
            "Api-Username": self._config.api_user,
        }

    def _get(self, path: str) -> dict[str, Any] | None:
        url = self._config.base_url + path
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        return self._send(req)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = self._config.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        return self._send(req)

    def _put(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = self._config.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="PUT")
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            # Log but don't crash — caller decides how to handle None
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            print(f"  Discourse API error: {req.method} {req.full_url} -> {status}: {body}")
            return None
        except urllib.error.URLError as e:
            print(f"  Discourse connection error: {req.full_url} -> {e.reason}")
            return None
