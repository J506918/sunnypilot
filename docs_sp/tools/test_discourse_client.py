#!/usr/bin/env python3
"""Tests for the Discourse API client (fully mocked, no live requests).

Run: python3 docs_sp/tools/test_discourse_client.py
"""

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from discourse_client import DiscourseClient, DiscourseConfig

TEST_CONFIG = DiscourseConfig(
    base_url="https://community.sunnypilot.ai",
    api_key="test-api-key-123",
    api_user="docs-bot",
    category_slug="documentation",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_response(data: dict, status: int = 200) -> MagicMock:
    """Create a mock urllib response with JSON body."""
    body = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def mock_http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    """Create a mock HTTPError."""
    return urllib.error.HTTPError(
        url="https://community.sunnypilot.ai/test",
        code=status,
        msg=f"HTTP {status}",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# DiscourseConfig tests
# ---------------------------------------------------------------------------


def test_config_from_env():
    env = {
        "DISCOURSE_URL": "https://forum.example.com/",
        "DISCOURSE_API_KEY": "secret-key",
        "DISCOURSE_API_USER": "bot",
        "DISCOURSE_CATEGORY": "docs",
    }
    with patch.dict("os.environ", env, clear=False):
        config = DiscourseConfig.from_env()

    assert config.base_url == "https://forum.example.com"  # trailing slash stripped
    assert config.api_key == "secret-key"
    assert config.api_user == "bot"
    assert config.category_slug == "docs"
    print("  PASS: config_from_env")


def test_config_from_env_defaults():
    env = {
        "DISCOURSE_URL": "https://forum.example.com",
        "DISCOURSE_API_KEY": "key",
    }
    with patch.dict("os.environ", env, clear=False):
        # Remove optional vars if present
        with patch.dict("os.environ", {"DISCOURSE_API_USER": "", "DISCOURSE_CATEGORY": ""}, clear=False):
            pass
        config = DiscourseConfig.from_env()

    assert config.api_user in ("system", os.environ.get("DISCOURSE_API_USER", "system"))
    print("  PASS: config_from_env_defaults")


def test_config_missing_url():
    env = {"DISCOURSE_API_KEY": "key"}
    with patch.dict("os.environ", env, clear=True):
        try:
            DiscourseConfig.from_env()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "DISCOURSE_URL" in str(e)
    print("  PASS: config_missing_url")


def test_config_missing_api_key():
    env = {"DISCOURSE_URL": "https://forum.example.com"}
    with patch.dict("os.environ", env, clear=True):
        try:
            DiscourseConfig.from_env()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "DISCOURSE_API_KEY" in str(e)
    print("  PASS: config_missing_api_key")


def test_config_immutable():
    try:
        TEST_CONFIG.base_url = "https://other.com"  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass
    print("  PASS: config_immutable")


# ---------------------------------------------------------------------------
# get_category_id
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_get_category_id_found(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"category": {"id": 42, "slug": "documentation"}})
    client = DiscourseClient(TEST_CONFIG)

    result = client.get_category_id("documentation")

    assert result == 42
    call_args = mock_urlopen.call_args[0][0]
    assert "/c/documentation/show.json" in call_args.full_url
    assert call_args.get_header("Api-key") == "test-api-key-123"
    assert call_args.get_header("Api-username") == "docs-bot"
    print("  PASS: get_category_id_found")


@patch("urllib.request.urlopen")
def test_get_category_id_not_found(mock_urlopen: MagicMock):
    mock_urlopen.side_effect = mock_http_error(404, "Not Found")
    client = DiscourseClient(TEST_CONFIG)

    result = client.get_category_id("nonexistent")
    assert result is None
    print("  PASS: get_category_id_not_found")


@patch("urllib.request.urlopen")
def test_get_category_id_uses_default_slug(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"category": {"id": 7}})
    client = DiscourseClient(TEST_CONFIG)

    client.get_category_id()  # no arg — uses config.category_slug

    call_args = mock_urlopen.call_args[0][0]
    assert "/c/documentation/show.json" in call_args.full_url
    print("  PASS: get_category_id_uses_default_slug")


# ---------------------------------------------------------------------------
# find_topic_by_sync_id
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_find_topic_found(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({
        "topics": [{"id": 101, "title": "ICBM Docs"}],
    })
    client = DiscourseClient(TEST_CONFIG)

    result = client.find_topic_by_sync_id("features/cruise/icbm.md")

    assert result is not None
    assert result["id"] == 101
    call_args = mock_urlopen.call_args[0][0]
    assert "/search.json?" in call_args.full_url
    assert "docs-sync-id" in call_args.full_url
    print("  PASS: find_topic_found")


@patch("urllib.request.urlopen")
def test_find_topic_not_found(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"topics": []})
    client = DiscourseClient(TEST_CONFIG)

    result = client.find_topic_by_sync_id("nonexistent.md")
    assert result is None
    print("  PASS: find_topic_not_found")


@patch("urllib.request.urlopen")
def test_find_topic_api_error(mock_urlopen: MagicMock):
    mock_urlopen.side_effect = mock_http_error(500, "Internal Server Error")
    client = DiscourseClient(TEST_CONFIG)

    result = client.find_topic_by_sync_id("features/icbm.md")
    assert result is None
    print("  PASS: find_topic_api_error")


# ---------------------------------------------------------------------------
# create_topic
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_create_topic_success(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({
        "id": 501,
        "topic_id": 201,
        "topic_slug": "icbm-sunnypilot-docs",
    })
    client = DiscourseClient(TEST_CONFIG)

    result = client.create_topic(
        title="ICBM - sunnypilot Docs",
        raw="# ICBM\n\nDoc content here.",
        category_id=42,
        tags=["docs", "auto-sync"],
    )

    assert result is not None
    assert result["topic_id"] == 201

    call_args = mock_urlopen.call_args[0][0]
    assert call_args.method == "POST"
    assert "/posts.json" in call_args.full_url

    sent_payload = json.loads(call_args.data.decode("utf-8"))
    assert sent_payload["title"] == "ICBM - sunnypilot Docs"
    assert sent_payload["category"] == 42
    assert sent_payload["tags"] == ["docs", "auto-sync"]
    assert "ICBM" in sent_payload["raw"]
    print("  PASS: create_topic_success")


@patch("urllib.request.urlopen")
def test_create_topic_no_tags(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"id": 502, "topic_id": 202})
    client = DiscourseClient(TEST_CONFIG)

    client.create_topic(title="Test", raw="body", category_id=1)

    sent_payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "tags" not in sent_payload
    print("  PASS: create_topic_no_tags")


@patch("urllib.request.urlopen")
def test_create_topic_failure(mock_urlopen: MagicMock):
    mock_urlopen.side_effect = mock_http_error(422, '{"errors":["Title too short"]}')
    client = DiscourseClient(TEST_CONFIG)

    result = client.create_topic(title="X", raw="body", category_id=1)
    assert result is None
    print("  PASS: create_topic_failure")


# ---------------------------------------------------------------------------
# update_post
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_update_post_success(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"post": {"id": 501, "version": 2}})
    client = DiscourseClient(TEST_CONFIG)

    result = client.update_post(post_id=501, raw="Updated content")

    assert result is not None
    call_args = mock_urlopen.call_args[0][0]
    assert call_args.method == "PUT"
    assert "/posts/501.json" in call_args.full_url

    sent_payload = json.loads(call_args.data.decode("utf-8"))
    assert sent_payload["post"]["raw"] == "Updated content"
    assert sent_payload["post"]["edit_reason"] == "Documentation sync"
    print("  PASS: update_post_success")


@patch("urllib.request.urlopen")
def test_update_post_custom_reason(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"post": {"id": 501}})
    client = DiscourseClient(TEST_CONFIG)

    client.update_post(post_id=501, raw="content", edit_reason="Manual fix")

    sent_payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert sent_payload["post"]["edit_reason"] == "Manual fix"
    print("  PASS: update_post_custom_reason")


@patch("urllib.request.urlopen")
def test_update_post_not_found(mock_urlopen: MagicMock):
    mock_urlopen.side_effect = mock_http_error(404)
    client = DiscourseClient(TEST_CONFIG)

    result = client.update_post(post_id=99999, raw="content")
    assert result is None
    print("  PASS: update_post_not_found")


# ---------------------------------------------------------------------------
# first_post_id
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_first_post_id_found(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({
        "post_stream": {
            "posts": [
                {"id": 501, "post_number": 1},
                {"id": 502, "post_number": 2},
            ],
        },
    })
    client = DiscourseClient(TEST_CONFIG)

    result = client.first_post_id(topic_id=201)
    assert result == 501
    print("  PASS: first_post_id_found")


@patch("urllib.request.urlopen")
def test_first_post_id_empty_stream(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"post_stream": {"posts": []}})
    client = DiscourseClient(TEST_CONFIG)

    result = client.first_post_id(topic_id=201)
    assert result is None
    print("  PASS: first_post_id_empty_stream")


@patch("urllib.request.urlopen")
def test_first_post_id_topic_not_found(mock_urlopen: MagicMock):
    mock_urlopen.side_effect = mock_http_error(404)
    client = DiscourseClient(TEST_CONFIG)

    result = client.first_post_id(topic_id=99999)
    assert result is None
    print("  PASS: first_post_id_topic_not_found")


# ---------------------------------------------------------------------------
# Headers / auth
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_headers_set_correctly(mock_urlopen: MagicMock):
    mock_urlopen.return_value = mock_response({"category": {"id": 1}})
    client = DiscourseClient(TEST_CONFIG)

    client.get_category_id()

    req = mock_urlopen.call_args[0][0]
    assert req.get_header("Api-key") == "test-api-key-123"
    assert req.get_header("Api-username") == "docs-bot"
    assert req.get_header("Content-type") == "application/json"
    print("  PASS: headers_set_correctly")


# ---------------------------------------------------------------------------
# Connection error
# ---------------------------------------------------------------------------


@patch("urllib.request.urlopen")
def test_connection_error_returns_none(mock_urlopen: MagicMock):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    client = DiscourseClient(TEST_CONFIG)

    result = client.get_category_id()
    assert result is None
    print("  PASS: connection_error_returns_none")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

import os  # noqa: E402 (needed for test_config_from_env_defaults)

if __name__ == "__main__":
    print("Testing Discourse API client:")
    tests = [
        # Config
        test_config_from_env,
        test_config_from_env_defaults,
        test_config_missing_url,
        test_config_missing_api_key,
        test_config_immutable,
        # get_category_id
        test_get_category_id_found,
        test_get_category_id_not_found,
        test_get_category_id_uses_default_slug,
        # find_topic_by_sync_id
        test_find_topic_found,
        test_find_topic_not_found,
        test_find_topic_api_error,
        # create_topic
        test_create_topic_success,
        test_create_topic_no_tags,
        test_create_topic_failure,
        # update_post
        test_update_post_success,
        test_update_post_custom_reason,
        test_update_post_not_found,
        # first_post_id
        test_first_post_id_found,
        test_first_post_id_empty_stream,
        test_first_post_id_topic_not_found,
        # Misc
        test_headers_set_correctly,
        test_connection_error_returns_none,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(1 if failed > 0 else 0)
