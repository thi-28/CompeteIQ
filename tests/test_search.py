from unittest.mock import MagicMock

import pytest
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

from tools.search import SearchError, TavilySearchTool

OK_RESPONSE = {
    "results": [
        {"title": "T1", "url": "https://a.com", "content": "alpha", "score": 0.9},
        {"title": "T2", "url": "", "content": "no url"},
        {"title": "T3", "url": "https://c.com", "content": "   "},
        {"title": None, "url": "https://d.com", "content": "delta", "score": None},
    ]
}


def make_tool(client, **kw):
    sleeps: list[float] = []
    tool = TavilySearchTool(client=client, jitter=0.0, sleep=sleeps.append, **kw)
    return tool, sleeps


def test_search_normalizes_and_drops_ungroundable_results():
    client = MagicMock()
    client.search.return_value = OK_RESPONSE
    tool, sleeps = make_tool(client, max_results=7)

    results = tool.search("anthropic news")

    assert [r["url"] for r in results] == ["https://a.com", "https://d.com"]
    assert results[1] == {"title": "", "url": "https://d.com", "content": "delta", "score": 0.0}
    client.search.assert_called_once_with(
        query="anthropic news", max_results=7, search_depth="basic"
    )
    assert sleeps == []


def test_retries_transient_errors_with_exponential_backoff():
    client = MagicMock()
    client.search.side_effect = [TimeoutError("t1"), ConnectionError("c"), OK_RESPONSE]
    tool, sleeps = make_tool(client, base_delay=1.0)

    assert len(tool.search("q")) == 2
    assert client.search.call_count == 3
    assert sleeps == [1.0, 2.0]


def test_raises_after_exhausting_retries():
    client = MagicMock()
    client.search.side_effect = Exception("429 Too Many Requests")
    tool, sleeps = make_tool(client, max_retries=4, base_delay=1.0)

    with pytest.raises(SearchError, match="after 4 attempts"):
        tool.search("q")
    assert client.search.call_count == 4
    assert sleeps == [1.0, 2.0, 4.0]  # no sleep after the final attempt


@pytest.mark.parametrize("exc", [InvalidAPIKeyError("bad key"), UsageLimitExceededError("quota")])
def test_non_retryable_errors_fail_fast(exc):
    client = MagicMock()
    client.search.side_effect = exc
    tool, sleeps = make_tool(client)

    with pytest.raises(SearchError, match="Non-retryable"):
        tool.search("q")
    assert client.search.call_count == 1
    assert sleeps == []


def test_backoff_is_capped_and_jitter_bounded():
    tool = TavilySearchTool(client=MagicMock(), base_delay=1.0, max_delay=5.0, jitter=0.0)
    assert [tool.backoff_delay(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]

    jittered = TavilySearchTool(client=MagicMock(), base_delay=2.0, jitter=0.5)
    for _ in range(50):
        assert 2.0 <= jittered.backoff_delay(1) <= 3.0


def test_empty_or_missing_results():
    client = MagicMock()
    client.search.return_value = {}
    tool, _ = make_tool(client)
    assert tool.search("q") == []


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        TavilySearchTool()


def test_invalid_max_retries():
    with pytest.raises(ValueError):
        TavilySearchTool(client=MagicMock(), max_retries=0)
