import hashlib
import math
import re
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import Settings
from memory.episodic import EpisodicMemory, OpenAIEmbedder, signal_id
from models import Signal

DIM = 256


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder: similar wording -> high cosine."""
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def sig(competitor="Anthropic", title="Claude 5 model launch", summary="Anthropic released Claude 5 with longer context.",
        url="https://a.com/1", signal_type="product_launch"):
    return Signal(competitor=competitor, signal_type=signal_type, title=title, summary=summary,
                  impact_assessment="high", source_url=url, confidence=0.9,
                  date_detected=datetime(2026, 11, 1, tzinfo=timezone.utc))


@pytest.fixture
def mem(tmp_path):
    settings = replace(Settings(), dedup_similarity_threshold=0.9, memory_top_k=5)
    return EpisodicMemory(persist_dir=str(tmp_path / "chroma"), embedder=fake_embed, settings=settings)


def test_store_and_count(mem):
    res = mem.store([sig(), sig("Meta", "Llama 5 open weights", "Meta released Llama 5.", "https://m.com/1")], "r1")
    assert len(res["stored"]) == 2 and res["duplicates"] == []
    assert mem.count() == 2 and mem.count("Anthropic") == 1 and mem.count("Nobody") == 0


def test_exact_repeat_is_skipped_by_deterministic_id(mem):
    mem.store([sig()], "r1")
    res = mem.store([sig(url="https://A.com/1/")], "r2")  # same signal, cosmetic URL diff
    assert res["stored"] == [] and res["duplicates"] == [signal_id(sig())]
    assert mem.count() == 1


def test_near_duplicate_same_competitor_is_skipped(mem):
    mem.store([sig()], "r1")
    near = sig(title="Claude 5 model launch announced", url="https://news.com/x")
    assert mem.store([near], "r2")["duplicates"]
    assert mem.count() == 1


def test_near_duplicate_within_same_batch_is_skipped(mem):
    res = mem.store([sig(), sig(title="Claude 5 model launch announced", url="https://b.com")], "r1")
    assert len(res["stored"]) == 1 and len(res["duplicates"]) == 1


def test_same_text_for_different_competitor_is_not_a_duplicate(mem):
    mem.store([sig("Anthropic", summary="Company released a new model with longer context.")], "r1")
    other = sig("Google", summary="Company released a new model with longer context.", url="https://g.com")
    assert mem.store([other], "r1")["stored"]
    assert mem.count() == 2


def test_distinct_signal_is_stored(mem):
    mem.store([sig()], "r1")
    different = sig(title="New pricing tier for enterprise", summary="Anthropic cut API prices by half.",
                    url="https://a.com/pricing", signal_type="pricing_change")
    assert mem.store([different], "r2")["stored"]


def test_retrieval_is_scoped_to_one_competitor(mem):
    mem.store([sig("Anthropic", f"Anthropic item {i}", f"Anthropic news number {i}", f"https://a.com/{i}") for i in range(3)], "r1")
    mem.store([sig("Meta", f"Meta item {i}", f"Meta news number {i}", f"https://m.com/{i}") for i in range(8)], "r1")

    ctx = mem.retrieve("Anthropic", ["Anthropic news"])
    assert len(ctx) == 3
    assert all("Anthropic |" in c for c in ctx)
    assert ctx[0].startswith("[2026-11-01]")
    assert len(mem.retrieve("Meta", ["Meta news"], top_k=4)) == 4


def test_retrieval_ranks_by_query_similarity(mem):
    mem.store([
        sig(title="Pricing change", summary="Anthropic cut API prices", url="https://a.com/p", signal_type="pricing_change"),
        sig(title="Partnership deal", summary="Anthropic partners with a cloud provider", url="https://a.com/d", signal_type="partnership"),
    ], "r1")
    assert "Partnership deal" in mem.retrieve("Anthropic", ["partners cloud provider"], top_k=1)[0]


def test_retrieval_empty_for_unknown_competitor_without_embedding_call(tmp_path):
    embed = MagicMock(side_effect=fake_embed)
    m = EpisodicMemory(persist_dir=str(tmp_path / "c"), embedder=embed)
    assert m.retrieve("Nobody") == []
    embed.assert_not_called()


def test_store_batches_embeddings_into_one_call(tmp_path):
    embed = MagicMock(side_effect=fake_embed)
    m = EpisodicMemory(persist_dir=str(tmp_path / "c"), embedder=embed)
    m.store([sig(), sig("Meta", "Other", "Other thing", "https://m.com")], "r1")
    assert embed.call_count == 1
    assert m.store([], "r1") == {"stored": [], "duplicates": []}


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "c")
    EpisodicMemory(persist_dir=path, embedder=fake_embed).store([sig()], "r1")
    assert EpisodicMemory(persist_dir=path, embedder=fake_embed).count("Anthropic") == 1


def test_openai_embedder_uses_configured_model_and_preserves_order():
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(data=[
        SimpleNamespace(index=1, embedding=[0.0, 1.0]), SimpleNamespace(index=0, embedding=[1.0, 0.0]),
    ])
    out = OpenAIEmbedder(Settings().embedding_model, client=client)(["a", "b"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]
    client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input=["a", "b"])
