from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tau2.knowledge.embedders.bailian_embedder import BailianEmbedder


def test_bailian_embedder_uses_dashscope_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.test/v1")
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])]
    )

    with patch(
        "tau2.knowledge.embedders.bailian_embedder.OpenAI", return_value=client
    ) as openai:
        embedder = BailianEmbedder()
        result = embedder.embed(["bank policy"])

    openai.assert_called_once_with(
        api_key="test-key", base_url="https://example.test/v1"
    )
    client.embeddings.create.assert_called_once_with(
        input=["bank policy"], model="qwen3.7-text-embedding"
    )
    np.testing.assert_array_equal(result, np.array([[0.1, 0.2]]))


def test_bailian_embedder_batches_at_api_limit(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.test/v1")
    client = MagicMock()
    client.embeddings.create.side_effect = [
        SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(i)]) for i in range(20)]
        ),
        SimpleNamespace(data=[SimpleNamespace(embedding=[20.0])]),
    ]

    with patch("tau2.knowledge.embedders.bailian_embedder.OpenAI", return_value=client):
        result = BailianEmbedder().embed([str(i) for i in range(21)])

    assert client.embeddings.create.call_count == 2
    assert len(client.embeddings.create.call_args_list[0].kwargs["input"]) == 20
    assert len(client.embeddings.create.call_args_list[1].kwargs["input"]) == 1
    np.testing.assert_array_equal(result[:, 0], np.arange(21, dtype=float))


@pytest.mark.parametrize("missing", ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"])
def test_bailian_embedder_requires_environment(monkeypatch, missing):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv(missing)

    with pytest.raises(ValueError, match=missing):
        BailianEmbedder()
