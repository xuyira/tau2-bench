from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tau2.knowledge.embedders.bailian_embedder import (
    DEFAULT_BAILIAN_BASE_URL,
    BailianEmbedder,
)


def test_bailian_embedder_uses_dashscope_compatible_endpoint_and_batches():
    response = MagicMock()
    response.data = [MagicMock(embedding=[1.0, 2.0]), MagicMock(embedding=[3.0, 4.0])]
    with patch.dict("os.environ", {}, clear=True), patch(
        "tau2.knowledge.embedders.bailian_embedder.OpenAI"
    ) as openai:
        openai.return_value.embeddings.create.return_value = response
        embedder = BailianEmbedder(api_key="test-key", batch_size=2)
        result = embedder.embed(["a", "b"])

    assert embedder.base_url == DEFAULT_BAILIAN_BASE_URL
    assert embedder.get_name() == "bailian_qwen3.7-text-embedding"
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    openai.return_value.embeddings.create.assert_called_once_with(
        input=["a", "b"], model="qwen3.7-text-embedding"
    )


def test_bailian_embedder_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            BailianEmbedder()
