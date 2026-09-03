"""Alibaba Cloud Model Studio (Bailian) OpenAI-compatible embedder."""

import os
from typing import List, Optional

import numpy as np
from openai import OpenAI

from tau2.knowledge.embedders.base import BaseEmbedder

DEFAULT_BAILIAN_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_BAILIAN_BATCH_SIZE = 20
DEFAULT_BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class BailianEmbedder(BaseEmbedder):
    """Embed text through Bailian's OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_BAILIAN_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = DEFAULT_BAILIAN_BATCH_SIZE,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BAILIAN_BASE_URL
        if batch_size < 1 or batch_size > DEFAULT_BAILIAN_BATCH_SIZE:
            raise ValueError(
                f"Bailian batch_size must be between 1 and "
                f"{DEFAULT_BAILIAN_BATCH_SIZE}."
            )
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError(
                "Bailian API key not provided. Set DASHSCOPE_API_KEY or pass api_key."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return one embedding vector for each supplied text."""
        if not texts:
            raise ValueError("No text to embed.")

        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            embeddings.extend(item.embedding for item in response.data)
        return np.array(embeddings)

    def get_name(self) -> str:
        """Return the configured provider and model name."""
        return f"bailian_{self.model}"
