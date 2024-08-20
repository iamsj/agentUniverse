# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/3/19 11:43
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: ollama_embedding.py

from typing import List, Optional
from pydantic import Field
import ollama

from agentuniverse.agent.action.knowledge.embedding.embedding import Embedding


class OllamaEmbedding(Embedding):
    """The Ollama embedding class."""

    ollama_model_name: Optional[str] = Field(default="mxbai-embed-large")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Ollama API."""
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.ollama_model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    async def async_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get embeddings using Ollama API."""
        embeddings = []
        for text in texts:
            response = await ollama.embeddings(model=self.ollama_model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def as_langchain(self):
        """This method would return an equivalent Langchain embedding class if needed."""
        # Since Langchain is more associated with OpenAI embeddings, you might want to convert this
        # or simply return `NotImplementedError` if not applicable.
        raise NotImplementedError("Conversion to Langchain embedding is not supported for Ollama.")


# 使用示例
if __name__ == "__main__":
    embedding = OllamaEmbedding()
    texts = [
        "Llamas are members of the camelid family...",
        # ... 其他文档文本
    ]
    embeddings = embedding.get_embeddings(texts)
    print(embeddings)
    prompt = "What animals are llamas related to?"
    related_doc = embedding.get_embeddings(prompt)
    print(related_doc)
