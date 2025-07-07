from typing import List
import os
import gc
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class EmbeddingClient:
    """Base class for embedding clients."""
    def generate_query_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    def generate_document_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    def cleanup(self):
        pass

class HuggingFaceEmbeddingClient(EmbeddingClient):
    """Embedding client for Hugging Face sentence-transformers models."""
    def __init__(self, model_name: str, huggingface_api_key: str | None = None):
        self.model_name = model_name
        self.huggingface_api_key = huggingface_api_key
        self._model = SentenceTransformer(model_name, device='cpu')

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self._model.encode(text)
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

    def generate_query_embedding(self, text: str) -> List[float]:
        return self.generate_embedding(text)

    def generate_document_embedding(self, text: str) -> List[float]:
        return self.generate_embedding(text)

    def cleanup(self):
        if self._model:
            del self._model
            gc.collect()

class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client for OpenAI models."""
    def __init__(self, model_name: str, openai_api_key: str):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self._client = OpenAI(api_key=self.openai_api_key)

    def generate_embedding(self, text: str) -> List[float]:
        response = self._client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def generate_query_embedding(self, text: str) -> List[float]:
        return self.generate_embedding(text)

    def generate_document_embedding(self, text: str) -> List[float]:
        return self.generate_embedding(text)

def get_embedding_client(
    model_name: str,
    huggingface_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> EmbeddingClient:
    """Factory function to get the correct embedding client."""
    if "ada" in model_name:
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for Ada models.")
        return OpenAIEmbeddingClient(model_name, openai_api_key)
    else:
        return HuggingFaceEmbeddingClient(model_name, huggingface_api_key)
