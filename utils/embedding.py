from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingClient:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", huggingface_api_key: str | None = None):
        # If a Hugging Face API key is provided, set it as an environment variable
        # This is useful for models that require authentication or for using Inference Endpoints
        if huggingface_api_key:
            import os
            os.environ["HF_TOKEN"] = huggingface_api_key
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        if not text:
            return []
        # The encode method returns a numpy array, convert to list
        embedding = self.model.encode(text).tolist()
        return embedding