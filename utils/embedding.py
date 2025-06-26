from openai import OpenAI
from typing import List

class EmbeddingClient:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        if not api_key:
            raise ValueError("OpenAI API key must be provided.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print("Error generating embedding:", e)
            return []