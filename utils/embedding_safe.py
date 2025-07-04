from typing import List, Dict, Optional
import os
import gc

class SafeEmbeddingClient:
    """
    Safe embedding client that uses only one model at a time to prevent memory issues.
    Falls back to using the document model for both queries and documents.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 huggingface_api_key: str | None = None):
        """
        Initialize with single model to prevent segmentation faults.
        
        Args:
            model_name: Single model to use for both queries and documents
            huggingface_api_key: Optional HuggingFace API key
        """
        # Set HuggingFace token
        if huggingface_api_key:
            os.environ["HF_TOKEN"] = huggingface_api_key
            os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_api_key
        
        self.model_name = model_name
        self._model = None
        
        print(f"[DEBUG] SafeEmbeddingClient initialized")
        print(f"[DEBUG] Model: {model_name}")

    @property
    def model(self):
        """Lazy load single model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[DEBUG] Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"[DEBUG] Model loaded successfully")
        return self._model

    def generate_embedding(self, text: str, use_document_model: bool = False) -> List[float]:
        """Generate embedding for text using single model."""
        if not text:
            return []
        
        try:
            embedding = self.model.encode(text).tolist()
            return embedding
        except Exception as e:
            print(f"[DEBUG] Error generating embedding: {e}")
            return []
    
    def generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding using model - for questions."""
        return self.generate_embedding(text, use_document_model=False)
    
    def generate_document_embedding(self, text: str) -> List[float]:
        """Generate embedding using model - for documents."""
        return self.generate_embedding(text, use_document_model=True)
    
    def cleanup(self):
        """Clean up model to free memory."""
        try:
            if self._model is not None:
                del self._model
                self._model = None
                print("[DEBUG] Model cleaned up")
                gc.collect()
                print("[DEBUG] Garbage collection completed")
        except Exception as e:
            print(f"[DEBUG] Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except:
            pass