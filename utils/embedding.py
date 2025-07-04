from typing import List, Dict, Optional
import os
import gc
import threading

class EmbeddingClient:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 document_model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 huggingface_api_key: str | None = None):
        """
        Initialize embedding client with lazy loading to prevent memory issues.
        
        Args:
            model_name: Model for queries (default: MiniLM for questions)
            document_model_name: Model for documents (default: MPNet for documents)
            huggingface_api_key: Optional HuggingFace API key
        """
        # Set HuggingFace token before importing SentenceTransformer
        if huggingface_api_key:
            os.environ["HF_TOKEN"] = huggingface_api_key
            os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_api_key
        
        # Store model names for lazy loading
        self.model_name = model_name
        self.document_model_name = document_model_name
        
        # Initialize models as None - will be loaded on first use
        self._query_model = None
        self._document_model = None
        self._model_lock = threading.Lock()
        
        print(f"[DEBUG] EmbeddingClient initialized with lazy loading")
        print(f"[DEBUG] Query model: {model_name}")
        print(f"[DEBUG] Document model: {document_model_name}")

    @property
    def query_model(self):
        """Lazy load query model."""
        if self._query_model is None:
            with self._model_lock:
                if self._query_model is None:
                    from sentence_transformers import SentenceTransformer
                    print(f"[DEBUG] Loading query model: {self.model_name}")
                    self._query_model = SentenceTransformer(self.model_name)
                    print(f"[DEBUG] Query model loaded successfully")
        return self._query_model

    @property
    def document_model(self):
        """Lazy load document model."""
        if self._document_model is None:
            with self._model_lock:
                if self._document_model is None:
                    from sentence_transformers import SentenceTransformer
                    print(f"[DEBUG] Loading document model: {self.document_model_name}")
                    self._document_model = SentenceTransformer(self.document_model_name)
                    print(f"[DEBUG] Document model loaded successfully")
        return self._document_model

    @property
    def model(self):
        """Backward compatibility - return query model."""
        return self.query_model

    def generate_embedding(self, text: str, use_document_model: bool = False) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            use_document_model: If True, use document model (MPNet), else use query model (MiniLM)
        """
        if not text:
            return []
        
        # Choose appropriate model
        model = self.document_model if use_document_model else self.query_model
        
        # The encode method returns a numpy array, convert to list
        embedding = model.encode(text).tolist()
        return embedding
    
    def generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding using query model (MiniLM) - for questions."""
        return self.generate_embedding(text, use_document_model=False)
    
    def generate_document_embedding(self, text: str) -> List[float]:
        """Generate embedding using document model (MPNet) - for documents."""
        return self.generate_embedding(text, use_document_model=True)
    
    def cleanup(self):
        """Clean up models to free memory."""
        try:
            with self._model_lock:
                if self._query_model is not None:
                    del self._query_model
                    self._query_model = None
                    print("[DEBUG] Query model cleaned up")
                
                if self._document_model is not None:
                    del self._document_model
                    self._document_model = None
                    print("[DEBUG] Document model cleaned up")
                
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