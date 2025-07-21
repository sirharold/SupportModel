import os
import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from functools import lru_cache
import time
from dataclasses import dataclass
from config import DEBUG_MODE

def debug_print(message: str, force: bool = False):
    """Print debug message only if DEBUG_MODE is enabled or force is True."""
    if DEBUG_MODE or force:
        print(message)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChromaDBConfig:
    """Configuration for ChromaDB connection."""
    openai_api_key: str  # Still needed for LLM reranker and OpenAI comparison
    huggingface_api_key: str | None = None  # Added for Hugging Face models if needed
    gemini_api_key: str | None = None
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    persist_directory: str = "/Users/haroldgomez/chromadb2"
    #persist_directory: str = "/Volumes/DiscoEXTMac/chromadb2"
    
    @classmethod
    def from_env(cls, env_path: str = ".env") -> "ChromaDBConfig":
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            
            required_keys = ["OPENAI_API_KEY"]
            optional_keys = ["HUGGINGFACE_API_KEY", "GEMINI_API_KEY", "CHROMA_HOST", "CHROMA_PORT", "CHROMA_PERSIST_DIR"]
            config_dict = {}
            
            for key in required_keys:
                value = os.getenv(key)
                if not value:
                    raise ValueError(f"Missing required environment variable: {key}")
                config_dict[key.lower()] = value

            for key in optional_keys:
                value = os.getenv(key)
                config_dict[key.lower()] = value  # Store even if None
            
            return cls(
                openai_api_key=config_dict["openai_api_key"],
                huggingface_api_key=config_dict.get("huggingface_api_key"),
                gemini_api_key=config_dict.get("gemini_api_key"),
                chroma_host=config_dict.get("chroma_host") or "localhost",
                chroma_port=int(config_dict.get("chroma_port") or 8000),
                #persist_directory=config_dict.get("chroma_persist_dir") or "/Volumes/DiscoEXTMac/chromadb2"
                persist_directory=config_dict.get("chroma_persist_dir") or "/Users/haroldgomez/chromadb2"
            )
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

def get_chromadb_client(config: ChromaDBConfig):
    """Create cached ChromaDB client connection."""
    try:
        # Try to connect to ChromaDB server first
        try:
            client = chromadb.HttpClient(
                host=config.chroma_host,
                port=config.chroma_port,
                settings=Settings(allow_reset=True)
            )
            # Test the connection
            client.heartbeat()
            logger.info(f"Successfully connected to ChromaDB server at {config.chroma_host}:{config.chroma_port}")
            return client
        except Exception as server_error:
            logger.warning(f"Failed to connect to ChromaDB server: {server_error}")
            logger.info("Falling back to persistent client")
            
            # Fallback to persistent client
            client = chromadb.PersistentClient(path=config.persist_directory)
            logger.info(f"Successfully connected to ChromaDB persistent client at {config.persist_directory}")
            return client
            
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise

class ChromaDBClientWrapper:
    """ChromaDB client wrapper with equivalent functionality to WeaviateClientWrapper."""
    
    def __init__(self, client, documents_class: str, questions_class: str, retry_attempts: int = 3, retry_delay: float = 1.0):
        if not client:
            raise ValueError("ChromaDB client cannot be None")
        
        self.client = client
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Map collection names to ChromaDB collection names
        self.docs_collection_name = self._map_collection_name(documents_class)
        self.questions_collection_name = self._map_collection_name(questions_class)
        
        # Get or create collections
        try:
            self._docs_collection = self.client.get_collection(name=self.docs_collection_name)
            debug_print(f"[DEBUG] Retrieved existing docs collection: {self.docs_collection_name}")
        except Exception as e:
            # List available collections for debugging
            try:
                collections = self.client.list_collections()
                available_names = [col.name for col in collections]
                debug_print(f"[DEBUG] Available collections: {available_names}")
                raise ValueError(f"Collection '{self.docs_collection_name}' not found in ChromaDB. Available collections: {available_names}")
            except Exception:
                raise ValueError(f"Collection '{self.docs_collection_name}' not found in ChromaDB and couldn't list collections: {e}")
            
        try:
            self._questions_collection = self.client.get_collection(name=self.questions_collection_name)
            debug_print(f"[DEBUG] Retrieved existing questions collection: {self.questions_collection_name}")
        except Exception as e:
            # List available collections for debugging
            try:
                collections = self.client.list_collections()
                available_names = [col.name for col in collections]
                debug_print(f"[DEBUG] Available collections: {available_names}")
                raise ValueError(f"Collection '{self.questions_collection_name}' not found in ChromaDB. Available collections: {available_names}")
            except Exception:
                raise ValueError(f"Collection '{self.questions_collection_name}' not found in ChromaDB and couldn't list collections: {e}")
    
    def _map_collection_name(self, weaviate_class: str) -> str:
        """Map Weaviate class names to ChromaDB collection names."""
        mapping = {
            "DocumentsMpnet": "docs_mpnet",
            "DocumentsMiniLM": "docs_minilm", 
            "Documentation": "docs_ada",
            "QuestionsMlpnet": "questions_mpnet",
            "QuestionsMiniLM": "questions_minilm",
            "Questions": "questions_ada"
        }
        return mapping.get(weaviate_class, weaviate_class.lower())
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff."""
        for attempt in range(self.retry_attempts):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                logger.warning(f"Operation failed (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def close(self) -> None:
        """Close the ChromaDB client connection."""
        try:
            # ChromaDB doesn't require explicit closing, but we log it for consistency
            logger.info("ChromaDB client connection closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB client: {e}")
    
    def search_questions_by_vector(
        self, 
        vector: List[float], 
        top_k: int = 10,
        include_distance: bool = False
    ) -> List[Dict]:
        """Search questions by vector similarity."""
        debug_print(f"[DEBUG] search_questions_by_vector: Called with vector length {len(vector)}, top_k {top_k}")
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        def _search_questions():
            debug_print(f"[DEBUG] search_questions_by_vector: Searching in collection: {self.questions_collection_name}")
            
            results = self._questions_collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances'] if include_distance else ['metadatas', 'documents']
            )
            
            debug_print(f"[DEBUG] search_questions_by_vector: ChromaDB returned {len(results['metadatas'][0])} objects.")
            questions = []
            
            for i in range(len(results['metadatas'][0])):
                question_data = results['metadatas'][0][i].copy()
                # Add the document content if available
                if results['documents'][0][i]:
                    question_data['content'] = results['documents'][0][i]
                if include_distance and 'distances' in results:
                    question_data['distance'] = results['distances'][0][i]
                questions.append(question_data)
            
            return questions
        
        try:
            return self._retry_operation(_search_questions)
        except Exception as e:
            logger.error(f"Error searching questions by vector: {e}")
            debug_print(f"[DEBUG] search_questions_by_vector: Error: {e}")
            return []
    
    def search_docs_by_vector(
        self, 
        vector: List[float], 
        top_k: int = 10,
        diversity_threshold: float = 0.85,
        include_distance: bool = False
    ) -> List[Dict]:
        """Enhanced document search with diversity control."""
        debug_print(f"[DEBUG] search_docs_by_vector: Called with vector length {len(vector)}, top_k {top_k}")
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        def _search_docs():
            debug_print(f"[DEBUG] search_docs_by_vector: Searching in collection: {self.docs_collection_name}")
            # Fetch more documents for better diversity
            fetch_limit = max(top_k * 3, 30)

            results = self._docs_collection.query(
                query_embeddings=[vector],
                n_results=fetch_limit,
                include=['metadatas', 'documents', 'distances'] if include_distance else ['metadatas', 'documents']
            )
            
            debug_print(f"[DEBUG] search_docs_by_vector: ChromaDB returned {len(results['metadatas'][0])} objects.")
            
            # Convert results to objects format for diversity filtering
            objects = []
            for i in range(len(results['metadatas'][0])):
                obj_data = {
                    'properties': results['metadatas'][0][i].copy(),
                    'metadata': {'distance': results['distances'][0][i]} if include_distance and 'distances' in results else None
                }
                # Add document content
                if results['documents'][0][i]:
                    obj_data['properties']['content'] = results['documents'][0][i]
                objects.append(obj_data)
            
            debug_print("[DEBUG] Applying diversity filtering.")
            return self._apply_diversity_filtering(
                objects,
                top_k,
                diversity_threshold,
                include_distance
            )
        
        try:
            return self._retry_operation(_search_docs)
        except Exception as e:
            logger.error(f"Error searching docs by vector: {e}")
            debug_print(f"[DEBUG] search_docs_by_vector: Top-level error: {e}")
            return []
    
    def _apply_diversity_filtering(
        self, 
        objects, 
        top_k: int, 
        diversity_threshold: float,
        include_distance: bool
    ) -> List[Dict]:
        """Apply diversity filtering to ensure varied results."""
        filtered_docs = []
        seen_links = set()
        seen_titles = set()
        
        for obj in objects:
            props = obj['properties'].copy()
            link = props.get("link", "").strip()
            title = props.get("title", "").strip().lower()
            
            # Skip if we've seen this link
            if link in seen_links:
                continue
            
            # Apply diversity filtering based on title similarity
            if title and any(
                self._calculate_title_similarity(title, seen_title) > diversity_threshold 
                for seen_title in seen_titles
            ):
                continue
            
            if link:
                seen_links.add(link)
            if title:
                seen_titles.add(title)
            
            if include_distance and obj.get('metadata') and obj['metadata']:
                props['distance'] = obj['metadata']['distance']
            
            filtered_docs.append(props)
            
            if len(filtered_docs) >= top_k:
                break
        
        return filtered_docs
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple Jaccard similarity between titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def lookup_docs_by_links_batch(
        self, 
        links: List[str], 
        batch_size: int = 50
    ) -> List[Dict]:
        """Optimized batch lookup of documents by links."""
        if not links:
            return []
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(link.strip() for link in links if link.strip()))
        
        def _batch_lookup(link_batch: List[str]) -> List[Dict]:
            # ChromaDB doesn't have complex filtering like Weaviate, so we get all docs
            # and filter by links in memory
            results = self._docs_collection.get(
                include=['metadatas', 'documents']
            )
            
            filtered_docs = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata.get('link') in link_batch:
                    doc_data = metadata.copy()
                    if results['documents'][i]:
                        doc_data['content'] = results['documents'][i]
                    filtered_docs.append(doc_data)
            
            return filtered_docs
        
        try:
            all_results = []
            for i in range(0, len(unique_links), batch_size):
                batch = unique_links[i:i + batch_size]
                batch_results = self._retry_operation(_batch_lookup, batch)
                all_results.extend(batch_results)
            
            # Deduplicate by link
            seen_links = set()
            deduplicated = []
            for doc in all_results:
                link = doc.get("link", "").strip()
                if link and link not in seen_links:
                    deduplicated.append(doc)
                    seen_links.add(link)
            
            logger.info(f"Retrieved {len(deduplicated)} unique documents from {len(unique_links)} links")
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error in batch lookup: {e}")
            return []
    
    def lookup_docs_by_links(self, links: List[str]) -> List[Dict]:
        """Legacy method - lookup documents by links (non-batch version)."""
        if not links:
            return []
        
        # Use batch method for better performance
        return self.lookup_docs_by_links_batch(links, batch_size=50)
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about collections."""
        try:
            stats = {}
            
            # Get count for questions collection
            questions_count = self._questions_collection.count()
            stats[f"{self.questions_collection_name}_count"] = questions_count
            
            # Get count for docs collection  
            docs_count = self._docs_collection.count()
            stats[f"{self.docs_collection_name}_count"] = docs_count
            
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def search_docs_by_keyword(
        self, 
        keyword: str, 
        limit: int = 10
    ) -> List[Dict]:
        """Search documents by keyword (text search) in the docs collection."""
        debug_print(f"[DEBUG] search_docs_by_keyword: Searching for keyword: '{keyword}'")
        if not keyword:
            debug_print("[DEBUG] search_docs_by_keyword: Keyword is empty.")
            return []
        
        def _search():
            # ChromaDB doesn't have BM25, so we'll use text search on document content
            results = self._docs_collection.get(
                include=['metadatas', 'documents']
            )
            
            # Filter documents that contain the keyword
            matching_docs = []
            for i, doc_content in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                
                # Search in document content and metadata fields
                search_fields = [
                    doc_content or "",
                    metadata.get('title', ''),
                    metadata.get('content', ''),
                    str(metadata.get('text', ''))
                ]
                
                if any(keyword.lower() in field.lower() for field in search_fields):
                    doc_data = metadata.copy()
                    if doc_content:
                        doc_data['content'] = doc_content
                    matching_docs.append(doc_data)
                
                if len(matching_docs) >= limit:
                    break
            
            debug_print(f"[DEBUG] search_docs_by_keyword: Found {len(matching_docs)} documents for keyword '{keyword}'.")
            return matching_docs
        
        try:
            return self._retry_operation(_search)
        except Exception as e:
            logger.error(f"Error searching docs by keyword: {e}")
            debug_print(f"[DEBUG] search_docs_by_keyword: Error: {e}")
            return []

    def search_questions_by_keyword(
        self, 
        keyword: str, 
        limit: int = 10
    ) -> List[Dict]:
        """Search questions by keyword (text search) in the questions collection."""
        debug_print(f"[DEBUG] search_questions_by_keyword: Searching for keyword: '{keyword}'")
        if not keyword:
            debug_print("[DEBUG] search_questions_by_keyword: Keyword is empty.")
            return []
        
        def _search():
            # ChromaDB doesn't have BM25, so we'll use text search on question content
            results = self._questions_collection.get(
                include=['metadatas', 'documents']
            )
            
            # Filter questions that contain the keyword
            matching_questions = []
            for i, doc_content in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                
                # Search in question content and metadata fields
                search_fields = [
                    doc_content or "",
                    metadata.get('title', ''),
                    metadata.get('question_content', ''),
                    metadata.get('accepted_answer', ''),
                    str(metadata.get('text', ''))
                ]
                
                if any(keyword.lower() in field.lower() for field in search_fields):
                    question_data = metadata.copy()
                    if doc_content:
                        question_data['content'] = doc_content
                    matching_questions.append(question_data)
                
                if len(matching_questions) >= limit:
                    break
            
            debug_print(f"[DEBUG] search_questions_by_keyword: Found {len(matching_questions)} questions for keyword '{keyword}'.")
            return matching_questions
        
        try:
            return self._retry_operation(_search)
        except Exception as e:
            logger.error(f"Error searching questions by keyword: {e}")
            debug_print(f"[DEBUG] search_questions_by_keyword: Error: {e}")
            return []

    def get_sample_questions(
        self, 
        limit: int = 20, 
        random_sample: bool = False
    ) -> List[Dict]:
        """Get sample questions from the collection."""
        debug_print(f"[DEBUG] get_sample_questions: Getting {limit} questions, random: {random_sample}")
        debug_print(f"[DEBUG] get_sample_questions: Using collection: {self.questions_collection_name}")
        
        def _get_questions():
            # First check how many items are in the collection
            try:
                count = self._questions_collection.count()
                debug_print(f"[DEBUG] get_sample_questions: Collection has {count} items")
                if count == 0:
                    debug_print("[DEBUG] get_sample_questions: Collection is empty!")
                    return []
            except Exception as e:
                debug_print(f"[DEBUG] get_sample_questions: Could not count items: {e}")
            
            if random_sample:
                # For random sampling, we fetch more and then sample
                # MEJORADO: Eliminar límite artificial de 1000, usar el límite solicitado
                fetch_limit = limit  # Usar directamente el límite solicitado
                results = self._questions_collection.get(
                    limit=fetch_limit,
                    include=['metadatas', 'documents']
                )
                
                debug_print(f"[DEBUG] get_sample_questions: Fetched {len(results.get('metadatas', []))} items for random sampling")
                
                # Randomly sample from results
                import random
                indices = list(range(len(results['metadatas'])))
                random.shuffle(indices)
                selected_indices = indices[:limit]
            else:
                # Get first N questions
                results = self._questions_collection.get(
                    limit=limit,
                    include=['metadatas', 'documents']
                )
                debug_print(f"[DEBUG] get_sample_questions: Fetched {len(results.get('metadatas', []))} items")
                selected_indices = list(range(len(results['metadatas'])))
            
            questions = []
            for i in selected_indices:
                if i < len(results['metadatas']):
                    question_data = results['metadatas'][i].copy()
                    # Add document content if available
                    if i < len(results['documents']) and results['documents'][i]:
                        question_data['content'] = results['documents'][i]
                    # Add an ID for reference
                    question_data['id'] = f"q_{i}"
                    questions.append(question_data)
            
            debug_print(f"[DEBUG] get_sample_questions: Returning {len(questions)} questions")
            return questions
        
        try:
            return self._retry_operation(_get_questions)
        except Exception as e:
            logger.error(f"Error getting sample questions: {e}")
            debug_print(f"[DEBUG] get_sample_questions: Error: {e}")
            return []

# Convenience functions for backward compatibility
def cargar_credenciales(ruta_env: str = ".env") -> Dict[str, str]:
    """Load credentials - maintained for backward compatibility."""
    config = ChromaDBConfig.from_env(ruta_env)
    return {
        "OPENAI_API_KEY": config.openai_api_key
    }

def conectar(client_config: Dict[str, str]):
    """Connect to ChromaDB - maintained for backward compatibility."""
    config = ChromaDBConfig(
        openai_api_key=client_config["OPENAI_API_KEY"]
    )
    return get_chromadb_client(config)

def list_chromadb_collections():
    """List all available ChromaDB collections for debugging."""
    try:
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        logger.error(f"Failed to list ChromaDB collections: {e}")
        return []

def create_collections_if_needed():
    """Create the required ChromaDB collections if they don't exist."""
    try:
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        required_collections = [
            "questions_minilm", "questions_mpnet", "docs_minilm", 
            "questions_ada", "docs_ada", "docs_mpnet"
        ]
        
        existing_collections = [col.name for col in client.list_collections()]
        
        for collection_name in required_collections:
            if collection_name not in existing_collections:
                print(f"Creating collection: {collection_name}")
                client.create_collection(name=collection_name)
            else:
                print(f"Collection already exists: {collection_name}")
                
        return True
    except Exception as e:
        logger.error(f"Failed to create ChromaDB collections: {e}")
        return False

# Example usage with improved error handling and logging
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Available collections:", list_chromadb_collections())
        
        # Create collections if needed
        create_collections_if_needed()
        
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        wrapper = ChromaDBClientWrapper(client, "DocumentsMpnet", "QuestionsMlpnet", retry_attempts=3)
        
        # Example operations
        stats = wrapper.get_collection_stats()
        print(f"Collection stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")