import os
import logging
from typing import List, Dict
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.filters import Filter as WeaviateFilter
from functools import lru_cache
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class WeaviateConfig:
    """Configuration for Weaviate connection."""
    wcs_url: str
    wcs_api_key: str
    openai_api_key: str # Still needed for LLM reranker and OpenAI comparison
    huggingface_api_key: str | None = None # Added for Hugging Face models if needed
    gemini_api_key: str | None = None
    
    @classmethod
    def from_env(cls, env_path: str = ".env") -> "WeaviateConfig":
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            
            required_keys = ["WCS_URL", "WCS_API_KEY", "OPENAI_API_KEY"]
            optional_keys = ["HUGGINGFACE_API_KEY", "GEMINI_API_KEY"]
            config_dict = {}
            
            for key in required_keys:
                value = os.getenv(key)
                if not value:
                    raise ValueError(f"Missing required environment variable: {key}")
                config_dict[key.lower()] = value

            for key in optional_keys:
                value = os.getenv(key)
                config_dict[key.lower()] = value # Store even if None
            
            return cls(
                wcs_url=config_dict["wcs_url"],
                wcs_api_key=config_dict["wcs_api_key"],
                openai_api_key=config_dict["openai_api_key"],
                huggingface_api_key=config_dict.get("huggingface_api_key"),
                gemini_api_key=config_dict.get("gemini_api_key")
            )
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

@lru_cache(maxsize=1)
def get_weaviate_client(config: WeaviateConfig):
    """Create cached Weaviate client connection."""
    try:
        client = connect_to_weaviate_cloud(
            cluster_url=config.wcs_url,
            auth_credentials=AuthApiKey(config.wcs_api_key),
            headers={"X-OpenAI-Api-Key": config.openai_api_key}
        )
        
        if not hasattr(client, "collections"):
            raise RuntimeError("Weaviate client missing 'collections' attribute")
        
        logger.info("Successfully connected to Weaviate")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise

class WeaviateClientWrapper:
    """Enhanced Weaviate client wrapper with improved performance and error handling."""
    
    def __init__(self, client, documents_class: str, questions_class: str, retry_attempts: int = 3, retry_delay: float = 1.0):
        if not client:
            raise ValueError("Weaviate client cannot be None")
        if not hasattr(client, "collections"):
            raise TypeError("Invalid client: missing 'collections' attribute")
        
        self.client = client
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._docs_collection = self.client.collections.get(documents_class)
        self._questions_collection = self.client.collections.get(questions_class)
    
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
        """Close the Weaviate client connection."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
                logger.info("Weaviate client connection closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
    
    def search_questions_by_vector(
        self, 
        vector: List[float], 
        top_k: int = 10,
        include_distance: bool = False
    ) -> List[Dict]:
        """Search questions by vector similarity."""
        print(f"[DEBUG] search_questions_by_vector: Called with vector length {len(vector)}, top_k {top_k}")
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        def _search_questions():
            print(f"[DEBUG] search_questions_by_vector: Searching in collection: {self._questions_collection.name}")
            if include_distance:
                results = self._questions_collection.query.near_vector(
                    near_vector=vector,
                    limit=top_k,
                    return_metadata=['distance']
                )
            else:
                results = self._questions_collection.query.near_vector(
                    near_vector=vector,
                    limit=top_k
                )
            
            print(f"[DEBUG] search_questions_by_vector: Weaviate returned {len(results.objects)} objects.")
            questions = []
            for obj in results.objects:
                question_data = obj.properties.copy()
                if include_distance and obj.metadata:
                    question_data['distance'] = obj.metadata.distance
                questions.append(question_data)
            
            return questions
        
        try:
            return self._retry_operation(_search_questions)
        except Exception as e:
            logger.error(f"Error searching questions by vector: {e}")
            print(f"[DEBUG] search_questions_by_vector: Error: {e}")
            return []
    
    def search_docs_by_vector(
        self, 
        vector: List[float], 
        top_k: int = 10,
        diversity_threshold: float = 0.85,
        include_distance: bool = False
    ) -> List[Dict]:
        """Enhanced document search with diversity control."""
        print(f"[DEBUG] search_docs_by_vector: Called with vector length {len(vector)}, top_k {top_k}")
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        def _search_docs():
            print(f"[DEBUG] search_docs_by_vector: Searching in collection: {self._docs_collection.name}")
            # Fetch more documents for better diversity
            fetch_limit = max(top_k * 3, 30)

            try:
                # Try the new API with filters parameter
                print("[DEBUG] Attempting near_vector query with filters.")
                results = self._docs_collection.query.near_vector(
                    near_vector=vector,
                    limit=fetch_limit,
                    return_metadata=['distance'] if include_distance else None
                )
                print(f"[DEBUG] search_docs_by_vector: Weaviate returned {len(results.objects)} objects.")
            except TypeError as e:
                # Fallback to old API without filters parameter (if TypeError is due to filters)
                logger.warning(f"Using fallback API without filters parameter due to: {e}")
                print(f"[DEBUG] Fallback to old API: {e}")
                results = self._docs_collection.query.near_vector(
                    near_vector=vector,
                    limit=fetch_limit,
                    return_metadata=['distance'] if include_distance else None
                )
                print(f"[DEBUG] search_docs_by_vector: Weaviate returned {len(results.objects)} objects (fallback).")
                # Filter results manually
                filtered_objects = []
                for obj in results.objects:
                    # No chunk_index filter here, as it's removed
                    filtered_objects.append(obj)
                # Create a mock results object
                class MockResults:
                    def __init__(self, objects):
                        self.objects = objects
                results = MockResults(filtered_objects)
            
            print("[DEBUG] Applying diversity filtering.")
            return self._apply_diversity_filtering(
                results.objects,
                top_k,
                diversity_threshold,
                include_distance
            )
        
        try:
            return self._retry_operation(_search_docs)
        except Exception as e:
            logger.error(f"Error searching docs by vector: {e}")
            print(f"[DEBUG] search_docs_by_vector: Top-level error: {e}")
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
            props = obj.properties.copy()
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
            
            if include_distance and obj.metadata:
                props['distance'] = obj.metadata.distance
            
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
            # Create OR filter for batch of links
            if len(link_batch) == 1:
                link_filter = WeaviateFilter.by_property("link").equal(link_batch[0])
            else:
                link_filters = [
                    WeaviateFilter.by_property("link").equal(link) 
                    for link in link_batch
                ]
                link_filter = link_filters[0]
                for f in link_filters[1:]:
                    link_filter = link_filter | f
            
            # No chunk_index filter needed for DocumentsMpnet collection
            combined_filter = link_filter
            
            results = self._docs_collection.query.fetch_objects(
                filters=combined_filter,
                limit=len(link_batch)
            )
            
            return [obj.properties for obj in results.objects]
        
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
            stats["QuestionsMiniLM_count"] = self._questions_collection.aggregate.over_all(total_count=True).total_count
            stats["DocumentsMpnet_count"] = self._docs_collection.aggregate.over_all(total_count=True).total_count
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def search_docs_by_keyword(
        self, 
        keyword: str, 
        limit: int = 10
    ) -> List[Dict]:
        """Search documents by keyword (BM25) in the Documentation collection."""
        print(f"[DEBUG] search_docs_by_keyword: Searching for keyword: '{keyword}'")
        if not keyword:
            print("[DEBUG] search_docs_by_keyword: Keyword is empty.")
            return []
        
        def _search():
            results = self._docs_collection.query.bm25(
                query=keyword,
                limit=limit
            )
            docs = []
            for obj in results.objects:
                docs.append(obj.properties.copy())
            print(f"[DEBUG] search_docs_by_keyword: Found {len(docs)} documents for keyword '{keyword}'.")
            return docs
        
        try:
            return self._retry_operation(_search)
        except Exception as e:
            logger.error(f"Error searching docs by keyword: {e}")
            print(f"[DEBUG] search_docs_by_keyword: Error: {e}")
            return []

    def search_questions_by_keyword(
        self, 
        keyword: str, 
        limit: int = 10
    ) -> List[Dict]:
        """Search questions by keyword (BM25) in the Questions collection."""
        print(f"[DEBUG] search_questions_by_keyword: Searching for keyword: '{keyword}'")
        if not keyword:
            print("[DEBUG] search_questions_by_keyword: Keyword is empty.")
            return []
        
        def _search():
            results = self._questions_collection.query.bm25(
                query=keyword,
                limit=limit
            )
            questions = []
            for obj in results.objects:
                questions.append(obj.properties.copy())
            print(f"[DEBUG] search_questions_by_keyword: Found {len(questions)} questions for keyword '{keyword}'.")
            return questions
        
        try:
            return self._retry_operation(_search)
        except Exception as e:
            logger.error(f"Error searching questions by keyword: {e}")
            print(f"[DEBUG] search_questions_by_keyword: Error: {e}")
            return []

    def get_sample_questions(
        self, 
        limit: int = 20, 
        random_sample: bool = False
    ) -> List[Dict]:
        """Get sample questions from the collection."""
        print(f"[DEBUG] get_sample_questions: Getting {limit} questions, random: {random_sample}")
        
        def _get_questions():
            properties_to_return = ["title", "question_content", "accepted_answer"]
            
            if random_sample:
                # For random sampling, we fetch more and then sample
                fetch_limit = min(limit * 5, 1000)  # Get more for better randomization
                results = self._questions_collection.query.fetch_objects(
                    limit=fetch_limit,
                    return_properties=properties_to_return
                )
            else:
                # Get first N questions
                results = self._questions_collection.query.fetch_objects(
                    limit=limit,
                    return_properties=properties_to_return
                )
            
            questions = []
            for obj in results.objects:
                question_data = obj.properties.copy()
                # Add the object ID for reference
                if hasattr(obj, 'uuid'):
                    question_data['id'] = str(obj.uuid)
                questions.append(question_data)
            
            print(f"[DEBUG] get_sample_questions: Found {len(questions)} questions")
            return questions
        
        try:
            return self._retry_operation(_get_questions)
        except Exception as e:
            logger.error(f"Error getting sample questions: {e}")
            print(f"[DEBUG] get_sample_questions: Error: {e}")
            return []

# Convenience functions for backward compatibility
def cargar_credenciales(ruta_env: str = ".env") -> Dict[str, str]:
    """Load credentials - maintained for backward compatibility."""
    config = WeaviateConfig.from_env(ruta_env)
    return {
        "WCS_API_KEY": config.wcs_api_key,
        "WCS_URL": config.wcs_url,
        "OPENAI_API_KEY": config.openai_api_key
    }

def conectar(client_config: Dict[str, str]):
    """Connect to Weaviate - maintained for backward compatibility."""
    config = WeaviateConfig(
        wcs_url=client_config["WCS_URL"],
        wcs_api_key=client_config["WCS_API_KEY"],
        openai_api_key=client_config["OPENAI_API_KEY"]
    )
    return get_weaviate_client(config)

# Example usage with improved error handling and logging
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        config = WeaviateConfig.from_env()
        client = get_weaviate_client(config)
        wrapper = WeaviateClientWrapper(client, retry_attempts=3)
        
        # Example operations
        stats = wrapper.get_collection_stats()
        print(f"Collection stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Weaviate client: {e}")