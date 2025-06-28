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
    openai_api_key: str
    
    @classmethod
    def from_env(cls, env_path: str = ".env") -> "WeaviateConfig":
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            
            required_keys = ["WCS_URL", "WCS_API_KEY", "OPENAI_API_KEY"]
            config_dict = {}
            
            for key in required_keys:
                value = os.getenv(key)
                if not value:
                    raise ValueError(f"Missing required environment variable: {key}")
                config_dict[key.lower()] = value
            
            return cls(
                wcs_url=config_dict["wcs_url"],
                wcs_api_key=config_dict["wcs_api_key"],
                openai_api_key=config_dict["openai_api_key"]
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
    
    def __init__(self, client, retry_attempts: int = 3, retry_delay: float = 1.0):
        if not client:
            raise ValueError("Weaviate client cannot be None")
        if not hasattr(client, "collections"):
            raise TypeError("Invalid client: missing 'collections' attribute")
        
        self.client = client
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._questions_collection = None
        self._docs_collection = None
    
    @property
    def questions_collection(self):
        """Lazy-loaded questions collection."""
        if self._questions_collection is None:
            self._questions_collection = self.client.collections.get("Questions")
        return self._questions_collection
    
    @property
    def docs_collection(self):
        """Lazy-loaded documentation collection."""
        if self._docs_collection is None:
            self._docs_collection = self.client.collections.get("Documentation")
        return self._docs_collection
    
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
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        def _search():
            if include_distance:
                results = self.questions_collection.query.near_vector(
                    near_vector=vector,
                    limit=top_k,
                    return_metadata=['distance']
                )
            else:
                results = self.questions_collection.query.near_vector(
                    near_vector=vector,
                    limit=top_k
                )
            
            questions = []
            for obj in results.objects:
                question_data = obj.properties.copy()
                if include_distance and obj.metadata:
                    question_data['distance'] = obj.metadata.distance
                questions.append(question_data)
            
            return questions
        
        try:
            return self._retry_operation(_search)
        except Exception as e:
            logger.error(f"Error searching questions by vector: {e}")
            return []
    
    def search_docs_by_vector(
        self, 
        vector: List[float], 
        top_k: int = 10,
        chunk_index: int = 1,
        diversity_threshold: float = 0.85,
        include_distance: bool = False
    ) -> List[Dict]:
        """Enhanced document search with diversity control."""
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        def _search():
            # Fetch more documents for better diversity
            fetch_limit = max(top_k * 3, 30)
            
            filter_chunk = WeaviateFilter.by_property("chunk_index").equal(chunk_index)
            
            try:
                # Try the new API with where parameter
                results = self.docs_collection.query.near_vector(
                    near_vector=vector,
                    limit=fetch_limit,
                    where=filter_chunk,
                    return_metadata=['distance'] if include_distance else None
                )
            except TypeError:
                # Fallback to old API without where parameter
                logger.warning("Using fallback API without where parameter")
                results = self.docs_collection.query.near_vector(
                    near_vector=vector,
                    limit=fetch_limit,
                    return_metadata=['distance'] if include_distance else None
                )
                # Filter results manually
                filtered_objects = []
                for obj in results.objects:
                    if obj.properties.get("chunk_index") == chunk_index:
                        filtered_objects.append(obj)
                # Create a mock results object
                class MockResults:
                    def __init__(self, objects):
                        self.objects = objects
                results = MockResults(filtered_objects)
            
            return self._apply_diversity_filtering(
                results.objects, 
                top_k, 
                diversity_threshold,
                include_distance
            )
        
        try:
            return self._retry_operation(_search)
        except Exception as e:
            logger.error(f"Error searching docs by vector: {e}")
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
        chunk_index: int = 1,
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
            
            chunk_filter = WeaviateFilter.by_property("chunk_index").equal(chunk_index)
            combined_filter = link_filter & chunk_filter
            
            results = self.docs_collection.query.fetch_objects(
                where=combined_filter,
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
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about collections."""
        try:
            stats = {}
            for collection_name in ["Questions", "Documentation"]:
                # This is a simplified stat - in practice you might use aggregation queries
                stats[collection_name] = "Available"
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

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