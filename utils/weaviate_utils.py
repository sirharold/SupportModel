import os
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey
from typing import List
from weaviate.collections.classes.filters import Filter as WeaviateFilter

def cargar_credenciales(ruta_env: str = ".env") -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(ruta_env)
        return {
            "WCS_API_KEY": os.getenv("WCS_API_KEY"),
            "WCS_URL": os.getenv("WCS_URL"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        }
    except Exception as e:
        print("Error loading environment variables:", e)
        return {}

def conectar(client_config: dict):
    try:
        if not all(k in client_config for k in ("WCS_URL", "WCS_API_KEY", "OPENAI_API_KEY")):
            raise ValueError("Missing one or more required configuration keys: 'WCS_URL', 'WCS_API_KEY', 'OPENAI_API_KEY'")

        client = connect_to_weaviate_cloud(
            cluster_url=client_config["WCS_URL"],
            auth_credentials=AuthApiKey(client_config["WCS_API_KEY"]),
            headers={"X-OpenAI-Api-Key": client_config["OPENAI_API_KEY"]}
        )

        if not hasattr(client, "collections"):
            raise RuntimeError("Weaviate client initialized but 'collections' attribute is missing.")

        return client

    except Exception as e:
        print(client_config)
        print("Error connecting to Weaviate:", e)
        return None

class WeaviateClientWrapper:
    def __init__(self, client):
        if not client:
            raise ValueError("Weaviate client is None. Ensure connection is established before wrapping.")
        if not hasattr(client, "collections") or not callable(getattr(client.collections, "get", None)):
            raise TypeError("Provided client is invalid or missing the 'collections.get' method.")
        self.client = client

    def close(self):
        """Close the underlying weaviate client connection."""
        if hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception as e:
                print("Error closing Weaviate client:", e)

    def search_questions_by_vector(self, vector: List[float], top_k: int = 10) -> List[dict]:
        try:
            
            questions_collection = self.client.collections.get("Questions")
            results = questions_collection.query.near_vector(
                near_vector=vector,
                limit=top_k
            )
            return [obj.properties for obj in results.objects]
        except Exception as e:
            print("Error in search_questions_by_vector:", e)
            return []

    def search_docs_by_vector(self, vector: List[float], top_k: int = 10) -> List[dict]:
        """Search documents by vector with optimized filtering and higher fetch limit."""
        try:
            doc_collection = self.client.collections.get("Documentation")
            
            # Filter for chunk_index=1 at query time for better performance
            filter_chunk = WeaviateFilter.by_property("chunk_index").equal(1)
            
            # Increased fetch limit for better coverage
            fetch_limit = max(top_k * 5, 50)
            results = doc_collection.query.near_vector(
                near_vector=vector,
                limit=fetch_limit,
                where=filter_chunk
            )

            # Deduplicate by link while preserving order
            filtered = []
            seen_links = set()
            for obj in results.objects:
                props = obj.properties
                link = props.get("link")
                if link and link not in seen_links:
                    filtered.append(props)
                    seen_links.add(link)

            return filtered
        except Exception as e:
            print("Error in search_docs_by_vector:", e)
            return []

    def lookup_docs_by_links(self, links: List[str], max_per_link: int = 10) -> List[dict]:
        """Fetch documents for provided links returning only chunk_index 1 results."""
        results = []
        seen_links = set()
        try:
            doc_collection = self.client.collections.get("Documentation")
            for link in links:
                if link in seen_links:
                    continue
                try:
                    filter_link = WeaviateFilter.by_property("link").equal(link)
                    filter_chunk = WeaviateFilter.by_property("chunk_index").equal(1)
                    filter_obj = filter_link & filter_chunk
                    query_result = doc_collection.query.fetch_objects(filters=filter_obj, limit=1)
                    if query_result.objects:
                        results.append(query_result.objects[0].properties)
                        seen_links.add(link)
                except Exception as e:
                    print(f"Error fetching documents for link '{link}':", e)
        except Exception as e:
            print("Error in lookup_docs_by_links:", e)
        return results
