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
        try:
            doc_collection = self.client.collections.get("Documentation")
            results = doc_collection.query.near_vector(
                near_vector=vector,
                limit=top_k
            )
            return [obj.properties for obj in results.objects]
        except Exception as e:
            print("Error in search_docs_by_vector:", e)
            return []

    def lookup_docs_by_links(self, links: List[str], max_per_link: int = 10) -> List[dict]:
        results = []
        try:
            doc_collection = self.client.collections.get("Documentation")
            for link in links:
                try:
                    filter_obj = WeaviateFilter.by_property("link").equal(link)
                    query_result = doc_collection.query.fetch_objects(filters=filter_obj, limit=max_per_link)
                    results.extend([obj.properties for obj in query_result.objects])
                except Exception as e:
                    print(f"Error fetching documents for link '{link}':", e)
        except Exception as e:
            print("Error in lookup_docs_by_links:", e)
        return results
