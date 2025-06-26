import os
from utils.weaviate_utils import conectar, WeaviateClientWrapper
from utils.embedding import EmbeddingClient
from utils.qa_pipeline import answer_question

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

def safe_main():
    client = None
    try:
        config = cargar_credenciales()

        if not all(config.values()):
            raise ValueError("Missing required environment variables.")

        client = conectar(config)
        weaviate_wrapper = WeaviateClientWrapper(client)

        embedding_client = EmbeddingClient(api_key=config["OPENAI_API_KEY"], model="text-embedding-ada-002")

        question = "How can I enable and use Managed Identity in an Azure Function to securely connect to Azure SQL, Key Vault, or Service Bus without using connection strings or secrets?"
        answer_question(question, weaviate_wrapper, embedding_client,10)

    except Exception as e:
        print("An error occurred during execution:")
        print(str(e))

    finally:
        if client:
            client.close()

if __name__ == "__main__":
    safe_main()
