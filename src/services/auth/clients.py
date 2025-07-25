# utils/clients.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from src.config.config import EMBEDDING_MODELS, CHROMADB_COLLECTION_CONFIG, GENERATIVE_MODELS
from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client, ChromaDBClientWrapper
from src.data.embedding_safe import get_embedding_client
from src.services.local_models import get_tinyllama_client, get_mistral_client
from src.services.auth.openrouter_client import get_cached_llama4_scout_client, get_cached_deepseek_openrouter_client

@st.cache_resource
def initialize_clients(model_name: str, generative_model_name: str = "llama-4-scout"):
    """
    Initializes and caches all required clients based on the selected models.
    
    Args:
        model_name (str): The key for the selected embedding model.
        generative_model_name (str): The key for the selected generative model.
        
    Returns:
        Tuple: A tuple containing the chromadb_wrapper, embedding_client,
               openai_client, gemini_client, local_tinyllama_client, local_mistral_client, 
               openrouter_client, and the raw chromadb client.
    """
    config = ChromaDBConfig.from_env()
    client = get_chromadb_client(config)
    
    chromadb_collections = CHROMADB_COLLECTION_CONFIG[model_name]
    chromadb_wrapper = ChromaDBClientWrapper(
        client,
        documents_class=chromadb_collections["documents"],
        questions_class=chromadb_collections["questions"],
        retry_attempts=3
    )
    
    embedding_client = get_embedding_client(
        model_name=EMBEDDING_MODELS[model_name],
        huggingface_api_key=config.huggingface_api_key,
        openai_api_key=config.openai_api_key
    )
    
    # Initialize OpenAI client (only if API key is available)
    openai_client = None
    if hasattr(config, 'openai_api_key') and config.openai_api_key:
        openai_client = OpenAI(api_key=config.openai_api_key)
    
    # Initialize Gemini client (only if API key is available)
    gemini_api_key = getattr(config, 'gemini_api_key', None)
    gemini_client = None
    if gemini_api_key and generative_model_name == "gemini-1.5-flash":
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai.GenerativeModel(GENERATIVE_MODELS[generative_model_name])
    
    # Initialize local model clients
    local_tinyllama_client = None
    local_mistral_client = None
    
    if generative_model_name == "tinyllama-1.1b":
        local_tinyllama_client = get_tinyllama_client()
    elif generative_model_name == "deepseek-v3-chat":
        # Use OpenRouter DeepSeek V3 Chat instead of local
        try:
            openrouter_client = get_cached_deepseek_openrouter_client()
            if openrouter_client and hasattr(openrouter_client, 'test_connection'):
                if not openrouter_client.test_connection():
                    st.warning("⚠️ OpenRouter DeepSeek client inicializado pero la conexión falló. Verifica tu API key.")
        except Exception as e:
            st.error(f"❌ No se pudo inicializar OpenRouter client para DeepSeek: {e}")
            st.info("💡 **Solución**: Verifica que OPEN_ROUTER_KEY esté configurado en tu archivo .env")
            import os
            if os.getenv('OPEN_ROUTER_KEY'):
                st.info("✅ OPEN_ROUTER_KEY encontrado en variables de entorno")
            else:
                st.error("❌ OPEN_ROUTER_KEY no encontrado en variables de entorno")
    elif generative_model_name == "mistral-7b":
        local_mistral_client = get_mistral_client()
    
    # Initialize OpenRouter client
    openrouter_client = None
    if generative_model_name == "llama-3.3-70b":
        try:
            openrouter_client = get_cached_llama4_scout_client()
            # Test the connection
            if openrouter_client and hasattr(openrouter_client, 'test_connection'):
                if not openrouter_client.test_connection():
                    st.warning("⚠️ OpenRouter client inicializado pero la conexión falló. Verifica tu API key.")
        except Exception as e:
            st.error(f"❌ No se pudo inicializar OpenRouter client: {e}")
            st.info("💡 **Solución**: Verifica que OPEN_ROUTER_KEY esté configurado en tu archivo .env")
            # Check if the API key is available in environment
            import os
            if os.getenv('OPEN_ROUTER_KEY'):
                st.info("✅ OPEN_ROUTER_KEY encontrado en variables de entorno")
            else:
                st.error("❌ OPEN_ROUTER_KEY no encontrado en variables de entorno")
    
    return chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client
