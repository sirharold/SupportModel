# utils/clients.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from config import EMBEDDING_MODELS, WEAVIATE_CLASS_CONFIG, GENERATIVE_MODELS
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper
from utils.embedding_safe import get_embedding_client
from utils.local_models import get_tinyllama_client, get_mistral_client
from utils.openrouter_client import get_cached_llama4_scout_client

@st.cache_resource
def initialize_clients(model_name: str, generative_model_name: str = "llama-4-scout"):
    """
    Initializes and caches all required clients based on the selected models.
    
    Args:
        model_name (str): The key for the selected embedding model.
        generative_model_name (str): The key for the selected generative model.
        
    Returns:
        Tuple: A tuple containing the weaviate_wrapper, embedding_client,
               openai_client, gemini_client, local_tinyllama_client, local_mistral_client, 
               openrouter_client, and the raw weaviate client.
    """
    config = WeaviateConfig.from_env()
    client = get_weaviate_client(config)
    
    weaviate_classes = WEAVIATE_CLASS_CONFIG[model_name]
    weaviate_wrapper = WeaviateClientWrapper(
        client,
        documents_class=weaviate_classes["documents"],
        questions_class=weaviate_classes["questions"],
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
    if gemini_api_key and generative_model_name == "gemini-pro":
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai.GenerativeModel(GENERATIVE_MODELS[generative_model_name])
    
    # Initialize local model clients
    local_tinyllama_client = None
    local_mistral_client = None
    
    if generative_model_name == "tinyllama-1.1b":
        local_tinyllama_client = get_tinyllama_client()
    elif generative_model_name == "mistral-7b":
        local_mistral_client = get_mistral_client()
    
    # Initialize OpenRouter client
    openrouter_client = None
    if generative_model_name == "llama-4-scout":
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
    
    return weaviate_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client
