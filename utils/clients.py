# utils/clients.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from config import EMBEDDING_MODELS, WEAVIATE_CLASS_CONFIG, GENERATIVE_MODELS
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper
from utils.embedding_safe import get_embedding_client
from utils.local_models import get_llama_client, get_mistral_client

@st.cache_resource
def initialize_clients(model_name: str, generative_model_name: str = "llama-3.1-8b"):
    """
    Initializes and caches all required clients based on the selected models.
    
    Args:
        model_name (str): The key for the selected embedding model.
        generative_model_name (str): The key for the selected generative model.
        
    Returns:
        Tuple: A tuple containing the weaviate_wrapper, embedding_client,
               openai_client, gemini_client, local_llama_client, local_mistral_client, and the raw weaviate client.
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
    local_llama_client = None
    local_mistral_client = None
    
    if generative_model_name == "llama-3.1-8b":
        local_llama_client = get_llama_client()
    elif generative_model_name == "mistral-7b":
        local_mistral_client = get_mistral_client()
    
    return weaviate_wrapper, embedding_client, openai_client, gemini_client, local_llama_client, local_mistral_client, client
