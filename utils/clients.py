# utils/clients.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from config import EMBEDDING_MODELS, WEAVIATE_CLASS_CONFIG, GENERATIVE_MODELS
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper
from utils.embedding_safe import get_embedding_client

@st.cache_resource
def initialize_clients(model_name: str, generative_model_name: str = "gpt-4"):
    """
    Initializes and caches all required clients based on the selected models.
    
    Args:
        model_name (str): The key for the selected embedding model.
        generative_model_name (str): The key for the selected generative model.
        
    Returns:
        Tuple: A tuple containing the weaviate_wrapper, embedding_client,
               openai_client, gemini_client, and the raw weaviate client.
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
    
    openai_client = OpenAI(api_key=config.openai_api_key)
    
    # Initialize Gemini client
    gemini_api_key = getattr(config, 'gemini_api_key', None)
    gemini_client = None
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai.GenerativeModel(GENERATIVE_MODELS[generative_model_name])

    return weaviate_wrapper, embedding_client, openai_client, gemini_client, client
