# config.py

# Embedding model configurations
EMBEDDING_MODELS = {
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "ada": "text-embedding-ada-002"
}

DEFAULT_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

# Generative model configurations
GENERATIVE_MODELS = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
    "gpt-4": "gpt-4",
    "gemini-pro": "gemini-pro",
    "tinyllama-1.1b": "tinyllama-1.1b",
    "deepseek-v3-chat": "deepseek/deepseek-chat:free"
}

DEFAULT_GENERATIVE_MODEL = "llama-3.3-70b"  # Changed to OpenRouter model for better performance

# Debug configuration
DEBUG_MODE = False  # Set to True to enable debug messages, False to disable them

# Weaviate class names for different models
WEAVIATE_CLASS_CONFIG = {
    "multi-qa-mpnet-base-dot-v1": {
        "documents": "DocumentsMpnet",
        "questions": "QuestionsMlpnet"
    },
    "all-MiniLM-L6-v2": {
        "documents": "DocumentsMiniLM",
        "questions": "QuestionsMiniLM"
    },
    "ada": {
        "documents": "Documentation",
        "questions": "Questions"
    }
}

# Model descriptions for the comparison page
MODEL_DESCRIPTIONS = {
    "multi-qa-mpnet-base-dot-v1": {
        "description": "Modelo optimizado para Question Answering (QA). Ideal para cuando la consulta es una pregunta directa.",
        "dimensions": 768,
        "provider": "Hugging Face"
    },
    "all-MiniLM-L6-v2": {
        "description": "Modelo rápido y versátil, bueno para búsqueda semántica general y clustering. Un gran balance entre velocidad y precisión.",
        "dimensions": 384,
        "provider": "Hugging Face"
    },
    "ada": {
        "description": "Potente modelo de propósito general de OpenAI. Excelente en capturar el significado semántico en una amplia gama de textos.",
        "dimensions": 1536,
        "provider": "OpenAI"
    }
}

# Generative model descriptions
GENERATIVE_MODEL_DESCRIPTIONS = {
    "llama-3.3-70b": {
        "description": "Llama 3.3 70B Instruct vía OpenRouter. Modelo avanzado gratuito con excelente rendimiento.",
        "provider": "Meta (OpenRouter)",
        "cost": "Gratuito",
        "requirements": "API key de OpenRouter requerida"
    },
    "tinyllama-1.1b": {
        "description": "Modelo local TinyLlama 1.1B. Extremadamente liviano y rápido sin costos de API.",
        "provider": "TinyLlama (Local)",
        "cost": "Gratuito",
        "requirements": "1GB RAM, perfecto para laptops antiguos"
    },
    "deepseek-v3-chat": {
        "description": "DeepSeek V3 Chat vía OpenRouter. Modelo de última generación optimizado para QA y razonamiento con contexto de 163k tokens.",
        "provider": "DeepSeek (OpenRouter)",
        "cost": "Gratuito", 
        "requirements": "API key de OpenRouter requerida"
    },
    "gpt-4": {
        "description": "GPT-4 de OpenAI. Modelo de alta calidad con costos por uso.",
        "provider": "OpenAI",
        "cost": "Pagado",
        "requirements": "API key de OpenAI requerida"
    },
    "gemini-pro": {
        "description": "Gemini Pro de Google. Modelo multimodal con costos por uso.",
        "provider": "Google",
        "cost": "Pagado",
        "requirements": "API key de Google requerida"
    }
}

# Legacy alias for backward compatibility
LOCAL_MODEL_DESCRIPTIONS = {
    k: v for k, v in GENERATIVE_MODEL_DESCRIPTIONS.items() 
    if "Local" in v.get("provider", "")
}