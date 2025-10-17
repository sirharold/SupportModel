# config.py

# Embedding model configurations
EMBEDDING_MODELS = {
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "ada": "text-embedding-ada-002",
    "e5-large-v2": "intfloat/e5-large-v2"
}

DEFAULT_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

# Generative model configurations
GENERATIVE_MODELS = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
    "gpt-4": "gpt-4",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "tinyllama-1.1b": "tinyllama-1.1b",
    "deepseek-v3-chat": "deepseek/deepseek-r1:free",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet:beta"
}

DEFAULT_GENERATIVE_MODEL = "llama-3.3-70b"  # Free OpenRouter model - most reliable

# Debug configuration
DEBUG_MODE = False  # Set to True to enable debug messages, False to disable them

# ChromaDB collection names for different models
CHROMADB_COLLECTION_CONFIG = {
    "multi-qa-mpnet-base-dot-v1": {
        "documents": "docs_mpnet",
        "questions": "questions_mpnet",
        "questions_withlinks": "questions_withlinks"  # Colección optimizada
    },
    "all-MiniLM-L6-v2": {
        "documents": "docs_minilm",
        "questions": "questions_minilm",
        "questions_withlinks": "questions_withlinks"  # Colección optimizada
    },
    "ada": {
        "documents": "docs_ada",
        "questions": "questions_ada",
        "questions_withlinks": "questions_withlinks"  # Colección optimizada
    },
    "e5-large-v2": {
        "documents": "docs_e5large",
        "questions": "questions_e5large",
        "questions_withlinks": "questions_withlinks"  # Colección optimizada
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
    },
    "e5-large-v2": {
        "description": "Modelo E5-Large-v2 de Microsoft. Excelente rendimiento en tareas de retrieval con embeddings de 1024 dimensiones.",
        "dimensions": 1024,
        "provider": "Microsoft (Hugging Face)"
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
        "description": "DeepSeek V3 vía OpenRouter. Modelo de última generación optimizado para QA y razonamiento con contexto de 163k tokens.",
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
    "gemini-1.5-flash": {
        "description": "Gemini 1.5 Flash de Google. Modelo rápido y gratuito optimizado para tareas de evaluación.",
        "provider": "Google",
        "cost": "Gratuito",
        "requirements": "API key de Google requerida"
    },
    "claude-3.5-sonnet": {
        "description": "Claude 3.5 Sonnet de Anthropic vía OpenRouter. Modelo avanzado con excelente capacidad de análisis y razonamiento.",
        "provider": "Anthropic (OpenRouter)",
        "cost": "Pagado",
        "requirements": "API key de OpenRouter requerida"
    }
}

# Legacy alias for backward compatibility
LOCAL_MODEL_DESCRIPTIONS = {
    k: v for k, v in GENERATIVE_MODEL_DESCRIPTIONS.items() 
    if "Local" in v.get("provider", "")
}