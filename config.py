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
    "gpt-4": "gpt-4",
    "gemini-pro": "gemini-pro"
}

DEFAULT_GENERATIVE_MODEL = "gpt-4"

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