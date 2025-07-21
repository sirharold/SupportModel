from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from openai import OpenAI
from src.data.embedding_safe import EmbeddingClient
from sentence_transformers import CrossEncoder
from src.services.auth.auth import ensure_huggingface_login
from src.services.storage.weaviate_utils import WeaviateConfig
import numpy as np

def rerank_with_llm(question: str, docs: List[dict], openai_client: OpenAI, top_k: int = 10, embedding_model: str = None) -> List[dict]:
    """
    Reranks documents using a local CrossEncoder model with absolute score normalization.
    
    Uses sigmoid normalization instead of softmax to ensure scores are comparable
    across different embedding models regardless of the number of documents returned.
    
    Args:
        question: The query string
        docs: List of documents to rerank
        openai_client: OpenAI client (for compatibility)
        top_k: Number of top documents to return
        embedding_model: Name of the embedding model used (for logging/debugging)
    """
    if not docs:
        return []

    # Ensure we are logged in to Hugging Face before downloading the model
    config = WeaviateConfig.from_env()
    ensure_huggingface_login(token=config.huggingface_api_key)

    # The CrossEncoder model expects pairs of [query, passage]
    model_inputs = [[question, doc.get("content", "")] for doc in docs]
    
    # Initialize a light-weight, fast, and effective cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    
    # Predict the raw logit scores
    raw_scores = cross_encoder.predict(model_inputs)
    
    # Apply sigmoid normalization to CrossEncoder scores
    # NOTE: CrossEncoder scores ARE comparable across embedding models because
    # they use the same CrossEncoder model to evaluate all query-document pairs
    # regardless of which embedding model retrieved them initially
    try:
        raw_scores = np.array(raw_scores)
        # Apply sigmoid: 1 / (1 + e^(-x))
        # This maps CrossEncoder logits to [0,1] probabilities
        final_scores = 1 / (1 + np.exp(-raw_scores))
    except (OverflowError, ZeroDivisionError):
        # Fallback: Min-max normalization if sigmoid fails
        raw_scores = np.array(raw_scores)
        min_score = np.min(raw_scores)
        max_score = np.max(raw_scores)
        if max_score > min_score:
            final_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            final_scores = np.ones_like(raw_scores) * 0.5  # All equal scores
        print(f"[WARNING] Sigmoid normalization failed for {embedding_model}, using min-max normalization")

    # Add final scores to the documents
    for doc, score in zip(docs, final_scores):
        doc["score"] = float(score)
        
    # Sort documents by the new score in descending order
    sorted_docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
    
    return sorted_docs[:top_k]



def rerank_documents(query: str, docs: List[dict], embedding_client: EmbeddingClient, top_k: int = 10) -> List[dict]:
    if not docs:
        return []

    query_vec = embedding_client.generate_query_embedding(query)
    if not query_vec:
        raise ValueError("Query embedding could not be generated.")

    doc_vecs = [embedding_client.generate_document_embedding(doc.get("content", "")) for doc in docs]
    
    valid_docs = [doc for doc, vec in zip(docs, doc_vecs) if vec]
    valid_vecs = [vec for vec in doc_vecs if vec]

    if not valid_vecs:
        return []

    scores = cosine_similarity([query_vec], valid_vecs)[0]

    for doc, score in zip(valid_docs, scores):
        doc["score"] = float(score)

    return sorted(valid_docs, key=lambda d: d["score"], reverse=True)[:top_k]

