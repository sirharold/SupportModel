from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from openai import OpenAI
from utils.embedding_safe import EmbeddingClient
from sentence_transformers import CrossEncoder
from utils.auth import ensure_huggingface_login
from utils.weaviate_utils_improved import WeaviateConfig
import numpy as np

def rerank_with_llm(question: str, docs: List[dict], openai_client: OpenAI, top_k: int = 10) -> List[dict]:
    """
    Reranks documents using a local CrossEncoder model and normalizes scores.
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
    
    # Normalize the scores using a Softmax function to get probabilities
    softmax_scores = np.exp(raw_scores) / np.sum(np.exp(raw_scores))

    # Add normalized scores to the documents
    for doc, score in zip(docs, softmax_scores):
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

