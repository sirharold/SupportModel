from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from utils.embedding import EmbeddingClient

def rerank_documents(query: str, docs: List[dict], embedding_client: EmbeddingClient, top_k: int = 10) -> List[dict]:
    try:
        query_vec = embedding_client.generate_embedding(query)
        if not query_vec:
            raise ValueError("Query embedding could not be generated.")

        doc_vecs = [embedding_client.generate_embedding(doc.get("content", "")) for doc in docs]
        if any(len(vec) == 0 for vec in doc_vecs):
            raise ValueError("Some document embeddings could not be generated.")

        scores = cosine_similarity([query_vec], doc_vecs)[0]
        for doc, score in zip(docs, scores):
            doc["score"] = float(score)

        ranked_docs = sorted(docs, key=lambda d: d["score"], reverse=True)
        return ranked_docs[:top_k]

    except Exception as e:
        print("Error in rerank_documents:", e)
        return []
