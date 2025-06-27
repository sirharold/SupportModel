from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from utils.embedding import EmbeddingClient

def rerank_documents(query: str, docs: List[dict], embedding_client: EmbeddingClient, top_k: int = 10) -> List[dict]:
    try:
        query_vec = embedding_client.generate_embedding(query)
        if not query_vec:
            raise ValueError("Query embedding could not be generated.")

        valid_docs = []
        valid_vecs = []

        for doc in docs:
            content = doc.get("content", "")
            vec = embedding_client.generate_embedding(content)
            if vec:  # solo incluimos si el embedding es válido
                valid_docs.append(doc)
                valid_vecs.append(vec)

        if not valid_vecs:
            print("⚠️ No valid document embeddings generated.")
            return []

        scores = cosine_similarity([query_vec], valid_vecs)[0]

        for doc, score in zip(valid_docs, scores):
            doc["score"] = float(score)

        ranked_docs = sorted(valid_docs, key=lambda d: d["score"], reverse=True)
        return ranked_docs[:top_k]

    except Exception as e:
        print("❌ Error in rerank_documents:", e)
        return []
