import re
from typing import List, Tuple
from utils.extract_links import extract_urls_from_answer
from utils.reranker import rerank_documents
from utils.embedding import EmbeddingClient
from utils.weaviate_utils import WeaviateClientWrapper

def answer_question(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_client: EmbeddingClient,
    top_k: int = 10
) -> Tuple[List[dict], str]:
    """
    Realiza el pipeline completo para responder una pregunta:
    1. Embedding de la pregunta
    2. Búsqueda de preguntas similares (Questions)
    3. Extracción de links desde respuestas aceptadas
    4. Recuperación de documentos vinculados y búsqueda por vector
    5. Reranking de documentos
    6. Devolver documentos + info de debug
    """
    debug_logs = []

    try:
        # 1. Embedding de la pregunta
        vector = embedding_client.generate_embedding(question)
        debug_logs.append(f"🔹 Query vector length: {len(vector)}")
        debug_logs.append(f"🔹 top_k: {top_k}")

        # 2. Buscar preguntas similares (Questions)
        similar_questions = weaviate_wrapper.search_questions_by_vector(vector, top_k=top_k*5)
        debug_logs.append(f"🔹 Questions found: {len(similar_questions)}")

        # 3. Extraer links desde respuestas aceptadas
        all_links = []
        for q in similar_questions:
            extracted = extract_urls_from_answer(q.get("accepted_answer", ""))
            all_links.extend(extracted)

        debug_logs.append(f"🔹 Links extracted from answers: {len(all_links)}")
        debug_logs.append(f"🔹 Unique links extracted: {len(set(all_links))}")
        debug_logs.append(f"🔹 Unique links extracted: {all_links}")

        # 4. Recuperar documentos vinculados
        linked_docs = weaviate_wrapper.lookup_docs_by_links(all_links)
        debug_logs.append(f"🔹 Linked documents found: {len(linked_docs)}")

        # 5. Buscar documentos directamente por vector
        vector_docs = weaviate_wrapper.search_docs_by_vector(vector, top_k=top_k)
        debug_logs.append(f"🔹 Vector-retrieved documents: {len(vector_docs)}")

        combined_docs = linked_docs + vector_docs
        unique_docs_dict = {}
        for doc in combined_docs:
            link = doc.get("link", "").strip()
            if link and link not in unique_docs_dict:
                unique_docs_dict[link] = doc

        unique_docs = list(unique_docs_dict.values())
        debug_logs.append(f"🔹 Unique documents after deduplication: {len(unique_docs)}")

        if not unique_docs:
            debug_logs.append("⚠️ No unique documents retrieved.")
            return [], "\n".join(debug_logs)

        # 6. Rerankear
        reranked = rerank_documents(question, unique_docs, embedding_client, top_k=top_k)
        debug_logs.append(f"🔹 Documents after reranking: {len(reranked)}")
        return reranked, "\n".join(debug_logs)

    except Exception as e:
        debug_logs.append(f"❌ Error: {e}")
        return [], "\n".join(debug_logs)
