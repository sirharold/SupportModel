import re
from typing import List, Tuple
from utils.extract_links import extract_urls_from_answer
from utils.reranker import rerank_documents
from utils.embedding import EmbeddingClient
from utils.weaviate_utils_improved import WeaviateClientWrapper

def answer_question(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_client: EmbeddingClient,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
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

        # 2. Buscar preguntas similares (Questions) - optimized limit
        similar_questions = weaviate_wrapper.search_questions_by_vector(vector, top_k=min(top_k*3, 30))
        debug_logs.append(f"🔹 Questions found: {len(similar_questions)}")

        # 3. Extraer links desde respuestas aceptadas con deduplicación temprana
        unique_links = set()
        for q in similar_questions:
            extracted = extract_urls_from_answer(q.get("accepted_answer", ""))
            unique_links.update(extracted)

        all_links = list(unique_links)
        debug_logs.append(f"🔹 Links extracted from answers: {len(all_links)}")
        debug_logs.append(f"🔹 Sample links: {all_links[:3]}")

        # 4. Recuperar documentos vinculados usando batch operation when available
        if hasattr(weaviate_wrapper, "lookup_docs_by_links_batch"):
            linked_docs = weaviate_wrapper.lookup_docs_by_links_batch(
                all_links, batch_size=50
            )
        else:
            linked_docs = weaviate_wrapper.lookup_docs_by_links(all_links)
        debug_logs.append(f"🔹 Linked documents found: {len(linked_docs)}")

        # 5. Buscar documentos directamente por vector con diversity filtering
        search_kwargs = {
            "vector": vector,
            "top_k": max(top_k * 2, 20),
        }
        if "diversity_threshold" in weaviate_wrapper.search_docs_by_vector.__code__.co_varnames:
            search_kwargs["diversity_threshold"] = diversity_threshold
            search_kwargs["include_distance"] = True
        vector_docs = weaviate_wrapper.search_docs_by_vector(**search_kwargs)
        debug_logs.append(f"🔹 Vector-retrieved documents: {len(vector_docs)}")

        # 6. Combinar y deduplicar con prioridad a documentos linked
        unique_docs_dict = {}
        
        # Primero agregar documentos linked (mayor prioridad)
        for doc in linked_docs:
            link = doc.get("link", "").strip()
            if link:
                unique_docs_dict[link] = doc
        
        # Luego agregar documentos de vector search si no existen
        for doc in vector_docs:
            link = doc.get("link", "").strip()
            if link and link not in unique_docs_dict:
                unique_docs_dict[link] = doc

        unique_docs = list(unique_docs_dict.values())
        debug_logs.append(f"🔹 Unique documents after optimized deduplication: {len(unique_docs)}")

        if not unique_docs:
            debug_logs.append("⚠️ No unique documents retrieved.")
            return [], "\n".join(debug_logs)

        # 7. Rerankear con límite optimizado
        max_docs_to_rerank = min(len(unique_docs), top_k*3)
        docs_to_rerank = unique_docs[:max_docs_to_rerank]
        reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        debug_logs.append(f"🔹 Documents sent to reranking: {len(docs_to_rerank)}")
        debug_logs.append(f"🔹 Documents after reranking: {len(reranked)}")
        return reranked, "\n".join(debug_logs)

    except Exception as e:
        debug_logs.append(f"❌ Error: {e}")
        return [], "\n".join(debug_logs)
