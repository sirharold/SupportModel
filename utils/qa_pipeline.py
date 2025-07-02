import re
from typing import List, Tuple
from openai import OpenAI
from utils.extract_links import extract_urls_from_answer
from utils.reranker import rerank_documents, rerank_with_llm
from utils.embedding import EmbeddingClient
from utils.weaviate_utils_improved import WeaviateClientWrapper

def _expand_query_with_llm(question: str, openai_client: OpenAI) -> Tuple[str, str]:
    """
    Usa un LLM para expandir la pregunta del usuario con variantes técnicas.
    Retorna la pregunta expandida y un log de la operación.
    """
    try:
        prompt = (
            "You are an Azure expert acting as a search query optimizer. "
            "Expand the following user question to improve search results in a technical documentation database. "
            "Generate 3 more detailed or alternative technical phrasings. "
            "Do NOT answer the question. ONLY generate the 3 variations, each on a new line.\n\n"
            f"Original Question: {question}"
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful Azure documentation expert and search query optimizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            n=1
        )
        
        variations = response.choices[0].message.content.strip()
        expanded_query = f"{question}\n{variations}"
        log_message = f"🔹 Query expanded successfully.\nExpanded Query:\n---\n{expanded_query}\n---"
        return expanded_query, log_message
    except Exception as e:
        log_message = f"⚠️ Query expansion failed: {e}. Falling back to original question."
        return question, log_message

def answer_question(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = False
) -> Tuple[List[dict], str]:
    print("[DEBUG] Entering answer_question function.")
    """
    Realiza el pipeline completo para responder una pregunta:
    1. Expansión de la pregunta con LLM (NUEVO)
    2. Embedding de la pregunta expandida
    3. Búsqueda de preguntas similares (Questions)
    4. Extracción de links desde respuestas aceptadas
    5. Recuperación de documentos vinculados y búsqueda por vector
    6. Reranking de documentos
    7. Devolver documentos + info de debug
    """
    debug_logs = []

    try:
        # 1. Expansión de la pregunta con LLM
        expanded_question, expansion_log = _expand_query_with_llm(question, openai_client)
        debug_logs.append(expansion_log)
        print(f"[DEBUG] Expanded Question used for embedding: {expanded_question}")

        # 2. Embedding de la pregunta expandida
        vector = embedding_client.generate_embedding(expanded_question)
        print(f"[DEBUG] Query vector generated. Length: {len(vector)}")
        debug_logs.append(f"🔹 Query vector length: {len(vector)}")
        debug_logs.append(f"🔹 top_k: {top_k}")

        # 3. Buscar preguntas similares (Questions) - optimized limit
        if use_questions_collection:
            similar_questions = weaviate_wrapper.search_questions_by_vector(vector, top_k=min(top_k*3, 30))
            debug_logs.append(f"🔹 Questions found: {len(similar_questions)}")
            print("[DEBUG] Similar Questions retrieved:")
            for i, q in enumerate(similar_questions):
                print(f"[DEBUG]   Q {i+1}: Title: {q.get('title', 'N/A')}, Accepted Answer: {q.get('accepted_answer', 'N/A')[:100]}...")

            # 4. Extraer links desde respuestas aceptadas con deduplicación temprana
            unique_links = set()
            for q in similar_questions:
                extracted = extract_urls_from_answer(q.get("accepted_answer", ""))
                unique_links.update(extracted)

            all_links = list(unique_links)
            debug_logs.append(f"🔹 Links extracted from answers: {len(all_links)}")
            debug_logs.append(f"🔹 Sample links: {all_links[:3]}")

            # 5. Recuperar documentos vinculados usando batch operation when available
            if hasattr(weaviate_wrapper, "lookup_docs_by_links_batch"):
                linked_docs = weaviate_wrapper.lookup_docs_by_links_batch(
                    all_links, batch_size=50
                )
            else:
                linked_docs = weaviate_wrapper.lookup_docs_by_links(all_links)
            debug_logs.append(f"🔹 Linked documents found: {len(linked_docs)}")
            print("[DEBUG] Linked Documents retrieved:")
            for i, doc in enumerate(linked_docs):
                print(f"[DEBUG]   Linked Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")
        else:
            debug_logs.append("🔹 Skipping Questions collection search.")
            similar_questions = []
            linked_docs = []

        # 6. Buscar documentos directamente por vector con diversity filtering
        search_kwargs = {
            "vector": vector,
            "top_k": max(top_k * 2, 20),
        }
        if "diversity_threshold" in weaviate_wrapper.search_docs_by_vector.__code__.co_varnames:
            search_kwargs["diversity_threshold"] = diversity_threshold
            search_kwargs["include_distance"] = True
        vector_docs = weaviate_wrapper.search_docs_by_vector(**search_kwargs)
        debug_logs.append(f"🔹 Vector-retrieved documents: {len(vector_docs)}")
        print("[DEBUG] Vector-Retrieved Documents:")
        for i, doc in enumerate(vector_docs):
            print(f"[DEBUG]   Vector Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")

        # 7. Combinar y deduplicar con prioridad a documentos linked
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

        print("[DEBUG] Documents retrieved from Weaviate (before reranking):")
        for i, doc in enumerate(unique_docs):
            print(f"[DEBUG]   Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")

        # 8. Reranking (condicional)
        debug_logs.append(f"🔹 Preparing for reranking. LLM Reranker enabled: {use_llm_reranker}")
        print(f"[DEBUG] In qa_pipeline: LLM Reranker enabled: {use_llm_reranker}")
        print(f"[DEBUG] In qa_pipeline: Number of unique_docs: {len(unique_docs)}")
        print(f"[DEBUG] In qa_pipeline: openai_client is None: {openai_client is None}")
        max_docs_to_rerank = min(len(unique_docs), 40 if use_llm_reranker else top_k * 3)
        docs_to_rerank = unique_docs[:max_docs_to_rerank]

        if use_llm_reranker:
            try:
                debug_logs.append(f"🔹 Using LLM to rerank {len(docs_to_rerank)} documents...")
                reranked = rerank_with_llm(question, docs_to_rerank, openai_client, top_k=top_k)
            except Exception as e:
                print(f"[DEBUG] ERROR during LLM reranking: {e}")
                debug_logs.append(f"❌ Error during LLM reranking: {e}. Falling back to standard reranking.")
                reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        else:
            debug_logs.append(f"🔹 Using standard embedding similarity to rerank {len(docs_to_rerank)} documents...")
            reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        
        debug_logs.append(f"🔹 Documents after reranking: {len(reranked)}")
        return reranked, "\n".join(debug_logs)

    except Exception as e:
        debug_logs.append(f"❌ Error: {e}")
        return [], "\n".join(debug_logs)

