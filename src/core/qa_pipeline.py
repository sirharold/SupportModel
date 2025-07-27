from typing import List, Tuple, Dict, Union
from openai import OpenAI
import google.generativeai as genai
from src.data.extract_links import extract_urls_from_answer
from src.core.reranker import rerank_documents, rerank_with_llm
from src.data.embedding import EmbeddingClient
from src.services.storage.chromadb_utils import ChromaDBClientWrapper
from src.services.answer_generation.base import generate_final_answer, evaluate_answer_quality
from src.services.answer_generation.gemini import generate_final_answer_gemini
from src.services.answer_generation.local import generate_final_answer_local, refine_query_local
from src.config.config import DEBUG_MODE

def debug_print(message: str, force: bool = False):
    """Print debug message only if DEBUG_MODE is enabled or force is True."""
    if DEBUG_MODE or force:
        print(message)

def refine_and_prepare_query(question: str, gemini_client: genai.GenerativeModel, model_name: str, 
                           local_mistral_client=None, openrouter_client=None, use_local_refinement: bool = True) -> Tuple[str, str]:
    """
    Cleans, distills, and conditionally prefixes a user query for optimal performance.
    Now supports local model refinement for cost optimization.
    """
    logs = []
    try:
        # Use local model refinement if available and enabled
        if use_local_refinement and local_mistral_client:
            refined_query, refinement_log = refine_query_local(question, "mistral-7b")
            logs.append(refinement_log)
        # Use OpenRouter model refinement if available (DISABLED for free models to save API calls)
        elif use_local_refinement and openrouter_client:
            from src.services.answer_generation.openrouter import refine_query_openrouter
            refined_query, refinement_log = refine_query_openrouter(question, "deepseek-v3-chat")
            logs.append(refinement_log)
            
            # Apply conditional prefixing
            if "multi-qa-mpnet-base-dot-v1" in model_name:
                final_query = "query: " + refined_query
                logs.append("🔹 Added 'query: ' prefix for mpnet model.")
            else:
                final_query = refined_query
                logs.append("🔹 No prefix added for this model.")
            
            return final_query, "\n".join(logs)
        
        # Fallback to original Gemini refinement if local not available
        elif gemini_client:
            # Step 1: Noise and Salutation Removal
            cleaning_prompt = (
                "You are a text cleaning expert. Your task is to remove all greetings, "
                "pleasantries, signatures, and any other conversational filler from the "
                "following user query. Output ONLY the core technical question."
                f"\n\nOriginal Query: {question}"
            )
            
            cleaned_response = gemini_client.generate_content(cleaning_prompt)
            cleaned_question = cleaned_response.text.strip()
            logs.append(f"🔹 Query after cleaning: {cleaned_question}")

            # Step 2: Core Question Distillation
            distillation_prompt = (
                "Based on the following text, distill it into a single, clear, and "
                "concise technical question suitable for a vector database search. "
                "Remove any ambiguity and focus on the essential problem."
                f"\n\nCleaned Text: {cleaned_question}"
            )

            distilled_response = gemini_client.generate_content(distillation_prompt)
            distilled_question = distilled_response.text.strip()
            logs.append(f"🔹 Query after distillation: {distilled_question}")

            # Step 3: Conditional Prefixing
            if "multi-qa-mpnet-base-dot-v1" in model_name:
                final_query = "query: " + distilled_question
                logs.append("🔹 Added 'query: ' prefix for mpnet model.")
            else:
                final_query = distilled_question
                logs.append("🔹 No prefix added for this model.")
                
            return final_query, "\n".join(logs)
        
        # No refinement available, use original query
        else:
            logs.append("🔹 No refinement available, using original query.")
            final_query = question
            if "multi-qa-mpnet-base-dot-v1" in model_name:
                final_query = "query: " + question
                logs.append("🔹 Added 'query: ' prefix for mpnet model.")
            
            return final_query, "\n".join(logs)

    except Exception as e:
        logs.append(f"⚠️ Query refinement failed: {e}. Falling back to original question.")
        return question, "\n".join(logs)

def answer_question(
    question: str,
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    gemini_client: genai.GenerativeModel = None,
    local_tinyllama_client = None,
    local_mistral_client = None,
    openrouter_client = None,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = True,
    generate_answer: bool = True,
    evaluate_quality: bool = False,
    documents_class: str = "Documents",
    questions_class: str = "Questions",
    generative_model_name: str = "tinyllama-1.1b",
    use_local_refinement: bool = True
) -> Union[Tuple[List[dict], str], Tuple[List[dict], str, str, Dict]]:
    debug_print("[DEBUG] Entering answer_question function.")
    """
    Realiza el pipeline completo RAG para responder una pregunta:
    1. Expansión de la pregunta con LLM
    2. Embedding de la pregunta expandida
    3. Búsqueda de preguntas similares (Questions)
    4. Extracción de links desde respuestas aceptadas
    5. Recuperación de documentos vinculados y búsqueda por vector
    6. Reranking de documentos
    7. GENERACIÓN: Síntesis de respuesta final usando documentos (NUEVO)
    8. EVALUACIÓN: Métricas de calidad RAG (opcional)
    
    Args:
        generate_answer: Si generar respuesta final (True) o solo retornar documentos (False)
        evaluate_quality: Si evaluar calidad de la respuesta generada
    
    Returns:
        Si generate_answer=False: (documentos, debug_info)
        Si generate_answer=True: (documentos, debug_info, respuesta_generada, rag_metrics)
    """
    debug_logs = []

    try:
        # 1. Conditionally refine and prepare the query
        if "ada" in embedding_client.model_name:
            refined_query = question
            refinement_log = "🔹 Skipping query refinement for Ada model."
        else:
            refined_query, refinement_log = refine_and_prepare_query(
                question, gemini_client, embedding_client.model_name, 
                local_mistral_client, openrouter_client, use_local_refinement
            )
        
        debug_logs.append(refinement_log)
        debug_print(f"[DEBUG] Query used for embedding: {refined_query}")

        # 2. Embedding of the prepared question
        query_vector = embedding_client.generate_query_embedding(refined_query)
        if "ada" in embedding_client.model_name:
            debug_print(f"[DEBUG-ADA] Generated Ada query vector. Length: {len(query_vector)}. First 5 dims: {query_vector[:5]}")
        
        debug_print(f"[DEBUG] Query vector generated. Length: {len(query_vector)}")
        debug_logs.append(f"🔹 Query vector length: {len(query_vector)}")
        debug_logs.append(f"🔹 top_k: {top_k}")

        # 3. Buscar preguntas similares (Questions)
        if use_questions_collection:
            debug_print(f"[DEBUG] Searching for similar questions in collection: {questions_class}")
            similar_questions = chromadb_wrapper.search_questions_by_vector(query_vector, top_k=min(top_k*3, 30))
            debug_logs.append(f"🔹 Questions found: {len(similar_questions)}")
            debug_print(f"[DEBUG] Similar Questions retrieved: {len(similar_questions)}")
            #for i, q in enumerate(similar_questions):
            #    print(f"[DEBUG]   Q {i+1}: Title: {q.get('title', 'N/A')}, Accepted Answer: {q.get('accepted_answer', 'N/A')[:100]}...")

            # 4. Extraer links desde respuestas aceptadas con deduplicación temprana
            unique_links = set()
            for q in similar_questions:
                extracted = extract_urls_from_answer(q.get("accepted_answer", ""))
                unique_links.update(extracted)

            all_links = list(unique_links)
            debug_logs.append(f"🔹 Links extracted from answers: {len(all_links)}")
            debug_print(f"[DEBUG] Extracted {len(all_links)} unique links from similar questions.")
            debug_logs.append(f"🔹 Sample links: {all_links[:3]}")

            # 5. Recuperar documentos vinculados usando batch operation when available
            if hasattr(chromadb_wrapper, "lookup_docs_by_links_batch"):
                debug_print(f"[DEBUG] Looking up documents by links in collection: {documents_class}")
                linked_docs = chromadb_wrapper.lookup_docs_by_links_batch(
                    all_links, batch_size=50
                )
            else:
                linked_docs = chromadb_wrapper.lookup_docs_by_links(all_links)
            debug_logs.append(f"🔹 Linked documents found: {len(linked_docs)}")
            debug_print(f"[DEBUG] Linked Documents retrieved: {len(linked_docs)}")
            #for i, doc in enumerate(linked_docs):
            #    print(f"[DEBUG]   Linked Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")
        else:
            debug_logs.append("🔹 Skipping Questions collection search.")
            similar_questions = []
            linked_docs = []

        # 6. Buscar documentos directamente por vector
        document_vector = embedding_client.generate_document_embedding(refined_query)
        debug_print(f"[DEBUG] Document vector generated. Length: {len(document_vector)}")
        debug_logs.append(f"🔹 Document vector length: {len(document_vector)}")
        
        debug_print(f"[DEBUG] Searching for documents by vector in collection: {documents_class}")
        vector_docs = chromadb_wrapper.search_docs_by_vector(
            vector=document_vector,
            top_k=max(top_k * 2, 20),
            diversity_threshold=diversity_threshold,
            include_distance=True
        )
        if "ada" in embedding_client.model_name:
            debug_print(f"[DEBUG-ADA] Weaviate search returned {len(vector_docs)} documents.")
        #debug_logs.append(f"🔹 Vector-retrieved documents: {len(vector_docs)}")
        #print(f"[DEBUG] Vector-Retrieved Documents: {len(vector_docs)}")
        #for i, doc in enumerate(vector_docs):
        #    print(f"[DEBUG]   Vector Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")

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
        debug_print(f"[DEBUG] Total unique documents after deduplication: {len(unique_docs)}")

        if not unique_docs:
            debug_logs.append("⚠️ No unique documents retrieved.")
            debug_print("[DEBUG] No unique documents found. Returning empty list.")
            return [], "\n".join(debug_logs)

        debug_print("[DEBUG] Documents retrieved from Weaviate (before reranking):")
        #for i, doc in enumerate(unique_docs):
        #    print(f"[DEBUG]   Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")

        # 8. Reranking (condicional)
        debug_logs.append(f"🔹 Preparing for reranking. LLM Reranker enabled: {use_llm_reranker}")
        debug_print(f"[DEBUG] In qa_pipeline: LLM Reranker enabled: {use_llm_reranker}")
        debug_print(f"[DEBUG] In qa_pipeline: Number of unique_docs to rerank: {len(unique_docs)}")
        max_docs_to_rerank = min(len(unique_docs), 40 if use_llm_reranker else top_k * 3)
        docs_to_rerank = unique_docs[:max_docs_to_rerank]

        if use_llm_reranker:
            try:
                debug_logs.append(f"🔹 Using LLM to rerank {len(docs_to_rerank)} documents...")
                debug_print(f"[DEBUG] Reranking {len(docs_to_rerank)} documents with LLM.")
                reranked = rerank_with_llm(question, docs_to_rerank, openai_client, top_k=top_k, embedding_model=embedding_client.model_name)
            except Exception as e:
                debug_print(f"[DEBUG] ERROR during LLM reranking: {e}")
                debug_logs.append(f"❌ Error during LLM reranking: {e}. Falling back to standard reranking.")
                debug_print("[DEBUG] Falling back to standard reranking after LLM error.")
                reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        else:
            debug_logs.append(f"🔹 Using standard embedding similarity to rerank {len(docs_to_rerank)} documents...")
            debug_print(f"[DEBUG] Reranking {len(docs_to_rerank)} documents with standard reranker.")
            reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        
        debug_logs.append(f"🔹 Documents after reranking: {len(reranked)}")
        debug_print(f"[DEBUG] Documents after reranking: {len(reranked)}")
        
        # 9. Generación de respuesta final (NUEVO)
        if generate_answer:
            if generative_model_name == "tinyllama-1.1b" and local_tinyllama_client:
                generated_answer, generation_info = generate_final_answer_local(
                    question=question,
                    retrieved_docs=reranked,
                    model_name="tinyllama-1.1b"
                )
            elif generative_model_name == "mistral-7b" and local_mistral_client:
                generated_answer, generation_info = generate_final_answer_local(
                    question=question,
                    retrieved_docs=reranked,
                    model_name="mistral-7b"
                )
            elif generative_model_name == "deepseek-v3-chat":
                from src.services.answer_generation.openrouter import generate_final_answer_openrouter
                generated_answer, generation_info = generate_final_answer_openrouter(
                    question=question,
                    retrieved_docs=reranked,
                    model_name="deepseek-v3-chat"
                )
            elif generative_model_name == "gemini-1.5-flash" and gemini_client:
                generated_answer, generation_info = generate_final_answer_gemini(
                    question=question,
                    retrieved_docs=reranked,
                    gemini_client=gemini_client
                )
            elif openai_client:
                generated_answer, generation_info = generate_final_answer(
                    question=question,
                    retrieved_docs=reranked,
                    openai_client=openai_client
                )
            else:
                generated_answer = "Error: No generative model available"
                generation_info = {"status": "error", "error": "No generative model configured"}
            
            rag_metrics = {}
            if evaluate_quality and generation_info.get('status') == 'success':
                try:
                    rag_metrics = evaluate_answer_quality(
                        question=question,
                        answer=generated_answer,
                        source_docs=reranked,
                        openai_client=openai_client
                    )
                except Exception as e:
                    rag_metrics = {"evaluation_error": str(e)}
            
            rag_metrics.update(generation_info)
            
            return reranked, "\n".join(debug_logs), generated_answer, rag_metrics
        else:
            # Modo tradicional: solo documentos
            debug_logs.append("🔹 Skipping answer generation (generate_answer=False)")
            return reranked, "\n".join(debug_logs)

    except Exception as e:
        debug_logs.append(f"❌ Error: {e}")
        debug_print(f"[DEBUG] Unhandled error in answer_question: {e}")
        if generate_answer:
            return [], "\n".join(debug_logs), f"Error en el pipeline: {e}", {"status": "pipeline_error", "error": str(e)}
        else:
            return [], "\n".join(debug_logs)

def answer_question_documents_only(
    question: str,
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = True,
    documents_class: str = "Documents",
    questions_class: str = "Questions"
) -> Tuple[List[dict], str]:
    """
    Función de compatibilidad que mantiene el comportamiento original.
    Solo retorna documentos sin generar respuesta.
    """
    return answer_question(
        question=question,
        chromadb_wrapper=chromadb_wrapper,
        embedding_client=embedding_client,
        openai_client=openai_client,
        gemini_client=None,
        local_tinyllama_client=None,
        local_mistral_client=None,
        top_k=top_k,
        diversity_threshold=diversity_threshold,
        use_llm_reranker=use_llm_reranker,
        use_questions_collection=use_questions_collection,
        generate_answer=False,
        documents_class=documents_class,
        questions_class=questions_class
    )


def answer_question_documents_with_comparison(
    question: str,
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = True,
    documents_class: str = "Documents",
    questions_class: str = "Questions"
) -> Tuple[List[dict], List[dict], str]:
    """
    Retorna documentos antes y después del reranking para comparación.
    
    Returns:
        Tuple de (pre_reranking_docs, post_reranking_docs, debug_info)
    """
    debug_logs = []
    
    try:
        # 1. Búsqueda inicial (igual que answer_question)
        debug_logs.append(f"🔹 Starting search for: {question}")
        
        # 2. Buscar en colección de preguntas
        if use_questions_collection:
            question_vector = embedding_client.generate_query_embedding(question)
            similar_questions = chromadb_wrapper.search_questions_by_vector(
                vector=question_vector,
                top_k=3,
                threshold=0.7
            )
            
            linked_docs = []
            if similar_questions:
                all_links = []
                for q in similar_questions:
                    ms_links = q.get("ms_links", [])
                    all_links.extend(ms_links)
                
                if all_links:
                    linked_docs = chromadb_wrapper.search_docs_by_links(all_links)
        else:
            similar_questions = []
            linked_docs = []
        
        # 3. Buscar documentos por vector
        document_vector = embedding_client.generate_document_embedding(question)
        vector_docs = chromadb_wrapper.search_docs_by_vector(
            vector=document_vector,
            top_k=max(top_k * 2, 20),
            diversity_threshold=diversity_threshold,
            include_distance=True
        )
        
        # 4. Combinar y deduplicar
        unique_docs_dict = {}
        
        # Primero agregar documentos linked
        for doc in linked_docs:
            link = doc.get("link", "").strip()
            if link:
                unique_docs_dict[link] = doc
        
        # Luego agregar documentos de vector search
        for doc in vector_docs:
            link = doc.get("link", "").strip()
            if link and link not in unique_docs_dict:
                unique_docs_dict[link] = doc
        
        unique_docs = list(unique_docs_dict.values())
        
        if not unique_docs:
            return [], [], "No se encontraron documentos"
        
        # 5. Guardar copia de documentos pre-reranking con scores originales
        pre_reranking_docs = []
        for doc in unique_docs[:top_k]:  # Solo los top_k para comparación justa
            pre_reranking_docs.append({
                **doc,
                'original_score': doc.get('score', 0)  # Preservar score original
            })
        
        # 6. Aplicar reranking
        max_docs_to_rerank = min(len(unique_docs), 40 if use_llm_reranker else top_k * 3)
        docs_to_rerank = unique_docs[:max_docs_to_rerank]
        
        if use_llm_reranker:
            try:
                debug_logs.append(f"🔹 Using LLM to rerank {len(docs_to_rerank)} documents...")
                reranked = rerank_with_llm(question, docs_to_rerank, openai_client, top_k=top_k, embedding_model=embedding_client.model_name)
            except Exception as e:
                debug_logs.append(f"❌ Error during LLM reranking: {e}. Falling back to standard reranking.")
                reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        else:
            debug_logs.append(f"🔹 Using standard embedding similarity to rerank {len(docs_to_rerank)} documents...")
            reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        
        debug_logs.append(f"🔹 Pre-reranking docs: {len(pre_reranking_docs)}, Post-reranking docs: {len(reranked)}")
        
        return pre_reranking_docs, reranked, "\n".join(debug_logs)
        
    except Exception as e:
        debug_logs.append(f"❌ Error: {e}")
        return [], [], "\n".join(debug_logs)

def answer_question_with_rag(
    question: str,
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    gemini_client: genai.GenerativeModel = None,
    local_tinyllama_client = None,
    local_mistral_client = None,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = True,
    evaluate_quality: bool = True,
    documents_class: str = "Documents",
    questions_class: str = "Questions",
    generative_model_name: str = "tinyllama-1.1b"
) -> Tuple[List[dict], str, str, Dict]:
    """
    Función que ejecuta el pipeline RAG completo con generación de respuesta.
    
    Returns:
        Tuple de (documentos, debug_info, respuesta_generada, rag_metrics)
    """
    result = answer_question(
        question=question,
        chromadb_wrapper=chromadb_wrapper,
        embedding_client=embedding_client,
        openai_client=openai_client,
        gemini_client=gemini_client,
        local_tinyllama_client=local_tinyllama_client,
        local_mistral_client=local_mistral_client,
        top_k=top_k,
        diversity_threshold=diversity_threshold,
        use_llm_reranker=use_llm_reranker,
        use_questions_collection=use_questions_collection,
        generate_answer=True,
        evaluate_quality=evaluate_quality,
        documents_class=documents_class,
        questions_class=questions_class,
        generative_model_name=generative_model_name
    )
    
    # Garantizar que retornamos 4 elementos
    if len(result) == 4:
        return result
    else:
        # Fallback si algo sale mal
        docs, debug = result
        return docs, debug, "Error: No se pudo generar respuesta", {"status": "error"}

