from typing import List, Tuple, Dict, Union
from openai import OpenAI
import google.generativeai as genai
from utils.extract_links import extract_urls_from_answer
from utils.reranker import rerank_documents, rerank_with_llm
from utils.embedding import EmbeddingClient
from utils.weaviate_utils_improved import WeaviateClientWrapper
from utils.answer_generator import generate_final_answer, evaluate_answer_quality
from utils.gemini_answer_generator import generate_final_answer_gemini

def refine_and_prepare_query(question: str, openai_client: OpenAI, model_name: str) -> Tuple[str, str]:
    """
    Cleans, distills, and conditionally prefixes a user query for optimal performance.
    """
    logs = []
    try:
        # Step 1: Noise and Salutation Removal
        cleaning_prompt = (
            "You are a text cleaning expert. Your task is to remove all greetings, "
            "pleasantries, signatures, and any other conversational filler from the "
            "following user query. Output ONLY the core technical question."
            f"\n\nOriginal Query: {question}"
        )
        
        cleaned_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a text cleaning expert."},
                {"role": "user", "content": cleaning_prompt}
            ],
            temperature=0.0,
            n=1
        )
        cleaned_question = cleaned_response.choices[0].message.content.strip()
        logs.append(f"🔹 Query after cleaning: {cleaned_question}")

        # Step 2: Core Question Distillation
        distillation_prompt = (
            "Based on the following text, distill it into a single, clear, and "
            "concise technical question suitable for a vector database search. "
            "Remove any ambiguity and focus on the essential problem."
            f"\n\nCleaned Text: {cleaned_question}"
        )

        distilled_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at distilling technical questions."},
                {"role": "user", "content": distillation_prompt}
            ],
            temperature=0.0,
            n=1
        )
        distilled_question = distilled_response.choices[0].message.content.strip()
        logs.append(f"🔹 Query after distillation: {distilled_question}")

        # Step 3: Conditional Prefixing
        if "multi-qa-mpnet-base-dot-v1" in model_name:
            final_query = "query: " + distilled_question
            logs.append("🔹 Added 'query: ' prefix for mpnet model.")
        else:
            final_query = distilled_question
            logs.append("🔹 No prefix added for this model.")
            
        return final_query, "\n".join(logs)

    except Exception as e:
        logs.append(f"⚠️ Query refinement failed: {e}. Falling back to original question.")
        return question, "\n".join(logs)

def answer_question(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    gemini_client: genai.GenerativeModel = None,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = True,
    generate_answer: bool = True,
    evaluate_quality: bool = False,
    documents_class: str = "Documents",
    questions_class: str = "Questions",
    generative_model_name: str = "gpt-4"
) -> Union[Tuple[List[dict], str], Tuple[List[dict], str, str, Dict]]:
    print("[DEBUG] Entering answer_question function.")
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
            print("[DEBUG-ADA] Using original question for Ada.")
            refined_query = question
            refinement_log = "🔹 Skipping query refinement for Ada model."
        else:
            refined_query, refinement_log = refine_and_prepare_query(question, openai_client, embedding_client.model_name)
        
        debug_logs.append(refinement_log)
        print(f"[DEBUG] Query used for embedding: {refined_query}")

        # 2. Embedding of the prepared question
        query_vector = embedding_client.generate_query_embedding(refined_query)
        if "ada" in embedding_client.model_name:
            print(f"[DEBUG-ADA] Generated Ada query vector. Length: {len(query_vector)}. First 5 dims: {query_vector[:5]}")
        print(f"[DEBUG] Query vector generated. Length: {len(query_vector)}")
        debug_logs.append(f"🔹 Query vector length: {len(query_vector)}")
        debug_logs.append(f"🔹 top_k: {top_k}")

        # 3. Buscar preguntas similares (Questions)
        if use_questions_collection:
            print(f"[DEBUG] Searching for similar questions in collection: {questions_class}")
            similar_questions = weaviate_wrapper.search_questions_by_vector(query_vector, top_k=min(top_k*3, 30))
            debug_logs.append(f"🔹 Questions found: {len(similar_questions)}")
            print(f"[DEBUG] Similar Questions retrieved: {len(similar_questions)}")
            #for i, q in enumerate(similar_questions):
            #    print(f"[DEBUG]   Q {i+1}: Title: {q.get('title', 'N/A')}, Accepted Answer: {q.get('accepted_answer', 'N/A')[:100]}...")

            # 4. Extraer links desde respuestas aceptadas con deduplicación temprana
            unique_links = set()
            for q in similar_questions:
                extracted = extract_urls_from_answer(q.get("accepted_answer", ""))
                unique_links.update(extracted)

            all_links = list(unique_links)
            debug_logs.append(f"🔹 Links extracted from answers: {len(all_links)}")
            print(f"[DEBUG] Extracted {len(all_links)} unique links from similar questions.")
            debug_logs.append(f"🔹 Sample links: {all_links[:3]}")

            # 5. Recuperar documentos vinculados usando batch operation when available
            if hasattr(weaviate_wrapper, "lookup_docs_by_links_batch"):
                print(f"[DEBUG] Looking up documents by links in collection: {documents_class}")
                linked_docs = weaviate_wrapper.lookup_docs_by_links_batch(
                    all_links, batch_size=50
                )
            else:
                linked_docs = weaviate_wrapper.lookup_docs_by_links(all_links)
            debug_logs.append(f"🔹 Linked documents found: {len(linked_docs)}")
            print(f"[DEBUG] Linked Documents retrieved: {len(linked_docs)}")
            #for i, doc in enumerate(linked_docs):
            #    print(f"[DEBUG]   Linked Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")
        else:
            debug_logs.append("🔹 Skipping Questions collection search.")
            similar_questions = []
            linked_docs = []

        # 6. Buscar documentos directamente por vector
        document_vector = embedding_client.generate_document_embedding(refined_query)
        print(f"[DEBUG] Document vector generated. Length: {len(document_vector)}")
        debug_logs.append(f"🔹 Document vector length: {len(document_vector)}")
        
        print(f"[DEBUG] Searching for documents by vector in collection: {documents_class}")
        vector_docs = weaviate_wrapper.search_docs_by_vector(
            vector=document_vector,
            top_k=max(top_k * 2, 20),
            diversity_threshold=diversity_threshold,
            include_distance=True
        )
        if "ada" in embedding_client.model_name:
            print(f"[DEBUG-ADA] Weaviate search returned {len(vector_docs)} documents.")
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
        print(f"[DEBUG] Total unique documents after deduplication: {len(unique_docs)}")

        if not unique_docs:
            debug_logs.append("⚠️ No unique documents retrieved.")
            print("[DEBUG] No unique documents found. Returning empty list.")
            return [], "\n".join(debug_logs)

        print("[DEBUG] Documents retrieved from Weaviate (before reranking):")
        #for i, doc in enumerate(unique_docs):
        #    print(f"[DEBUG]   Doc {i+1}: Title: {doc.get('title', 'N/A')}, Link: {doc.get('link', 'N/A')}")

        # 8. Reranking (condicional)
        debug_logs.append(f"🔹 Preparing for reranking. LLM Reranker enabled: {use_llm_reranker}")
        print(f"[DEBUG] In qa_pipeline: LLM Reranker enabled: {use_llm_reranker}")
        print(f"[DEBUG] In qa_pipeline: Number of unique_docs to rerank: {len(unique_docs)}")
        max_docs_to_rerank = min(len(unique_docs), 40 if use_llm_reranker else top_k * 3)
        docs_to_rerank = unique_docs[:max_docs_to_rerank]

        if use_llm_reranker:
            try:
                debug_logs.append(f"🔹 Using LLM to rerank {len(docs_to_rerank)} documents...")
                print(f"[DEBUG] Reranking {len(docs_to_rerank)} documents with LLM.")
                reranked = rerank_with_llm(question, docs_to_rerank, openai_client, top_k=top_k)
            except Exception as e:
                print(f"[DEBUG] ERROR during LLM reranking: {e}")
                debug_logs.append(f"❌ Error during LLM reranking: {e}. Falling back to standard reranking.")
                print("[DEBUG] Falling back to standard reranking after LLM error.")
                reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        else:
            debug_logs.append(f"🔹 Using standard embedding similarity to rerank {len(docs_to_rerank)} documents...")
            print(f"[DEBUG] Reranking {len(docs_to_rerank)} documents with standard reranker.")
            reranked = rerank_documents(question, docs_to_rerank, embedding_client, top_k=top_k)
        
        debug_logs.append(f"🔹 Documents after reranking: {len(reranked)}")
        print(f"[DEBUG] Documents after reranking: {len(reranked)}")
        
        # 9. Generación de respuesta final (NUEVO)
        if generate_answer:
            if generative_model_name == "gemini-pro" and gemini_client:
                generated_answer, generation_info = generate_final_answer_gemini(
                    question=question,
                    retrieved_docs=reranked,
                    gemini_client=gemini_client
                )
            else:
                generated_answer, generation_info = generate_final_answer(
                    question=question,
                    retrieved_docs=reranked,
                    openai_client=openai_client
                )
            
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
        print(f"[DEBUG] Unhandled error in answer_question: {e}")
        if generate_answer:
            return [], "\n".join(debug_logs), f"Error en el pipeline: {e}", {"status": "pipeline_error", "error": str(e)}
        else:
            return [], "\n".join(debug_logs)

def answer_question_documents_only(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
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
        weaviate_wrapper=weaviate_wrapper,
        embedding_client=embedding_client,
        openai_client=openai_client,
        top_k=top_k,
        diversity_threshold=diversity_threshold,
        use_llm_reranker=use_llm_reranker,
        use_questions_collection=use_questions_collection,
        generate_answer=False,
        documents_class=documents_class,
        questions_class=questions_class
    )

def answer_question_with_rag(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    gemini_client: genai.GenerativeModel = None,
    top_k: int = 10,
    *,
    diversity_threshold: float = 0.85,
    use_llm_reranker: bool = True,
    use_questions_collection: bool = True,
    evaluate_quality: bool = True,
    documents_class: str = "Documents",
    questions_class: str = "Questions",
    generative_model_name: str = "gpt-4"
) -> Tuple[List[dict], str, str, Dict]:
    """
    Función que ejecuta el pipeline RAG completo con generación de respuesta.
    
    Returns:
        Tuple de (documentos, debug_info, respuesta_generada, rag_metrics)
    """
    result = answer_question(
        question=question,
        weaviate_wrapper=weaviate_wrapper,
        embedding_client=embedding_client,
        openai_client=openai_client,
        gemini_client=gemini_client,
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

