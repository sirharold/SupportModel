"""
Pipeline QA extendido con cálculo de métricas de recuperación antes y después del reranking.
"""

from typing import List, Tuple, Dict, Union
from openai import OpenAI
import google.generativeai as genai
from utils.extract_links import extract_urls_from_answer
from utils.reranker import rerank_documents, rerank_with_llm
from utils.embedding import EmbeddingClient
from utils.weaviate_utils_improved import WeaviateClientWrapper
from utils.answer_generator import generate_final_answer, evaluate_answer_quality
from utils.gemini_answer_generator import generate_final_answer_gemini
from utils.local_answer_generator import generate_final_answer_local, refine_query_local
from utils.retrieval_metrics import calculate_before_after_reranking_metrics, format_metrics_for_display
import copy


def answer_question_with_retrieval_metrics(
    question: str,
    weaviate_wrapper: WeaviateClientWrapper,
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
    use_local_refinement: bool = True,
    ground_truth_answer: str = "",
    ms_links: List[str] = None,
    calculate_metrics: bool = True
) -> Union[Tuple[List[dict], str], Tuple[List[dict], str, str, Dict], Tuple[List[dict], str, str, Dict, Dict]]:
    """
    Pipeline QA completo con cálculo de métricas de recuperación antes y después del reranking.
    
    Args:
        calculate_metrics: Si calcular métricas de recuperación (requiere ground_truth_answer)
        ground_truth_answer: Respuesta aceptada de referencia
        ms_links: Enlaces de Microsoft Learn extraídos previamente
        ... otros parámetros del pipeline original
        
    Returns:
        Si calculate_metrics=True: (documentos, debug_info, respuesta_generada, rag_metrics, retrieval_metrics)
        Si calculate_metrics=False: resultado del pipeline original
    """
    
    # Importar la función original del pipeline
    from utils.qa_pipeline import refine_and_prepare_query
    
    print("[DEBUG] Entering answer_question_with_retrieval_metrics function.")
    debug_logs = []
    
    # Variables para almacenar documentos antes y después del reranking
    docs_before_reranking = []
    docs_after_reranking = []
    retrieval_metrics = {}

    try:
        # 1. Conditionally refine and prepare the query
        if "ada" in embedding_client.model_name:
            refined_query = question
            refinement_log = "🔹 Skipping query refinement for Ada model."
        else:
            refined_query, refinement_log = refine_and_prepare_query(
                question, gemini_client, embedding_client.model_name, 
                local_mistral_client, use_local_refinement
            )
        
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
            if calculate_metrics:
                return [], "\n".join(debug_logs), "", {}, {}
            elif generate_answer:
                return [], "\n".join(debug_logs), "", {}
            else:
                return [], "\n".join(debug_logs)

        print("[DEBUG] Documents retrieved from Weaviate (before reranking):")
        
        # 8. Guardar documentos ANTES del reranking para métricas
        if calculate_metrics:
            docs_before_reranking = copy.deepcopy(unique_docs)

        # 9. Reranking (condicional)
        debug_logs.append(f"🔹 Preparing for reranking. LLM Reranker enabled: {use_llm_reranker}")
        print(f"[DEBUG] In qa_pipeline: LLM Reranker enabled: {use_llm_reranker}")
        print(f"[DEBUG] In qa_pipeline: Number of unique_docs to rerank: {len(unique_docs)}")
        max_docs_to_rerank = min(len(unique_docs), 40 if use_llm_reranker else top_k * 3)
        docs_to_rerank = unique_docs[:max_docs_to_rerank]

        if use_llm_reranker:
            try:
                debug_logs.append(f"🔹 Using LLM to rerank {len(docs_to_rerank)} documents...")
                print(f"[DEBUG] Reranking {len(docs_to_rerank)} documents with LLM.")
                reranked = rerank_with_llm(question, docs_to_rerank, openai_client, top_k=top_k, embedding_model=embedding_client.model_name)
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
        
        # 10. Guardar documentos DESPUÉS del reranking para métricas
        if calculate_metrics:
            docs_after_reranking = copy.deepcopy(reranked)

        # 11. Calcular métricas de recuperación si se solicita
        if calculate_metrics and ground_truth_answer:
            try:
                debug_logs.append("🔹 Calculating retrieval metrics (before/after reranking)...")
                retrieval_metrics = calculate_before_after_reranking_metrics(
                    question=question,
                    docs_before_reranking=docs_before_reranking,
                    docs_after_reranking=docs_after_reranking,
                    ground_truth_answer=ground_truth_answer,
                    ms_links=ms_links,
                    k_values=[1, 3, 5, 10]
                )
                debug_logs.append(f"🔹 Retrieval metrics calculated successfully")
                
                # Log de métricas principales
                metrics_before = retrieval_metrics['before_reranking']
                metrics_after = retrieval_metrics['after_reranking']
                debug_logs.append(f"🔹 MRR improvement: {metrics_before['MRR']:.4f} → {metrics_after['MRR']:.4f}")
                debug_logs.append(f"🔹 Precision@5 improvement: {metrics_before.get('Precision@5', 0):.4f} → {metrics_after.get('Precision@5', 0):.4f}")
                
            except Exception as e:
                debug_logs.append(f"❌ Error calculating retrieval metrics: {e}")
                retrieval_metrics = {"error": str(e)}
        
        # 12. Generación de respuesta final (NUEVO)
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
            elif generative_model_name == "gemini-pro" and gemini_client:
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
            
            if calculate_metrics:
                return reranked, "\n".join(debug_logs), generated_answer, rag_metrics, retrieval_metrics
            else:
                return reranked, "\n".join(debug_logs), generated_answer, rag_metrics
        else:
            # Modo tradicional: solo documentos
            debug_logs.append("🔹 Skipping answer generation (generate_answer=False)")
            if calculate_metrics:
                return reranked, "\n".join(debug_logs), retrieval_metrics
            else:
                return reranked, "\n".join(debug_logs)

    except Exception as e:
        debug_logs.append(f"❌ Error: {e}")
        print(f"[DEBUG] Unhandled error in answer_question_with_retrieval_metrics: {e}")
        if calculate_metrics:
            if generate_answer:
                return [], "\n".join(debug_logs), f"Error en el pipeline: {e}", {"status": "pipeline_error", "error": str(e)}, {"error": str(e)}
            else:
                return [], "\n".join(debug_logs), {"error": str(e)}
        elif generate_answer:
            return [], "\n".join(debug_logs), f"Error en el pipeline: {e}", {"status": "pipeline_error", "error": str(e)}
        else:
            return [], "\n".join(debug_logs)


def batch_calculate_retrieval_metrics(
    questions_and_answers: List[Dict],
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_client: EmbeddingClient,
    openai_client: OpenAI,
    gemini_client: genai.GenerativeModel = None,
    local_tinyllama_client = None,
    local_mistral_client = None,
    top_k: int = 10,
    use_llm_reranker: bool = True,
    generative_model_name: str = "tinyllama-1.1b"
) -> List[Dict]:
    """
    Calcula métricas de recuperación para un lote de preguntas y respuestas.
    
    Args:
        questions_and_answers: Lista de diccionarios con 'question', 'accepted_answer', 'ms_links'
        ... otros parámetros del pipeline
        
    Returns:
        Lista de diccionarios con métricas para cada pregunta
    """
    all_metrics = []
    
    for i, qa_pair in enumerate(questions_and_answers):
        print(f"\n[BATCH] Processing question {i+1}/{len(questions_and_answers)}")
        
        question = qa_pair.get('question', '')
        accepted_answer = qa_pair.get('accepted_answer', '')
        ms_links = qa_pair.get('ms_links', [])
        
        try:
            # Ejecutar pipeline con métricas
            result = answer_question_with_retrieval_metrics(
                question=question,
                weaviate_wrapper=weaviate_wrapper,
                embedding_client=embedding_client,
                openai_client=openai_client,
                gemini_client=gemini_client,
                local_tinyllama_client=local_tinyllama_client,
                local_mistral_client=local_mistral_client,
                top_k=top_k,
                use_llm_reranker=use_llm_reranker,
                generate_answer=False,  # Solo documentos para métricas
                calculate_metrics=True,
                ground_truth_answer=accepted_answer,
                ms_links=ms_links,
                generative_model_name=generative_model_name
            )
            
            # Extraer métricas del resultado
            if len(result) >= 3:
                docs, debug_info, retrieval_metrics = result
                
                # Añadir información contextual
                retrieval_metrics['question'] = question
                retrieval_metrics['question_index'] = i
                retrieval_metrics['embedding_model'] = embedding_client.model_name
                retrieval_metrics['use_llm_reranker'] = use_llm_reranker
                
                all_metrics.append(retrieval_metrics)
                
                # Log de progreso
                if retrieval_metrics.get('before_reranking'):
                    mrr_before = retrieval_metrics['before_reranking'].get('MRR', 0)
                    mrr_after = retrieval_metrics['after_reranking'].get('MRR', 0)
                    print(f"[BATCH] Q{i+1} MRR: {mrr_before:.4f} → {mrr_after:.4f}")
                
        except Exception as e:
            print(f"[BATCH] Error processing question {i+1}: {e}")
            all_metrics.append({
                'question': question,
                'question_index': i,
                'error': str(e)
            })
    
    return all_metrics


def print_batch_metrics_summary(all_metrics: List[Dict]):
    """
    Imprime un resumen de las métricas calculadas en lote.
    
    Args:
        all_metrics: Lista de métricas calculadas por batch_calculate_retrieval_metrics
    """
    if not all_metrics:
        print("No metrics to display.")
        return
    
    # Filtrar métricas válidas (sin errores)
    valid_metrics = [m for m in all_metrics if 'before_reranking' in m and 'after_reranking' in m]
    
    if not valid_metrics:
        print("No valid metrics found.")
        return
    
    print(f"\n📊 RESUMEN DE MÉTRICAS - {len(valid_metrics)} CONSULTAS PROCESADAS")
    print("=" * 80)
    
    # Calcular promedios
    from utils.retrieval_metrics import calculate_aggregated_metrics
    aggregated = calculate_aggregated_metrics(valid_metrics)
    
    if aggregated:
        print(f"{'Métrica':<15} {'Before (μ)':<12} {'After (μ)':<12} {'Mejora (μ)':<12} {'% Mejora':<12}")
        print("-" * 80)
        
        for metric_key in ['MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 
                          'Precision@1', 'Precision@3', 'Precision@5', 'Precision@10',
                          'F1@1', 'F1@3', 'F1@5', 'F1@10']:
            if metric_key in aggregated['before_reranking']:
                before_mean = aggregated['before_reranking'][metric_key]['mean']
                after_mean = aggregated['after_reranking'][metric_key]['mean']
                improvement_mean = aggregated['improvement'][metric_key]['mean']
                
                pct_improvement = (improvement_mean / before_mean * 100) if before_mean > 0 else 0
                
                print(f"{metric_key:<15} {before_mean:<12.4f} {after_mean:<12.4f} {improvement_mean:<12.4f} {pct_improvement:<12.2f}%")
    
    # Mostrar casos destacados
    print("\n🎯 CASOS DESTACADOS:")
    print("-" * 50)
    
    # Mejor mejora en MRR
    best_mrr_improvement = max(valid_metrics, 
                              key=lambda x: x['after_reranking']['MRR'] - x['before_reranking']['MRR'])
    mrr_improvement = best_mrr_improvement['after_reranking']['MRR'] - best_mrr_improvement['before_reranking']['MRR']
    print(f"Mayor mejora MRR: +{mrr_improvement:.4f} (Q{best_mrr_improvement.get('question_index', 'N/A')})")
    
    # Mejor mejora en Precision@5
    best_p5_improvement = max(valid_metrics, 
                             key=lambda x: x['after_reranking'].get('Precision@5', 0) - x['before_reranking'].get('Precision@5', 0))
    p5_improvement = best_p5_improvement['after_reranking'].get('Precision@5', 0) - best_p5_improvement['before_reranking'].get('Precision@5', 0)
    print(f"Mayor mejora Precision@5: +{p5_improvement:.4f} (Q{best_p5_improvement.get('question_index', 'N/A')})")
    
    print("-" * 80)