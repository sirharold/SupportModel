"""
Módulo para evaluación cumulativa de métricas.
"""

import streamlit as st
import numpy as np
import time
import gc
from typing import List, Dict, Any
from utils.clients import initialize_clients
from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from utils.memory_utils import get_memory_usage, cleanup_memory
from utils.metrics import calculate_average_metrics, create_question_data_record
from utils.data_processing import fetch_random_questions_from_weaviate


def run_cumulative_metrics_for_models(
    num_questions: int,
    model_names: List[str],
    generative_model_name: str,
    top_k: int = 10,
    use_llm_reranker: bool = True,
    batch_size: int = 50
) -> Dict[str, Dict[str, Any]]:
    """Run cumulative evaluation for several embedding models."""
    results = {}
    total_models = len(model_names)

    for i, model in enumerate(model_names):
        with st.spinner(f"Evaluando {model} ({i+1}/{total_models})..."):
            model_result = run_cumulative_metrics_evaluation(
                num_questions=num_questions,
                model_name=model,
                generative_model_name=generative_model_name,
                top_k=top_k,
                use_llm_reranker=use_llm_reranker,
                batch_size=batch_size
            )
            results[model] = model_result
            
            # Limpiar memoria entre modelos
            gc.collect()

    return results


def run_cumulative_metrics_evaluation(
    num_questions: int,
    model_name: str,
    generative_model_name: str,
    top_k: int = 10,
    use_llm_reranker: bool = True,
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Ejecuta evaluación cumulativa de métricas para un modelo específico.
    
    Args:
        num_questions: Número de preguntas a evaluar
        model_name: Nombre del modelo de embedding
        generative_model_name: Nombre del modelo generativo
        top_k: Número de documentos top a recuperar
        use_llm_reranker: Si usar reranking con LLM
        batch_size: Tamaño del lote para procesamiento
        
    Returns:
        Diccionario con resultados de la evaluación
    """
    
    # Inicializar clientes
    weaviate_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(
        model_name, generative_model_name
    )
    
    # Extraer preguntas desde Weaviate
    st.info(f"🔍 Extrayendo {num_questions} preguntas aleatorias desde Weaviate...")
    selected_questions = fetch_random_questions_from_weaviate(
        weaviate_wrapper=weaviate_wrapper,
        embedding_model_name=model_name,
        num_questions=num_questions,
        sample_size=max(num_questions * 5, 500)  # Sample size más grande para tener suficientes con links
    )
    
    if not selected_questions:
        st.error("❌ No se pudieron extraer preguntas desde Weaviate")
        return {
            'num_questions_evaluated': 0,
            'avg_before_metrics': {},
            'avg_after_metrics': {},
            'avg_rag_stats': {},
            'all_questions_data': [],
            'individual_before_metrics': [],
            'individual_after_metrics': [],
            'memory_stats': {'initial_memory': 0, 'final_memory': 0, 'memory_increase': 0}
        }
    
    actual_questions = len(selected_questions)
    if actual_questions < num_questions:
        st.warning(f"⚠️ Solo se encontraron {actual_questions} preguntas con enlaces MS Learn (se solicitaron {num_questions})")
    else:
        st.success(f"✅ Extraídas {actual_questions} preguntas con enlaces MS Learn")
    
    # Listas para almacenar métricas
    before_reranking_metrics = []
    after_reranking_metrics = []
    rag_stats_list = []
    all_questions_data = []
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Procesar preguntas en lotes usando el número real de preguntas extraídas
    num_batches = (actual_questions + batch_size - 1) // batch_size
    initial_memory = get_memory_usage()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, actual_questions)
        batch_questions = selected_questions[start_idx:end_idx]
        
        status_text.text(f"Procesando lote {batch_idx + 1}/{num_batches} ({len(batch_questions)} preguntas)")
        
        for local_idx, qa in enumerate(batch_questions):
            global_idx = start_idx + local_idx
            question = qa.get('question', qa.get('question_content', qa.get('title', '')))
            accepted_answer = qa.get('accepted_answer', '')
            ms_links = qa.get('ms_links', [])
            
            try:
                result = answer_question_with_retrieval_metrics(
                    question=question,
                    weaviate_wrapper=weaviate_wrapper,
                    embedding_client=embedding_client,
                    openai_client=openai_client,
                    gemini_client=gemini_client,
                    local_tinyllama_client=local_tinyllama_client,
                    local_mistral_client=local_mistral_client,
                    openrouter_client=openrouter_client,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    generate_answer=False,
                    ground_truth_answer=accepted_answer,
                    ms_links=ms_links,
                    calculate_metrics=True,
                    generative_model_name=generative_model_name
                )
                
                if len(result) >= 3:
                    docs, debug_info, retrieval_metrics = result
                    
                    # Extraer métricas antes y después del reranking
                    before_metrics = retrieval_metrics.get('before_reranking', {})
                    after_metrics = retrieval_metrics.get('after_reranking', {})
                    
                    # Extraer estadísticas RAG
                    rag_stats = {
                        'ground_truth_links_count': retrieval_metrics.get('ground_truth_links_count', 0),
                        'docs_before_count': retrieval_metrics.get('docs_before_count', 0),
                        'docs_after_count': retrieval_metrics.get('docs_after_count', 0)
                    }
                    
                    before_reranking_metrics.append(before_metrics)
                    after_reranking_metrics.append(after_metrics)
                    rag_stats_list.append(rag_stats)
                    
                    # Almacenar datos de la pregunta para referencia usando función optimizada
                    question_data = create_question_data_record(
                        question_num=global_idx + 1,
                        question=question,
                        ms_links=ms_links,
                        docs_count=len(docs),
                        before_metrics=before_metrics,
                        after_metrics=after_metrics
                    )
                    all_questions_data.append(question_data)
                    
            except Exception as e:
                st.error(f"Error evaluando pregunta {global_idx + 1}: {e}")
                continue
            
            progress_bar.progress((global_idx + 1) / actual_questions)
        
        # Limpiar memoria después de cada lote
        cleanup_memory()
        
        # Pausa breve para permitir que el sistema respire
        time.sleep(0.1)
        
        # Mostrar información de memoria cada 5 lotes
        if (batch_idx + 1) % 5 == 0:
            memory_used = get_memory_usage()
            memory_increase = memory_used - initial_memory
            if memory_increase > 0:
                st.info(f"📊 Memoria utilizada: {memory_used:.1f} MB (+{memory_increase:.1f} MB desde inicio)")
    
    # Limpieza final
    cleanup_memory()
    final_memory = get_memory_usage()
    total_memory_increase = final_memory - initial_memory
    
    # Calcular métricas promedio
    avg_before_metrics = calculate_average_metrics(before_reranking_metrics)
    avg_after_metrics = calculate_average_metrics(after_reranking_metrics)
    avg_rag_stats = calculate_average_metrics(rag_stats_list)
    
    # Limpiar interfaz
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar estadísticas de memoria
    if total_memory_increase > 50:  # Solo mostrar si el incremento es significativo
        st.info(f"💾 Evaluación completada. Incremento total de memoria: {total_memory_increase:.1f} MB")
    
    return {
        'num_questions_evaluated': len(before_reranking_metrics),
        'avg_before_metrics': avg_before_metrics,
        'avg_after_metrics': avg_after_metrics,
        'avg_rag_stats': avg_rag_stats,
        'all_questions_data': all_questions_data,
        'individual_before_metrics': before_reranking_metrics,
        'individual_after_metrics': after_reranking_metrics,
        'memory_stats': {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_increase': total_memory_increase
        }
    }