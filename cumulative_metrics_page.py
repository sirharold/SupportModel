"""
Página de Métricas Acumulativas - Evalúa múltiples preguntas y calcula promedios
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import re
from typing import List, Dict, Any
from utils.clients import initialize_clients
from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from config import EMBEDDING_MODELS, GENERATIVE_MODELS, WEAVIATE_CLASS_CONFIG
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_ms_links(accepted_answer: str) -> List[str]:
    """
    Extrae links de Microsoft Learn de la respuesta aceptada.
    
    Args:
        accepted_answer: Texto de la respuesta aceptada
        
    Returns:
        Lista de links de Microsoft Learn encontrados
    """
    # Patrón para encontrar links de Microsoft Learn
    pattern = r'https://learn\.microsoft\.com[\w/\-\?=&%\.]+'
    links = re.findall(pattern, accepted_answer)
    return list(set(links))  # Eliminar duplicados

def filter_questions_with_links(questions_and_answers: List[Dict]) -> List[Dict]:
    """
    Filtra preguntas que tienen links en la respuesta aceptada.
    
    Args:
        questions_and_answers: Lista de preguntas y respuestas
        
    Returns:
        Lista filtrada de preguntas con links en respuesta aceptada
    """
    filtered_questions = []
    
    for qa in questions_and_answers:
        accepted_answer = qa.get('accepted_answer', '')
        
        # Extraer links de Microsoft Learn
        ms_links = extract_ms_links(accepted_answer)
        
        # Solo incluir si hay links
        if ms_links and len(ms_links) > 0:
            # Añadir los links extraídos al diccionario
            qa_copy = qa.copy()
            qa_copy['ms_links'] = ms_links
            qa_copy['question'] = qa.get('question_content', qa.get('title', ''))
            filtered_questions.append(qa_copy)
    
    return filtered_questions

def calculate_average_metrics(all_metrics: List[Dict]) -> Dict[str, float]:
    """
    Calcula métricas promedio de una lista de métricas.
    
    Args:
        all_metrics: Lista de diccionarios con métricas
        
    Returns:
        Diccionario con métricas promedio
    """
    if not all_metrics:
        return {}
    
    # Inicializar contadores
    metric_sums = {}
    metric_counts = {}
    
    # Sumar todas las métricas
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if key not in metric_sums:
                    metric_sums[key] = 0
                    metric_counts[key] = 0
                metric_sums[key] += value
                metric_counts[key] += 1
    
    # Calcular promedios
    average_metrics = {}
    for key in metric_sums:
        if metric_counts[key] > 0:
            average_metrics[key] = metric_sums[key] / metric_counts[key]
    
    return average_metrics

def run_cumulative_metrics_evaluation(
    questions_and_answers: List[Dict],
    num_questions: int,
    model_name: str,
    generative_model_name: str,
    top_k: int = 10,
    use_llm_reranker: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta evaluación de métricas acumulativas para múltiples preguntas.
    
    Args:
        questions_and_answers: Lista de preguntas y respuestas
        num_questions: Número de preguntas a evaluar
        model_name: Nombre del modelo de embedding
        generative_model_name: Nombre del modelo generativo
        top_k: Número de documentos top-k
        use_llm_reranker: Si usar LLM reranking
        
    Returns:
        Diccionario con resultados de evaluación
    """
    # Filtrar preguntas con links
    filtered_questions = filter_questions_with_links(questions_and_answers)
    
    if len(filtered_questions) < num_questions:
        st.warning(f"⚠️ Solo hay {len(filtered_questions)} preguntas con links disponibles. Se usarán todas.")
        num_questions = len(filtered_questions)
    
    # Seleccionar preguntas aleatoriamente
    if len(filtered_questions) > num_questions:
        selected_indices = np.random.choice(len(filtered_questions), num_questions, replace=False)
        selected_questions = [filtered_questions[i] for i in selected_indices]
    else:
        selected_questions = filtered_questions
    
    # Inicializar clientes
    weaviate_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(
        model_name, generative_model_name
    )
    
    # Listas para almacenar métricas
    before_reranking_metrics = []
    after_reranking_metrics = []
    all_questions_data = []
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Evaluar cada pregunta
    for i, qa in enumerate(selected_questions):
        question = qa['question']
        ground_truth_answer = qa.get('accepted_answer', '')
        ms_links = qa.get('ms_links', [])
        
        status_text.text(f"Evaluando pregunta {i+1}/{num_questions}...")
        
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
                openrouter_client=openrouter_client,
                top_k=top_k,
                use_llm_reranker=use_llm_reranker,
                generate_answer=False,  # Sin RAG como se solicitó
                calculate_metrics=True,
                ground_truth_answer=ground_truth_answer,
                ms_links=ms_links,
                generative_model_name=generative_model_name
            )
            
            if len(result) >= 3:
                docs, debug_info, retrieval_metrics = result
                
                # Extraer métricas antes y después del reranking
                before_metrics = retrieval_metrics.get('before_reranking', {})
                after_metrics = retrieval_metrics.get('after_reranking', {})
                
                before_reranking_metrics.append(before_metrics)
                after_reranking_metrics.append(after_metrics)
                
                # Almacenar datos de la pregunta para referencia
                all_questions_data.append({
                    'question_num': i + 1,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'ground_truth_links': len(ms_links),
                    'docs_retrieved': len(docs),
                    'before_precision_5': before_metrics.get('Precision@5', 0),
                    'after_precision_5': after_metrics.get('Precision@5', 0),
                    'before_recall_5': before_metrics.get('Recall@5', 0),
                    'after_recall_5': after_metrics.get('Recall@5', 0),
                    'before_f1_5': before_metrics.get('F1@5', 0),
                    'after_f1_5': after_metrics.get('F1@5', 0)
                })
                
        except Exception as e:
            st.error(f"Error evaluando pregunta {i+1}: {e}")
            continue
        
        progress_bar.progress((i + 1) / num_questions)
    
    # Calcular métricas promedio
    avg_before_metrics = calculate_average_metrics(before_reranking_metrics)
    avg_after_metrics = calculate_average_metrics(after_reranking_metrics)
    
    # Limpiar interfaz
    progress_bar.empty()
    status_text.empty()
    
    return {
        'num_questions_evaluated': len(before_reranking_metrics),
        'avg_before_metrics': avg_before_metrics,
        'avg_after_metrics': avg_after_metrics,
        'all_questions_data': all_questions_data,
        'individual_before_metrics': before_reranking_metrics,
        'individual_after_metrics': after_reranking_metrics
    }

def display_cumulative_metrics(results: Dict[str, Any], model_name: str, use_llm_reranker: bool):
    """
    Muestra los resultados de métricas acumulativas en la interfaz.
    
    Args:
        results: Resultados de la evaluación
        model_name: Nombre del modelo usado
        use_llm_reranker: Si se usó LLM reranking
    """
    num_questions = results['num_questions_evaluated']
    avg_before = results['avg_before_metrics']
    avg_after = results['avg_after_metrics']
    
    st.success(f"✅ Evaluación completada para {num_questions} preguntas")
    
    # Métricas principales en columnas
    st.subheader("📊 Métricas Promedio")
    
    # Métricas principales
    main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Antes del Reranking**")
        for metric in main_metrics:
            if metric in avg_before:
                st.metric(
                    label=metric,
                    value=f"{avg_before[metric]:.3f}",
                    help=f"Promedio de {metric} antes del reranking LLM"
                )
    
    with col2:
        if use_llm_reranker:
            st.markdown("**🤖 Después del Reranking LLM**")
            for metric in main_metrics:
                if metric in avg_after:
                    # Calcular delta
                    delta = avg_after[metric] - avg_before.get(metric, 0)
                    st.metric(
                        label=metric,
                        value=f"{avg_after[metric]:.3f}",
                        delta=f"{delta:+.3f}",
                        help=f"Promedio de {metric} después del reranking LLM"
                    )
        else:
            st.info("ℹ️ Reranking LLM deshabilitado")
    
    # Gráfico de comparación
    if use_llm_reranker and avg_before and avg_after:
        st.subheader("📈 Comparación Visual")
        
        # Preparar datos para el gráfico
        metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
        before_values = [avg_before.get(m, 0) for m in metrics_to_plot]
        after_values = [avg_after.get(m, 0) for m in metrics_to_plot]
        
        # Crear gráfico de barras comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Antes del Reranking',
            x=metrics_to_plot,
            y=before_values,
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Después del Reranking',
            x=metrics_to_plot,
            y=after_values,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title=f'Comparación de Métricas Promedio ({num_questions} preguntas)',
            xaxis_title='Métricas',
            yaxis_title='Valor Promedio',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Métricas adicionales en expander
    with st.expander("📋 Métricas Detalladas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Antes del Reranking**")
            before_df = pd.DataFrame([avg_before]).T
            before_df.columns = ['Valor Promedio']
            before_df.index.name = 'Métrica'
            st.dataframe(before_df.style.format({'Valor Promedio': '{:.4f}'}))
        
        with col2:
            if use_llm_reranker and avg_after:
                st.markdown("**Después del Reranking**")
                after_df = pd.DataFrame([avg_after]).T
                after_df.columns = ['Valor Promedio']
                after_df.index.name = 'Métrica'
                st.dataframe(after_df.style.format({'Valor Promedio': '{:.4f}'}))
    
    # Tabla de evolución por pregunta
    with st.expander("📊 Evolución por Pregunta"):
        questions_df = pd.DataFrame(results['all_questions_data'])
        if not questions_df.empty:
            # Renombrar columnas para mejor legibilidad
            column_names = {
                'question_num': 'Pregunta #',
                'ground_truth_links': 'Links GT',
                'docs_retrieved': 'Docs Recuperados',
                'before_precision_5': 'Precision@5 (Antes)',
                'after_precision_5': 'Precision@5 (Después)',
                'before_f1_5': 'F1@5 (Antes)',
                'after_f1_5': 'F1@5 (Después)',
                'before_recall_5': 'Recall@5 (Antes)',
                'after_recall_5': 'Recall@5 (Después)'
            }
            
            # Mostrar tabla con métricas por pregunta
            if use_llm_reranker:
                display_columns = ['question_num', 'ground_truth_links', 'docs_retrieved', 
                                 'before_precision_5', 'after_precision_5', 
                                 'before_f1_5', 'after_f1_5']
            else:
                display_columns = ['question_num', 'ground_truth_links', 'docs_retrieved', 
                                 'before_precision_5', 'before_f1_5']
            
            questions_df_display = questions_df[display_columns].rename(columns=column_names)
            
            st.dataframe(questions_df_display.style.format({
                'Precision@5 (Antes)': '{:.3f}',
                'Precision@5 (Después)': '{:.3f}',
                'F1@5 (Antes)': '{:.3f}',
                'F1@5 (Después)': '{:.3f}'
            }), use_container_width=True)

def load_questions_from_json(file_path: str) -> List[Dict]:
    """
    Carga preguntas desde archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Lista de preguntas y respuestas
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo '{file_path}'")
        return []
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al leer el archivo JSON: {e}")
        return []

def show_cumulative_metrics_page():
    """Muestra la página de métricas acumulativas."""
    st.title("📈 Métricas Acumulativas")
    st.markdown("""
    Esta página evalúa múltiples preguntas y calcula **métricas promedio** para obtener una visión general 
    del rendimiento del sistema. Solo se evalúan preguntas que tienen links en la respuesta aceptada.
    """)
    
    # Cargar preguntas desde JSON
    questions_and_answers = load_questions_from_json('data/val_set.json')
    
    if not questions_and_answers:
        st.error("❌ No se pudieron cargar las preguntas")
        st.stop()
    
    # Filtrar preguntas con links
    filtered_questions = filter_questions_with_links(questions_and_answers)
    
    st.info(f"📊 {len(filtered_questions)} preguntas disponibles con links en respuesta aceptada")
    
    # Configuración
    st.subheader("⚙️ Configuración de Evaluación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Número de preguntas
        num_questions = st.slider(
            "Número de preguntas a evaluar",
            min_value=1,
            max_value=min(50, len(filtered_questions)),
            value=5,
            help="Cantidad de preguntas para evaluar (tomadas aleatoriamente)"
        )
        
        # Modelo de embedding
        model_name = st.selectbox(
            "Modelo de Embedding",
            options=list(EMBEDDING_MODELS.keys()),
            index=0,
            help="Modelo para generar embeddings de documentos"
        )
    
    with col2:
        # Configuración de retrieval
        top_k = st.slider(
            "Top-K documentos",
            min_value=5,
            max_value=20,
            value=10,
            help="Número de documentos a recuperar"
        )
        
        # LLM Reranking
        use_llm_reranker = st.checkbox(
            "Usar LLM Reranking",
            value=True,
            help="Usar GPT-4 para reordenar documentos (necesario para métricas antes/después)"
        )
    
    # Modelo generativo (solo para reranking)
    generative_model_name = st.selectbox(
        "Modelo Generativo (para reranking)",
        options=list(GENERATIVE_MODELS.keys()),
        index=0,
        help="Modelo usado para el reranking LLM"
    )
    
    # Botón para ejecutar evaluación
    if st.button("🚀 Ejecutar Evaluación", type="primary"):
        if len(filtered_questions) < num_questions:
            st.error(f"❌ Solo hay {len(filtered_questions)} preguntas con links disponibles")
            st.stop()
        
        with st.spinner(f"🔍 Evaluando {num_questions} preguntas..."):
            start_time = time.time()
            
            results = run_cumulative_metrics_evaluation(
                questions_and_answers=filtered_questions,
                num_questions=num_questions,
                model_name=model_name,
                generative_model_name=generative_model_name,
                top_k=top_k,
                use_llm_reranker=use_llm_reranker
            )
            
            evaluation_time = time.time() - start_time
            
            # Mostrar resultados
            display_cumulative_metrics(results, model_name, use_llm_reranker)
            
            # Estadísticas adicionales
            st.markdown("---")
            st.subheader("📊 Estadísticas de Evaluación")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📝 Preguntas Evaluadas",
                    f"{results['num_questions_evaluated']}",
                    help="Número total de preguntas procesadas"
                )
            
            with col2:
                if results['all_questions_data']:
                    avg_gt_links = np.mean([q['ground_truth_links'] for q in results['all_questions_data']])
                    st.metric(
                        "🔗 Links GT Promedio",
                        f"{avg_gt_links:.1f}",
                        help="Número promedio de links de ground truth por pregunta"
                    )
            
            with col3:
                st.metric(
                    "⏱️ Tiempo Total",
                    f"{evaluation_time:.1f}s",
                    help="Tiempo total de evaluación"
                )
            
            # Opción para descargar resultados
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Descargar Resultados Detallados"):
                    results_df = pd.DataFrame(results['all_questions_data'])
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar CSV Detallado",
                        data=csv,
                        file_name=f"cumulative_metrics_detailed_{model_name}_{num_questions}q.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Descargar Métricas Promedio"):
                    # Crear CSV con métricas promedio
                    avg_metrics_data = {
                        'Metric': [],
                        'Before_Reranking': [],
                        'After_Reranking': []
                    }
                    
                    for metric in ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']:
                        avg_metrics_data['Metric'].append(metric)
                        avg_metrics_data['Before_Reranking'].append(results['avg_before_metrics'].get(metric, 0))
                        avg_metrics_data['After_Reranking'].append(results['avg_after_metrics'].get(metric, 0))
                    
                    avg_df = pd.DataFrame(avg_metrics_data)
                    csv = avg_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar CSV Promedio",
                        data=csv,
                        file_name=f"cumulative_metrics_avg_{model_name}_{num_questions}q.csv",
                        mime="text/csv"
                    )