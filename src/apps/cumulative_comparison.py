"""
Página de Comparación Acumulativa: Métricas para N Preguntas
Evalúa múltiples preguntas y genera métricas consolidadas
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import io
from datetime import datetime
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.apps.question_answer_comparison import (
    MODEL_NAME_MAPPING, 
    load_random_questions,
    initialize_clients_no_cache
)
from src.config.config import GENERATIVE_MODELS, DEFAULT_GENERATIVE_MODEL


def show_cumulative_comparison_page():
    """Página principal de comparación acumulativa"""
    
    st.title("📊 Comparación Acumulativa: Análisis de N Preguntas")
    st.markdown("""
    Esta herramienta evalúa múltiples preguntas para generar métricas consolidadas y comparar 
    el rendimiento de diferentes modelos de embedding a gran escala.
    """)
    
    # Information about the analysis
    with st.expander("📚 Información sobre el Análisis Acumulativo"):
        st.markdown("""
        **🔍 Qué hace este análisis:**
        
        - Evalúa N preguntas aleatorias de la base de datos
        - Compara documentos recuperados usando pregunta vs respuesta para cada modelo
        - Calcula métricas tradicionales (Jaccard, nDCG, Precision) y RAG (Faithfulness, Relevancia, etc.)
        - Genera estadísticas consolidadas: promedio, mediana, desviación estándar
        - Permite exportar resultados detallados
        
        **📈 Métricas Incluidas:**
        - **IR Tradicionales**: Jaccard Similarity, nDCG@10, Precision@5
        - **Evaluación LLM**: Calidad de contenido recuperado
        - **RAG Avanzadas**: Faithfulness, Relevancia, Correctness, Similarity
        - **Score Compuesto**: Métrica final consolidada
        """)
    
    # Add load/save section before configuration
    st.markdown("---")
    show_load_save_section()
    st.markdown("---")
    
    # Configuration Section
    config = get_configuration_parameters()
    
    # Processing Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Ejecutar Análisis Acumulativo", type="primary", key="run_analysis"):
            with st.spinner("Ejecutando análisis acumulativo..."):
                # Load questions
                questions = load_random_questions(config['num_questions'])
                
                if not questions:
                    st.error("❌ No se pudieron cargar preguntas para el análisis")
                    return
                
                # Process all questions
                results = process_cumulative_analysis(questions, config)
                
                # Store results in session state
                st.session_state.cumulative_results = results
                st.session_state.cumulative_config = config
                st.success("✅ Análisis completado exitosamente")
    
    with col2:
        # Show save button if results exist
        if 'cumulative_results' in st.session_state:
            show_save_results_button()
    
    # Results Section
    if 'cumulative_results' in st.session_state:
        display_cumulative_results(
            st.session_state.cumulative_results, 
            st.session_state.cumulative_config
        )


def get_configuration_parameters() -> Dict[str, Any]:
    """Sección de configuración de parámetros"""
    
    st.subheader("⚙️ Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of questions
        num_questions = st.slider(
            "Número de preguntas a analizar",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Cantidad de preguntas aleatorias para el análisis"
        )
        
        # Top-k documents
        top_k = st.slider(
            "Documentos a recuperar (top-k)",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Cantidad de documentos a recuperar para cada query"
        )
    
    with col2:
        # Reranking option
        use_reranking = st.checkbox(
            "🔄 Usar Reranking",
            value=True,
            help="Aplicar reranking a los documentos recuperados"
        )
        
        # Generative model selection
        generative_model = st.selectbox(
            "🤖 Modelo Generativo",
            options=list(GENERATIVE_MODELS.keys()),
            index=list(GENERATIVE_MODELS.keys()).index(DEFAULT_GENERATIVE_MODEL),
            help="Modelo generativo para evaluación de calidad"
        )
    
    # Models to compare
    st.markdown("### 🎯 Modelos a Comparar")
    model_selection = {}
    
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    for i, (short_name, full_name) in enumerate(MODEL_NAME_MAPPING.items()):
        with columns[i]:
            model_selection[short_name] = st.checkbox(
                f"{short_name.upper()}",
                value=True,
                help=f"Incluir {full_name} en la comparación"
            )
    
    return {
        'num_questions': num_questions,
        'top_k': top_k,
        'use_reranking': use_reranking,
        'generative_model': generative_model,
        'selected_models': [k for k, v in model_selection.items() if v]
    }


def process_cumulative_analysis(questions: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
    """Lógica principal de procesamiento del análisis acumulativo"""
    
    from src.apps.question_answer_comparison import perform_comparison
    
    results = {
        'individual_results': {},  # Resultados por pregunta
        'consolidated_metrics': {},  # Métricas consolidadas por modelo
        'execution_stats': {
            'total_questions': len(questions),
            'total_time': 0,
            'questions_processed': 0,
            'questions_failed': 0
        }
    }
    
    start_time = time.time()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each question
    for i, question in enumerate(questions):
        try:
            status_text.text(f"Procesando pregunta {i+1}/{len(questions)}: {question['title'][:50]}...")
            
            # Use the existing comparison logic but without displaying results
            comparison_results = execute_single_question_analysis(
                question, 
                config,
                display_progress=False
            )
            
            if comparison_results:
                results['individual_results'][question['id']] = {
                    'question': question,
                    'results': comparison_results
                }
                results['execution_stats']['questions_processed'] += 1
            else:
                results['execution_stats']['questions_failed'] += 1
                
        except Exception as e:
            st.warning(f"Error procesando pregunta {i+1}: {str(e)}")
            results['execution_stats']['questions_failed'] += 1
        
        # Update progress
        progress_bar.progress((i + 1) / len(questions))
    
    # Calculate consolidated metrics
    results['consolidated_metrics'] = calculate_consolidated_metrics(
        results['individual_results'], 
        config['selected_models']
    )
    
    results['execution_stats']['total_time'] = time.time() - start_time
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results


def execute_single_question_analysis(question: Dict, config: Dict[str, Any], display_progress: bool = True) -> Dict[str, Any]:
    """Ejecuta análisis para una sola pregunta (lógica de retrieval, reranking, RAG)"""
    
    from src.apps.question_answer_comparison import (
        calculate_rag_metrics_for_documents,
        evaluate_retrieved_content_quality,
        calculate_comparison_metrics,
        process_query_results
    )
    
    results = {}
    
    for short_model_name in config['selected_models']:
        try:
            full_model_name = MODEL_NAME_MAPPING[short_model_name]
            
            # Initialize clients
            chromadb_wrapper, embedding_client = initialize_clients_no_cache(full_model_name)
            
            expected_dimensions = {
                "multi-qa-mpnet-base-dot-v1": 768,
                "all-MiniLM-L6-v2": 384, 
                "ada": 1536,
                "e5-large-v2": 1024
            }
            
            # Get collection
            from src.config.config import CHROMADB_COLLECTION_CONFIG
            docs_collection_name = CHROMADB_COLLECTION_CONFIG[full_model_name]["documents"]
            docs_collection_obj = chromadb_wrapper.client.get_collection(name=docs_collection_name)
            
            # Generate embeddings
            question_query = f"{question['title']} {question['content']}"
            question_embedding = embedding_client.generate_query_embedding(question_query)
            answer_embedding = embedding_client.generate_query_embedding(question['accepted_answer'])
            
            # Retrieve documents
            n_retrieve = config['top_k'] * 2 if config['use_reranking'] else config['top_k']
            
            question_results = docs_collection_obj.query(
                query_embeddings=[question_embedding],
                n_results=n_retrieve,
                include=["metadatas", "distances", "documents"]
            )
            
            answer_results = docs_collection_obj.query(
                query_embeddings=[answer_embedding],
                n_results=n_retrieve,
                include=["metadatas", "distances", "documents"]
            )
            
            # Apply reranking if enabled
            if config['use_reranking']:
                try:
                    from src.core.reranker import rerank_with_llm
                    
                    # Process initial results
                    question_docs_pre = process_query_results(question_results, "question")
                    answer_docs_pre = process_query_results(answer_results, "answer")
                    
                    # Prepare for reranking
                    question_docs_for_rerank = [
                        {
                            'content': doc['content'], 
                            'metadata': {
                                'title': doc['title'], 
                                'chunk_index': doc['chunk_index'],
                                'link': doc.get('link', '')
                            }
                        }
                        for doc in question_docs_pre
                    ]
                    answer_docs_for_rerank = [
                        {
                            'content': doc['content'], 
                            'metadata': {
                                'title': doc['title'], 
                                'chunk_index': doc['chunk_index'],
                                'link': doc.get('link', '')
                            }
                        }
                        for doc in answer_docs_pre
                    ]
                    
                    # Rerank documents
                    reranked_question_docs = rerank_with_llm(
                        question_query, 
                        question_docs_for_rerank, 
                        openai_client=None,
                        top_k=config['top_k'],
                        embedding_model=full_model_name
                    )
                    reranked_answer_docs = rerank_with_llm(
                        question['accepted_answer'], 
                        answer_docs_for_rerank,
                        openai_client=None, 
                        top_k=config['top_k'],
                        embedding_model=full_model_name
                    )
                    
                    # Convert reranked results
                    question_docs = []
                    for i, doc in enumerate(reranked_question_docs):
                        score = doc.get('score', 0.5)
                        question_docs.append({
                            'title': doc['metadata']['title'],
                            'chunk_index': doc['metadata']['chunk_index'],
                            'link': doc['metadata'].get('link', ''),
                            'content': doc['content'],
                            'distance': 1 - score,
                            'similarity': max(0, min(1, score)),
                            'source': 'question',
                            'rank': i + 1
                        })
                    
                    answer_docs = []
                    for i, doc in enumerate(reranked_answer_docs):
                        score = doc.get('score', 0.5)
                        answer_docs.append({
                            'title': doc['metadata']['title'],
                            'chunk_index': doc['metadata']['chunk_index'],
                            'link': doc['metadata'].get('link', ''),
                            'content': doc['content'],
                            'distance': 1 - score,
                            'similarity': max(0, min(1, score)),
                            'source': 'answer',
                            'rank': i + 1
                        })
                        
                except Exception as rerank_error:
                    # Fallback to non-reranked results
                    question_docs = process_query_results(question_results, "question")[:config['top_k']]
                    answer_docs = process_query_results(answer_results, "answer")[:config['top_k']]
            else:
                # Process results without reranking
                question_docs = process_query_results(question_results, "question")[:config['top_k']]
                answer_docs = process_query_results(answer_results, "answer")[:config['top_k']]
            
            # Calculate all metrics
            question_full = f"{question['title']} {question['content']}"
            
            # RAG metrics
            question_rag_metrics = calculate_rag_metrics_for_documents(
                question_full,
                question['accepted_answer'],
                question_docs,
                config['generative_model'],
                chromadb_wrapper,
                embedding_client
            )
            
            answer_rag_metrics = calculate_rag_metrics_for_documents(
                question_full,
                question['accepted_answer'], 
                answer_docs,
                config['generative_model'],
                chromadb_wrapper,
                embedding_client
            )
            
            # Content quality
            question_quality = evaluate_retrieved_content_quality(
                question_full, 
                question['accepted_answer'], 
                question_docs, 
                config['generative_model']
            )
            
            answer_quality = evaluate_retrieved_content_quality(
                question_full, 
                question['accepted_answer'], 
                answer_docs, 
                config['generative_model']
            )
            
            # Traditional metrics
            metrics = calculate_comparison_metrics(question_docs, answer_docs, config['top_k'])
            
            # Consolidate all metrics
            metrics['question_quality'] = question_quality
            metrics['answer_quality'] = answer_quality
            metrics['avg_quality'] = (question_quality + answer_quality) / 2
            
            # RAG metrics (averaged)
            metrics['faithfulness'] = (question_rag_metrics['faithfulness'] + answer_rag_metrics['faithfulness']) / 2
            metrics['answer_relevance'] = (question_rag_metrics['answer_relevance'] + answer_rag_metrics['answer_relevance']) / 2
            metrics['answer_correctness'] = (question_rag_metrics['answer_correctness'] + answer_rag_metrics['answer_correctness']) / 2
            metrics['answer_similarity'] = (question_rag_metrics['answer_similarity'] + answer_rag_metrics['answer_similarity']) / 2
            
            # Update composite score
            original_composite = metrics['composite_score']
            metrics['composite_score'] = (
                0.4 * metrics['jaccard_similarity'] + 
                0.25 * metrics['ndcg_at_10'] + 
                0.15 * metrics['precision_at_5'] +
                0.2 * metrics['avg_quality']
            )
            metrics['original_composite'] = original_composite
            
            results[short_model_name] = {
                'metrics': metrics,
                'used_reranking': config['use_reranking']
            }
            
        except Exception as e:
            if display_progress:
                st.warning(f"Error con modelo {short_model_name}: {str(e)}")
            results[short_model_name] = {
                'metrics': {},
                'used_reranking': config['use_reranking']
            }
    
    return results


def calculate_consolidated_metrics(individual_results: Dict, selected_models: List[str]) -> Dict[str, Any]:
    """Calcula métricas consolidadas de todos los resultados individuales"""
    
    consolidated = {}
    
    # Métricas a consolidar
    metrics_to_consolidate = [
        'jaccard_similarity', 'ndcg_at_10', 'precision_at_5', 'common_docs',
        'composite_score', 'question_quality', 'answer_quality', 'avg_quality',
        'faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity'
    ]
    
    for model in selected_models:
        model_metrics = {metric: [] for metric in metrics_to_consolidate}
        
        # Collect all values for each metric
        for question_id, result in individual_results.items():
            if model in result['results'] and result['results'][model]['metrics']:
                metrics = result['results'][model]['metrics']
                for metric in metrics_to_consolidate:
                    if metric in metrics:
                        model_metrics[metric].append(metrics[metric])
        
        # Calculate statistics
        consolidated[model] = {}
        for metric, values in model_metrics.items():
            if values:  # Only if we have values
                consolidated[model][metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                consolidated[model][metric] = {
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }
    
    return consolidated


def display_cumulative_results(results: Dict[str, Any], config: Dict[str, Any]):
    """Sección de visualización de resultados acumulativos"""
    
    st.subheader("📊 Resultados del Análisis Acumulativo")
    
    # Execution Summary
    stats = results['execution_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Preguntas Procesadas", stats['questions_processed'])
    with col2:
        st.metric("Preguntas Fallidas", stats['questions_failed'])
    with col3:
        st.metric("Tiempo Total", f"{stats['total_time']:.1f}s")
    with col4:
        st.metric("Promedio por Pregunta", f"{stats['total_time']/max(1, stats['questions_processed']):.1f}s")
    
    if stats['questions_processed'] == 0:
        st.error("❌ No se procesaron preguntas exitosamente")
        return
    
    # Consolidated Metrics Table
    display_consolidated_metrics_table(results['consolidated_metrics'], config)
    
    # Visualizations
    display_cumulative_visualizations(results['consolidated_metrics'], config)
    
    # Export functionality
    display_export_section(results, config)


def display_consolidated_metrics_table(consolidated_metrics: Dict, config: Dict[str, Any]):
    """Muestra tabla de métricas consolidadas"""
    
    st.markdown("### 📋 Métricas Consolidadas")
    
    # Create comprehensive table
    table_data = []
    
    for model in config['selected_models']:
        if model in consolidated_metrics:
            metrics = consolidated_metrics[model]
            table_data.append({
                'Modelo': model.upper(),
                'Jaccard (μ±σ)': f"{metrics['jaccard_similarity']['mean']:.3f}±{metrics['jaccard_similarity']['std']:.3f}",
                'nDCG@10 (μ±σ)': f"{metrics['ndcg_at_10']['mean']:.3f}±{metrics['ndcg_at_10']['std']:.3f}",
                'Precision@5 (μ±σ)': f"{metrics['precision_at_5']['mean']:.3f}±{metrics['precision_at_5']['std']:.3f}",
                'Calidad LLM (μ±σ)': f"{metrics['avg_quality']['mean']:.3f}±{metrics['avg_quality']['std']:.3f}",
                'Faithfulness (μ±σ)': f"{metrics['faithfulness']['mean']:.3f}±{metrics['faithfulness']['std']:.3f}",
                'Relevancia (μ±σ)': f"{metrics['answer_relevance']['mean']:.3f}±{metrics['answer_relevance']['std']:.3f}",
                'Correctness (μ±σ)': f"{metrics['answer_correctness']['mean']:.3f}±{metrics['answer_correctness']['std']:.3f}",
                'Similarity (μ±σ)': f"{metrics['answer_similarity']['mean']:.3f}±{metrics['answer_similarity']['std']:.3f}",
                'Score Final (μ±σ)': f"{metrics['composite_score']['mean']:.3f}±{metrics['composite_score']['std']:.3f}",
                'N': metrics['composite_score']['count']
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)


def display_cumulative_visualizations(consolidated_metrics: Dict, config: Dict[str, Any]):
    """Muestra visualizaciones de métricas acumulativas"""
    
    st.markdown("### 📈 Visualizaciones")
    
    # Prepare data for plotting
    models = config['selected_models']
    
    if not models or not consolidated_metrics:
        st.warning("No hay datos para visualizar")
        return
    
    # Score Comparison Chart
    st.markdown("#### Comparación de Score Final")
    
    mean_scores = [consolidated_metrics[model]['composite_score']['mean'] for model in models]
    std_scores = [consolidated_metrics[model]['composite_score']['std'] for model in models]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[model.upper() for model in models],
        y=mean_scores,
        error_y=dict(type='data', array=std_scores),
        marker_color='lightblue',
        name='Score Final'
    ))
    
    fig.update_layout(
        title="Score Final por Modelo (Media ± Desviación Estándar)",
        xaxis_title="Modelo",
        yaxis_title="Score Final",
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RAG Metrics Radar Chart
    st.markdown("#### Comparación de Métricas RAG")
    
    fig = go.Figure()
    
    rag_metrics = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
    rag_labels = ['Faithfulness', 'Relevancia', 'Correctness', 'Similarity']
    
    for model in models:
        if model in consolidated_metrics:
            values = [consolidated_metrics[model][metric]['mean'] for metric in rag_metrics]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=rag_labels,
                fill='toself',
                name=model.upper()
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Comparación de Métricas RAG por Modelo"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_export_section(results: Dict[str, Any], config: Dict[str, Any]):
    """Sección de exportación de resultados"""
    
    st.markdown("### 📥 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exportar Métricas Consolidadas"):
            consolidated_df = create_consolidated_export_df(results['consolidated_metrics'], config)
            csv_consolidated = consolidated_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV Consolidado",
                data=csv_consolidated,
                file_name=f"metricas_consolidadas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Exportar Resultados Detallados"):
            detailed_df = create_detailed_export_df(results['individual_results'], config)
            csv_detailed = detailed_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV Detallado",
                data=csv_detailed,
                file_name=f"resultados_detallados_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def create_consolidated_export_df(consolidated_metrics: Dict, config: Dict[str, Any]) -> pd.DataFrame:
    """Crea DataFrame para exportación de métricas consolidadas"""
    
    export_data = []
    
    for model in config['selected_models']:
        if model in consolidated_metrics:
            metrics = consolidated_metrics[model]
            
            # Flatten all metrics
            row = {'model': model}
            for metric_name, stats in metrics.items():
                row[f'{metric_name}_mean'] = stats['mean']
                row[f'{metric_name}_std'] = stats['std']
                row[f'{metric_name}_median'] = stats['median']
                row[f'{metric_name}_min'] = stats['min']
                row[f'{metric_name}_max'] = stats['max']
                row[f'{metric_name}_count'] = stats['count']
            
            export_data.append(row)
    
    return pd.DataFrame(export_data)


def create_detailed_export_df(individual_results: Dict, config: Dict[str, Any]) -> pd.DataFrame:
    """Crea DataFrame para exportación de resultados detallados"""
    
    export_data = []
    
    for question_id, result in individual_results.items():
        question = result['question']
        
        for model in config['selected_models']:
            if model in result['results'] and result['results'][model]['metrics']:
                metrics = result['results'][model]['metrics']
                
                row = {
                    'question_id': question_id,
                    'question_title': question['title'],
                    'model': model,
                    **metrics  # Include all metrics
                }
                
                export_data.append(row)
    
    return pd.DataFrame(export_data)


def show_load_save_section():
    """Sección para cargar resultados previos"""
    st.subheader("💾 Gestión de Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📥 Cargar Resultados Previos**")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo de resultados",
            type=['json'],
            help="Carga un archivo JSON con resultados de análisis previos para evitar recomputar",
            key="load_results_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Load the JSON data
                results_data = json.load(uploaded_file)
                
                # Validate the structure
                if validate_results_structure(results_data):
                    # Store in session state
                    st.session_state.cumulative_results = results_data['results']
                    st.session_state.cumulative_config = results_data['config']
                    st.session_state.loaded_filename = uploaded_file.name
                    st.session_state.loaded_timestamp = results_data.get('metadata', {}).get('timestamp', 'Desconocido')
                    
                    st.success(f"✅ Resultados cargados exitosamente")
                    st.info(f"📄 Archivo: {uploaded_file.name}")
                    st.info(f"🕒 Fecha: {st.session_state.loaded_timestamp}")
                    st.info(f"📊 Preguntas: {results_data['config']['num_questions']}")
                    st.info(f"🤖 Modelos: {len(results_data['config']['models'])}")
                    
                    # Option to clear loaded results
                    if st.button("🗑️ Limpiar resultados cargados", key="clear_loaded"):
                        for key in ['cumulative_results', 'cumulative_config', 'loaded_filename', 'loaded_timestamp']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                        
                else:
                    st.error("❌ El archivo no tiene el formato correcto de resultados")
                    
            except json.JSONDecodeError:
                st.error("❌ Error al leer el archivo JSON")
            except Exception as e:
                st.error(f"❌ Error cargando resultados: {str(e)}")
    
    with col2:
        st.markdown("**ℹ️ Información**")
        if 'loaded_filename' in st.session_state:
            st.info(f"🔄 **Resultados Cargados:**")
            st.write(f"📄 Archivo: `{st.session_state.loaded_filename}`")
            st.write(f"🕒 Fecha: {st.session_state.loaded_timestamp}")
        else:
            st.info("""
            **💡 Consejos:**
            - Los archivos guardados contienen todos los datos del análisis
            - Puedes cargar resultados previos para evitar recalcular
            - Los resultados incluyen configuración y métricas detalladas
            - El formato es JSON estándar
            """)


def show_save_results_button():
    """Botón para descargar resultados actuales"""
    if st.button("💾 Guardar Resultados", type="secondary", key="save_results"):
        save_data = prepare_results_for_save()
        
        # Create download button
        json_str = json.dumps(save_data, indent=2, ensure_ascii=False)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cumulative_analysis_results_{timestamp}.json"
        
        st.download_button(
            label="📥 Descargar archivo JSON",
            data=json_str,
            file_name=filename,
            mime="application/json",
            key="download_results"
        )
        
        st.success("✅ Archivo preparado para descarga")
        st.info(f"📄 Nombre: {filename}")
        st.info(f"📊 Tamaño: {len(json_str):,} caracteres")


def prepare_results_for_save() -> Dict[str, Any]:
    """Prepara los resultados para guardar en JSON"""
    if 'cumulative_results' not in st.session_state or 'cumulative_config' not in st.session_state:
        return {}
    
    # Create comprehensive save data
    save_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'Cumulative N Questions Analysis Results',
            'generated_by': 'Streamlit Cumulative Analysis Tool'
        },
        'config': st.session_state.cumulative_config,
        'results': st.session_state.cumulative_results
    }
    
    # Add summary statistics
    if st.session_state.cumulative_results.get('individual_results'):
        results = st.session_state.cumulative_results['individual_results']
        save_data['summary'] = {
            'total_questions': len(results),
            'models_analyzed': list(st.session_state.cumulative_config['models']),
            'metrics_included': list(results[list(results.keys())[0]]['results'][list(st.session_state.cumulative_config['models'])[0]]['metrics'].keys()) if results else [],
            'analysis_completion_time': datetime.now().isoformat()
        }
    
    return save_data


def validate_results_structure(data: Dict[str, Any]) -> bool:
    """Valida que los datos cargados tengan la estructura correcta"""
    try:
        # Check for required top-level keys
        required_keys = ['results', 'config']
        if not all(key in data for key in required_keys):
            return False
        
        # Check config structure
        config = data['config']
        config_keys = ['num_questions', 'models', 'top_k']
        if not all(key in config for key in config_keys):
            return False
        
        # Check results structure
        results = data['results']
        if not isinstance(results, dict):
            return False
        
        # Basic validation passed
        return True
        
    except Exception:
        return False


if __name__ == "__main__":
    show_cumulative_comparison_page()