import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.core.qa_pipeline import answer_question_documents_only, answer_question_with_rag
from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client, ChromaDBClientWrapper
from src.data.embedding_safe import get_embedding_client
from openai import OpenAI
import time
import random
import numpy as np
from datetime import datetime
import statistics
from src.config.config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, CHROMADB_COLLECTION_CONFIG

def show_batch_queries_page():
    """Página para consultas en lote."""
    
    # Configuración con cache
    @st.cache_resource
    def initialize_clients(model_name: str):
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        weaviate_classes = CHROMADB_COLLECTION_CONFIG[model_name]
        chromadb_wrapper = ChromaDBClientWrapper(
            client,
            documents_class=weaviate_classes["documents"],
            questions_class=weaviate_classes["questions"],
            retry_attempts=3
        )
        
        embedding_client = get_embedding_client(
            model_name=EMBEDDING_MODELS[model_name],
            huggingface_api_key=config.huggingface_api_key,
            openai_api_key=config.openai_api_key
        )
        
        # Initialize OpenAI client (required for some functions, but only used when needed)
        openai_client = OpenAI(api_key=config.openai_api_key)
        
        # Warn if using costly OpenAI embeddings
        if "ada" in model_name.lower():
            st.error("⚠️ **ADVERTENCIA**: Estás usando el modelo OpenAI 'ada' que genera costos por cada consulta!")
            st.info("💡 **Recomendación**: Cambia a 'multi-qa-mpnet-base-dot-v1' o 'all-MiniLM-L6-v2' para evitar costos.")
        else:
            st.success(f"✅ Usando modelo local: {model_name} (sin costos)")
        
        # Note: Gemini client removed to eliminate API costs
        # All operations now use local models or heuristics
        return chromadb_wrapper, embedding_client, openai_client, client

    st.subheader("📊 Consultas en Lote")
    st.markdown("**Realiza múltiples consultas de documentos y obtén métricas comprehensivas**")
    
    # Configuración de la consulta en lote
    config_col1, config_col2 = st.columns([2, 1])
    
    with config_col1:
        st.markdown("### ⚙️ Configuración de Consulta")
        
        # Selección de modelo de embedding
        model_name = st.selectbox(
            "Selecciona el modelo de embedding:",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(DEFAULT_EMBEDDING_MODEL),
            key="batch_model_select"
        )

        chromadb_wrapper, embedding_client, openai_client, client = initialize_clients(model_name)
        
        # Número de preguntas
        num_questions = st.number_input(
            "📊 Número de preguntas a consultar:",
            min_value=1,
            max_value=1000,
            value=20,
            step=1,
            help="Número de preguntas a extraer de QuestionsMiniLM para realizar consultas"
        )
        
        # Filtros de preguntas
        st.markdown("### 🔍 Filtros de Preguntas")
        
        # Opciones de selección
        selection_method = st.radio(
            "Método de selección:",
            ["🎲 Aleatorio", "📊 Primeras N", "🎯 Palabras clave"],
            help="Cómo seleccionar las preguntas de la base de datos"
        )
        
        keyword_filter = ""
        if selection_method == "🎯 Palabras clave":
            keyword_filter = st.text_input(
                "Palabra clave para filtrar:",
                placeholder="ej: Azure Functions, Key Vault, Storage...",
                help="Buscar preguntas que contengan esta palabra clave"
            )
        
        # Parámetros de búsqueda para lote
        st.markdown("### 🔧 Parámetros de Búsqueda")
        batch_top_k = st.slider("Documentos por consulta", 1, 20, 5, key="batch_top_k")
        batch_use_llm = st.checkbox("Usar LLM Reranking", value=False, key="batch_llm", 
                                  help="⚠️ Más preciso pero MUY lento y costoso para lotes grandes")
        batch_use_questions = st.checkbox("Usar Colección Questions", value=True, key="batch_questions")
        
        # Configuración RAG para lotes
        st.markdown("### 🤖 Configuración RAG")
        batch_enable_rag = st.checkbox("Activar RAG Completo", value=False, key="batch_rag",
                                     help="⚠️ Genera respuestas para cada consulta. MUY LENTO y COSTOSO para lotes grandes")
        batch_evaluate_rag = st.checkbox("Evaluar Calidad RAG", value=False, key="batch_rag_eval",
                                       help="⚠️ Evalúa cada respuesta generada. EXTREMADAMENTE lento y costoso")
        
        if batch_enable_rag:
            st.warning("⚠️ **Advertencia**: RAG en lotes es muy lento y costoso. Se recomienda usar con pocos documentos (≤10).")
        
        # Botón para ejecutar
        if st.button("🚀 Ejecutar Consultas en Lote", type="primary", use_container_width=True):
            if selection_method == "🎯 Palabras clave" and not keyword_filter.strip():
                st.error("⚠️ Especifica una palabra clave para este método de selección")
            else:
                # Guardar configuración en session state
                st.session_state.batch_config = {
                    'num_questions': num_questions,
                    'top_k': batch_top_k,
                    'use_llm': batch_use_llm,
                    'use_questions': batch_use_questions,
                    'selection_method': selection_method,
                    'keyword_filter': keyword_filter,
                    'enable_rag': batch_enable_rag,
                    'evaluate_rag': batch_evaluate_rag,
                    'model_name': model_name
                }
                
                # Ejecutar consultas
                execute_batch_queries(chromadb_wrapper, embedding_client, openai_client)
    
    with config_col2:
        st.markdown("### 📋 Estado Actual")
        
        # Mostrar configuración si existe
        if 'batch_config' in st.session_state:
            config = st.session_state.batch_config
            st.metric("🔢 Preguntas", config['num_questions'])
            st.metric("📄 Docs/consulta", config['top_k'])
            st.metric("🤖 LLM Reranking", "✅" if config['use_llm'] else "❌")
            st.metric("❓ Use Questions", "✅" if config['use_questions'] else "❌")
            st.metric("🎯 Método", config['selection_method'])
            
            if config.get('keyword_filter'):
                st.text(f"🔍 Filtro: {config['keyword_filter']}")
        else:
            st.info("Configura y ejecuta una consulta en lote para ver los parámetros")
        
        # Mostrar progreso si hay una ejecución en curso
        if st.session_state.get('batch_running', False):
            st.warning("⏳ Consulta en lote ejecutándose...")
            if 'batch_progress' in st.session_state:
                progress = st.session_state.batch_progress
                st.progress(progress['current'] / progress['total'])
                st.text(f"{progress['current']}/{progress['total']} preguntas procesadas")
    
    # Mostrar resultados si existen
    if 'batch_results' in st.session_state:
        show_comprehensive_metrics()

def execute_batch_queries(chromadb_wrapper, embedding_client, openai_client):
    """Ejecuta las consultas en lote según la configuración."""
    config = st.session_state.batch_config
    
    # Obtener preguntas según el método seleccionado
    with st.spinner("📋 Obteniendo preguntas de la base de datos..."):
        questions = get_questions_from_db(
            chromadb_wrapper, 
            config['num_questions'], 
            config['selection_method'], 
            config.get('keyword_filter', '')
        )
    
    if not questions:
        st.error("❌ No se encontraron preguntas con los criterios especificados")
        return
    
    st.success(f"✅ Obtenidas {len(questions)} preguntas para procesar")
    
    # Inicializar progreso
    st.session_state.batch_running = True
    st.session_state.batch_progress = {'current': 0, 'total': len(questions)}
    
    # Contenedor para el progreso
    progress_container = st.empty()
    results_container = st.empty()
    
    # Ejecutar consultas
    batch_results = []
    start_time = time.time()
    
    for i, question_data in enumerate(questions):
        # Actualizar progreso
        st.session_state.batch_progress['current'] = i + 1
        
        with progress_container.container():
            st.progress((i + 1) / len(questions))
            st.text(f"Procesando pregunta {i + 1}/{len(questions)}: {question_data.get('title', 'Sin título')[:50]}...")
        
        # Ejecutar consulta
        question_text = question_data.get('body', question_data.get('title', ''))
        if question_text:
            query_start = time.time()
            
            try:
                if config.get('enable_rag', False):
                    # Usar RAG completo
                    model_name = st.session_state.batch_config['model_name']
                    documents_class=CHROMADB_COLLECTION_CONFIG[model_name]["documents"]
                    questions_class=CHROMADB_COLLECTION_CONFIG[model_name]["questions"]
                    results, debug_info, generated_answer, rag_metrics = answer_question_with_rag(
                        question_text,
                        chromadb_wrapper,
                        embedding_client,
                        openai_client,
                        top_k=config['top_k'],
                        use_llm_reranker=config['use_llm'],
                        use_questions_collection=config['use_questions'],
                        evaluate_quality=config.get('evaluate_rag', False),
                        documents_class=documents_class,
                        questions_class=questions_class
                    )
                else:
                    # Solo documentos (modo tradicional)
                    model_name = st.session_state.batch_config['model_name']
                    documents_class=CHROMADB_COLLECTION_CONFIG[model_name]["documents"]
                    questions_class=CHROMADB_COLLECTION_CONFIG[model_name]["questions"]
                    results, debug_info = answer_question_documents_only(
                        question_text,
                        chromadb_wrapper,
                        embedding_client,
                        openai_client,
                        top_k=config['top_k'],
                        use_llm_reranker=config['use_llm'],
                        use_questions_collection=config['use_questions'],
                        documents_class=documents_class,
                        questions_class=questions_class
                    )
                    generated_answer = None
                    rag_metrics = {}
                
                query_time = time.time() - query_start
                
                batch_results.append({
                    'question_id': question_data.get('id', f'q_{i}'),
                    'question_title': question_data.get('title', 'Sin título'),
                    'question_body': question_text,
                    'results_count': len(results),
                    'query_time': query_time,
                    'results': results,
                    'debug_info': debug_info,
                    'generated_answer': generated_answer,
                    'rag_metrics': rag_metrics,
                    'has_rag_answer': generated_answer is not None
                })
                
            except Exception as e:
                st.error(f"Error en pregunta {i + 1}: {e}")
                batch_results.append({
                    'question_id': question_data.get('id', f'q_{i}'),
                    'question_title': question_data.get('title', 'Sin título'),
                    'question_body': question_text,
                    'results_count': 0,
                    'query_time': 0,
                    'results': [],
                    'debug_info': f"Error: {e}",
                    'generated_answer': None,
                    'rag_metrics': {'status': 'error', 'error': str(e)},
                    'has_rag_answer': False
                })
    
    total_time = time.time() - start_time
    
    # Guardar resultados
    st.session_state.batch_results = {
        'results': batch_results,
        'total_time': total_time,
        'config': config,
        'timestamp': time.time()
    }
    
    # Limpiar estado de ejecución
    st.session_state.batch_running = False
    if 'batch_progress' in st.session_state:
        del st.session_state.batch_progress
    
    progress_container.empty()
    st.success(f"✅ Consulta en lote completada en {total_time:.2f} segundos")
    st.rerun()

def get_questions_from_db(chromadb_wrapper, num_questions, selection_method, keyword_filter):
    """Obtiene preguntas de la base de datos según el método especificado."""
    try:
        model_name = st.session_state.batch_config['model_name']
        questions_class = CHROMADB_COLLECTION_CONFIG[model_name]["questions"]
        
        if selection_method == "🎯 Palabras clave":
            # Búsqueda por palabra clave
            questions = chromadb_wrapper.search_questions_by_keyword(keyword_filter, limit=num_questions)
        else:
            # Obtener preguntas (primeras N o aleatorias)
            questions = chromadb_wrapper.get_sample_questions(
                limit=num_questions if selection_method == "📊 Primeras N" else num_questions * 3,
                random_sample=(selection_method == "🎲 Aleatorio")
            )
            
            if selection_method == "🎲 Aleatorio" and len(questions) > num_questions:
                questions = random.sample(questions, num_questions)
        
        return questions
    except Exception as e:
        st.error(f"Error obteniendo preguntas: {e}")
        return []

def show_comprehensive_metrics():
    """Muestra métricas comprehensivas de las consultas en lote."""
    st.markdown("---")
    st.markdown("### 📊 Análisis Comprehensivo de Consultas en Lote")
    
    batch_data = st.session_state.batch_results
    results = batch_data['results']
    config = batch_data['config']
    
    # Calcular métricas comprehensivas
    metrics = calculate_comprehensive_metrics(results)
    
    # Dashboard de métricas principales
    st.markdown("#### 🎯 Métricas Principales")
    main_col1, main_col2, main_col3, main_col4, main_col5 = st.columns(5)
    
    with main_col1:
        st.metric("📝 Total Preguntas", metrics['total_questions'])
        st.metric("✅ Tasa de Éxito", f"{metrics['success_rate']:.1f}%")
    with main_col2:
        st.metric("⏱️ Tiempo Promedio", f"{metrics['avg_time']:.2f}s")
        st.metric("📊 Mediana Tiempo", f"{metrics['median_time']:.2f}s")
    with main_col3:
        st.metric("📄 Docs Promedio", f"{metrics['avg_docs']:.1f}")
        st.metric("🎯 Docs Mediana", f"{metrics['median_docs']:.1f}")
    with main_col4:
        st.metric("⚡ Throughput", f"{metrics['throughput']:.1f} q/min")
        st.metric("🔍 Total Docs", metrics['total_docs'])
    with main_col5:
        st.metric("🏃 Tiempo Total", f"{metrics['total_time']:.1f}s")
        st.metric("❌ Fallos", metrics['failed_queries'])
    
    # Tabs para análisis detallado
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Rendimiento", "📊 Distribuciones", "🎯 Calidad", "📋 Resumen", "🔍 Detalles"
    ])
    
    with tab1:
        show_performance_metrics(results, metrics)
    
    with tab2:
        show_distribution_analysis(results, metrics)
    
    with tab3:
        show_quality_metrics(results, metrics)
    
    with tab4:
        show_summary_table(results)
    
    with tab5:
        show_detailed_analysis(results)

def calculate_comprehensive_metrics(results):
    """Calcula métricas comprehensivas de los resultados."""
    total_questions = len(results)
    
    # Métricas básicas
    query_times = [r['query_time'] for r in results]
    docs_counts = [r['results_count'] for r in results]
    successful_queries = sum(1 for r in results if r['results_count'] > 0)
    failed_queries = total_questions - successful_queries
    
    # Métricas de tiempo
    total_time = sum(query_times)
    avg_time = total_time / total_questions if total_questions > 0 else 0
    median_time = statistics.median(query_times) if query_times else 0
    min_time = min(query_times) if query_times else 0
    max_time = max(query_times) if query_times else 0
    std_time = statistics.stdev(query_times) if len(query_times) > 1 else 0
    
    # Métricas de documentos
    total_docs = sum(docs_counts)
    avg_docs = total_docs / total_questions if total_questions > 0 else 0
    median_docs = statistics.median(docs_counts) if docs_counts else 0
    min_docs = min(docs_counts) if docs_counts else 0
    max_docs = max(docs_counts) if docs_counts else 0
    std_docs = statistics.stdev(docs_counts) if len(docs_counts) > 1 else 0
    
    # Métricas de rendimiento
    success_rate = (successful_queries / total_questions * 100) if total_questions > 0 else 0
    throughput = (total_questions / total_time * 60) if total_time > 0 else 0  # queries per minute
    
    # Distribución de scores si están disponibles
    all_scores = []
    for result in results:
        for doc in result.get('results', []):
            score = doc.get('score', 0)
            if score > 0:
                all_scores.append(score)
    
    avg_score = statistics.mean(all_scores) if all_scores else 0
    median_score = statistics.median(all_scores) if all_scores else 0
    min_score = min(all_scores) if all_scores else 0
    max_score = max(all_scores) if all_scores else 0
    
    return {
        'total_questions': total_questions,
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'success_rate': success_rate,
        'total_time': total_time,
        'avg_time': avg_time,
        'median_time': median_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'total_docs': total_docs,
        'avg_docs': avg_docs,
        'median_docs': median_docs,
        'min_docs': min_docs,
        'max_docs': max_docs,
        'std_docs': std_docs,
        'throughput': throughput,
        'avg_score': avg_score,
        'median_score': median_score,
        'min_score': min_score,
        'max_score': max_score,
        'total_scores': len(all_scores)
    }

def show_performance_metrics(results, metrics):
    """Muestra métricas de rendimiento."""
    st.markdown("#### ⚡ Análisis de Rendimiento")
    
    # Métricas de tiempo detalladas
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("**⏱️ Estadísticas de Tiempo de Respuesta**")
        time_metrics_df = pd.DataFrame({
            'Métrica': ['Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Desv. Estándar'],
            'Tiempo (s)': [
                f"{metrics['avg_time']:.3f}",
                f"{metrics['median_time']:.3f}",
                f"{metrics['min_time']:.3f}",
                f"{metrics['max_time']:.3f}",
                f"{metrics['std_time']:.3f}"
            ]
        })
        st.dataframe(time_metrics_df, use_container_width=True)
        
        # Percentiles de tiempo
        query_times = [r['query_time'] for r in results]
        if query_times:
            percentiles = [50, 75, 90, 95, 99]
            perc_data = []
            for p in percentiles:
                perc_data.append({
                    'Percentil': f'P{p}',
                    'Tiempo (s)': f"{np.percentile(query_times, p):.3f}"
                })
            st.markdown("**📊 Percentiles de Tiempo**")
            st.dataframe(pd.DataFrame(perc_data), use_container_width=True)
    
    with perf_col2:
        st.markdown("**📄 Estadísticas de Documentos Recuperados**")
        docs_metrics_df = pd.DataFrame({
            'Métrica': ['Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Desv. Estándar'],
            'Documentos': [
                f"{metrics['avg_docs']:.1f}",
                f"{metrics['median_docs']:.1f}",
                f"{metrics['min_docs']}",
                f"{metrics['max_docs']}",
                f"{metrics['std_docs']:.1f}"
            ]
        })
        st.dataframe(docs_metrics_df, use_container_width=True)
        
        # Distribución de éxito/fallo
        success_data = pd.DataFrame({
            'Estado': ['Exitosas', 'Fallidas'],
            'Cantidad': [metrics['successful_queries'], metrics['failed_queries']],
            'Porcentaje': [metrics['success_rate'], 100 - metrics['success_rate']]
        })
        st.markdown("**✅ Distribución de Éxito**")
        st.dataframe(success_data, use_container_width=True)
    
    # Gráfico de tiempo vs documentos
    st.markdown("**📈 Correlación Tiempo vs Documentos Encontrados**")
    scatter_data = pd.DataFrame({
        'Tiempo (s)': [r['query_time'] for r in results],
        'Documentos': [r['results_count'] for r in results],
        'Pregunta': [f"Q{i+1}" for i in range(len(results))]
    })
    
    fig_scatter = px.scatter(
        scatter_data, 
        x='Tiempo (s)', 
        y='Documentos',
        hover_data=['Pregunta'],
        title="Relación entre Tiempo de Respuesta y Documentos Encontrados",
        color='Documentos',
        size='Documentos',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_distribution_analysis(results, metrics):
    """Muestra análisis de distribuciones."""
    st.markdown("#### 📊 Análisis de Distribuciones")
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribución de Tiempos de Respuesta',
            'Distribución de Documentos Encontrados',
            'Evolución Temporal de Consultas',
            'Distribución de Scores (si disponible)'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Histograma de tiempos
    query_times = [r['query_time'] for r in results]
    fig.add_trace(
        go.Histogram(x=query_times, name="Tiempo (s)", nbinsx=20),
        row=1, col=1
    )
    
    # Histograma de documentos
    docs_counts = [r['results_count'] for r in results]
    fig.add_trace(
        go.Histogram(x=docs_counts, name="Documentos", nbinsx=max(docs_counts)+1 if docs_counts else 1),
        row=1, col=2
    )
    
    # Evolución temporal
    question_ids = list(range(1, len(results)+1))
    fig.add_trace(
        go.Scatter(x=question_ids, y=query_times, mode='lines+markers', name="Tiempo"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=question_ids, y=docs_counts, mode='lines+markers', name="Docs", yaxis="y2"),
        row=2, col=1
    )
    
    # Distribución de scores
    all_scores = []
    for result in results:
        for doc in result.get('results', []):
            score = doc.get('score', 0)
            if score > 0:
                all_scores.append(score)
    
    if all_scores:
        fig.add_trace(
            go.Histogram(x=all_scores, name="Scores", nbinsx=20),
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=['No scores disponibles'], name="Scores"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Análisis de Distribuciones")
    st.plotly_chart(fig, use_container_width=True)

def show_quality_metrics(results, metrics):
    """Muestra métricas de calidad."""
    st.markdown("#### 🎯 Métricas de Calidad")
    
    # Add ranking metrics evaluation if ground truth is available
    # For now, show comprehensive score analysis
    st.markdown("**🎯 Métricas de Ranking - Top-5 vs Completo**")
    
    # Calculate metrics for top-5 documents across all queries
    top5_scores = []
    all_scores = []
    for result in results:
        query_scores = [doc.get('score', 0) for doc in result.get('results', []) if doc.get('score', 0) > 0]
        if query_scores:
            top5_scores.extend(query_scores[:5])  # Top-5 scores per query
            all_scores.extend(query_scores)      # All scores per query
    
    if top5_scores:
        import numpy as np
        
        # Display top-5 vs all comparison
        comparison_col1, comparison_col2 = st.columns(2)
        
        with comparison_col1:
            st.markdown("**📊 Métricas Top-5**")
            top5_stats = pd.DataFrame({
                'Métrica': ['Precision@5*', 'Recall@5*', 'MRR@5*', 'F1-Score@5*', 'nDCG@5*'],
                'Ubicación': [
                    '→ Tab "Detalles"',
                    '→ Tab "Detalles"', 
                    '→ Tab "Detalles"',
                    '→ Tab "Detalles"',
                    '→ Tab "Detalles"'
                ]
            })
            st.dataframe(top5_stats, use_container_width=True)
            st.info("💡 **Ver evaluación individual:** Ve al tab 'Detalles' y selecciona una pregunta para ver sus métricas específicas de Precision@5, Recall@5, MRR@5, F1-Score@5 y nDCG@5.")
            
            # Top-5 score statistics
            st.markdown("**📈 Estadísticas Top-5 Scores**")
            top5_score_stats = pd.DataFrame({
                'Métrica': ['Promedio', 'Mediana', 'Máximo', 'Mínimo', 'Desv. Estándar'],
                'Top-5': [
                    f"{np.mean(top5_scores):.4f}",
                    f"{np.median(top5_scores):.4f}",
                    f"{max(top5_scores):.4f}",
                    f"{min(top5_scores):.4f}",
                    f"{np.std(top5_scores):.4f}"
                ]
            })
            st.dataframe(top5_score_stats, use_container_width=True)
            
        with comparison_col2:
            st.markdown("**📊 Métricas Completas**")
            if metrics['total_scores'] > 0:
                score_stats = pd.DataFrame({
                    'Métrica': ['Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Total Scores'],
                    'Valor': [
                        f"{metrics['avg_score']:.4f}",
                        f"{metrics['median_score']:.4f}",
                        f"{metrics['min_score']:.4f}",
                        f"{metrics['max_score']:.4f}",
                        f"{metrics['total_scores']}"
                    ]
                })
                st.dataframe(score_stats, use_container_width=True)
            else:
                st.info("No hay scores disponibles en los resultados")
        
        # Advanced score analysis
        st.markdown("**📈 Análisis Avanzado de Scores**")
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Score distribution for all vs top-5
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Histogram(
                x=all_scores, 
                name="Todos los docs", 
                opacity=0.7,
                nbinsx=20
            ))
            fig_comparison.add_trace(go.Histogram(
                x=top5_scores, 
                name="Top-5 docs", 
                opacity=0.7,
                nbinsx=20
            ))
            fig_comparison.update_layout(
                title="Distribución: Todos vs Top-5",
                xaxis_title="Score de Similitud",
                yaxis_title="Frecuencia",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
        with analysis_col2:
            # Quality distribution analysis
            high_quality_all = len([s for s in all_scores if s >= 0.8])
            medium_quality_all = len([s for s in all_scores if 0.6 <= s < 0.8])
            low_quality_all = len([s for s in all_scores if s < 0.6])
            
            high_quality_top5 = len([s for s in top5_scores if s >= 0.8])
            medium_quality_top5 = len([s for s in top5_scores if 0.6 <= s < 0.8])
            low_quality_top5 = len([s for s in top5_scores if s < 0.6])
            
            quality_comparison = pd.DataFrame({
                'Calidad': ['Alta (≥0.8)', 'Media (0.6-0.8)', 'Baja (<0.6)'],
                'Todos': [high_quality_all, medium_quality_all, low_quality_all],
                'Top-5': [high_quality_top5, medium_quality_top5, low_quality_top5],
                '% Todos': [f"{high_quality_all/len(all_scores)*100:.1f}%", 
                           f"{medium_quality_all/len(all_scores)*100:.1f}%", 
                           f"{low_quality_all/len(all_scores)*100:.1f}%"],
                '% Top-5': [f"{high_quality_top5/len(top5_scores)*100:.1f}%", 
                           f"{medium_quality_top5/len(top5_scores)*100:.1f}%", 
                           f"{low_quality_top5/len(top5_scores)*100:.1f}%"]
            })
            
            st.markdown("**🎯 Distribución por Calidad**")
            st.dataframe(quality_comparison, use_container_width=True)
            
            # Performance improvement in top-5
            avg_improvement = np.mean(top5_scores) - np.mean(all_scores)
            median_improvement = np.median(top5_scores) - np.median(all_scores)
            
            st.markdown("**📈 Mejora en Top-5**")
            improvement_stats = pd.DataFrame({
                'Métrica': ['Mejora Promedio', 'Mejora Mediana'],
                'Valor': [f"{avg_improvement:+.4f}", f"{median_improvement:+.4f}"],
                'Porcentaje': [f"{avg_improvement/np.mean(all_scores)*100:+.1f}%", 
                              f"{median_improvement/np.median(all_scores)*100:+.1f}%"]
            })
            st.dataframe(improvement_stats, use_container_width=True)
    else:
        st.info("No hay scores disponibles para análisis de calidad.")
    
    # Document count distribution (always show this)
    st.markdown("---")
    st.markdown("**📄 Distribución por Cantidad de Documentos**")
    
    doc_ranges = {'0 docs': 0, '1-2 docs': 0, '3-5 docs': 0, '6-10 docs': 0, '10+ docs': 0}
    
    for result in results:
        count = result['results_count']
        if count == 0:
            doc_ranges['0 docs'] += 1
        elif count <= 2:
            doc_ranges['1-2 docs'] += 1
        elif count <= 5:
            doc_ranges['3-5 docs'] += 1
        elif count <= 10:
            doc_ranges['6-10 docs'] += 1
        else:
            doc_ranges['10+ docs'] += 1
    
    range_df = pd.DataFrame(list(doc_ranges.items()), columns=['Rango', 'Cantidad'])
    
    # Create side-by-side layout for document distribution
    doc_col1, doc_col2 = st.columns(2)
    
    with doc_col1:
        st.dataframe(range_df, use_container_width=True)
    
    with doc_col2:
        # Gráfico de pie para distribución de éxito
        fig_pie = px.pie(
            values=[metrics['successful_queries'], metrics['failed_queries']],
            names=['Exitosas', 'Fallidas'],
            title="Distribución de Consultas Exitosas vs Fallidas",
            color_discrete_map={'Exitosas': '#2E8B57', 'Fallidas': '#DC143C'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top consultas por tiempo de respuesta
        st.markdown("**⚡ Top 5 Consultas Más Rápidas**")
        sorted_by_time = sorted(results, key=lambda x: x['query_time'])
        fast_queries = []
        for i, result in enumerate(sorted_by_time[:5]):
            fast_queries.append({
                'Rank': i+1,
                'Tiempo (s)': f"{result['query_time']:.3f}",
                'Docs': result['results_count'],
                'Título': result['question_title'][:40] + "..."
            })
        st.dataframe(pd.DataFrame(fast_queries), use_container_width=True)
        
        # Top consultas por documentos encontrados
        st.markdown("**🏆 Top 5 Consultas con Más Documentos**")
        sorted_by_docs = sorted(results, key=lambda x: x['results_count'], reverse=True)
        top_docs = []
        for i, result in enumerate(sorted_by_docs[:5]):
            top_docs.append({
                'Rank': i+1,
                'Docs': result['results_count'],
                'Tiempo (s)': f"{result['query_time']:.3f}",
                'Título': result['question_title'][:40] + "..."
            })
        st.dataframe(pd.DataFrame(top_docs), use_container_width=True)

def show_summary_table(results):
    """Muestra tabla resumen."""
    st.markdown("#### 📋 Tabla Resumen de Todas las Consultas")
    
    summary_data = []
    for i, result in enumerate(results):
        summary_data.append({
            'ID': i + 1,
            'Título': result['question_title'][:60] + "..." if len(result['question_title']) > 60 else result['question_title'],
            'Docs': result['results_count'],
            'Tiempo (s)': f"{result['query_time']:.3f}",
            'Estado': "✅" if result['results_count'] > 0 else "❌",
            'Método': result.get('method', 'N/A')
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Filtros para la tabla
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        show_only_successful = st.checkbox("Solo consultas exitosas", value=False)
    with filter_col2:
        min_docs = st.slider("Mínimo documentos", 0, max([r['results_count'] for r in results]), 0)
    with filter_col3:
        max_time = st.slider("Máximo tiempo (s)", 0.0, max([r['query_time'] for r in results]), max([r['query_time'] for r in results]))
    
    # Aplicar filtros
    filtered_df = df_summary.copy()
    if show_only_successful:
        filtered_df = filtered_df[filtered_df['Estado'] == "✅"]
    filtered_df = filtered_df[filtered_df['Docs'] >= min_docs]
    filtered_df = filtered_df[filtered_df['Tiempo (s)'].astype(float) <= max_time]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Estadísticas de la tabla filtrada
    if len(filtered_df) > 0:
        st.markdown(f"**Mostrando {len(filtered_df)} de {len(df_summary)} consultas**")

def show_detailed_analysis(results):
    """Muestra análisis detallado de consultas individuales."""
    st.markdown("#### 🔍 Análisis Detallado por Consulta")
    
    # Selector de pregunta
    selected_question = st.selectbox(
        "Selecciona una pregunta para análisis detallado:",
        range(len(results)),
        format_func=lambda x: f"#{x+1}: {results[x]['question_title'][:60]}..."
    )
    
    if selected_question is not None:
        result = results[selected_question]
        
        # Información básica
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown(f"**🆔 Pregunta #{selected_question + 1}**")
            st.markdown(f"**📝 Título:** {result['question_title']}")
            st.markdown("**💭 Contenido:**")
            st.text_area("Pregunta completa:", result['question_body'], height=100, disabled=True)
            
        with detail_col2:
            st.markdown("**📊 Métricas de la Consulta**")
            st.metric("⏱️ Tiempo de respuesta", f"{result['query_time']:.3f}s")
            st.metric("📄 Documentos encontrados", result['results_count'])
            st.metric("✅ Estado", "Exitosa" if result['results_count'] > 0 else "Fallida")
            
            if result.get('debug_info'):
                with st.expander("🔧 Información de Debug"):
                    st.text(result['debug_info'])
        
        # Métricas de Evaluación Individual
        st.markdown("---")
        st.markdown("**🎯 Evaluación de Calidad Individual**")
        
        if result['results']:
            # Calculate quality-based metrics using score thresholds
            docs = result['results']
            scores = [doc.get('score', 0) for doc in docs]
            
            # Define relevance based on score thresholds (adaptable approach)
            high_relevance_threshold = 0.75  # Highly relevant
            medium_relevance_threshold = 0.6  # Moderately relevant
            
            eval_col1, eval_col2 = st.columns(2)
            
            with eval_col1:
                st.markdown("**📊 Métricas Top-5 (Basadas en Scores)**")
                
                # Get top-5 results
                top5_docs = docs[:5] if len(docs) >= 5 else docs
                top5_scores = scores[:5] if len(scores) >= 5 else scores
                
                # Calculate metrics based on score thresholds
                highly_relevant_top5 = len([s for s in top5_scores if s >= high_relevance_threshold])
                moderately_relevant_top5 = len([s for s in top5_scores if s >= medium_relevance_threshold])
                
                # Quality-based pseudo metrics
                precision_high_5 = highly_relevant_top5 / len(top5_docs) if top5_docs else 0
                precision_med_5 = moderately_relevant_top5 / len(top5_docs) if top5_docs else 0
                
                # Score-based ranking quality
                avg_score_top5 = sum(top5_scores) / len(top5_scores) if top5_scores else 0
                max_score_top5 = max(top5_scores) if top5_scores else 0
                
                # Find position of first highly relevant document
                first_relevant_pos = next((i+1 for i, s in enumerate(top5_scores) if s >= high_relevance_threshold), 0)
                mrr_5 = 1.0 / first_relevant_pos if first_relevant_pos > 0 else 0.0
                
                metrics_df = pd.DataFrame({
                    'Métrica': [
                        'Precision@5 (Alta)',
                        'Precision@5 (Media)', 
                        'MRR@5 (Score≥0.75)',
                        'Avg Score Top-5',
                        'Max Score Top-5'
                    ],
                    'Valor': [
                        f"{precision_high_5:.3f}",
                        f"{precision_med_5:.3f}",
                        f"{mrr_5:.3f}",
                        f"{avg_score_top5:.4f}",
                        f"{max_score_top5:.4f}"
                    ],
                    'Descripción': [
                        f'{highly_relevant_top5}/5 docs con score ≥ 0.75',
                        f'{moderately_relevant_top5}/5 docs con score ≥ 0.60',
                        'Posición del primer doc altamente relevante',
                        'Promedio de scores en top-5',
                        'Mejor score en top-5'
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                # Score distribution in top-5
                st.markdown("**📈 Distribución de Calidad Top-5**")
                quality_dist = pd.DataFrame({
                    'Calidad': ['Alta (≥0.75)', 'Media (0.6-0.75)', 'Baja (<0.6)'],
                    'Cantidad': [
                        len([s for s in top5_scores if s >= 0.75]),
                        len([s for s in top5_scores if 0.6 <= s < 0.75]),
                        len([s for s in top5_scores if s < 0.6])
                    ]
                })
                st.dataframe(quality_dist, use_container_width=True)
            
            with eval_col2:
                st.markdown("**📊 Análisis Comparativo Completo**")
                
                # Compare top-5 vs all results
                all_scores = scores
                avg_score_all = sum(all_scores) / len(all_scores) if all_scores else 0
                highly_relevant_all = len([s for s in all_scores if s >= high_relevance_threshold])
                
                comparison_df = pd.DataFrame({
                    'Métrica': [
                        'Documentos',
                        'Score Promedio',
                        'Score Máximo',
                        'Docs Alta Calidad',
                        '% Alta Calidad'
                    ],
                    'Top-5': [
                        len(top5_docs),
                        f"{avg_score_top5:.4f}",
                        f"{max_score_top5:.4f}",
                        highly_relevant_top5,
                        f"{precision_high_5*100:.1f}%"
                    ],
                    'Todos': [
                        len(docs),
                        f"{avg_score_all:.4f}",
                        f"{max(all_scores):.4f}" if all_scores else "0.0000",
                        highly_relevant_all,
                        f"{highly_relevant_all/len(docs)*100:.1f}%" if docs else "0.0%"
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualización de scores
                if len(top5_scores) > 0:
                    fig_individual = px.bar(
                        x=[f"Doc {i+1}" for i in range(len(top5_scores))],
                        y=top5_scores,
                        title=f"Scores Top-5 - Pregunta #{selected_question + 1}",
                        labels={'x': 'Documento', 'y': 'Score de Similitud'},
                        color=top5_scores,
                        color_continuous_scale='RdYlGn',
                        text=[f"{s:.3f}" for s in top5_scores]
                    )
                    fig_individual.update_traces(textposition='outside')
                    fig_individual.update_layout(height=350, showlegend=False)
                    
                    # Add threshold lines
                    fig_individual.add_hline(y=high_relevance_threshold, 
                                           line_dash="dash", line_color="green",
                                           annotation_text="Alta Relevancia (0.75)")
                    fig_individual.add_hline(y=medium_relevance_threshold, 
                                           line_dash="dash", line_color="orange",
                                           annotation_text="Media Relevancia (0.60)")
                    
                    st.plotly_chart(fig_individual, use_container_width=True)
                
                # Ground truth evaluation option
                st.markdown("**🎯 Evaluación con Ground Truth**")
                with st.expander("Configurar evaluación con referencia"):
                    st.info("💡 Para evaluación completa de Precision, Recall, F1 y nDCG, proporciona enlaces de referencia:")
                    
                    reference_links = st.text_area(
                        "Enlaces de referencia (uno por línea):",
                        help="Ingresa los enlaces que consideras relevantes para esta pregunta",
                        key=f"ref_links_{selected_question}"
                    )
                    
                    if reference_links and st.button("Calcular Métricas", key=f"calc_metrics_{selected_question}"):
                        ref_links_list = [link.strip() for link in reference_links.split('\n') if link.strip()]
                        
                        if ref_links_list:
                            # Import metrics functions
                            from src.evaluation.metrics import compute_precision_recall_f1, compute_mrr, compute_ndcg
                            
                            # Prepare docs for metrics calculation
                            our_docs_for_metrics = [{"link": doc.get("link")} for doc in docs]
                            
                            # Calculate metrics at k=5
                            precision_5, recall_5, f1_5 = compute_precision_recall_f1(our_docs_for_metrics, ref_links_list, k=5)
                            mrr_gt_5 = compute_mrr(our_docs_for_metrics, ref_links_list, k=5)
                            ndcg_gt_5 = compute_ndcg(our_docs_for_metrics, ref_links_list, k=5)
                            
                            # Display ground truth metrics
                            gt_metrics_df = pd.DataFrame({
                                'Métrica': ['Precision@5', 'Recall@5', 'F1-Score@5', 'MRR@5', 'nDCG@5'],
                                'Valor': [
                                    f"{precision_5:.3f}",
                                    f"{recall_5:.3f}",
                                    f"{f1_5:.3f}",
                                    f"{mrr_gt_5:.3f}",
                                    f"{ndcg_gt_5:.3f}"
                                ]
                            })
                            st.dataframe(gt_metrics_df, use_container_width=True)
                            
                            # Show matches
                            our_links = [doc.get('link', '') for doc in docs[:5]]
                            matches = [link for link in our_links if link in ref_links_list]
                            st.success(f"✅ Coincidencias encontradas: {len(matches)}/{len(ref_links_list)}")
                            if matches:
                                st.write("**Enlaces coincidentes:**")
                                for match in matches:
                                    st.write(f"- {match}")
        else:
            st.warning("⚠️ No hay documentos para evaluar en esta consulta.")
        
        # Documentos encontrados
        if result['results']:
            st.markdown("**📄 Documentos Encontrados:**")
            
            docs_data = []
            for i, doc in enumerate(result['results']):
                docs_data.append({
                    'Rank': i + 1,
                    'Título': doc.get('title', 'Sin título')[:50] + "...",
                    'Score': f"{doc.get('score', 0):.4f}",
                    'Link': doc.get('link', 'N/A')
                })
            
            docs_df = pd.DataFrame(docs_data)
            st.dataframe(docs_df, use_container_width=True)
            
            # Mostrar contenido de documentos seleccionados
            doc_to_show = st.selectbox(
                "Ver contenido del documento:",
                range(len(result['results'])),
                format_func=lambda x: f"Doc {x+1}: {result['results'][x].get('title', 'Sin título')[:40]}..."
            )
            
            if doc_to_show is not None:
                selected_doc = result['results'][doc_to_show]
                st.markdown(f"**📖 Contenido del Documento {doc_to_show + 1}:**")
                st.markdown(f"**🔗 Link:** [{selected_doc.get('link', '#')}]({selected_doc.get('link', '#')})")
                if selected_doc.get('content'):
                    st.text_area(
                        "Contenido:", 
                        selected_doc['content'][:1000] + "..." if len(selected_doc.get('content', '')) > 1000 else selected_doc.get('content', ''),
                        height=200,
                        disabled=True
                    )
        else:
            st.warning("⚠️ No se encontraron documentos para esta consulta")

if __name__ == "__main__":
    show_batch_queries_page()