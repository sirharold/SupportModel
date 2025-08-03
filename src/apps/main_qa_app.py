import multiprocessing
import os
import time
import pandas as pd

# Set multiprocessing start method to 'spawn' and disable tokenizers parallelism
# This must be done before any other imports that might initialize multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import plotly.express as px
import json
import numpy as np
from src.core.qa_pipeline import answer_question_documents_only, answer_question_with_rag
from src.services.auth.clients import initialize_clients
from src.services.local_models import preload_tinyllama_model
from src.apps.comparison_page import show_comparison_page
# from src.apps.batch_queries_page import show_batch_queries_page  # Module doesn't exist
from src.apps.data_analysis_page import show_data_analysis_page
from src.apps.cumulative_metrics_create import show_cumulative_metrics_create_page
from src.apps.cumulative_metrics_results import show_cumulative_metrics_results_page
from src.apps.cumulative_metrics_results_matplotlib import show_cumulative_metrics_results_page as show_cumulative_metrics_results_matplotlib_page
from src.apps.question_answer_comparison import show_question_answer_comparison_page
from src.apps.cumulative_comparison import show_cumulative_comparison_page
# from src.apps.cumulative_n_questions_config import show_cumulative_n_questions_config_page  # Module doesn't exist
# from src.apps.cumulative_n_questions_results import show_cumulative_n_questions_results_page  # Module doesn't exist
from src.apps.chapter_4_visualizations import main as show_chapter_4_visualizations
from src.apps.chapter_7_figures import main as show_chapter_7_figures
from src.config.config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, CHROMADB_COLLECTION_CONFIG, GENERATIVE_MODELS, DEFAULT_GENERATIVE_MODEL, GENERATIVE_MODEL_DESCRIPTIONS

def _sanitize_json_string(json_string: str) -> str:
    """Sanitiza una cadena JSON eliminando caracteres de control inválidos."""
    import re
    
    # Método más robusto: usar regex para remover todos los caracteres de control
    # ASCII control characters (0-31) except \t(9), \n(10), \r(13)
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_string)
    
    # También remover caracteres Unicode problemáticos
    sanitized = re.sub(r'[\u0080-\u009F]', '', sanitized)  # C1 control characters
    sanitized = re.sub(r'[\u2028\u2029]', '', sanitized)   # Line/Paragraph separators
    
    # Remove any remaining non-printable characters
    sanitized = re.sub(r'[^\x20-\x7E\t\n\r]', '', sanitized)
    
    return sanitized

# Configuración de página
st.set_page_config(
    page_title="Azure Q&A Expert System", 
    layout="wide",
    page_icon="☁️"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0078d4 0%, #00bcf2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0078d4;
    }
    .doc-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-high { border-left: 4px solid #28a745; }
    .confidence-medium { border-left: 4px solid #ffc107; }
    .confidence-low { border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Cache del modelo TinyLlama para mejor performance
@st.cache_resource
def get_cached_tinyllama_client():
    """Get cached TinyLlama client for better performance."""
    from src.services.local_models import get_tinyllama_client
    return get_tinyllama_client()

@st.cache_resource
def preload_model_on_startup():
    """Preload model when app starts."""
    try:
        success = preload_tinyllama_model()
        return success
    except Exception as e:
        st.error(f"❌ Error precargando modelo: {e}")
        return False

# Precargar modelo al iniciar la aplicación
with st.spinner("🔄 Inicializando modelo TinyLlama..."):
    model_preloaded = preload_model_on_startup()

# Header principal
st.markdown("""
<div class="main-header">
    <h1>☁️ Azure Q&A Expert System</h1>
    <p>Encuentra documentación oficial de Microsoft Azure para tus preguntas técnicas</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para navegación y configuración
st.sidebar.title("🧭 Navegación")
page = st.sidebar.radio(
    "Selecciona una página:",
    [
        "🔍 Búsqueda Individual",
        "📊 Visualizaciones Capítulo 4",
        "📊 Figuras Capítulo 7",
        "📈 Análisis de Datos",
        "🔬 Comparación de Modelos",
        "🔄 Comparador Pregunta vs Respuesta",
        "📊 Análisis Acumulativo N Preguntas",
        "⚙️ Configuración Métricas Acumulativas",
        "📈 Resultados Métricas Acumulativas",
        "📊 Resultados Matplotlib",
    ],
    index=0
)
st.sidebar.markdown("---")

st.sidebar.title("⚙️ Configuración")

# Selección de modelo de embedding
model_name = st.sidebar.selectbox(
    "Selecciona el modelo de embedding:",
    options=list(EMBEDDING_MODELS.keys()),
    index=list(EMBEDDING_MODELS.keys()).index(DEFAULT_EMBEDDING_MODEL)
)

# Selección de modelo generativo
generative_model_name = st.sidebar.selectbox(
    "Selecciona el modelo generativo:",
    options=list(GENERATIVE_MODELS.keys()),
    index=list(GENERATIVE_MODELS.keys()).index(DEFAULT_GENERATIVE_MODEL),
    help="TinyLlama es gratuito y funciona sin configuración. Mistral requiere autorización en Hugging Face."
)

# Mostrar información del modelo seleccionado
if generative_model_name in GENERATIVE_MODEL_DESCRIPTIONS:
    model_info = GENERATIVE_MODEL_DESCRIPTIONS[generative_model_name]
    st.sidebar.success(f"🎯 **{model_info['cost']}** - {model_info['description']}")
    st.sidebar.info(f"📋 **Requisitos**: {model_info['requirements']}")
    
    # Advertencia especial para Mistral
    if generative_model_name == "mistral-7b":
        st.sidebar.warning("⚠️ **Atención**: Mistral (7B) es muy pesado para laptops. Recomendamos usar TinyLlama (1.1B).")
        st.sidebar.info("📁 Mistral requiere ~14GB de descarga y 6-8GB RAM.")
        
        # Verificar memoria disponible
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)
            if available_memory < 6:
                st.sidebar.error(f"🚫 Memoria insuficiente: {available_memory:.1f}GB disponible, 6GB+ requerido")
            else:
                st.sidebar.info(f"✅ Memoria disponible: {available_memory:.1f}GB")
        except:
            pass
elif generative_model_name == "llama-4-scout":
    st.sidebar.success("🌟 **Modelo de API Gratuito** - Llama-4-Scout via OpenRouter")
    st.sidebar.info("ℹ️ Si el modelo no está disponible temporalmente, intenta con TinyLlama como alternativa local.")
elif generative_model_name == "gemini-1.5-flash":
    st.sidebar.warning("💰 **Modelo de API** - Incurre en costos por uso")
elif generative_model_name == "gpt-4":
    st.sidebar.warning("💰 **Modelo de API** - Incurre en costos altos por uso")

# Páginas principales
if page == "🔍 Búsqueda Individual":
    
    # Parámetros de búsqueda
    search_params = st.sidebar.expander("🔍 Parámetros de Búsqueda", expanded=True)
    with search_params:
        top_k = st.slider("Documentos a retornar", 5, 20, 10)
        use_llm_reranker = st.checkbox("Usar Re-Ranking con LLM", value=True, help="Usa CrossEncoder (ms-marco-MiniLM-L-6-v2) para reordenar documentos con mayor precisión.")
        diversity_threshold = st.slider(
            "Umbral de diversidad", 0.5, 0.95, 0.85, 0.05,
            help="Controla la diversidad de resultados (más alto = más diverso)"
        )
    
    # Configuración RAG
    rag_params = st.sidebar.expander("🤖 Configuración RAG", expanded=False)
    with rag_params:
        enable_rag = st.checkbox("Activar RAG Completo", value=True, 
                                help="Genera respuestas sintetizadas usando los documentos encontrados")
        
        # Opción para generar respuesta en búsqueda individual
        if not enable_rag:
            generate_individual_answer = st.checkbox("Generar Respuesta Individual", value=True,
                                                    help="Genera respuesta usando documentos con score ≥ 0.8 o mínimo 3 documentos con el modelo seleccionado")
            if generate_individual_answer:
                st.info("🎯 Criterio de selección: Documentos con score ≥ 0.8 o mínimo 3 documentos")
                
                # Opción de precarga manual
                if generative_model_name == "tinyllama-1.1b":
                    if st.button("🚀 Precargar TinyLlama (Más Rápido)", help="Carga el modelo en memoria para respuestas más rápidas"):
                        with st.spinner("🔄 Precargando modelo..."):
                            client = get_cached_tinyllama_client()
                            success = client.ensure_loaded()
                            if not success:
                                st.error("❌ Error precargando modelo")
        else:
            generate_individual_answer = False
            
        evaluate_rag_quality = st.checkbox("Evaluar Calidad RAG", value=True,
                                         help="Calcula métricas RAGAS (faithfulness, answer relevancy, context precision/recall) y BERTScore usando modelos reales")
        show_rag_metrics = st.checkbox("Mostrar Métricas RAG", value=True,
                                     help="Muestra métricas de confianza y completitud de la respuesta generada")

    # Métricas de evaluación
    eval_params = st.sidebar.expander("📊 Evaluación", expanded=False)
    with eval_params:
        enable_openai_comparison = st.checkbox("Comparar con OpenAI", value=False)
        show_debug_info = st.checkbox("Mostrar información de debug", value=True)

    chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client = initialize_clients(model_name, generative_model_name)
    
    # Área principal
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Haz tu pregunta sobre Azure")
    
    # Question input with examples
    question_examples = [
        "¿Cómo configurar Managed Identity en Azure Functions?",
        "¿Cómo conectar Azure Functions a Key Vault sin secrets?",
        "¿Cuáles son las mejores prácticas para Azure Storage?",
        "¿Cómo implementar autenticación en Azure App Service?"
    ]
    
    # Initialize session state for persistence
    if 'last_title' not in st.session_state:
        st.session_state.last_title = ""
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""
    if 'keyword_search_results' not in st.session_state:
        st.session_state.keyword_search_results = []
    if 'keyword_search_query' not in st.session_state:
        st.session_state.keyword_search_query = ""
    
    # Update inputs if example was selected or random question loaded
    if 'selected_title' in st.session_state:
        title_value = st.session_state.selected_title
        # Update last_title to persist the value
        st.session_state.last_title = st.session_state.selected_title
        del st.session_state.selected_title
    else:
        title_value = st.session_state.last_title
    
    if 'selected_question' in st.session_state:
        selected_question = st.session_state.selected_question
        # Update last_question to persist the value
        st.session_state.last_question = st.session_state.selected_question
        del st.session_state.selected_question
    else:
        selected_question = st.session_state.last_question

    title = st.text_input(
        "📝 Título (opcional):",
        value=title_value,
        placeholder="e.g., Azure Functions Authentication, Virtual Machine Setup, etc.",
        help="Un título descriptivo para tu consulta",
        key="title_input"
    )

    question = st.text_area(
        "❓ Tu pregunta:",
        value=selected_question,
        height=120,
        placeholder="Describe tu pregunta técnica sobre Azure en detalle...",
        help="Sé específico sobre el servicio de Azure y lo que quieres lograr",
        key="question_input"
    )

    # Botón para cargar pregunta aleatoria
    if st.button("🎲 Cargar Pregunta Aleatoria", 
                help="Carga una pregunta aleatoria con enlaces válidos de Microsoft Learn",
                key="load_random_question"):
        
        with st.spinner("🔍 Cargando pregunta optimizada con enlaces válidos..."):
            try:
                # Importar función optimizada y utilidades necesarias
                from src.data.optimized_questions import get_optimized_random_question
                from src.services.auth.clients import initialize_clients
                
                # Obtener el modelo de embedding desde la configuración actual
                embedding_model = model_name  # Usar directamente el modelo seleccionado en el selectbox
                
                # Inicializar cliente ChromaDB
                chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client = initialize_clients(
                    model_name=embedding_model,
                    generative_model_name="gpt-4"  # No se usa para esta operación
                )
                
                # Obtener pregunta aleatoria validada (versión optimizada ultra-rápida)
                random_question = get_optimized_random_question(
                    chromadb_wrapper=chromadb_wrapper,
                    embedding_model_name=embedding_model,
                    max_attempts=3  # Solo 3 intentos necesarios con la colección optimizada
                )
                
                if random_question:
                    # Actualizar session state para cargar en los campos
                    st.session_state.selected_title = random_question.get('title', '')
                    st.session_state.selected_question = random_question.get('question_content', random_question.get('question', ''))
                    
                    # Mostrar información de la pregunta cargada
                    st.success("✅ Pregunta cargada exitosamente! Los campos se actualizarán automáticamente.")
                    
                    # Mostrar estadísticas de validación en un expander
                    with st.expander("📊 Información de la pregunta cargada"):
                        st.write(f"**Título:** {random_question.get('title', 'Sin título')}")
                        st.write(f"**Enlaces totales:** {random_question.get('total_links', 0)}")
                        st.write(f"**Enlaces válidos:** {random_question.get('valid_links', 0)}")
                        if random_question.get('validation_success_rate'):
                            rate = random_question['validation_success_rate'] * 100
                            st.write(f"**Tasa de validación:** {rate:.1f}%")
                        
                        if random_question.get('validated_links'):
                            st.write("**Enlaces validados:**")
                            for link in random_question['validated_links'][:3]:  # Mostrar primeros 3
                                st.write(f"- {link}")
                            if len(random_question['validated_links']) > 3:
                                st.write(f"... y {len(random_question['validated_links']) - 3} más")
                    
                    # Forzar recarga de la página para actualizar los campos
                    st.rerun()
                    
                else:
                    st.warning("⚠️ No se pudo encontrar una pregunta con enlaces válidos. Intenta nuevamente.")
                    st.info("💡 Tip: Las preguntas válidas requieren que los enlaces de Microsoft Learn en la respuesta aceptada existan como documentos en la base de datos.")
                    
            except Exception as e:
                st.error(f"❌ Error al cargar pregunta aleatoria: {str(e)}")
                st.info("🔧 Verifica que la base de datos esté disponible y configurada correctamente.")

    with col2:
        st.subheader("📈 Métricas de Sesión")

        # Inicializar métricas de sesión
        if 'session_metrics' not in st.session_state:
            st.session_state.session_metrics = {
                'queries_made': 0,
                'avg_response_time': 0,
                'total_docs_retrieved': 0
            }

        metrics_container = st.container()

    # Botón de búsqueda
    if st.button("🔍 Buscar Documentación", type="primary", use_container_width=True):
        # Save current title and question to session state for persistence
        st.session_state.last_title = title
        st.session_state.last_question = question

        # Debug info para entender el problema
        if st.session_state.get('debug_mode', False):
            st.write(f"Debug - question value: '{question}'")
            st.write(f"Debug - title value: '{title}'")
            st.write(f"Debug - last_question: '{st.session_state.get('last_question', '')}'")
        
        # Usar la pregunta del input field o del session state si está vacía
        actual_question = question.strip() or st.session_state.get('last_question', '').strip()
        
        if not actual_question:
            st.warning("⚠️ Por favor ingresa una pregunta.")
            if st.session_state.get('last_question', ''):
                st.info(f"💡 Pregunta detectada en session: {st.session_state['last_question'][:100]}...")
        else:
            # Combine title and question for better search
            full_query = f"{title.strip()} {actual_question}".strip()
            
            # Medir tiempo de respuesta
            start_time = time.time()
            
            with st.spinner("🔍 Buscando documentación relevante..." + (" y generando respuesta..." if enable_rag else "")):
                # Ejecutar búsqueda con o sin RAG
                if enable_rag:
                    results, debug_info, generated_answer, rag_metrics = answer_question_with_rag(
                        full_query,
                        chromadb_wrapper,
                        embedding_client,
                        openai_client,
                        gemini_client,
                        local_tinyllama_client,
                        local_mistral_client,
                        top_k=top_k,
                        diversity_threshold=diversity_threshold,
                        use_llm_reranker=use_llm_reranker,
                        use_questions_collection=False,
                        evaluate_quality=evaluate_rag_quality,
                        documents_class=CHROMADB_COLLECTION_CONFIG[model_name]["documents"],
                        questions_class=CHROMADB_COLLECTION_CONFIG[model_name]["questions"],
                        generative_model_name=generative_model_name
                    )
                    # For RAG mode, we don't have pre-reranking docs, so use results
                    pre_reranking_docs = results
                else:
                    # Get both pre and post reranking results for comparison
                    from src.core.qa_pipeline import answer_question_documents_with_comparison
                    pre_reranking_docs, post_reranking_docs, debug_info = answer_question_documents_with_comparison(
                        full_query,
                        chromadb_wrapper,
                        embedding_client,
                        openai_client,
                        top_k=top_k,
                        diversity_threshold=diversity_threshold,
                        use_llm_reranker=use_llm_reranker,
                        use_questions_collection=False,
                        documents_class=CHROMADB_COLLECTION_CONFIG[model_name]["documents"],
                        questions_class=CHROMADB_COLLECTION_CONFIG[model_name]["questions"]
                    )
                    # Use post-reranking results as main results for compatibility
                    results = post_reranking_docs
                    
                    # Generate final answer using local model for individual search
                    generated_answer = None
                    rag_metrics = {}
                    
                    if results and generate_individual_answer:
                        # Filter documents with score >= 0.8 or take at least 3 documents
                        high_score_docs = [doc for doc in results if doc.get('score', 0) >= 0.8]
                        
                        if len(high_score_docs) >= 3:
                            selected_docs = high_score_docs
                        else:
                            # Take at least 3 documents (or all if less than 3)
                            selected_docs = results[:max(3, len(high_score_docs))]
                        
                        # Generate answer using selected model
                        if generative_model_name == "llama-4-scout" and openrouter_client:
                            # Use OpenRouter client for Llama-4-Scout
                            try:
                                # Prepare context from selected documents with links
                                context_parts = []
                                for i, doc in enumerate(selected_docs):
                                    title = doc.get('title', f'Documento {i+1}')
                                    content = doc.get('content', '')
                                    link = doc.get('link', '')
                                    
                                    context_part = f"Documento {i+1}:\nTítulo: {title}\n"
                                    if link:
                                        context_part += f"Enlace: {link}\n"
                                    context_part += f"Contenido: {content}"
                                    context_parts.append(context_part)
                                
                                context = "\n\n".join(context_parts)
                                
                                generated_answer = openrouter_client.generate_answer(
                                    question=full_query,
                                    context=context,
                                    max_length=512
                                )
                                
                                # Add Microsoft Learn links to the response
                                if generated_answer and not generated_answer.startswith("Error"):
                                    ms_links = []
                                    for doc in selected_docs[:6]:
                                        link = doc.get('link', '')
                                        title = doc.get('title', 'Documento')
                                        if link and 'learn.microsoft.com' in link:
                                            ms_links.append(f"- **{title}**  \n  {link}")
                                    
                                    if ms_links:
                                        generated_answer += "\n\n## Enlaces y Referencias\n\n"
                                        generated_answer += "\n\n".join(ms_links[:max(3, len(ms_links))])
                                        generated_answer += "\n\n*Consulta la documentación oficial de Microsoft Learn para información más detallada.*"
                                
                                # Calculate real RAGAS and BERTScore metrics
                                rag_metrics = {
                                    'docs_used': len(selected_docs),
                                    'high_score_docs': len(high_score_docs),
                                    'min_score': min([doc.get('score', 0) for doc in selected_docs]),
                                    'max_score': max([doc.get('score', 0) for doc in selected_docs]),
                                    'model_provider': 'OpenRouter'
                                }
                                
                                # Calculate RAGAS + BERTScore if enabled
                                if evaluate_rag_quality and show_rag_metrics:
                                    try:
                                        from src.services.answer_generation.ragas_evaluation import evaluate_answer_with_ragas_and_bertscore
                                        ragas_bert_metrics = evaluate_answer_with_ragas_and_bertscore(
                                            question=full_query,
                                            answer=generated_answer,
                                            source_docs=selected_docs,
                                            openai_client=openai_client
                                        )
                                        rag_metrics.update(ragas_bert_metrics)
                                    except Exception as e:
                                        st.warning(f"⚠️ Error calculando métricas RAGAS/BERTScore: {e}")
                                        # Add default values
                                        rag_metrics.update({
                                            'faithfulness': 0.0,
                                            'answer_relevancy': 0.0,
                                            'context_precision': 0.0,
                                            'context_recall': 0.0,
                                            'bert_precision': 0.0,
                                            'bert_recall': 0.0,
                                            'bert_f1': 0.0
                                        })
                            except Exception as e:
                                st.error(f"Error generando respuesta con OpenRouter: {e}")
                                generated_answer = None
                                rag_metrics = {}
                        elif (generative_model_name == "tinyllama-1.1b" and local_tinyllama_client) or \
                           (generative_model_name == "mistral-7b" and local_mistral_client):
                            from src.services.answer_generation.local import generate_final_answer_local
                            
                            try:
                                # Optimized length for faster generation
                                max_len = 256 if generative_model_name == "tinyllama-1.1b" else 512
                                
                                generated_answer, generation_info = generate_final_answer_local(
                                    question=full_query,
                                    retrieved_docs=selected_docs,
                                    model_name=generative_model_name,
                                    max_length=max_len
                                )
                                
                                # Calculate real RAGAS and BERTScore metrics
                                rag_metrics = {
                                    'docs_used': len(selected_docs),
                                    'high_score_docs': len(high_score_docs),
                                    'min_score': min([doc.get('score', 0) for doc in selected_docs]),
                                    'max_score': max([doc.get('score', 0) for doc in selected_docs]),
                                    'model_provider': 'Local'
                                }
                                
                                # Calculate RAGAS + BERTScore if enabled
                                if evaluate_rag_quality and show_rag_metrics:
                                    try:
                                        from src.services.answer_generation.ragas_evaluation import evaluate_answer_with_ragas_and_bertscore
                                        ragas_bert_metrics = evaluate_answer_with_ragas_and_bertscore(
                                            question=full_query,
                                            answer=generated_answer,
                                            source_docs=selected_docs,
                                            openai_client=openai_client
                                        )
                                        rag_metrics.update(ragas_bert_metrics)
                                    except Exception as e:
                                        st.warning(f"⚠️ Error calculando métricas RAGAS/BERTScore: {e}")
                                        # Add default values
                                        rag_metrics.update({
                                            'faithfulness': 0.0,
                                            'answer_relevancy': 0.0,
                                            'context_precision': 0.0,
                                            'context_recall': 0.0,
                                            'bert_precision': 0.0,
                                            'bert_recall': 0.0,
                                            'bert_f1': 0.0
                                        })
                            except Exception as e:
                                st.error(f"Error generando respuesta con {generative_model_name}: {e}")
                                generated_answer = None
                                rag_metrics = {}
                        else:
                            # No hay cliente disponible para el modelo seleccionado
                            if generative_model_name == "llama-4-scout":
                                st.warning(f"⚠️ OpenRouter client no está disponible. Verifica tu API key OPEN_ROUTER_KEY.")
                            elif generative_model_name in ["tinyllama-1.1b", "mistral-7b"]:
                                st.warning(f"⚠️ Modelo local {generative_model_name} no está disponible. Asegúrate de que esté configurado correctamente.")
                            else:
                                st.warning(f"⚠️ Modelo {generative_model_name} no soportado para respuesta individual.")
                            generated_answer = None
                            rag_metrics = {}
            
            # Actualizar métricas de sesión
            response_time = time.time() - start_time
            st.session_state.session_metrics['queries_made'] += 1
            st.session_state.session_metrics['total_docs_retrieved'] += len(results) if 'results' in locals() and results else 0
            st.session_state.session_metrics['avg_response_time'] = (
                (st.session_state.session_metrics['avg_response_time'] * 
                 (st.session_state.session_metrics['queries_made'] - 1) + response_time) /
                st.session_state.session_metrics['queries_made']
            )

            # Mostrar resultados
            if 'results' in locals() and results:
                if generated_answer:
                    st.success(f"✅ Respuesta generada con {len(results)} documentos en {response_time:.2f}s")
                else:
                    st.success(f"✅ Encontrados {len(results)} documentos relevantes en {response_time:.2f}s")
                
                # Mostrar respuesta generada (RAG o búsqueda individual)
                if generated_answer:
                    st.markdown("---")
                    if enable_rag:
                        st.markdown("### 🤖 **Respuesta Generada (RAG)**")
                    else:
                        st.markdown("### 🤖 **Respuesta Generada (Búsqueda Individual)**")
                        high_score_count = rag_metrics.get('high_score_docs', 0)
                        docs_used = rag_metrics.get('docs_used', 0)
                        min_score = rag_metrics.get('min_score', 0)
                        max_score = rag_metrics.get('max_score', 0)
                        
                        if high_score_count >= 3:
                            st.info(f"🎯 Usando {docs_used} documentos con score ≥ 0.8 (rango: {min_score:.3f} - {max_score:.3f})")
                        else:
                            st.info(f"🎯 Usando {docs_used} documentos ({high_score_count} con score ≥ 0.8, completado con top documentos)")
                            st.warning(f"⚠️ Pocos documentos de alta calidad encontrados. Rango de scores: {min_score:.3f} - {max_score:.3f}")
                    
                    # Mostrar métricas RAGAS + BERTScore si están habilitadas
                    if show_rag_metrics and rag_metrics:
                        # Primera fila: Métricas RAGAS
                        st.markdown("##### 📊 Métricas RAGAS")
                        ragas_col1, ragas_col2, ragas_col3, ragas_col4 = st.columns(4)
                        
                        with ragas_col1:
                            faithfulness = rag_metrics.get('faithfulness', 0)
                            st.metric("🎯 Faithfulness", f"{faithfulness:.3f}", 
                                    help="¿La respuesta es fiel al contexto? (0-1)")
                        
                        with ragas_col2:
                            answer_relevancy = rag_metrics.get('answer_relevancy', 0)
                            st.metric("🔍 Answer Relevancy", f"{answer_relevancy:.3f}", 
                                    help="¿La respuesta aborda la pregunta? (0-1)")
                        
                        with ragas_col3:
                            context_precision = rag_metrics.get('context_precision', 0)
                            st.metric("📈 Context Precision", f"{context_precision:.3f}", 
                                    help="¿Los docs están bien ordenados por relevancia? (0-1)")
                        
                        with ragas_col4:
                            context_recall = rag_metrics.get('context_recall', 0)
                            st.metric("📚 Context Recall", f"{context_recall:.3f}", 
                                    help="¿El contexto contiene toda la info necesaria? (0-1)")
                        
                        # Segunda fila: BERTScore
                        st.markdown("##### 🤖 BERTScore")
                        bert_col1, bert_col2, bert_col3, bert_col4 = st.columns(4)
                        
                        with bert_col1:
                            bert_precision = rag_metrics.get('bert_precision', 0)
                            st.metric("🎯 BERT Precision", f"{bert_precision:.3f}", 
                                    help="Precisión semántica respecto a los documentos")
                        
                        with bert_col2:
                            bert_recall = rag_metrics.get('bert_recall', 0)
                            st.metric("📊 BERT Recall", f"{bert_recall:.3f}", 
                                    help="Cobertura semántica de los documentos")
                        
                        with bert_col3:
                            bert_f1 = rag_metrics.get('bert_f1', 0)
                            st.metric("⚡ BERT F1", f"{bert_f1:.3f}", 
                                    help="Balance entre precisión y recall semántico")
                        
                        with bert_col4:
                            docs_used = rag_metrics.get('docs_used', 0)
                            st.metric("📚 Docs Usados", f"{docs_used}/{len(results)}", 
                                    help="Documentos utilizados para generar la respuesta")
                    
                    # Mostrar la respuesta generada
                    st.markdown("#### 💬 Respuesta:")
                    st.markdown(generated_answer)
                    
                
                st.markdown("---")
                st.markdown("### 📄 **Documentos de Referencia**")
                
                # Create side-by-side comparison
                if enable_openai_comparison:
                    st.warning("⚠️ Comparación con OpenAI temporalmente deshabilitada debido a problemas de indentación")
                    openai_docs = []
                    openai_links = []
                    # TODO: Fix indentation issues in OpenAI comparison block
                    """
                    # Get OpenAI results first
                    with st.spinner("🤖 Consultando OpenAI para comparación..."):
                        try:
                            import json

                            tools = [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "list_azure_documentation",
                                        "description": "Provides a list of relevant Azure documentation links for a given user question.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "documents": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "score": {"type": "number", "description": "Relevance score between 0 and 1, where 1 represents maximum relevance to the question."},
                                                            "title": {"type": "string", "description": "The official title of the documentation page."},
                                                            "link": {"type": "string", "description": "The full URL to the documentation page, must be from learn.microsoft.com."},
                                                        },
                                                        "required": ["score", "title", "link"]
                                                    }
                                                }
                                            },
                                            "required": ["documents"]
                                        }
                                    }
                                }
                            ]
                            
                            response = openai_client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "Actúa como un experto en documentación de Azure. Tu tarea es buscar y listar exactamente 10 documentos desde el dominio oficial https://learn.microsoft.com que puedan ayudar a responder una pregunta técnica. Para cada documento, entrega un score (entre 0 y 1), donde 1 representa máxima relevancia respecto a la pregunta, el título del documento y el link completo. Los documentos deben estar ordenados de mayor a menor según el score. No inventes contenido ni enlaces. Solo acepta resultados del dominio learn.microsoft.com."},
                                    {"role": "user", "content": f"Aquí está la pregunta a responder: {full_query}"}
                                ],
                                tools=tools,
                                tool_choice={"type": "function", "function": {"name": "list_azure_documentation"}},
                                temperature=0.1
                            )
                            
                            message = response.choices[0].message
                            openai_docs = []
                            openai_links = []

                            if message.tool_calls:
                                tool_call = message.tool_calls[0]
                                if tool_call.function.name == "list_azure_documentation":
                                    # Sanitize JSON arguments to handle control characters
                                    raw_arguments = tool_call.function.arguments
                                sanitized_arguments = _sanitize_json_string(raw_arguments)
                                
                                try:
                                    tool_args = json.loads(sanitized_arguments)
                                except json.JSONDecodeError as e:
                                    st.error(f"Error procesando respuesta de OpenAI: {e}")
                                    tool_args = {"documents": []}
                                documents_data = tool_args.get("documents", [])
                                
                                for doc in documents_data:
                                    # Use the score provided by OpenAI
                                    openai_docs.append({
                                        'title': doc.get('title'),
                                        'link': doc.get('link'),
                                        'score': doc.get('score', 1.0),  # Default to 1.0 if score not provided
                                        'source': 'OpenAI GPT-4'
                                    })
                                
                                openai_links = [doc.get("link") for doc in documents_data if doc.get("link")]
                        else:
                            st.warning("OpenAI did not return documentation in the expected format.")

                    except Exception as e:
                        st.error(f"Error consultando OpenAI: {e}")
                        openai_docs = []
                        openai_links = []
                    """
                else:
                    openai_docs = []
                    openai_links = []
            
            # Show before/after reranking comparison
            st.markdown("### 🔄 Comparación: Antes vs Después del Reranking")
            
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.subheader("📥 ANTES del Reranking")
                st.markdown(f"*Top {len(pre_reranking_docs)} documentos (búsqueda inicial)*")
                
                # Show pre-reranking results
                for i, doc in enumerate(pre_reranking_docs[:10], 1):
                    score = doc.get('score', doc.get('original_score', 0))
                    score_color = "#28a745" if score > 0.8 else "#ffc107" if score > 0.6 else "#dc3545"
                    
                    st.markdown(f"""
                    <div class="doc-card" style="border-left: 4px solid {score_color};">
                        <p><strong>#{i}</strong> {doc.get('title', 'Sin título')[:50]}...</p>
                        <p><strong>📊 Score:</strong> <span style="color: {score_color}; font-weight: bold;">{score:.4f}</span></p>
                        <p style="font-size: 0.8em; color: #666;">🔗 {doc.get('link', 'N/A')[:60]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Stats for pre-reranking
                with st.expander("📊 Estadísticas Pre-Reranking"):
                    if pre_reranking_docs:
                        scores = [doc.get('score', doc.get('original_score', 0)) for doc in pre_reranking_docs]
                        st.write(f"- **Score Promedio:** {np.mean(scores):.3f}")
                        st.write(f"- **Score Máximo:** {max(scores):.3f}")
                        st.write(f"- **Score Mínimo:** {min(scores):.3f}")
                        st.write(f"- **Desv. Estándar:** {np.std(scores):.3f}")
            
            with col_after:
                st.subheader("📤 DESPUÉS del Reranking")
                reranking_method = "CrossEncoder" if use_llm_reranker else "Embedding Similarity"
                st.markdown(f"*Top {len(results)} documentos ({reranking_method})*")
                
                # Show post-reranking results  
                for i, doc in enumerate(results[:10], 1):
                    score = doc.get('score', 0)
                    pre_score = doc.get('pre_rerank_score', None)
                    score_color = "#28a745" if score > 0.8 else "#ffc107" if score > 0.6 else "#dc3545"
                    
                    # Check position change
                    original_pos = None
                    for j, pre_doc in enumerate(pre_reranking_docs):
                        if pre_doc.get('link') == doc.get('link'):
                            original_pos = j + 1
                            break
                    
                    position_change = ""
                    if original_pos:
                        if original_pos > i:
                            position_change = f"<span style='color: green;'>↑ {original_pos - i}</span>"
                        elif original_pos < i:
                            position_change = f"<span style='color: red;'>↓ {i - original_pos}</span>"
                        else:
                            position_change = "<span style='color: gray;'>→</span>"
                    else:
                        position_change = "<span style='color: blue;'>NEW</span>"
                    
                    st.markdown(f"""
                    <div class="doc-card" style="border-left: 4px solid {score_color};">
                        <p><strong>#{i}</strong> {doc.get('title', 'Sin título')[:50]}... {position_change}</p>
                        <p><strong>📊 Score:</strong> <span style="color: {score_color}; font-weight: bold;">{score:.4f}</span></p>
                        <p style="font-size: 0.8em; color: #666;">🔗 {doc.get('link', 'N/A')[:60]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Stats for post-reranking
                with st.expander("📊 Estadísticas Post-Reranking"):
                    if results:
                        scores = [doc.get('score', 0) for doc in results]
                        st.write(f"- **Score Promedio:** {np.mean(scores):.3f}")
                        st.write(f"- **Score Máximo:** {max(scores):.3f}")
                        st.write(f"- **Score Mínimo:** {min(scores):.3f}")
                        st.write(f"- **Desv. Estándar:** {np.std(scores):.3f}")
                        st.write(f"- **Método:** {reranking_method}")
            
            # Add comparison metrics
            st.markdown("---")
            st.markdown("### 📊 Análisis del Impacto del Reranking")
            
            # Calculate reranking impact metrics
            if pre_reranking_docs and results:
                # Position changes analysis
                position_changes = []
                for i, post_doc in enumerate(results[:10]):
                    for j, pre_doc in enumerate(pre_reranking_docs[:10]):
                        if post_doc.get('link') == pre_doc.get('link'):
                            position_changes.append(j - i)  # Positive = moved up
                            break
                
                if position_changes:
                    avg_position_change = np.mean(np.abs(position_changes))
                    docs_moved_up = sum(1 for change in position_changes if change > 0)
                    docs_moved_down = sum(1 for change in position_changes if change < 0)
                    docs_unchanged = sum(1 for change in position_changes if change == 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📈 Docs que Subieron", docs_moved_up,
                                help="Documentos que mejoraron su posición")
                    with col2:
                        st.metric("📉 Docs que Bajaron", docs_moved_down,
                                help="Documentos que bajaron de posición")
                    with col3:
                        st.metric("➡️ Sin Cambios", docs_unchanged,
                                help="Documentos que mantuvieron su posición")
                    with col4:
                        st.metric("🔄 Cambio Promedio", f"{avg_position_change:.1f} pos",
                                help="Cambio promedio de posición (absoluto)")
            
            # Add retrieval query info
            with st.expander("📝 Pregunta utilizada para el retrieval"):
                st.code(full_query, language="text")
                st.info("Esta es la pregunta procesada que se utilizó para buscar documentos relevantes.")
            
            # Keep OpenAI comparison if enabled
            if enable_openai_comparison:
                st.markdown("---")
                st.markdown("### 🤖 Comparación con OpenAI GPT-4")
                col_our_final, col_openai = st.columns(2)
                
                with col_our_final:
                    st.subheader("🔍 Nuestro Sistema (Final)")
                    st.markdown(f"*Top 5 después de reranking*")
                    
                    for i, doc in enumerate(results[:5], 1):
                        score = doc.get('score', 0)
                        score_color = "#28a745" if score > 0.8 else "#ffc107" if score > 0.6 else "#dc3545"
                        
                        st.markdown(f"""
                        <div class="doc-card" style="border-left: 4px solid {score_color};">
                            <p>#{i} {doc.get('title', 'Sin título')}</p>
                            <p><strong>📊 Score:</strong> <span style="color: {score_color}; font-weight: bold;">{score:.4f}</span></p>
                            <p><strong>🔗 Link:</strong> <a href="{doc.get('link', '#')}" target="_blank" style="color: #0078d4;">{doc.get('link', 'N/A')}</a></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_openai:
                    if enable_openai_comparison and openai_docs:
                        st.subheader("🤖 OpenAI GPT-4 Expert")
                        st.markdown(f"*{len(openai_docs)} documentos recomendados*")
                    
                        # Show OpenAI results with same styling
                        for i, doc in enumerate(openai_docs[:10], 1):
                            score = doc.get('score', 1.0)
                            score_color = "#0078d4"
                            
                            st.markdown(f"""
                            <div class="doc-card" style="border-left: 4px solid {score_color};">
                                <h4>#{i} {doc.get('title', 'Sin título')}</h4>
                                <p><strong>📊 Score:</strong> <span style="color: {score_color}; font-weight: bold;">{score:.4f}</span></p>
                                <p><strong>🔗 Link:</strong> <a href="{doc.get('link', '#')}" target="_blank" style="color: #0078d4;">{doc.get('link', 'N/A')}</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add accordion with retrieval query for OpenAI system
                        with st.expander("📝 Prompt completo utilizado para OpenAI"):
                            st.markdown("**System Prompt:**")
                            st.code("Actúa como un experto en documentación de Azure. Tu tarea es buscar y listar exactamente 10 documentos desde el dominio oficial https://learn.microsoft.com que puedan ayudar a responder una pregunta técnica. Para cada documento, entrega un score (entre 0 y 1), donde 1 representa máxima relevancia respecto a la pregunta, el título del documento y el link completo. Los documentos deben estar ordenados de mayor a menor según el score. No inventes contenido ni enlaces. Solo acepta resultados del dominio learn.microsoft.com.", language="text")
                            st.markdown("**User Prompt:**")
                            st.code(f"Aquí está la pregunta a responder: {full_query}", language="text")
                            st.info("Este es el prompt completo enviado a OpenAI GPT-4 para obtener recomendaciones de documentación con scores de relevancia.")
                    else:
                        st.subheader("🤖 OpenAI GPT-4 Expert")
                        if enable_openai_comparison:
                            st.warning("No se pudieron obtener resultados de OpenAI")
                        else:
                            st.info("💡 Habilita la comparación con OpenAI en la configuración para ver resultados paralelos")
            
            # Comparison metrics at the bottom
            if enable_openai_comparison and openai_links:
                st.markdown("---")
                st.subheader("📊 Análisis Comparativo")
                
                our_links = [doc["link"] for doc in results if "learn.microsoft.com" in doc.get("link", "")]
                matches = len(set(openai_links) & set(our_links))
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("🔍 Nuestros resultados", len(our_links))
                with metric_col2:
                    st.metric("🤖 OpenAI resultados", len(openai_links))
                with metric_col3:
                    st.metric("✅ Coincidencias", matches)
                with metric_col4:
                    st.metric("📈 % Coincidencia", f"{(matches/len(openai_links)*100) if openai_links else 0:.1f}%")
            
            # Additional tabs for analysis and debug
            st.markdown("---")
            tab1, tab2 = st.tabs(["📊 Análisis Detallado", "🔧 Debug Info"])
            
            with tab1:
                if results:
                    # Performance analysis chart
                    df_results = pd.DataFrame([
                        {
                            'Documento': f"Doc {i+1}",
                            'Score': doc.get('score', 0),
                            'Título': doc.get('title', 'Sin título')[:30] + "..."
                        }
                        for i, doc in enumerate(results[:10])
                    ])
                    
                    fig = px.bar(
                        df_results, 
                        x='Documento', 
                        y='Score',
                        hover_data=['Título'],
                        title="📈 Scores de Relevancia - Nuestro Sistema",
                        color='Score',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Always show quality metrics (work with or without OpenAI)
                    st.markdown("---")
                    st.subheader("📊 Métricas de Calidad del Sistema")
                    
                    # Calculate intrinsic quality metrics
                    scores = [doc.get('score', 0) for doc in results if doc.get('score', 0) > 0]
                    
                    if scores:
                        import numpy as np
                        
                        # Quality metrics
                        top5_scores = scores[:5] if len(scores) >= 5 else scores
                        top3_scores = scores[:3] if len(scores) >= 3 else scores
                        
                        score_avg_top5 = np.mean(top5_scores)
                        score_min_top3 = min(top3_scores) if top3_scores else 0
                        score_max = max(scores)
                        score_std = np.std(scores)
                        
                        # Coverage metrics
                        high_quality_docs = len([s for s in scores if s >= 0.8])
                        medium_quality_docs = len([s for s in scores if 0.6 <= s < 0.8])
                        coverage_quality = high_quality_docs / len(scores) * 100
                        
                        # Diversity metrics (simple entropy-based)
                        score_entropy = -np.sum([(s/sum(scores)) * np.log2(s/sum(scores)) for s in scores if s > 0])
                        
                        # Display primary quality metrics
                        st.markdown("**🎯 Métricas Principales de Calidad**")
                        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
                        
                        with q_col1:
                            st.metric("📈 Score Promedio Top-5", f"{score_avg_top5:.3f}", 
                                    help="Fórmula: Σ(scores_top5) / 5\n\nMide la calidad promedio de los 5 mejores documentos encontrados. Un valor alto (>0.8) indica que los primeros resultados son muy relevantes. Valores bajos (<0.6) sugieren que incluso los mejores documentos tienen poca relevancia.")
                        with q_col2:
                            st.metric("🎯 Score Mínimo Top-3", f"{score_min_top3:.3f}", 
                                    help="Fórmula: min(scores_top3)\n\nRepresenta el peor documento entre los 3 primeros resultados. Es un umbral de calidad mínima: si este valor es alto (>0.7), garantiza que incluso el tercer mejor documento es muy relevante. Si es bajo (<0.5), indica inconsistencia en la calidad.")
                        with q_col3:
                            st.metric("⭐ Score Máximo", f"{score_max:.3f}", 
                                    help="Fórmula: max(scores)\n\nScore del documento más relevante encontrado. Indica el mejor match posible en la base de datos. Valores cercanos a 1.0 significan una coincidencia casi perfecta. Valores bajos (<0.8) sugieren que no hay documentos altamente relevantes.")
                        with q_col4:
                            st.metric("📊 Cobertura Alta Calidad", f"{coverage_quality:.1f}%", 
                                    help="Fórmula: (docs_score≥0.8 / total_docs) × 100\n\nPorcentaje de documentos con alta relevancia (score ≥ 0.8). Alta cobertura (>60%) indica que la mayoría de resultados son útiles. Baja cobertura (<30%) sugiere que solo pocos documentos son realmente relevantes.")
                        
                        # User experience metrics
                        st.markdown("**🚀 Métricas de Experiencia del Usuario**")
                        ux_col1, ux_col2, ux_col3, ux_col4 = st.columns(4)
                        
                        with ux_col1:
                            st.metric("⚡ Tiempo de Respuesta", f"{response_time:.2f}s", 
                                    help="Fórmula: tiempo_fin - tiempo_inicio\n\nTiempo total desde que se envía la consulta hasta que se obtienen los resultados. Incluye búsqueda vectorial, re-ranking y generación de respuesta. Tiempos <2s son excelentes, 2-5s son buenos, >5s pueden afectar la experiencia del usuario.")
                        with ux_col2:
                            st.metric("📚 Documentos Encontrados", f"{len(results)}", 
                                    help="Fórmula: count(retrieved_docs)\n\nNúmero total de documentos recuperados por el sistema. Más documentos ofrecen mayor cobertura pero pueden incluir resultados menos relevantes. El valor óptimo depende del parámetro top_k configurado.")
                        with ux_col3:
                            st.metric("🎲 Diversidad (Entropía)", f"{score_entropy:.2f}", 
                                    help="Fórmula: -Σ(pi × log2(pi)) donde pi = score_i/Σ(scores)\n\nMide la variedad en los scores de relevancia. Alta entropía (>3) indica scores diversos, sugiriendo documentos con diferentes niveles de relevancia. Baja entropía (<1) indica scores similares, ya sea todos altos o todos bajos.")
                        with ux_col4:
                            variability = "Alta" if score_std > 0.1 else "Media" if score_std > 0.05 else "Baja"
                            st.metric("📈 Variabilidad", f"{variability}", 
                                    help=f"Fórmula: √(Σ(score_i - media)² / n)\n\nDesviación estándar: {score_std:.3f}\n\nMide la dispersión de los scores. Alta variabilidad (>0.1) indica mezcla de documentos muy relevantes y poco relevantes. Baja variabilidad (<0.05) indica consistencia en la calidad de resultados.")
                        
                        # Content quality breakdown
                        st.markdown("**📄 Análisis de Contenido**")
                        content_col1, content_col2 = st.columns(2)
                        
                        with content_col1:
                            st.metric("🔥 Docs Alta Calidad", f"{high_quality_docs}", 
                                    help="Fórmula: count(docs where score ≥ 0.8)\n\nCantidad absoluta de documentos con alta relevancia. Complementa el porcentaje de cobertura mostrando el número exacto. Valores altos (>5) indican buena cantidad de contenido relevante disponible.")
                        with content_col2:
                            st.metric("⚡ Docs Calidad Media", f"{medium_quality_docs}", 
                                    help="Fórmula: count(docs where 0.6 ≤ score < 0.8)\n\nCantidad de documentos con relevancia moderada. Estos documentos pueden ser útiles como información complementaria. Un balance entre docs de alta y media calidad indica un sistema robusto.")
                    
                    # OpenAI comparison section (only if enabled)
                    if enable_openai_comparison and openai_links:
                        st.markdown("---")
                        st.subheader("🤖 Comparación con OpenAI")
                        
                        # Simple overlap metrics (more reliable than complex ranking metrics)
                        our_links = [doc.get("link", "") for doc in results]
                        
                        # Normalize URLs for better comparison
                        def normalize_url(url):
                            return url.lower().replace('/en-us/', '/').replace('///', '//').rstrip('/')
                        
                        our_links_normalized = [normalize_url(link) for link in our_links if link]
                        openai_links_normalized = [normalize_url(link) for link in openai_links if link]
                        
                        # Calculate overlap
                        common_links = set(our_links_normalized) & set(openai_links_normalized)
                        overlap_percentage = len(common_links) / len(openai_links_normalized) * 100 if openai_links_normalized else 0
                        
                        # Jaccard similarity of titles
                        def get_keywords(text):
                            return set(text.lower().split()) if text else set()
                        
                        our_titles = " ".join([doc.get('title', '') for doc in results[:5]])
                        openai_titles = " ".join([doc.get('title', '') for doc in openai_docs[:5]])
                        
                        our_keywords = get_keywords(our_titles)
                        openai_keywords = get_keywords(openai_titles)
                        
                        jaccard_similarity = len(our_keywords & openai_keywords) / len(our_keywords | openai_keywords) * 100 if (our_keywords | openai_keywords) else 0
                        
                        # Display comparison metrics
                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                        
                        with comp_col1:
                            st.metric("🔍 Nuestros Resultados", len(our_links))
                        with comp_col2:
                            st.metric("🤖 OpenAI Resultados", len(openai_links))
                        with comp_col3:
                            st.metric("✅ Overlap de Enlaces", f"{overlap_percentage:.1f}%", 
                                    help="Fórmula: (|nuestros_links ∩ openai_links| / |openai_links|) × 100\n\nPorcentaje de enlaces que ambos sistemas encontraron en común. URLs normalizadas para evitar diferencias de formato. Alto overlap (>70%) indica concordancia, bajo overlap (<30%) sugiere enfoques diferentes.")
                        with comp_col4:
                            st.metric("📝 Similitud de Títulos", f"{jaccard_similarity:.1f}%", 
                                    help="Fórmula: (|palabras_comunes| / |palabras_totales|) × 100\n\nSimilitud Jaccard entre las palabras de los títulos top-5 de ambos sistemas. Mide si ambos sistemas encuentran documentos sobre temas similares, independientemente de los URLs exactos. Alta similitud (>50%) indica coherencia temática.")
                    else:
                        st.info("No hay scores disponibles para calcular métricas de calidad.")
            
            with tab2:
                if show_debug_info:
                    st.subheader("🔧 Información de Debug")
                    st.text(debug_info)
                else:
                    st.info("ℹ️ Debug info deshabilitado en configuración")

            # Removed tab3 (Database Inspection) as requested

                # Display results if any are stored in session state
                if st.session_state.keyword_search_results:
                    st.success(f"Encontrados {len(st.session_state.keyword_search_results)} documentos para '{st.session_state.keyword_search_query}':")
                    for i, doc in enumerate(st.session_state.keyword_search_results):
                        content_preview = doc.get('content', '')[:500]
                        st.markdown(f"**{i+1}. {doc.get('title', 'Sin título')}**")
                        st.markdown(f"🔗 [{doc.get('link', '#')}]({doc.get('link', '#')})")
                        st.markdown(f"""
```
{content_preview}
```
""")
                        st.markdown("--- ")
                elif st.session_state.keyword_search_query and not st.session_state.keyword_search_results:
                    st.info(f"No se encontraron documentos para '{st.session_state.keyword_search_query}'.")
                else:
                    st.warning("⚠️ No se encontraron documentos relevantes. Intenta reformular tu pregunta.")

    # Actualizar métricas en sidebar
    with metrics_container:
        st.metric("Consultas realizadas", st.session_state.session_metrics['queries_made'])
        st.metric("Tiempo promedio", f"{st.session_state.session_metrics['avg_response_time']:.2f}s")
        st.metric("Docs recuperados", st.session_state.session_metrics['total_docs_retrieved'])

# elif page == "📊 Consultas en Lote":
#     show_batch_queries_page()  # Module doesn't exist

elif page == "🔬 Comparación de Modelos":
    show_comparison_page()

elif page == "🔄 Comparador Pregunta vs Respuesta":
    show_question_answer_comparison_page()

elif page == "📊 Análisis Acumulativo N Preguntas":
    show_cumulative_comparison_page()

elif page == "📊 Visualizaciones Capítulo 4":
    show_chapter_4_visualizations()

elif page == "📊 Figuras Capítulo 7":
    show_chapter_7_figures()

elif page == "📈 Análisis de Datos":
    show_data_analysis_page()

elif page == "⚙️ Configuración Métricas Acumulativas":
    show_cumulative_metrics_create_page()
elif page == "📈 Resultados Métricas Acumulativas":
    show_cumulative_metrics_results_page()
elif page == "📊 Resultados Matplotlib":
    show_cumulative_metrics_results_matplotlib_page()

# elif page == "🔧 Configuración Análisis N Preguntas (Colab)":
#     show_cumulative_n_questions_config_page()  # Module doesn't exist
# elif page == "📊 Resultados Análisis N Preguntas (Colab)":
#     show_cumulative_n_questions_results_page()  # Module doesn't exist

# Footer común
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💡 <strong>Tip:</strong> Para mejores resultados, sé específico en tus preguntas e incluye el servicio de Azure de interés.</p>
    <p>🔧 Sistema desarrollado con ChromaDB + sentence-transformers</p>
</div>
""", unsafe_allow_html=True)
