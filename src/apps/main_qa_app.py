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
from src.core.qa_pipeline import answer_question_documents_only, answer_question_with_rag
from src.services.auth.clients import initialize_clients
from src.services.local_models import preload_tinyllama_model
from src.apps.comparison_page import show_comparison_page
from src.apps.batch_queries_page import show_batch_queries_page
from src.apps.data_analysis_page import show_data_analysis_page
from src.apps.cumulative_metrics_create import show_cumulative_metrics_create_page
from src.apps.cumulative_metrics_results import show_cumulative_metrics_results_page
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
        "📊 Consultas en Lote",
        "🔬 Comparación de Modelos",
        "📈 Análisis de Datos",
        "⚙️ Configuración Métricas Acumulativas",
        "📈 Resultados Métricas Acumulativas",
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
elif generative_model_name == "gemini-pro":
    st.sidebar.warning("💰 **Modelo de API** - Incurre en costos por uso")
elif generative_model_name == "gpt-4":
    st.sidebar.warning("💰 **Modelo de API** - Incurre en costos altos por uso")

# Páginas principales
if page == "🔍 Búsqueda Individual":
    
    # Parámetros de búsqueda
    search_params = st.sidebar.expander("🔍 Parámetros de Búsqueda", expanded=True)
    with search_params:
        top_k = st.slider("Documentos a retornar", 5, 20, 10)
        use_llm_reranker = st.checkbox("Usar Re-Ranking con LLM", value=False, help="Usa GPT-4 para un re-ranking más preciso pero más lento y costoso.")
        use_questions_collection = st.checkbox("Usar Colección de Preguntas", value=True, help="Habilita la búsqueda en la colección QuestionsMiniLM para encontrar preguntas similares y extraer enlaces relevantes.")
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
            
        evaluate_rag_quality = st.checkbox("Evaluar Calidad RAG", value=False,
                                         help="Calcula métricas de faithfulness, relevancy y utilización del contexto")
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
    
    # Update inputs if example was selected
    if 'selected_title' in st.session_state:
        title_value = st.session_state.selected_title
        del st.session_state.selected_title
    else:
        title_value = st.session_state.last_title
    
    if 'selected_question' in st.session_state:
        selected_question = st.session_state.selected_question
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

        if not question.strip():
            st.warning("⚠️ Por favor ingresa una pregunta.")
        else:
            # Combine title and question for better search
            full_query = f"{title.strip()} {question.strip()}".strip()
            
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
                        use_questions_collection=use_questions_collection,
                        evaluate_quality=evaluate_rag_quality,
                        documents_class=CHROMADB_COLLECTION_CONFIG[model_name]["documents"],
                        questions_class=CHROMADB_COLLECTION_CONFIG[model_name]["questions"],
                        generative_model_name=generative_model_name
                    )
                else:
                    results, debug_info = answer_question_documents_only(
                        full_query,
                        chromadb_wrapper,
                        embedding_client,
                        openai_client,
                        top_k=top_k,
                        diversity_threshold=diversity_threshold,
                        use_llm_reranker=use_llm_reranker,
                        use_questions_collection=use_questions_collection,
                        documents_class=CHROMADB_COLLECTION_CONFIG[model_name]["documents"],
                        questions_class=CHROMADB_COLLECTION_CONFIG[model_name]["questions"]
                    )
                    
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
                                
                                rag_metrics = {
                                    'confidence': 0.85,  # OpenRouter models are generally more reliable
                                    'completeness': 'complete' if len(selected_docs) >= 3 else 'partial',
                                    'docs_used': len(selected_docs),
                                    'high_score_docs': len(high_score_docs),
                                    'min_score': min([doc.get('score', 0) for doc in selected_docs]),
                                    'max_score': max([doc.get('score', 0) for doc in selected_docs]),
                                    'model_provider': 'OpenRouter'
                                }
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
                                
                                rag_metrics = {
                                    'confidence': generation_info.get('confidence', 0.8),
                                    'completeness': 'complete' if len(selected_docs) >= 3 else 'partial',
                                    'docs_used': len(selected_docs),
                                    'high_score_docs': len(high_score_docs),
                                    'min_score': min([doc.get('score', 0) for doc in selected_docs]),
                                    'max_score': max([doc.get('score', 0) for doc in selected_docs]),
                                    'model_provider': 'Local'
                                }
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
            st.session_state.session_metrics['total_docs_retrieved'] += len(results)
            st.session_state.session_metrics['avg_response_time'] = (
                (st.session_state.session_metrics['avg_response_time'] * 
                 (st.session_state.session_metrics['queries_made'] - 1) + response_time) /
                st.session_state.session_metrics['queries_made']
            )

        # Mostrar resultados
        if results:
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
                
                # Mostrar métricas RAG si están habilitadas
                if show_rag_metrics and rag_metrics:
                    rag_col1, rag_col2, rag_col3, rag_col4 = st.columns(4)
                    
                    with rag_col1:
                        confidence = rag_metrics.get('confidence', 0)
                        st.metric("🎯 Confianza", f"{confidence:.2f}", 
                                help="Confianza del modelo en la respuesta generada")
                    
                    with rag_col2:
                        completeness = rag_metrics.get('completeness', 'unknown')
                        completeness_emoji = {"complete": "✅", "partial": "⚠️", "limited": "❌"}.get(completeness, "❓")
                        st.metric("📊 Completitud", f"{completeness_emoji} {completeness.title()}", 
                                help="Si la documentación permite una respuesta completa")
                    
                    with rag_col3:
                        docs_used = rag_metrics.get('docs_used', 0)
                        st.metric("📚 Docs Usados", f"{docs_used}/{len(results)}", 
                                help="Documentos utilizados para generar la respuesta")
                    
                    with rag_col4:
                        if evaluate_rag_quality and 'faithfulness' in rag_metrics:
                            faithfulness = rag_metrics.get('faithfulness', 0)
                            st.metric("🔍 Fidelidad", f"{faithfulness:.2f}", 
                                    help="Fidelidad de la respuesta a los documentos fuente")
                        elif not enable_rag and 'generation_time' in rag_metrics:
                            gen_time = rag_metrics.get('generation_time', 0)
                            st.metric("⏱️ Tiempo Gen", f"{gen_time:.1f}s", 
                                    help="Tiempo de generación de la respuesta")
                        elif not enable_rag and 'max_score' in rag_metrics:
                            max_score = rag_metrics.get('max_score', 0)
                            st.metric("🎯 Score Máx", f"{max_score:.3f}", 
                                    help="Score máximo de los documentos utilizados")
                        else:
                            st.metric("⚡ Estado", "✅ Generada", help="Respuesta generada exitosamente")
                
                # Mostrar la respuesta generada
                st.markdown("#### 💬 Respuesta:")
                st.markdown(generated_answer)
                
                # Mostrar métricas adicionales de evaluación si están disponibles
                if evaluate_rag_quality and rag_metrics.get('answer_relevancy'):
                    with st.expander("📊 Métricas Detalladas de Calidad RAG"):
                        eval_col1, eval_col2, eval_col3 = st.columns(3)
                        
                        with eval_col1:
                            st.metric("🎯 Faithfulness", f"{rag_metrics.get('faithfulness', 0):.3f}",
                                    help="¿La respuesta es fiel a los documentos?")
                        
                        with eval_col2:
                            st.metric("🔍 Answer Relevancy", f"{rag_metrics.get('answer_relevancy', 0):.3f}",
                                    help="¿La respuesta responde la pregunta?")
                        
                        with eval_col3:
                            st.metric("📚 Context Utilization", f"{rag_metrics.get('context_utilization', 0):.3f}",
                                    help="¿Se utilizó bien el contexto?")
                
                st.markdown("---")
                st.markdown("### 📄 **Documentos de Referencia**")
            
            # Create side-by-side comparison
            if enable_openai_comparison:
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
            else:
                openai_docs = []
                openai_links = []
            
            # Side by side comparison
            col_our, col_openai = st.columns(2)
            
            with col_our:
                st.subheader("🔍 Nuestro Sistema de Búsqueda")
                st.markdown(f"*{len(results)} documentos encontrados*")
                
                # Show our results with same styling as OpenAI
                for i, doc in enumerate(results[:10], 1):
                    score = doc.get('score', 0)
                    score_color = "#28a745" if score > 0.8 else "#ffc107" if score > 0.6 else "#dc3545"
                    
                    st.markdown(f"""
                    <div class="doc-card" style="border-left: 4px solid {score_color};">
                        <p>#{i} {doc.get('title', 'Sin título')}</p>
                        <p><strong>📊 Score:</strong> <span style="color: {score_color}; font-weight: bold;">{score:.4f}</span></p>
                        <p><strong>🔗 Link:</strong> <a href="{doc.get('link', '#')}" target="_blank" style="color: #0078d4;">{doc.get('link', 'N/A')}</a></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add accordion with retrieval query for our system
                with st.expander("📝 Pregunta utilizada para el retrieval - Nuestro Sistema"):
                    st.code(full_query, language="text")
                    st.info("Esta es la pregunta procesada que se utilizó para buscar documentos relevantes en nuestro sistema.")
                    
                    # Content preview
                    #with st.expander(f"Ver contenido #{i}"):
                    #    content_preview = doc.get('content', '')[:500]
                    #    st.text(content_preview + "..." if len(doc.get('content', '')) > 500 else content_preview)
            
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

elif page == "📊 Consultas en Lote":
    show_batch_queries_page()

elif page == "🔬 Comparación de Modelos":
    show_comparison_page()

elif page == "📈 Análisis de Datos":
    show_data_analysis_page()

elif page == "⚙️ Configuración Métricas Acumulativas":
    show_cumulative_metrics_create_page()
elif page == "📈 Resultados Métricas Acumulativas":
    show_cumulative_metrics_results_page()

# Footer común
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💡 <strong>Tip:</strong> Para mejores resultados, sé específico en tus preguntas e incluye el servicio de Azure de interés.</p>
    <p>🔧 Sistema desarrollado con ChromaDB + sentence-transformers</p>
</div>
""", unsafe_allow_html=True)
