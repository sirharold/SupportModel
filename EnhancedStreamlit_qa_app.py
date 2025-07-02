import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.qa_pipeline import answer_question
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper
from utils.embedding import EmbeddingClient
from openai import OpenAI
import os
import atexit
import pandas as pd
import time

# Configuración de página
st.set_page_config(
    page_title="Azure Q&A Expert System", 
    layout="wide",
    page_icon="☁️"
)

# Configurar credenciales (con cache para evitar reconexiones)
@st.cache_resource
def initialize_clients(_config_hash: str):
    config = WeaviateConfig.from_env()
    client = get_weaviate_client(config)
    weaviate_wrapper = WeaviateClientWrapper(client, retry_attempts=3)
    embedding_client = EmbeddingClient(huggingface_api_key=config.huggingface_api_key)
    openai_client = OpenAI(api_key=config.openai_api_key)
    return weaviate_wrapper, embedding_client, openai_client, client

# Create a hash from environment variables for caching
import hashlib
env_hash = hashlib.md5(
    f"{os.getenv('WCS_URL', '')}"
    f"{os.getenv('WCS_API_KEY', '')}"
    f"{os.getenv('OPENAI_API_KEY', '')}"
    f"{os.getenv('HUGGINGFACE_API_KEY', '')}"
    .encode()
).hexdigest()

weaviate_wrapper, embedding_client, openai_client, client = initialize_clients(env_hash)
atexit.register(lambda: client and client.close())

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

# Header principal
st.markdown("""
<div class="main-header">
    <h1>☁️ Azure Q&A Expert System</h1>
    <p>Encuentra documentación oficial de Microsoft Azure para tus preguntas técnicas</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para configuración
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("---")

# Parámetros de búsqueda
search_params = st.sidebar.expander("🔍 Parámetros de Búsqueda", expanded=True)
with search_params:
    top_k = st.slider("Documentos a retornar", 5, 20, 10)
    use_llm_reranker = st.checkbox("Usar Re-Ranking con LLM", value=False, help="Usa GPT-4 para un re-ranking más preciso pero más lento y costoso.")
    use_questions_collection = st.checkbox("Usar Colección de Preguntas", value=False, help="Habilita la búsqueda en la colección de preguntas (puede causar errores si las dimensiones de embedding no coinciden).")
    diversity_threshold = st.slider(
        "Umbral de diversidad", 0.5, 0.95, 0.85, 0.05,
        help="Controla la diversidad de resultados (más alto = más diverso)"
    )

# Métricas de evaluación
eval_params = st.sidebar.expander("📊 Evaluación", expanded=False)
with eval_params:
    enable_openai_comparison = st.checkbox("Comparar con OpenAI", value=False)
    show_debug_info = st.checkbox("Mostrar información de debug", value=True)

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
        
        with st.spinner("🔍 Buscando documentación relevante..."):
            # Ejecutar búsqueda
            results, debug_info = answer_question(
                full_query,
                weaviate_wrapper,
                embedding_client,
                openai_client,
                top_k=top_k,
                diversity_threshold=diversity_threshold,
                use_llm_reranker=use_llm_reranker,
                use_questions_collection=use_questions_collection
            )
            
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
            st.success(f"✅ Encontrados {len(results)} documentos relevantes en {response_time:.2f}s")
            
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
                                                        "title": {"type": "string", "description": "The official title of the documentation page."},
                                                        "link": {"type": "string", "description": "The full URL to the documentation page, must be from learn.microsoft.com."},
                                                    },
                                                    "required": ["title", "link"]
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
                                {"role": "system", "content": "You are an Azure documentation expert. You ONLY recommend official Microsoft Learn documentation from the learn.microsoft.com domain. You must call the `list_azure_documentation` function with exactly 10 results."},
                                {"role": "user", "content": f"Please provide the top 10 Azure documentation links for the following question: {full_query}"}
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
                                tool_args = json.loads(tool_call.function.arguments)
                                documents_data = tool_args.get("documents", [])
                                
                                for doc in documents_data:
                                    openai_docs.append({
                                        'title': doc.get('title'),
                                        'link': doc.get('link'),
                                        'score': 1.0,
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
                        st.markdown(f"""
                        <div class="doc-card" style="border-left: 4px solid #0078d4;">
                            <h4>#{i} {doc.get('title', 'Sin título')}</h4>
                            <p><strong>🤖 Fuente:</strong> <span style="color: #0078d4; font-weight: bold;">{doc.get('source', 'OpenAI')}</span></p>
                            <p><strong>🔗 Link:</strong> <a href="{doc.get('link', '#')}" target="_blank" style="color: #0078d4;">{doc.get('link', 'N/A')}</a></p>
                        </div>
                        """, unsafe_allow_html=True)
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
            tab1, tab2, tab3 = st.tabs(["📊 Análisis Detallado", "🔧 Debug Info", "🗄️ Inspección DB"])
            
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

                    if enable_openai_comparison and openai_links:
                        st.markdown("---")
                        st.subheader("⚖️ Métricas de Ranking (vs. OpenAI)")
                        
                        from utils.metrics import compute_ndcg, compute_mrr, compute_precision_recall_f1
                        
                        # Ensure results are in the correct format for metrics
                        our_docs_for_metrics = [{"link": doc.get("link")} for doc in results]
                        
                        # Calculate metrics
                        precision, recall, f1 = compute_precision_recall_f1(our_docs_for_metrics, openai_links, k=top_k)
                        mrr = compute_mrr(our_docs_for_metrics, openai_links, k=top_k)
                        ndcg = compute_ndcg(our_docs_for_metrics, openai_links, k=top_k)
                        
                        # Display metrics in columns
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric("nDCG@10", f"{ndcg:.3f}", help="Mide la calidad del ranking (más alto es mejor).")
                            st.metric("MRR@10", f"{mrr:.3f}", help="Evalúa qué tan arriba aparece el primer resultado relevante.")
                        with m_col2:
                            st.metric("Precision", f"{precision:.3f}", help="De los documentos que mostramos, cuántos son relevantes.")
                            st.metric("Recall", f"{recall:.3f}", help="De todos los documentos relevantes, cuántos encontramos.")
                        with m_col3:
                            st.metric("F1-Score", f"{f1:.3f}", help="Balance entre Precision y Recall.")
            
            with tab2:
                if show_debug_info:
                    st.subheader("🔧 Información de Debug")
                    st.text(debug_info)
                else:
                    st.info("ℹ️ Debug info deshabilitado en configuración")

            with tab3:
                st.subheader("🗄️ Inspección de Base de Datos Weaviate")
                
                # Display collection stats
                try:
                    stats = weaviate_wrapper.get_collection_stats()
                    st.write(f"**Documentos en 'Questions':** {stats.get('Questions_count', 'N/A')}")
                    st.write(f"**Documentos en 'DocumentsMiniLM':** {stats.get('DocumentsMiniLM_count', 'N/A')}")
                except Exception as e:
                    st.error(f"Error al obtener estadísticas de la colección: {e}")

                st.markdown("--- ")
                st.subheader("🔍 Búsqueda por Palabra Clave (BM25)")
                keyword_query = st.text_input("Introduce una palabra clave para buscar en la documentación:", key="keyword_search_input")
                
                # Initialize keyword search results in session state
                if 'keyword_search_results' not in st.session_state:
                    st.session_state.keyword_search_results = []
                if 'keyword_search_query' not in st.session_state:
                    st.session_state.keyword_search_query = ""

                if st.button("Buscar por Palabra Clave", key="bm25_search_button"):
                    if keyword_query:
                        st.session_state.keyword_search_query = keyword_query # Store the query
                        with st.spinner(f"Buscando '{keyword_query}'..."):
                            st.session_state.keyword_search_results = weaviate_wrapper.search_docs_by_keyword(keyword_query, limit=20)
                    else:
                        st.warning("Por favor, introduce una palabra clave.")
                        st.session_state.keyword_search_results = [] # Clear previous results

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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💡 <strong>Tip:</strong> Para mejores resultados, sé específico en tus preguntas e incluye el servicio de Azure de interés.</p>
    <p>🔧 Sistema desarrollado con Weaviate + OpenAI Embeddings</p>
</div>
""", unsafe_allow_html=True)