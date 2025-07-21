import streamlit as st
from src.services.storage.weaviate_utils import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper
from src.data.embedding import EmbeddingClient
import os
import atexit
from src.config.config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, WEAVIATE_CLASS_CONFIG

st.set_page_config(page_title="Weaviate Keyword Search Debugger", layout="wide")

st.title("🔍 Weaviate Keyword Search Debugger")
st.markdown("---")

# Configurar credenciales (con cache para evitar reconexiones)
@st.cache_resource
def initialize_clients(model_name: str):
    config = WeaviateConfig.from_env()
    client = get_weaviate_client(config)
    chromadb_wrapper = WeaviateClientWrapper(client, retry_attempts=3)
    embedding_client = EmbeddingClient(huggingface_api_key=config.huggingface_api_key) # Pass HF API key
    return chromadb_wrapper, client, embedding_client

# Selección de modelo de embedding
model_name = st.selectbox(
    "Selecciona el modelo de embedding:",
    options=list(EMBEDDING_MODELS.keys()),
    index=list(EMBEDDING_MODELS.keys()).index(DEFAULT_EMBEDDING_MODEL)
)

chromadb_wrapper, client, embedding_client = initialize_clients(model_name)
atexit.register(lambda: client and client.close())

st.subheader("Búsqueda por Palabra Clave (BM25)")
keyword_query = st.text_input("Introduce una palabra clave para buscar en la documentación:", key="debug_keyword_search_input")

if st.button("Buscar por Palabra Clave", key="debug_bm25_search_button"):
    if keyword_query:
        with st.spinner(f"Buscando '{keyword_query}' en Weaviate (BM25)..."):
            try:
                documents_class = WEAVIATE_CLASS_CONFIG[model_name]["documents"]
                keyword_results = chromadb_wrapper.search_docs_by_keyword(keyword_query, limit=20, class_name=documents_class)
                
                if keyword_results:
                    st.success(f"Encontrados {len(keyword_results)} documentos para '{keyword_query}':")
                    for i, doc in enumerate(keyword_results):
                        st.markdown(f"**{i+1}. {doc.get('title', 'Sin título')}**")
                        st.markdown(f"🔗 [{doc.get('link', '#')}]({doc.get('link', '#')})")
                        st.markdown(f"""
```
{doc.get('content', '')[:500]}
```
""")
                        st.markdown("---")
                else:
                    st.info(f"No se encontraron documentos para '{keyword_query}'.")
            except Exception as e:
                st.error(f"Error durante la búsqueda BM25: {e}")
    else:
        st.warning("Por favor, introduce una palabra clave.")

st.markdown("---")
st.subheader("Estadísticas de la Colección")
try:
    stats = chromadb_wrapper.get_collection_stats()
    for model, classes in WEAVIATE_CLASS_CONFIG.items():
        questions_class_name = classes['questions']
        documents_class_name = classes['documents']
        st.write(f"**Documentos en '{questions_class_name}':** {stats.get(f'{questions_class_name}_count', 'N/A')}")
        st.write(f"**Documentos en '{documents_class_name}':** {stats.get(f'{documents_class_name}_count', 'N/A')}")
except Exception as e:
    st.error(f"Error al obtener estadísticas de la colección: {e}")

