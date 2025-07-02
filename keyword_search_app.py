import streamlit as st
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper
from utils.embedding import EmbeddingClient
import os
import atexit

st.set_page_config(page_title="Weaviate Keyword Search Debugger", layout="wide")

st.title("🔍 Weaviate Keyword Search Debugger")
st.markdown("---")

# Configurar credenciales (con cache para evitar reconexiones)
@st.cache_resource
def initialize_clients(_config_hash: str):
    config = WeaviateConfig.from_env()
    client = get_weaviate_client(config)
    weaviate_wrapper = WeaviateClientWrapper(client, retry_attempts=3)
    embedding_client = EmbeddingClient(huggingface_api_key=config.huggingface_api_key) # Pass HF API key
    return weaviate_wrapper, client, embedding_client

# Create a hash from environment variables for caching
import hashlib
env_hash = hashlib.md5(f"{os.getenv('WCS_URL', '')}{os.getenv('WCS_API_KEY', '')}{os.getenv('OPENAI_API_KEY', '')}".encode()).hexdigest()

weaviate_wrapper, client, embedding_client = initialize_clients(env_hash)
atexit.register(lambda: client and client.close())

st.subheader("Búsqueda por Palabra Clave (BM25)")
keyword_query = st.text_input("Introduce una palabra clave para buscar en la documentación:", key="debug_keyword_search_input")

if st.button("Buscar por Palabra Clave", key="debug_bm25_search_button"):
    if keyword_query:
        with st.spinner(f"Buscando '{keyword_query}' en Weaviate (BM25)..."):
            try:
                keyword_results = weaviate_wrapper.search_docs_by_keyword(keyword_query, limit=20)
                
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
    stats = weaviate_wrapper.get_collection_stats()
    st.write(f"**Documentos en 'Questions':** {stats.get('Questions_count', 'N/A')}")
    st.write(f"**Documentos en 'Documentation':** {stats.get('Documentation_count', 'N/A')}")
except Exception as e:
    st.error(f"Error al obtener estadísticas de la colección: {e}")
