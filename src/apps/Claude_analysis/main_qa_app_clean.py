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
from src.apps.data_analysis_page import show_data_analysis_page
from src.apps.cumulative_metrics_create import show_cumulative_metrics_create_page
from src.apps.cumulative_metrics_results_matplotlib import show_cumulative_metrics_results_page as show_cumulative_metrics_results_matplotlib_page
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

# Session state para la barra de progreso
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Initialize clients
openai_client, openai_embedding_client, _ = initialize_clients()

# Precargar modelos locales en la primera ejecución
if 'models_loaded' not in st.session_state:
    with st.spinner("🎯 Cargando componentes del sistema..."):
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        # Simulación de carga de componentes con progreso más detallado
        progress_bar.progress(25, text="🔧 Inicializando ChromaDB...")
        time.sleep(0.5)
        
        progress_bar.progress(50, text="🧠 Cargando modelos de embedding...")
        time.sleep(0.5)
        
        progress_bar.progress(75, text="🚀 Preparando modelo generativo TinyLlama...")
        preload_tinyllama_model()  # Asegurar que TinyLlama esté precargado
        
        progress_bar.progress(100, text="✅ Sistema listo!")
        time.sleep(0.5)
        
        progress_placeholder.empty()
        st.session_state.models_loaded = True

# Header con estilo mejorado
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0078d4 0%, #40e0d0 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
<div class="main-header">
    <h1>☁️ Azure Q&A Expert System</h1>
    <p>Sistema de Recuperación Aumentada con Generación (RAG) para documentación de Azure</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para navegación y configuración
st.sidebar.title("🧭 Navegación")
page = st.sidebar.radio(
    "Selecciona una página:",
    [
        "🔍 Búsqueda Individual",
        "📈 Análisis de Datos",
        "⚙️ Configuración Métricas Acumulativas",
        "📊 Resultados Métricas Acumulativas",
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
    # El contenido de búsqueda individual va aquí
    # (Copiar todo el contenido de la página de búsqueda individual del archivo original)
    pass  # Por ahora dejamos vacío
    
elif page == "📈 Análisis de Datos":
    show_data_analysis_page()
    
elif page == "⚙️ Configuración Métricas Acumulativas":
    show_cumulative_metrics_create_page()
    
elif page == "📊 Resultados Métricas Acumulativas":
    show_cumulative_metrics_results_matplotlib_page()

# Footer común
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💡 <strong>Tip:</strong> Para mejores resultados, sé específico en tus preguntas e incluye el servicio de Azure de interés.</p>
    <p>🔧 Sistema desarrollado con ChromaDB + sentence-transformers</p>
</div>
""", unsafe_allow_html=True)