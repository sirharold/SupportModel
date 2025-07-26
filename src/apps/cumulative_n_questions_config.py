"""
Página de Configuración para Análisis Acumulativo de N Preguntas
Permite configurar evaluaciones y enviarlas a Google Colab
"""

import streamlit as st
import time
import json
import os
from typing import List, Dict, Any
from src.config.config import EMBEDDING_MODELS, GENERATIVE_MODELS, CHROMADB_COLLECTION_CONFIG

# Importar utilidades
from src.data.memory_utils import get_memory_usage, cleanup_memory
from src.services.storage.real_gdrive_integration import (
    show_gdrive_status, create_evaluation_config_in_drive,
    check_evaluation_status_in_drive, show_gdrive_authentication_instructions, 
    show_gdrive_debug_info
)
from src.apps.cumulative_metrics_page import display_current_colab_status


def load_questions_for_config(num_questions: int):
    """Cargar N preguntas aleatorias desde ChromaDB que tengan links presentes en la colección de documentos."""
    import random
    import re
    from urllib.parse import urlparse, urlunparse
    from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client
    
    try:
        st.info(f"🔍 Buscando {num_questions} preguntas con links válidos en ChromaDB...")
        
        # Conectar a ChromaDB usando la configuración del sistema
        try:
            config_chroma = ChromaDBConfig.from_env()
            client = get_chromadb_client(config_chroma)
            st.success(f"✅ Conectado a ChromaDB: {config_chroma.persist_directory}")
            
        except Exception as e:
            st.error(f"❌ Error conectando a ChromaDB: {e}")
            return []
        
        # Función para normalizar URLs (quita query params y anchors)
        def normalize_link(url):
            try:
                parsed = urlparse(url)
                normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
                return normalized.rstrip('/')
            except:
                return None
        
        # Obtener links de la colección de documentos
        try:
            if 'selected_question_collection' not in st.session_state:
                st.error("❌ No se ha seleccionado una colección de preguntas")
                return []
            
            # Determinar colección de documentos basada en la colección de preguntas
            collection_name = st.session_state.selected_question_collection
            docs_collection_name = collection_name.replace('questions_', 'docs_')
            
            st.info(f"📊 Obteniendo links de colección '{docs_collection_name}'...")
            docs_collection = client.get_collection(name=docs_collection_name)
            docs_metadatas = docs_collection.get(include=["metadatas"], limit=200000)["metadatas"]
            
            doc_links = set()
            for meta in docs_metadatas:
                link = meta.get("link")
                if link:
                    normalized = normalize_link(link)
                    if normalized:
                        doc_links.add(normalized)
            
            st.success(f"✅ Obtenidos {len(doc_links):,} links únicos de documentos")
            
        except Exception as e:
            st.error(f"❌ Error accediendo a colección de documentos '{docs_collection_name}': {e}")
            return []
        
        # Obtener y filtrar preguntas con links válidos
        try:
            questions_collection = client.get_collection(name=collection_name)
            total_questions = questions_collection.count()
            
            st.info(f"📊 Filtrando preguntas de '{collection_name}' ({total_questions:,} total)...")
            
            # Obtener todas las preguntas para filtrar
            all_questions = questions_collection.get(include=["metadatas"], limit=15000)["metadatas"]
            matched_questions = []
            
            progress_bar = st.progress(0)
            for i, meta in enumerate(all_questions):
                if i % 100 == 0:
                    progress_bar.progress(i / len(all_questions))
                
                accepted_answer = meta.get("accepted_answer", "")
                if "http" in accepted_answer:
                    urls = re.findall(r'https?://[^\s\)\]]+', accepted_answer)
                    for url in urls:
                        norm_url = normalize_link(url)
                        if norm_url and norm_url in doc_links:
                            matched_questions.append(meta)
                            break  # basta con una coincidencia por pregunta
            
            progress_bar.progress(1.0)
            st.success(f"✅ Encontradas {len(matched_questions):,} preguntas con links válidos")
            
            if len(matched_questions) < num_questions:
                st.warning(f"⚠️ Solo hay {len(matched_questions)} preguntas con links válidos, ajustando cantidad...")
                num_questions = len(matched_questions)
            
            # Seleccionar aleatoriamente del subconjunto filtrado
            random.seed(42)  # Seed fijo para reproducibilidad
            selected_questions = random.sample(matched_questions, num_questions)
            
        except Exception as e:
            st.error(f"❌ Error filtrando preguntas: {e}")
            return []
        
        # Procesar preguntas al formato esperado
        questions_data = []
        
        for i, metadata in enumerate(selected_questions):
            try:
                # Extraer información de la pregunta desde metadata
                question = {
                    'id': f"filtered_q_{i}",  # ID único para preguntas filtradas
                    'title': metadata.get('title', metadata.get('question', 'Sin título')),
                    'question_content': metadata.get('question_content', metadata.get('question', '')),
                    'accepted_answer': metadata.get('accepted_answer', metadata.get('answer', 'Sin respuesta')),
                    'tags': metadata.get('tags', []),
                    'ms_links': metadata.get('ms_links', metadata.get('links', [])),
                    'metadata': metadata
                }
                
                # Extraer links válidos de la respuesta aceptada (ya sabemos que tiene al menos uno)
                if question['accepted_answer'] and "http" in question['accepted_answer']:
                    urls = re.findall(r'https?://[^\s\)\]]+', question['accepted_answer'])
                    valid_links = []
                    for url in urls:
                        norm_url = normalize_link(url)
                        if norm_url and norm_url in doc_links:
                            valid_links.append(url)
                    question['ms_links'] = list(set(valid_links))  # Eliminar duplicados
                
                # Validar que tenga al menos título
                if question['title'] and question['title'] != 'Sin título':
                    questions_data.append(question)
                else:
                    st.warning(f"⚠️ Pregunta {i} sin título válido - omitida")
                    
            except Exception as e:
                st.warning(f"⚠️ Error procesando pregunta {i}: {e}")
                continue
        
        if not questions_data:
            st.error("❌ No se pudieron procesar preguntas válidas")
            return []
        
        st.success(f"✅ {len(questions_data)} preguntas válidas con links verificados extraídas desde ChromaDB")
        
        # Mostrar muestra de preguntas
        with st.expander(f"📋 Muestra de preguntas seleccionadas (con links válidos)", expanded=False):
            for i, q in enumerate(questions_data[:3]):
                st.write(f"**{i+1}.** {q['title'][:80]}...")
                st.write(f"   - Enlaces válidos: {len(q['ms_links'])}")
                st.write(f"   - Tags: {', '.join(q['tags'][:3]) if q['tags'] else 'Sin tags'}...")
                # Mostrar un link de ejemplo para verificación
                if q['ms_links']:
                    st.write(f"     Ejemplo: {q['ms_links'][0][:60]}...")
            
            if len(questions_data) > 3:
                st.write(f"... y {len(questions_data)-3} preguntas más")
        
        return questions_data
        
    except Exception as e:
        st.error(f"❌ Error general cargando preguntas: {e}")
        st.exception(e)
        return []


def show_cumulative_n_questions_config_page():
    """Página principal para crear configuraciones de análisis acumulativo de N preguntas."""
    
    st.title("⚙️ Configuración Análisis Acumulativo N Preguntas")
    st.markdown("""
    Esta página permite configurar evaluaciones de múltiples preguntas con comparación pregunta vs respuesta 
    y enviarlas a Google Colab para procesamiento con GPU.
    """)
    
    # Verificar si ChromaDB está disponible usando la configuración del sistema
    chromadb_available = False
    chromadb_path = None
    total_questions = 0
    available_question_collections = []
    
    try:
        import chromadb
        from chromadb.config import Settings
        from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client
        
        # Usar la configuración del sistema
        config_chroma = ChromaDBConfig.from_env()
        client = get_chromadb_client(config_chroma)
        chromadb_path = config_chroma.persist_directory
        
        # Buscar colecciones de preguntas disponibles
        try:
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            # Filtrar colecciones de preguntas
            for collection_name in collection_names:
                if collection_name.startswith('questions_'):
                    try:
                        collection = client.get_collection(name=collection_name)
                        count = collection.count()
                        if count > 0:
                            available_question_collections.append({
                                'name': collection_name,
                                'count': count,
                                'model': collection_name.replace('questions_', '')
                            })
                    except:
                        continue
            
            if available_question_collections:
                # Usar la colección con más preguntas o una específica
                best_collection = max(available_question_collections, key=lambda x: x['count'])
                total_questions = best_collection['count']
                chromadb_available = True
                
                # Guardar en session_state para usar en load_questions_for_config
                st.session_state.selected_question_collection = best_collection['name']
                st.session_state.available_question_collections = available_question_collections
                
                st.success(f"✅ ChromaDB disponible: {total_questions:,} preguntas en '{chromadb_path}'")
                st.info(f"📋 Usando colección: '{best_collection['name']}' ({best_collection['model']})")
                
                # Mostrar otras colecciones disponibles
                if len(available_question_collections) > 1:
                    other_collections = [f"{c['name']} ({c['count']:,})" for c in available_question_collections]
                    st.info(f"📋 Colecciones de preguntas disponibles: {', '.join(other_collections)}")
            else:
                st.error("❌ No se encontraron colecciones de preguntas con datos")
                st.info(f"📋 Colecciones disponibles: {collection_names}")
                st.info("💡 Busca colecciones que empiecen con 'questions_' (ej: questions_mpnet, questions_e5large)")
                return
                
        except Exception as e:
            st.error(f"❌ Error listando colecciones: {e}")
            return
            
    except ImportError:
        st.error("❌ ChromaDB no está instalado. Ejecuta: pip install chromadb")
        return
    except Exception as e:
        st.error(f"❌ Error verificando ChromaDB: {e}")
        return
    
    # Mostrar información del flujo
    st.info(f"""
    📋 **Flujo de trabajo:**
    1. **Filtrar preguntas** con links válidos desde ChromaDB ({total_questions:,} total → ~2,067 con links válidos)
    2. **Seleccionar N preguntas** aleatoriamente del subconjunto filtrado
    3. **Configurar evaluación** en esta página
    4. **Enviar a Google Drive** para Colab (incluye las preguntas seleccionadas)
    5. **Ejecutar en Google Colab** con GPU usando embeddings pre-calculados
    6. **Ver resultados** en la página de resultados
    """)
    
    # Configuración inicial
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Configuración de Datos")
        
        # Información sobre la fuente de datos
        st.info(f"📊 Las preguntas se seleccionan aleatoriamente desde ChromaDB ({total_questions:,} total) pero **solo se usan preguntas con links presentes en la colección de documentos** (~2,067 estimadas) para comparar recuperación usando pregunta vs respuesta.")
        
        # Número de preguntas
        max_questions = min(2067, 3000)  # Límite basado en preguntas con links válidos
        num_questions = st.number_input(
            "Número de preguntas a evaluar:",
            min_value=10,
            max_value=max_questions,
            value=min(100, max_questions),
            step=10,
            help=f"Cantidad de preguntas aleatorias para analizar (máximo {max_questions:,} de ~2,067 con links válidos)"
        )
        
        # Top-k documentos
        top_k = st.number_input(
            "Documentos a recuperar (top-k):",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            help="Cantidad de documentos a recuperar para cada query"
        )
        
        # Método de reranking
        reranking_method = st.selectbox(
            "🔄 Método de Reranking:",
            options=["crossencoder", "standard", "none"],
            index=0,  # CrossEncoder por defecto
            format_func=lambda x: {
                "crossencoder": "🧠 CrossEncoder (Recomendado)",
                "standard": "📊 Reranking Estándar",
                "none": "❌ Sin Reranking"
            }[x],
            help="Método de reranking: CrossEncoder usa ms-marco-MiniLM-L-6-v2 para mejor calidad"
        )
    
    with col2:
        st.subheader("🤖 Configuración de Modelos")
        
        # Selección de modelo generativo
        generative_model = st.selectbox(
            "Modelo Generativo:",
            options=list(GENERATIVE_MODELS.keys()),
            index=list(GENERATIVE_MODELS.keys()).index("tinyllama-1.1b"),
            help="Modelo generativo para evaluación de calidad y RAG metrics"
        )
        
        # Modelos de embedding a evaluar
        st.markdown("**Modelos de Embedding a evaluar:**")
        
        # Mapeo de nombres cortos
        MODEL_NAME_MAPPING = {
            "mpnet": "multi-qa-mpnet-base-dot-v1",
            "minilm": "all-MiniLM-L6-v2",
            "ada": "ada",
            "e5-large": "e5-large-v2"
        }
        
        selected_models = []
        for short_name, full_name in MODEL_NAME_MAPPING.items():
            if st.checkbox(f"✅ {short_name.upper()}", value=True, key=f"model_{short_name}"):
                selected_models.append(short_name)
        
        if not selected_models:
            st.error("⚠️ Selecciona al menos un modelo de embedding")
    
    # Configuración avanzada
    with st.expander("🔧 Configuración Avanzada", expanded=False):
        col3, col4 = st.columns(2)
        
        with col3:
            # Configuración de métricas
            st.markdown("**Métricas a calcular:**")
            calculate_traditional_metrics = st.checkbox("📊 Métricas IR Tradicionales", value=True, help="Jaccard, nDCG@10, Precision@5")
            calculate_rag_metrics = st.checkbox("🧠 Métricas RAG", value=True, help="Faithfulness, Relevance, Correctness, Similarity")
            calculate_llm_quality = st.checkbox("⭐ Evaluación LLM", value=True, help="Calidad de contenido evaluada por LLM")
        
        with col4:
            # Configuración de procesamiento
            st.markdown("**Procesamiento:**")
            batch_size = st.number_input("Tamaño de lote:", min_value=1, max_value=50, value=10, help="Preguntas por lote")
            parallel_processing = st.checkbox("⚡ Procesamiento paralelo", value=True, help="Usar múltiples workers")
            save_intermediate = st.checkbox("💾 Guardar resultados intermedios", value=True, help="Guardar progreso cada N preguntas")
    
    # Sección de Google Drive
    st.markdown("---")
    st.subheader("☁️ Integración con Google Drive")
    
    # Estado de Google Drive
    show_gdrive_status()
    
    # Configuración de archivos
    st.markdown("**📁 Configuración de archivos:**")
    col5, col6 = st.columns(2)
    
    with col5:
        config_filename = st.text_input(
            "Nombre del archivo de configuración:",
            value=f"n_questions_config_{int(time.time())}.json",
            help="Nombre del archivo de configuración a subir a Google Drive"
        )
    
    with col6:
        results_filename = st.text_input(
            "Nombre del archivo de resultados:",
            value=f"n_questions_results_{int(time.time())}.json",
            help="Nombre esperado del archivo de resultados"
        )
    
    # Botón para crear configuración
    if st.button("🚀 Crear y Subir Configuración", type="primary", key="create_config"):
        if not selected_models:
            st.error("❌ Selecciona al menos un modelo de embedding")
            return
        
        # Crear configuración
        config = create_n_questions_evaluation_config(
            num_questions=num_questions,
            top_k=top_k,
            reranking_method=reranking_method,
            generative_model=generative_model,
            selected_models=selected_models,
            calculate_traditional_metrics=calculate_traditional_metrics,
            calculate_rag_metrics=calculate_rag_metrics,
            calculate_llm_quality=calculate_llm_quality,
            batch_size=batch_size,
            parallel_processing=parallel_processing,
            save_intermediate=save_intermediate,
            results_filename=results_filename
        )
        
        # Mostrar configuración creada en acordeón colapsado
        with st.expander("📋 Ver Configuración JSON (opcional)", expanded=False):
            st.json(config)
            st.caption("💡 Esta es la configuración que se enviará a Google Colab. Solo necesitas revisarla si quieres verificar los detalles técnicos.")
        
        # Subir a Google Drive usando función unificada
        try:
            # Importar función unificada
            from src.apps.cumulative_metrics_create import create_config_and_send_to_drive
            
            # Usar la función unificada que maneja ambos tipos de configuración
            create_config_and_send_to_drive(config)
            
            # Guardar en session state para seguimiento
            if 'n_questions_configs' not in st.session_state:
                st.session_state.n_questions_configs = []
            
            st.session_state.n_questions_configs.append({
                'config_file': config_filename,
                'results_file': results_filename,
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'configurado'
            })
            
            st.info(f"""
            🎯 **Próximos pasos específicos para N preguntas:**
            1. Ve a Google Colab y ejecuta el notebook de análisis acumulativo N preguntas
            2. Usa el archivo de configuración: `{config_filename}`
            3. Los resultados se guardarán como: `{results_filename}`
            4. Ve a la página de resultados para visualizar los datos
            """)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Mostrar configuraciones recientes
    if 'n_questions_configs' in st.session_state and st.session_state.n_questions_configs:
        st.markdown("---")
        st.subheader("📋 Configuraciones Recientes")
        
        for i, config_info in enumerate(reversed(st.session_state.n_questions_configs[-5:])):  # Mostrar últimas 5
            with st.expander(f"📄 {config_info['config_file']} - {config_info['created_at']}"):
                col7, col8, col9 = st.columns(3)
                with col7:
                    st.write(f"**Estado:** {config_info['status']}")
                with col8:
                    st.write(f"**Config:** {config_info['config_file']}")
                with col9:
                    st.write(f"**Resultados:** {config_info['results_file']}")
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 💡 Tips y Recomendaciones")
    
    col10, col11 = st.columns(2)
    
    with col10:
        st.info("""
        **🎯 Para mejores resultados:**
        - Usa al menos 50 preguntas para estadísticas confiables
        - Activa el reranking para mejor calidad
        - Incluye múltiples modelos para comparación
        """)
    
    with col11:
        st.warning("""
        **⚠️ Consideraciones:**
        - Más preguntas = más tiempo de procesamiento
        - RAG metrics requieren más recursos
        - Guarda los nombres de archivos para referencia
        """)


def create_n_questions_evaluation_config(
    num_questions: int,
    top_k: int,
    reranking_method: str,
    generative_model: str,
    selected_models: List[str],
    calculate_traditional_metrics: bool,
    calculate_rag_metrics: bool,
    calculate_llm_quality: bool,
    batch_size: int,
    parallel_processing: bool,
    save_intermediate: bool,
    results_filename: str
) -> Dict[str, Any]:
    """Crear configuración para evaluación de N preguntas."""
    
    # Mapeo de nombres de modelos
    MODEL_NAME_MAPPING = {
        "mpnet": "multi-qa-mpnet-base-dot-v1",
        "minilm": "all-MiniLM-L6-v2",
        "ada": "ada",
        "e5-large": "e5-large-v2"
    }
    
    # Cargar preguntas desde ChromaDB
    questions_data = load_questions_for_config(num_questions)
    if not questions_data:
        st.error("❌ No se pudieron cargar preguntas para la configuración")
        return None
    
    config = {
        "evaluation_type": "n_questions_cumulative_analysis",
        "version": "1.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        
        # Configuración de datos
        "data_config": {
            "num_questions": len(questions_data),  # Usar cantidad real de preguntas cargadas
            "top_k": top_k,
            "reranking_method": reranking_method,
            "use_reranking": reranking_method != "none",  # Compatibilidad hacia atrás
            "data_source": "chromadb_random_selection"
        },
        
        # Incluir las preguntas en la configuración
        "questions_data": questions_data,
        
        # Configuración de modelos
        "model_config": {
            "generative_model": generative_model,
            "embedding_models": {
                short_name: MODEL_NAME_MAPPING[short_name] 
                for short_name in selected_models
            }
        },
        
        # Configuración de métricas
        "metrics_config": {
            "calculate_traditional_metrics": calculate_traditional_metrics,  # Jaccard, nDCG, Precision
            "calculate_rag_metrics": calculate_rag_metrics,  # RAG metrics
            "calculate_llm_quality": calculate_llm_quality,  # LLM quality evaluation
            "metrics_included": []
        },
        
        # Configuración de procesamiento
        "processing_config": {
            "batch_size": batch_size,
            "parallel_processing": parallel_processing,
            "save_intermediate": save_intermediate,
            "max_retries": 3,
            "timeout_per_question": 120  # segundos
        },
        
        # Configuración de salida
        "output_config": {
            "results_filename": results_filename,
            "include_individual_results": True,
            "include_consolidated_metrics": True,
            "export_formats": ["json", "csv"]
        },
        
        # Metadatos
        "metadata": {
            "description": f"Análisis acumulativo de {len(questions_data)} preguntas con comparación pregunta vs respuesta",
            "expected_duration_minutes": estimate_processing_time(len(questions_data), len(selected_models), reranking_method != "none"),
            "models_count": len(selected_models),
            "total_comparisons": len(questions_data) * len(selected_models),
            "questions_source": "chromadb_random_selection"
        }
    }
    
    # Agregar métricas específicas según configuración
    if calculate_traditional_metrics:
        config["metrics_config"]["metrics_included"].extend([
            "jaccard_similarity", "ndcg_at_10", "precision_at_5", "common_docs"
        ])
    
    if calculate_rag_metrics:
        config["metrics_config"]["metrics_included"].extend([
            "faithfulness", "answer_relevance", "answer_correctness", "answer_similarity"
        ])
    
    if calculate_llm_quality:
        config["metrics_config"]["metrics_included"].extend([
            "question_quality", "answer_quality", "avg_quality"
        ])
    
    # Agregar score compuesto siempre
    config["metrics_config"]["metrics_included"].append("composite_score")
    
    return config


def estimate_processing_time(actual_questions: int, num_models: int, use_reranking: bool) -> int:
    """Estimar tiempo de procesamiento en minutos."""
    
    # Tiempo base por pregunta y modelo (en segundos)
    base_time_per_question_model = 2.0
    
    # Factor de reranking
    reranking_factor = 1.5 if use_reranking else 1.0
    
    # Factor de RAG metrics (más lento)
    rag_factor = 1.8
    
    # Cálculo total
    total_seconds = (
        actual_questions * 
        num_models * 
        base_time_per_question_model * 
        reranking_factor * 
        rag_factor
    )
    
    # Convertir a minutos y redondear hacia arriba
    total_minutes = int(total_seconds / 60) + 1
    
    return max(total_minutes, 5)  # Mínimo 5 minutos


if __name__ == "__main__":
    show_cumulative_n_questions_config_page()