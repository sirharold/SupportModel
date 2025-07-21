"""
Página de Métricas Acumulativas - Evalúa múltiples preguntas y calcula promedios (REFACTORIZADA)
"""

import streamlit as st
import time
import json
import os
from typing import List, Dict, Any
from src.config.config import EMBEDDING_MODELS, GENERATIVE_MODELS, CHROMADB_COLLECTION_CONFIG

# Importar utilidades refactorizadas
from src.data.memory_utils import get_memory_usage, cleanup_memory
# from utils.data_processing import filter_questions_with_links  # No longer needed
from src.ui.metrics_display import display_cumulative_metrics, display_models_comparison
from src.ui.enhanced_metrics_display import display_enhanced_cumulative_metrics, display_enhanced_models_comparison
from src.evaluation.cumulative import run_cumulative_metrics_evaluation, run_cumulative_metrics_for_models
from src.data.file_utils import display_download_section
from src.evaluation.metrics import validate_data_integrity
from src.services.storage.real_gdrive_integration import (
    show_gdrive_status, create_evaluation_config_in_drive,
    check_evaluation_status_in_drive, get_evaluation_results_from_drive,
    show_gdrive_authentication_instructions, show_gdrive_debug_info,
    get_all_results_files_from_drive, get_specific_results_file_from_drive
)


def show_cumulative_metrics_page():
    """Función principal que muestra la página de métricas acumulativas."""
    st.title("📊 Métricas Acumulativas de Recuperación RAG")
    st.markdown("""
    Esta página permite evaluar múltiples preguntas y calcular métricas promedio para diferentes modelos de embedding.
    """)
    
    # Variables globales para la configuración
    use_colab_processing = False
    
    # Configuración inicial
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Configuración de Evaluación")
        
        # Información sobre la fuente de datos
        st.info("📊 Las preguntas se extraen aleatoriamente desde ChromaDB, filtrando solo aquellas que tienen enlaces de Microsoft Learn en la respuesta aceptada.")
        
        # Número de preguntas
        num_questions = st.number_input(
            "🔢 Número de preguntas a evaluar:",
            min_value=1,
            max_value=2000,
            value=600,
            step=1,
            help="Número total de preguntas para la evaluación"
        )
        
        # Configuración de modelo generativo
        generative_model_name = st.selectbox(
            "🤖 Modelo Generativo:",
            list(GENERATIVE_MODELS.keys()),
            index=list(GENERATIVE_MODELS.keys()).index("gpt-4") if "gpt-4" in GENERATIVE_MODELS else 0,
            help="Modelo usado para reranking LLM"
        )
        
        # Configuración de recuperación
        top_k = st.number_input(
            "🔝 Top-K documentos:",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Número máximo de documentos a recuperar"
        )
        
        # Usar reranking LLM
        use_llm_reranker = st.checkbox(
            "🤖 Usar Reranking LLM",
            value=True,
            help="Activar reordenamiento de documentos con LLM"
        )
        
        # NUEVO: Opción de usar Google Colab
        use_colab_processing = st.checkbox(
            "☁️ Procesamiento en Google Colab",
            value=True,
            help="Exportar evaluación a Google Colab para procesamiento con GPU (más rápido)"
        )
        
        if use_colab_processing:
            st.info("📋 Con Colab: Se exportarán los datos, obtendrás un notebook para ejecutar en Colab con GPU, y luego importarás los resultados.")
            
            # Mostrar estado de Google Drive
            gdrive_ok = show_gdrive_status()
            
            if not gdrive_ok:
                show_gdrive_authentication_instructions()
                st.stop()
        
        # Tamaño de lote
        batch_size = st.number_input(
            "📦 Tamaño de lote:",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Número de preguntas a procesar por lote (para gestión de memoria)"
        )
    
    with col2:
        st.subheader("🎯 Selección de Modelos")
        
        # Opción para evaluar todos los modelos
        evaluate_all_models = st.checkbox(
            "🔄 Evaluar todos los modelos",
            value=True,
            help="Evaluar todos los modelos de embedding disponibles"
        )
        
        if evaluate_all_models:
            selected_models = list(EMBEDDING_MODELS.keys())
            st.info(f"📋 Se evaluarán {len(selected_models)} modelos")
            for model in selected_models:
                st.markdown(f"• {model}")
        else:
            # Selección manual de modelos
            selected_models = st.multiselect(
                "🎛️ Modelos de Embedding:",
                list(EMBEDDING_MODELS.keys()),
                default=[list(EMBEDDING_MODELS.keys())[0]],
                help="Selecciona uno o más modelos para evaluar"
            )
        
        if not selected_models:
            st.error("❌ Debes seleccionar al menos un modelo")
            return
        
        # Información sobre memoria estimada
        estimated_memory = len(selected_models) * num_questions * 0.1  # MB aproximados
        st.info(f"💾 Memoria estimada: ~{estimated_memory:.1f} MB")
    
    # Botones de control
    st.markdown("---")
    
    if use_colab_processing:
        # Interfaz para flujo Google Colab
        show_colab_workflow(
            num_questions, selected_models, 
            generative_model_name, top_k, use_llm_reranker, 
            batch_size, evaluate_all_models
        )
    else:
        # Interfaz para evaluación local
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Ejecutar Evaluación", type="primary"):
                run_evaluation(
                    num_questions, selected_models, 
                    generative_model_name, top_k, use_llm_reranker, 
                    batch_size, evaluate_all_models
                )
        
        with col2:
            if st.button("🧹 Limpiar Caché"):
                for key in list(st.session_state.keys()):
                    if 'cumulative_results' in key:
                        del st.session_state[key]
                st.success("✅ Caché limpiado")
                st.rerun()
        
        with col3:
            if st.button("📊 Mostrar Estadísticas de Memoria"):
                show_memory_stats()
    
    # Mostrar resultados si existen
    show_cached_results()


def run_evaluation(num_questions: int, selected_models: List[str],
                  generative_model_name: str, top_k: int, use_llm_reranker: bool,
                  batch_size: int, evaluate_all_models: bool):
    """Ejecuta la evaluación de métricas."""
    
    # Mostrar información de memoria inicial
    initial_memory = get_memory_usage()
    st.info(f"💾 Memoria inicial: {initial_memory:.1f} MB")
    
    # Ejecutar evaluación
    start_time = time.time()
    evaluation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if evaluate_all_models:
            with st.spinner(f"🔄 Evaluando {len(selected_models)} modelos..."):
                results = run_cumulative_metrics_for_models(
                    num_questions=num_questions,
                    model_names=selected_models,
                    generative_model_name=generative_model_name,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    batch_size=batch_size
                )
        else:
            # Evaluación de modelo único
            model_name = selected_models[0]
            with st.spinner(f"⚙️ Evaluando {model_name}..."):
                single_result = run_cumulative_metrics_evaluation(
                    num_questions=num_questions,
                    model_name=model_name,
                    generative_model_name=generative_model_name,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    batch_size=batch_size
                )
                results = {model_name: single_result}
        
        # Validar integridad de datos
        for model_name, result in results.items():
            results[model_name] = validate_data_integrity(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Guardar en caché
        cache_key = f"cumulative_results_{len(selected_models)}_{num_questions}_{int(start_time)}"
        st.session_state[cache_key] = {
            'results': results,
            'evaluation_time': evaluation_time,
            'execution_time': execution_time,
            'evaluate_all_models': evaluate_all_models,
            'params': {
                'num_questions': num_questions,
                'selected_models': selected_models,
                'embedding_model_name': selected_models[0] if len(selected_models) == 1 else 'Multi-Model',
                'generative_model_name': generative_model_name,
                'top_k': top_k,
                'use_llm_reranker': use_llm_reranker,
                'batch_size': batch_size
            }
        }
        
        st.success(f"✅ Evaluación completada en {execution_time:.2f} segundos")
        
        # Mostrar memoria final
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        st.info(f"💾 Memoria final: {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Forzar rerun para mostrar resultados
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error durante la evaluación: {str(e)}")
        st.exception(e)
    finally:
        # Limpieza de memoria
        cleanup_memory()


def show_colab_workflow(num_questions: int, selected_models: List[str],
                       generative_model_name: str, top_k: int, use_llm_reranker: bool,
                       batch_size: int, evaluate_all_models: bool):
    """Muestra la interfaz completa del flujo Google Colab"""
    
    st.subheader("☁️ Flujo de Evaluación con Google Colab")
    
    # Configuración de la evaluación
    evaluation_config = {
        'num_questions': num_questions,
        'selected_models': selected_models,
        'generative_model_name': generative_model_name,
        'top_k': top_k,
        'use_llm_reranker': use_llm_reranker,
        'batch_size': batch_size,
        'evaluate_all_models': evaluate_all_models,
        'evaluation_type': 'cumulative_metrics'
    }
    
    # Columnas para configuración
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Configuración de Evaluación:**")
        st.json({k: v for k, v in evaluation_config.items()})
    
    with col2:
        st.markdown("**📁 Google Drive:**")
        st.info("Carpeta: `/TesisMagister/acumulative/`")
    
    # Sección de resultados disponibles (mostrar siempre)
    st.markdown("---")
    show_available_results_section()
    
    # Botones del flujo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Crear Configuración y Enviar a Google Drive", type="primary"):
            create_config_and_send_to_drive(evaluation_config)
    
    with col2:
        if st.button("🔄 Verificar Estado"):
            check_colab_evaluation_status()
    
    with col3:
        if st.button("🔍 Debug Google Drive"):
            st.markdown("---")
            show_gdrive_debug_info()
    
    with col4:
        # Placeholder para futuro botón
        st.empty()
    
    # Mostrar estado actual
    st.markdown("---")
    display_current_colab_status()


def create_config_and_send_to_drive(evaluation_config: Dict):
    """Crea y envía configuración a Google Drive real"""
    
    st.info("📤 Creando configuración y enviando a Google Drive...")
    
    try:
        # Obtener preguntas reales de la base de datos
        with st.spinner("📥 Obteniendo preguntas de ChromaDB..."):
            from src.data.processing import fetch_random_questions_from_chromadb
            from src.services.auth.clients import initialize_clients
            
            try:
                # Usar el primer modelo seleccionado para obtener las preguntas
                first_model = evaluation_config['selected_models'][0]
                chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client = initialize_clients(
                    model_name=first_model,
                    generative_model_name=evaluation_config['generative_model_name']
                )
                questions = fetch_random_questions_from_chromadb(
                    chromadb_wrapper=chromadb_wrapper,
                    embedding_model_name=first_model,
                    num_questions=evaluation_config['num_questions']
                )
                
                if questions:
                    evaluation_config['questions_data'] = questions
                    st.success(f"✅ Obtenidas {len(questions)} preguntas con enlaces MS Learn")
                else:
                    st.warning("⚠️ No se encontraron preguntas, usando configuración sin datos")
                    evaluation_config['questions_data'] = None
                    
            except Exception as e:
                st.warning(f"⚠️ Error obteniendo preguntas: {e}")
                evaluation_config['questions_data'] = None
        
        # Enviar a Google Drive real
        with st.spinner("☁️ Enviando configuración a Google Drive..."):
            result = create_evaluation_config_in_drive(evaluation_config)
            
            if result['success']:
                st.success("✅ ¡Configuración enviada exitosamente a Google Drive!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📄 Archivo creado:**")
                    st.code(result['config_filename'])
                    st.markdown(f"[🔗 Ver en Drive]({result['web_link']})")
                
                with col2:
                    st.markdown("**📋 Próximos pasos:**")
                    st.markdown("""
                    1. Abrir Google Colab
                    2. Subir `Universal_Colab_Evaluator.ipynb` 
                    3. Activar GPU (T4)
                    4. Ejecutar todas las celdas
                    5. Volver aquí para ver resultados
                    """)
                
                # Botón para descargar notebook
                try:
                    with open('Universal_Colab_Evaluator.ipynb', 'r') as f:
                        notebook_content = f.read()
                    
                    st.download_button(
                        label="📓 Descargar Notebook Universal",
                        data=notebook_content,
                        file_name="Universal_Colab_Evaluator.ipynb",
                        mime="application/json"
                    )
                except FileNotFoundError:
                    st.warning("⚠️ Notebook no encontrado localmente")
                
                st.rerun()
                
            else:
                st.error(f"❌ Error enviando a Google Drive: {result['error']}")
                
    except Exception as e:
        st.error(f"❌ Error: {e}")


def check_colab_evaluation_status():
    """Verifica el estado de la evaluación en Google Drive"""
    
    with st.spinner("🔍 Verificando estado en Google Drive..."):
        result = check_evaluation_status_in_drive()
        
        if result['success']:
            status = result['status']
            
            if status == 'completed':
                st.success("✅ ¡Evaluación completada en Google Colab!")
                
                data = result['data']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Resumen:**")
                    st.write(f"🤖 Modelos: {data.get('models_evaluated', 'N/A')}")
                    st.write(f"❓ Preguntas: {data.get('questions_processed', 'N/A')}")
                    st.write(f"⏱️ Tiempo: {data.get('total_time_seconds', 0):.1f}s")
                    st.write(f"🚀 GPU: {'✅' if data.get('gpu_used') else '❌'}")
                
                with col2:
                    st.markdown("**📁 Archivos:**")
                    st.write(f"📄 {data.get('results_file', 'N/A')}")
                    st.write(f"📊 {data.get('summary_file', 'N/A')}")
                    
            elif status == 'config_created':
                st.info("📋 Configuración creada. Esperando ejecución en Colab...")
            elif status == 'no_status_file':
                st.warning("⚠️ No se encontró información de estado")
            else:
                st.info(f"🔄 Estado actual: {status}")
                
        else:
            st.error(f"❌ Error verificando estado: {result['error']}")


def show_available_results_section():
    """Muestra la sección de resultados disponibles con dropdown de selección"""
    
    st.subheader("📊 Resultados Disponibles")
    
    # Obtener archivos disponibles sin spinner para no interferir con la UI
    try:
        files_result = get_all_results_files_from_drive()
        
        if not files_result['success']:
            st.warning(f"🔍 No se pudieron obtener archivos de resultados")
            st.error(f"❌ {files_result['error']}")
            
            # Mostrar información de debug si está disponible
            if 'debug_info' in files_result:
                debug_info = files_result['debug_info']
                with st.expander("🔍 Información de Debug", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📁 **Folder ID:** `{debug_info.get('folder_id', 'N/A')}`")
                        st.write(f"🔍 **Búsqueda:** `{debug_info.get('search_query', 'N/A')}`")
                    with col2:
                        st.write(f"📄 **Total archivos:** {debug_info.get('total_files', 0)}")
                        st.write(f"📋 **Archivos JSON:** {debug_info.get('json_files', 0)}")
                    
                    # Mostrar algunos nombres de archivos para ayudar con debug
                    if 'all_files' in debug_info and debug_info['all_files']:
                        st.write("📋 **Algunos archivos JSON encontrados:**")
                        for filename in debug_info['all_files']:
                            st.write(f"- `{filename}`")
            
            # Botón para ejecutar debug completo
            if st.button("🔍 Ejecutar Debug Completo de Google Drive"):
                st.markdown("---")
                show_gdrive_debug_info()
            
            st.info("💡 **Posibles soluciones:**")
            st.markdown("""
            1. 🚀 **Ejecuta una evaluación en Colab** para generar archivos de resultados
            2. 🔧 **Verifica la conexión** con el botón 'Debug Google Drive' 
            3. 📁 **Confirma la carpeta** acumulative en Google Drive
            4. 📋 **Revisa el formato** de archivos (deben ser `cumulative_results_*.json`)
            """)
            return
        
        available_files = files_result['files']
        
        if not available_files:
            st.info("📭 No hay archivos de resultados disponibles")
            st.info("💡 Ejecuta una evaluación en Colab para generar resultados")
            return
            
    except Exception as e:
        st.warning(f"⚠️ Error verificando archivos: {str(e)}")
        return
    
    # Mostrar información de archivos disponibles
    st.success(f"✅ Encontrados {len(available_files)} archivos de resultados")
    
    # Crear dropdown con todos los archivos disponibles
    file_options = {file_info['display_name']: file_info['file_id'] for file_info in available_files}
    
    # Usar session_state para mantener la selección
    if 'selected_results_file' not in st.session_state:
        st.session_state.selected_results_file = list(file_options.keys())[0]
    
    selected_display_name = st.selectbox(
        "📁 Seleccionar archivo de resultados:",
        options=list(file_options.keys()),
        index=list(file_options.keys()).index(st.session_state.selected_results_file) if st.session_state.selected_results_file in file_options else 0,
        help="Los archivos están ordenados por fecha de modificación (más reciente primero)",
        key="results_file_selector"
    )
    
    # Actualizar session_state
    st.session_state.selected_results_file = selected_display_name
    selected_file_id = file_options[selected_display_name]
    
    # Botón para mostrar resultados del archivo seleccionado
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("📊 Mostrar Resultados del Archivo Seleccionado", type="primary", use_container_width=True):
            load_and_display_selected_results(selected_file_id)
    
    with col2:
        if st.button("🔄 Actualizar Lista", help="Buscar nuevos archivos de resultados"):
            st.rerun()


def load_and_display_selected_results(file_id: str):
    """Carga y muestra los resultados del archivo seleccionado"""
    
    with st.spinner("📊 Cargando resultados del archivo seleccionado..."):
        result = get_specific_results_file_from_drive(file_id)
        
        if not result['success']:
            st.error(f"❌ Error cargando archivo: {result['error']}")
            return
        
        st.success("✅ Resultados cargados exitosamente!")
        
        # Usar un expander para los resultados para que no ocupen todo el espacio
        with st.expander("📊 Resultados de Evaluación", expanded=True):
            display_results_content(result['results'])


def display_results_content(results_data):
    """Muestra el contenido de los resultados de evaluación"""
    
    # Debug: Mostrar estructura de datos si está habilitado
    if st.checkbox("🔍 Mostrar estructura de datos (debug)", value=False):
        with st.expander("📋 Estructura de datos completa"):
            st.json(results_data)
    
    # Convertir formato para compatibilidad con funciones existentes
    processed_results = {}
    for model_name, model_data in results_data['results'].items():
        processed_results[model_name] = model_data
    
    # Generar visualizaciones usando funciones existentes
    # Extraer información de tiempo y configuración
    evaluation_info = results_data.get('evaluation_info', {})
    config_info = results_data.get('config', {})
    
    # Formatear fecha y hora
    timestamp = evaluation_info.get('timestamp')
    if timestamp:
        try:
            from datetime import datetime
            if isinstance(timestamp, str):
                # Intentar parsear diferentes formatos de timestamp
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                dt = datetime.fromtimestamp(timestamp)
            
            formatted_date = dt.strftime('%d/%m/%Y %H:%M:%S')
        except:
            formatted_date = str(timestamp)
    else:
        formatted_date = "Fecha no disponible"
    
    # Obtener número de preguntas
    num_questions = config_info.get('num_questions', evaluation_info.get('questions_processed', 'N/A'))
    
    # Obtener método de autenticación si está disponible
    auth_method = evaluation_info.get('auth_method', '')
    auth_icon = '🔐' if auth_method == 'API' else '📁' if auth_method == 'Mount' else ''
    
    # Mostrar header con información detallada
    st.subheader(f"📊 Resultados de Evaluación Acumulativa")
    
    # Información de la evaluación en una caja destacada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📅 **Fecha:** {formatted_date}")
    
    with col2:
        st.info(f"❓ **Preguntas:** {num_questions}")
    
    with col3:
        models_count = len(results_data['results'])
        st.info(f"🤖 **Modelos:** {models_count}")
    
    # Información adicional si está disponible
    if evaluation_info.get('total_time_seconds'):
        total_time = evaluation_info['total_time_seconds']
        if total_time >= 60:
            time_str = f"{total_time/60:.1f} min"
        else:
            time_str = f"{total_time:.1f}s"
        
        gpu_used = evaluation_info.get('gpu_used', False)
        gpu_icon = '🚀' if gpu_used else '💻'
        
        st.caption(f"⏱️ Tiempo de ejecución: {time_str} {gpu_icon} | {auth_icon} {auth_method}")
    
    st.markdown("---")
    
    # Determinar si usar reranker desde config
    use_llm_reranker = results_data['config'].get('use_llm_reranker', True)
    
    if len(processed_results) == 1:
        # Para un solo modelo, usar display_enhanced_cumulative_metrics
        model_name = list(processed_results.keys())[0]
        model_results = processed_results[model_name]
        
        # Adaptar formato según la estructura de datos disponible
        # Nuevo formato del notebook actualizado tiene avg_before_metrics y avg_after_metrics
        if 'avg_before_metrics' in model_results and 'avg_after_metrics' in model_results:
            # Formato actualizado con before/after metrics
            adapted_results = {
                'num_questions_evaluated': model_results.get('num_questions_evaluated', results_data['config']['num_questions']),
                'avg_before_metrics': model_results['avg_before_metrics'],
                'avg_after_metrics': model_results['avg_after_metrics'],
                'individual_metrics': model_results.get('individual_before_metrics', [])
            }
            
            # Verificar si realmente hay métricas after
            if use_llm_reranker and not model_results['avg_after_metrics']:
                st.info("ℹ️ Reranking LLM estaba habilitado en configuración, pero no se generaron métricas after. Mostrando solo métricas before.")
                use_llm_reranker = False
                
        else:
            # Formato anterior - solo avg_metrics (mantener compatibilidad)
            avg_metrics = model_results.get('avg_metrics', {})
            
            adapted_results = {
                'num_questions_evaluated': results_data['config']['num_questions'],
                'avg_before_metrics': avg_metrics,  # Usar las métricas como "before" 
                'avg_after_metrics': {},  # Vacío porque no hay reranking
                'individual_metrics': model_results.get('individual_metrics', [])
            }
            
            if use_llm_reranker:
                st.warning("⚠️ Reranking LLM estaba habilitado pero estos resultados usan formato antiguo. Mostrando solo métricas base.")
                use_llm_reranker = False
        
        # Use enhanced display with cleaner before/after LLM separation
        display_enhanced_cumulative_metrics(adapted_results, model_name, use_llm_reranker)
        
    else:
        # Para múltiples modelos, usar display_enhanced_models_comparison
        st.markdown("### 🏆 Comparación de Modelos")
        
        # Adaptar formato para múltiples modelos según estructura disponible
        adapted_multi_results = {}
        has_new_format = False
        
        for model_name, model_data in processed_results.items():
            # Verificar si usa el nuevo formato con before/after metrics
            if 'avg_before_metrics' in model_data and 'avg_after_metrics' in model_data:
                # Formato actualizado
                adapted_multi_results[model_name] = {
                    'num_questions_evaluated': model_data.get('num_questions_evaluated', results_data['config']['num_questions']),
                    'avg_before_metrics': model_data['avg_before_metrics'],
                    'avg_after_metrics': model_data['avg_after_metrics'],
                    'individual_metrics': model_data.get('individual_before_metrics', [])
                }
                has_new_format = True
            else:
                # Formato anterior - compatibilidad
                avg_metrics = model_data.get('avg_metrics', {})
                adapted_multi_results[model_name] = {
                    'num_questions_evaluated': results_data['config']['num_questions'],
                    'avg_before_metrics': avg_metrics,  # Usar métricas como "before"
                    'avg_after_metrics': {},  # Vacío porque no hay reranking
                    'individual_metrics': model_data.get('individual_metrics', [])
                }
        
        # Verificar si hay métricas after disponibles para LLM reranking
        if use_llm_reranker:
            has_after_metrics = any(adapted_multi_results[model]['avg_after_metrics'] for model in adapted_multi_results)
            if not has_after_metrics:
                if has_new_format:
                    st.info("ℹ️ Reranking LLM estaba habilitado pero no se generaron métricas after. Mostrando solo métricas before.")
                else:
                    st.warning("⚠️ Reranking LLM estaba habilitado pero estos resultados usan formato antiguo. Mostrando solo métricas base.")
                use_llm_reranker = False
        
        # Use enhanced display for cleaner multi-model comparison
        display_enhanced_models_comparison(adapted_multi_results, use_llm_reranker)
    
    # Sección de descarga
    st.markdown("---")
    
    # Preparar datos para la sección de descarga en el formato esperado
    cached_results = {
        'results': processed_results,
        'evaluation_time': results_data['evaluation_info'].get('timestamp'),
        'execution_time': results_data['evaluation_info'].get('total_time_seconds'),
        'evaluate_all_models': len(processed_results) > 1,
        'params': {
            'num_questions': results_data['config']['num_questions'],
            'selected_models': list(processed_results.keys()),
            'embedding_model_name': list(processed_results.keys())[0] if len(processed_results) == 1 else 'Multi-Model',
            'generative_model_name': results_data['config']['generative_model_name'],
            'top_k': results_data['config']['top_k'],
            'use_llm_reranker': results_data['config']['use_llm_reranker'],
            'batch_size': results_data['config']['batch_size']
        }
    }
    
    display_download_section(cached_results)




def display_current_colab_status():
    """Muestra el estado actual del sistema Google Colab"""
    
    st.subheader("📊 Estado Actual")
    
    # Verificar estado sin spinner (solo para mostrar info)
    try:
        result = check_evaluation_status_in_drive()
        
        if result['success']:
            status = result['status']
            
            if status == 'completed':
                st.success("✅ Evaluación completada - Lista para mostrar resultados")
            elif status == 'config_created':
                st.info("📋 Configuración lista - Ejecutar en Google Colab")
            elif status == 'no_status_file':
                st.info("⚪ Sin evaluaciones pendientes")
            else:
                st.info(f"🔄 Estado: {status}")
        else:
            st.warning("⚠️ No se pudo verificar estado")
            
    except Exception as e:
        st.warning(f"⚠️ Error verificando estado: {e}")


def create_drive_config_file(evaluation_config: Dict) -> bool:
    """Crea archivo de configuración simulando Google Drive"""
    
    try:
        import os
        
        # Simular la estructura de carpetas de Google Drive localmente
        # En producción, esto se conectaría realmente a Google Drive
        local_drive_path = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/simulated_drive"
        config_path = f"{local_drive_path}/evaluation_config.json"
        
        # Crear directorio si no existe
        os.makedirs(local_drive_path, exist_ok=True)
        
        # Guardar configuración
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_config, f, indent=2, ensure_ascii=False)
        
        st.success(f"✅ Configuración guardada en: {config_path}")
        st.info("📝 En producción, esto se guardaría en `/content/drive/MyDrive/TesisMagister/acumulative/evaluation_config.json`")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error creando archivo de configuración: {e}")
        return False


def show_colab_status_and_results():
    """Muestra el estado de la evaluación en Colab y botón para mostrar resultados"""
    
    st.subheader("📊 Estado de la Evaluación en Colab")
    
    # Verificar si existe archivo de status
    status_file = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/simulated_drive/evaluation_status.json"
    results_dir = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/simulated_drive/results"
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            
            if status_data.get('status') == 'completed':
                st.success("✅ ¡Evaluación completada en Google Colab!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Resumen de Resultados:**")
                    st.write(f"🤖 Modelos evaluados: {status_data.get('models_evaluated', 'N/A')}")
                    st.write(f"❓ Preguntas procesadas: {status_data.get('questions_processed', 'N/A')}")
                    st.write(f"⏱️ Tiempo total: {status_data.get('total_time_seconds', 0):.2f}s")
                    st.write(f"🚀 GPU utilizada: {'✅' if status_data.get('gpu_used') else '❌'}")
                
                with col2:
                    st.markdown("**📁 Archivos generados:**")
                    st.write(f"📄 {status_data.get('results_file', 'N/A')}")
                    st.write(f"📊 {status_data.get('summary_file', 'N/A')}")
                
                # Botón para mostrar resultados
                if st.button("📊 Mostrar Resultados y Generar Visualizaciones", type="primary"):
                    show_colab_results_and_generate_visuals(status_data)
                
            else:
                st.info(f"🔄 Estado: {status_data.get('status', 'unknown')}")
        
        except Exception as e:
            st.error(f"❌ Error leyendo estado: {e}")
    
    else:
        st.info("⏳ Esperando resultados de Google Colab...")
        st.markdown("**📋 Para que aparezcan los resultados:**")
        st.markdown("1. Ejecuta el notebook en Google Colab")
        st.markdown("2. Espera a que termine la evaluación")
        st.markdown("3. Los resultados aparecerán automáticamente aquí")
        
        # Botón para refrescar estado
        if st.button("🔄 Verificar Estado"):
            st.rerun()




def generate_colab_notebook_code(config: Dict) -> str:
    """Genera el código completo para ejecutar en Colab."""
    
    models_str = ', '.join([f'"{m}"' for m in config['selected_models']])
    
    code = f'''# 🚀 Evaluación de Embeddings en Google Colab con GPU
# Generado automáticamente el {config['timestamp']}
# Configuración: {config['num_questions']} preguntas, {len(config['selected_models'])} modelos

import os
import time
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# 📊 Configuración de la evaluación
EVALUATION_CONFIG = {{
    'num_questions': {config['num_questions']},
    'selected_models': [{models_str}],
    'generative_model_name': "{config['generative_model_name']}",
    'top_k': {config['top_k']},
    'use_llm_reranker': {config['use_llm_reranker']},
    'batch_size': {config['batch_size']},
    'evaluate_all_models': {config['evaluate_all_models']},
    'timestamp': "{config['timestamp']}"
}}

print("🚀 Iniciando evaluación de embeddings en Google Colab")
print("📊 Configuración:")
for key, value in EVALUATION_CONFIG.items():
    print(f"   {{key}}: {{value}}")

# ✅ 1. Verificar GPU
print("\\n🔧 Verificando hardware disponible...")
try:
    import torch
    print(f"CUDA disponible: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        print(f"GPU: {{torch.cuda.get_device_name(0)}}")
        print(f"Memoria GPU: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
        print("✅ GPU T4 detectada - procesamiento acelerado habilitado!")
    else:
        print("⚠️  GPU no disponible - usando CPU (más lento)")
except ImportError:
    print("⚠️  PyTorch no instalado - se instalará en el siguiente paso")

# 📦 2. Instalar dependencias necesarias
print("\\n📦 Instalando dependencias...")
!pip install -q sentence-transformers pandas numpy scikit-learn openai python-dotenv tqdm

# 📚 3. Importar librerías
print("\\n📚 Importando librerías...")
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm
    import warnings
    warnings.filterwarnings('ignore')
    print("✅ Librerías importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando librerías: {{e}}")
    print("💡 Reinicia el runtime y vuelve a ejecutar")

# 🎲 4. Generar datos de prueba (simulando tu base de datos)
print("\\n🎲 Generando datos de prueba para demostración...")

def generate_sample_questions(num_questions: int) -> List[Dict]:
    """Genera preguntas de ejemplo que simulan tu base de datos"""
    
    sample_questions = [
        "¿Cómo configurar Azure Storage Blob?",
        "¿Cuál es la diferencia entre SQL Database y Cosmos DB?",
        "¿Cómo implementar autenticación en Azure Functions?",
        "¿Qué es Azure Container Instances?",
        "¿Cómo usar Azure DevOps para CI/CD?",
        "¿Cuáles son las mejores prácticas para Azure Security?",
        "¿Cómo configurar Application Insights?",
        "¿Qué es Azure Service Bus?",
        "¿Cómo usar Azure Logic Apps?",
        "¿Cuál es la diferencia entre VM y App Service?",
        "¿Cómo configurar Azure Active Directory?",
        "¿Qué es Azure Kubernetes Service?",
        "¿Cómo usar Azure Key Vault?",
        "¿Cuáles son los tipos de Azure Storage?",
        "¿Cómo implementar Azure API Management?"
    ]
    
    # Expandir la lista repitiendo y variando
    questions = []
    for i in range(num_questions):
        base_question = sample_questions[i % len(sample_questions)]
        # Añadir variación
        variations = [
            f"{{base_question}}",
            f"Tutorial: {{base_question}}",
            f"Guía paso a paso: {{base_question}}",
            f"Mejores prácticas: {{base_question}}",
            f"Solución de problemas: {{base_question}}"
        ]
        
        question_text = variations[i % len(variations)]
        
        questions.append({{'question': question_text, 'id': f'q_{{i+1}}'}})
    
    return questions

# Generar preguntas de prueba
test_questions = generate_sample_questions(EVALUATION_CONFIG['num_questions'])
print(f"✅ Generadas {{len(test_questions)}} preguntas de prueba")

# 🤖 5. Función de evaluación acelerada con GPU
def run_gpu_accelerated_evaluation(questions: List[Dict], models: List[str]) -> Dict:
    """Ejecuta evaluación usando GPU para máximo rendimiento"""
    
    print(f"\\n🚀 Iniciando evaluación acelerada...")
    print(f"📊 Preguntas: {{len(questions)}}")
    print(f"🤖 Modelos: {{models}}")
    
    results = {{}}
    
    for model_name in tqdm(models, desc="Evaluando modelos"):
        print(f"\\n⚙️ Procesando modelo: {{model_name}}")
        
        # Simular carga del modelo
        model_start = time.time()
        
        # En una implementación real, aquí cargarías el modelo:
        # model = SentenceTransformer(model_name)
        # if torch.cuda.is_available():
        #     model = model.to('cuda')
        
        print(f"   📥 Modelo cargado en {{time.time() - model_start:.2f}}s")
        
        # Simular procesamiento por lotes
        batch_size = EVALUATION_CONFIG['batch_size']
        batch_results = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Lotes {{model_name}}", leave=False):
            batch_questions = questions[i:i+batch_size]
            
            # Simular embeddings y métricas (en implementación real usarías el modelo)
            batch_metrics = {{
                'precision': random.uniform(0.65, 0.95),
                'recall': random.uniform(0.60, 0.90),
                'f1': random.uniform(0.62, 0.92),
                'map': random.uniform(0.55, 0.88),
                'mrr': random.uniform(0.60, 0.92),
                'ndcg': random.uniform(0.65, 0.95)
            }}
            
            batch_results.append(batch_metrics)
            
            # Simular tiempo de procesamiento GPU
            time.sleep(0.01)  # Simular procesamiento rápido con GPU
        
        # Calcular métricas promedio
        avg_metrics = {{}}
        for metric in ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']:
            values = [br[metric] for br in batch_results]
            avg_metrics[f'avg_{{metric}}'] = np.mean(values)
        
        # Guardar resultados del modelo
        model_results = {{
            'model_name': model_name,
            'avg_metrics': avg_metrics,
            'total_questions': len(questions),
            'batch_size': batch_size,
            'processing_time_seconds': time.time() - model_start,
            'evaluation_time': datetime.now().isoformat(),
            'gpu_accelerated': torch.cuda.is_available() if 'torch' in globals() else False
        }}
        
        results[model_name] = model_results
        
        print(f"   ✅ {{model_name}} completado - F1: {{avg_metrics['avg_f1']:.3f}}")
    
    return results

# 🚀 6. Ejecutar evaluación completa
print("\\n" + "="*60)
print("🚀 EJECUTANDO EVALUACIÓN COMPLETA")
print("="*60)

start_time = time.time()

try:
    # Ejecutar evaluación
    evaluation_results = run_gpu_accelerated_evaluation(
        test_questions, 
        EVALUATION_CONFIG['selected_models']
    )
    
    total_time = time.time() - start_time
    
    # 💾 7. Guardar resultados
    print(f"\\n💾 Guardando resultados...")
    
    # Archivo JSON completo
    output_file = f"cumulative_results_colab_{{int(time.time())}}.json"
    
    final_results = {{
        'config': EVALUATION_CONFIG,
        'results': evaluation_results,
        'execution_summary': {{
            'total_time_seconds': total_time,
            'questions_processed': len(test_questions),
            'models_evaluated': len(EVALUATION_CONFIG['selected_models']),
            'gpu_used': torch.cuda.is_available() if 'torch' in globals() else False,
            'timestamp': datetime.now().isoformat()
        }}
    }}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Archivo CSV resumen
    csv_data = []
    for model_name, results in evaluation_results.items():
        metrics = results['avg_metrics']
        row = {{'Model': model_name}}
        row.update({{k.replace('avg_', '').upper(): f"{{v:.4f}}" for k, v in metrics.items()}})
        row['Processing_Time'] = f"{{results['processing_time_seconds']:.2f}}s"
        csv_data.append(row)
    
    df_summary = pd.DataFrame(csv_data)
    csv_file = f"results_summary_{{int(time.time())}}.csv"
    df_summary.to_csv(csv_file, index=False)
    
    # 📊 8. Mostrar resumen final
    print("\\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    print(f"⏱️  Tiempo total: {{total_time:.2f}} segundos")
    print(f"📊 Preguntas procesadas: {{len(test_questions):,}}")
    print(f"🤖 Modelos evaluados: {{len(EVALUATION_CONFIG['selected_models'])}}")
    print(f"🚀 GPU acelerado: {{'✅ Sí' if torch.cuda.is_available() if 'torch' in globals() else False else '❌ No'}}")
    
    print(f"\\n🏆 RANKING DE MODELOS:")
    print("-"*50)
    
    # Ordenar modelos por F1-Score
    model_ranking = sorted(
        evaluation_results.items(),
        key=lambda x: x[1]['avg_metrics']['avg_f1'],
        reverse=True
    )
    
    for i, (model_name, results) in enumerate(model_ranking, 1):
        metrics = results['avg_metrics']
        print(f"{{i}}. {{model_name}}")
        print(f"   F1: {{metrics['avg_f1']:.4f}} | MAP: {{metrics['avg_map']:.4f}} | MRR: {{metrics['avg_mrr']:.4f}}")
    
    print(f"\\n📁 ARCHIVOS GENERADOS:")
    print("-"*30)
    print(f"📄 {{output_file}} (resultados completos)")
    print(f"📊 {{csv_file}} (resumen CSV)")
    
    print(f"\\n💾 Para descargar:")
    print("1. Haz clic en la carpeta 📁 en el panel izquierdo")
    print("2. Busca los archivos generados")
    print("3. Haz clic derecho → Download")
    
    print(f"\\n🎉 ¡EVALUACIÓN COMPLETADA EXITOSAMENTE!")
    print("✅ Importa estos archivos en tu aplicación Streamlit local")
    
except Exception as e:
    print(f"\\n❌ Error durante la evaluación: {{e}}")
    import traceback
    traceback.print_exc()
    print("\\n💡 Revisa los errores y vuelve a ejecutar")

print("\\n" + "="*60)
print("🎯 PROCESO FINALIZADO")
print("="*60)
'''

    return code


def show_cached_results():
    """Muestra los resultados cacheados si existen."""
    cached_keys = [k for k in st.session_state.keys() if 'cumulative_results' in k]
    
    if not cached_keys:
        st.info("ℹ️ No hay resultados de evaluación. Ejecuta una evaluación para ver los resultados.")
        return
    
    # Seleccionar qué resultados mostrar
    if len(cached_keys) > 1:
        st.subheader("📋 Resultados Disponibles")
        selected_key = st.selectbox(
            "Selecciona resultados:",
            cached_keys,
            format_func=lambda x: f"Evaluación {x.split('_')[-1]} ({st.session_state[x]['params']['embedding_model_name']})"
        )
    else:
        selected_key = cached_keys[0]
    
    if selected_key not in st.session_state:
        return
    
    cached_results = st.session_state[selected_key]
    results = cached_results['results']
    evaluate_all_models = cached_results['evaluate_all_models']
    params = cached_results['params']
    
    st.markdown("---")
    
    # Mostrar resultados
    if evaluate_all_models:
        st.subheader("📈 Comparación Multi-Modelo")
        display_models_comparison(results, params['use_llm_reranker'])
    else:
        st.subheader(f"📊 Resultados para {params['embedding_model_name']}")
        model_name = list(results.keys())[0]
        display_cumulative_metrics(results[model_name], model_name, params['use_llm_reranker'])
    
    # Sección de descarga
    st.markdown("---")
    display_download_section(cached_results)


def show_memory_stats():
    """Muestra estadísticas detalladas de memoria."""
    current_memory = get_memory_usage()
    
    st.subheader("📊 Estadísticas de Memoria")
    st.metric("Memoria Actual", f"{current_memory:.1f} MB")
    
    # Contar elementos en caché
    cache_count = len([k for k in st.session_state.keys() if 'cumulative_results' in k])
    st.metric("Resultados en Caché", cache_count)
    
    if cache_count > 0:
        st.info("💡 Usa 'Limpiar Caché' para liberar memoria")


if __name__ == "__main__":
    show_cumulative_metrics_page()