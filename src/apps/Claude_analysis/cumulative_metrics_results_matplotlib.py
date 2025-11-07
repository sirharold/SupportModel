"""
Página de Visualización de Resultados de Métricas Acumulativas
Permite ver y analizar resultados de evaluaciones completadas en Google Colab
"""

import streamlit as st
import json
import os
import time
from typing import List, Dict, Any

# Importar utilidades
from src.ui.enhanced_metrics_display import display_enhanced_cumulative_metrics, display_enhanced_models_comparison, generate_analysis_with_llm, display_methodology_section
from src.data.file_utils import display_download_section
from src.evaluation.metrics import validate_data_integrity
from src.services.storage.real_gdrive_integration import (
    show_gdrive_status, check_evaluation_status_in_drive,
    show_gdrive_authentication_instructions, show_gdrive_debug_info,
    get_all_results_files_from_drive, get_specific_results_file_from_drive
)
from src.apps.Claude_analysis.cumulative_metrics_page import display_current_colab_status


def get_local_results_files():
    """Busca archivos de resultados locales como fallback."""
    import pytz
    from datetime import datetime
    
    local_results_folders = [
        ".",  # Directorio raíz del proyecto (donde están los archivos reales)
        "data",  # Carpeta data donde están los archivos de resultados
        "simulated_drive/results",
        "results"
    ]
    
    found_files = []
    
    # Timezone de Chile
    chile_tz = pytz.timezone('America/Santiago')
    
    for folder in local_results_folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(folder_path):
            try:
                files = os.listdir(folder_path)
                for file in files:
                    if file.startswith('cumulative_results_') and file.endswith('.json'):
                        file_path = os.path.join(folder_path, file)
                        file_stats = os.stat(file_path)
                        
                        # Convertir timestamp UTC a hora de Chile
                        utc_time = datetime.fromtimestamp(file_stats.st_mtime, tz=pytz.UTC)
                        chile_time = utc_time.astimezone(chile_tz)
                        
                        found_files.append({
                            'file_id': file_path,
                            'file_name': file,
                            'modified_time': chile_time.strftime('%Y-%m-%d %H:%M'),  # Formato hora Chile
                            'size': str(file_stats.st_size),
                            'source': 'local'
                        })
            except Exception as e:
                continue
    
    # Ordenar por fecha de modificación (más reciente primero)
    found_files.sort(key=lambda x: x['modified_time'], reverse=True)
    return found_files


def get_local_results_file_content(file_path):
    """Lee un archivo de resultados local."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {'success': True, 'results': data}
    except Exception as e:
        return {'success': False, 'error': f'Error leyendo archivo local: {str(e)}'}


def show_cumulative_metrics_results_page():
    """Página principal para visualizar resultados de métricas acumulativas."""
    
    st.title("📊 Métricas Acumulativas - Resultados Matplotlib")
    st.markdown("""
    Esta página permite visualizar y analizar los resultados de evaluaciones completadas en Google Colab.
    **📈 Versión con gráficos Matplotlib** - Incluye visualizaciones adicionales con matplotlib.
    """)
    
    # Sección principal de resultados
    st.markdown("---")
    show_available_results_section()


def show_available_results_section():
    """Muestra la sección de resultados disponibles desde Google Drive y archivos locales"""
    
    st.subheader("📂 Resultados Disponibles")
    
    # Intentar obtener archivos de Google Drive primero
    try:
        with st.spinner("🔍 Buscando archivos de resultados en Google Drive..."):
            files_result = get_all_results_files_from_drive()
    except Exception as e:
        st.warning(f"⚠️ Error al conectar con Google Drive: {str(e)}")
        files_result = {'success': False, 'error': str(e)}
    
    files = []
    gdrive_success = files_result['success'] and files_result.get('files')
    
    if gdrive_success:
        files = files_result['files']
        # Asegurar que todos los archivos de Google Drive tengan el campo 'source'
        for file in files:
            if 'source' not in file:
                file['source'] = 'gdrive'
        st.success(f"✅ Encontrados {len(files)} archivos de resultados en Google Drive")
    else:
        # Fallback a archivos locales
        st.warning("⚠️ No se pudieron obtener archivos de Google Drive, buscando archivos locales...")
        try:
            with st.spinner("🔍 Buscando archivos de resultados locales..."):
                local_files = get_local_results_files()
            
            if local_files:
                files = local_files
                st.info(f"📁 Encontrados {len(files)} archivos de resultados locales")
            else:
                st.error("❌ No se encontraron archivos de resultados ni en Google Drive ni localmente")
        except Exception as e:
            st.error(f"❌ Error buscando archivos locales: {str(e)}")
            st.exception(e)
            
    if files:
        # Mostrar archivos en formato mejorado
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**📋 Seleccionar archivo de resultados:**")
            
            # Crear opciones de display mejoradas
            file_options = []
            file_mapping = {}
            
            for file in files:
                file_name = file.get('file_name', 'N/A') # Use .get() for robustness
                file_size = file.get('size', 0)
                modified_time = file.get('modified_time', '') # Use .get() for robustness
                is_local = file.get('source') == 'local'
                
                # Formatear la opción de display
                if file_size:
                    try:
                        # Asegurar que file_size sea un número válido
                        size_bytes = int(str(file_size))  # Convertir a string primero, luego a int
                        size_mb = size_bytes / (1024 * 1024)
                        size_str = f"{size_mb:.1f} MB"
                    except (ValueError, TypeError):
                        size_str = "N/A"
                else:
                    size_str = "N/A"
                
                # Formatear fecha
                if modified_time:
                    if is_local:
                        # Para archivos locales, modified_time ya está en formato Chilean time
                        date_str = modified_time
                    else:
                        # Para archivos de Google Drive, modified_time ya viene convertido a Chilean time
                        date_str = modified_time
                else:
                    date_str = "N/A"
                
                # Añadir indicador de fuente
                source_emoji = "📁" if is_local else "☁️"
                display_name = f"{source_emoji} {file_name} | {size_str} | {date_str}"
                file_options.append(display_name)
                file_mapping[display_name] = file
            
            # Selector de archivo
            selected_file_display = st.selectbox(
                "Archivos disponibles:",
                options=file_options,
                index=0,
                help="Selecciona un archivo de resultados para visualizar"
            )
            
            selected_file = file_mapping[selected_file_display]
            
        with col2:
            st.markdown("**📊 Información del archivo:**")
            if selected_file:
                # Indicador de fuente
                is_local = selected_file.get('source') == 'local'
                source_emoji = "📁" if is_local else "☁️"
                source_text = "Local" if is_local else "Google Drive"
                st.write(f"{source_emoji} **Fuente:** {source_text}")
                
                st.write(f"📄 **Nombre:** {selected_file['file_name']}")
                if 'size' in selected_file:
                    try:
                        # Manejar tanto archivos locales como de Google Drive
                        size_bytes = int(str(selected_file['size']))
                        size_mb = size_bytes / (1024 * 1024)
                        st.write(f"📏 **Tamaño:** {size_mb:.1f} MB")
                    except (ValueError, TypeError):
                        st.write(f"📏 **Tamaño:** N/A")
                        
                if 'modified_time' in selected_file:
                    st.write(f"📅 **Modificado:** {selected_file['modified_time']}")
                elif 'modifiedTime' in selected_file:
                    st.write(f"📅 **Modificado:** {selected_file['modifiedTime'][:16]}")
                    
                if not is_local and 'webViewLink' in selected_file:
                    st.markdown(f"[🔗 Ver en Drive]({selected_file['webViewLink']})")
        
        # Opción para generar conclusiones con LLM
        generate_llm = st.checkbox(
            "🤖 Generar conclusiones con LLM (DeepSeek/Gemini)",
            value=False,
            key="generate_llm_analysis",
            help="Si se marca, se generarán automáticamente las conclusiones y"
                 " posibles mejoras usando DeepSeek R1 o Gemini 1.5 Flash"
        )

        # Botón para mostrar resultados
        if st.button("📊 Mostrar Resultados", type="primary"):
            if selected_file:
                # Store selected file and show results
                st.session_state.current_results_file = selected_file
                st.session_state.show_results = True
            else:
                st.error("❌ Por favor selecciona un archivo de resultados")
        
        # Show results if we have a file loaded (persists across checkbox changes)
        if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
            if hasattr(st.session_state, 'current_results_file'):
                show_selected_results(st.session_state.current_results_file, generate_llm)
    else:
        # No se encontraron archivos en ningún lado
        st.error("❌ No se encontraron archivos de resultados")
        st.markdown("""
        **💡 Para generar resultados:**
        1. Ve a la página de **Crear Configuración**
        2. Configura y envía una evaluación a Google Drive/Local
        3. Ejecuta el notebook en Google Colab o localmente
        4. Los resultados aparecerán aquí automáticamente
        """)
        
        if not gdrive_success:
            st.markdown("**🔧 Posibles problemas con Google Drive:**")
            st.markdown(f"- Error: {files_result.get('error', 'Error desconocido')}")
            st.markdown("- Verifica la autenticación con Google Drive")
            st.markdown("- Revisa que la carpeta `acumulative` existe")
            st.markdown("- Asegúrate de tener permisos de lectura")
        
        if st.button("⚙️ Ir a Crear Configuración"):
            st.switch_page("src/apps/cumulative_metrics_create.py")


def show_selected_results(selected_file: Dict, generate_llm_analysis: bool) -> None:
    """Muestra los resultados del archivo seleccionado.

    Args:
        selected_file: Información del archivo seleccionado en Google Drive.
        generate_llm_analysis: Si True, se generarán conclusiones con un modelo LLM y se mostrarán las secciones de conclusiones.
    """
    
    st.markdown("---")
    st.subheader(f"📊 Resultados: {selected_file['file_name']}")
    
    # Descargar y procesar archivo
    with st.spinner("📥 Descargando y procesando resultados..."):
        try:
            # Determinar si es archivo local o de Google Drive
            is_local = selected_file.get('source') == 'local'
            
            if is_local:
                # Obtener contenido del archivo local
                file_result = get_local_results_file_content(selected_file['file_id'])
            else:
                # Obtener contenido del archivo de Google Drive
                file_result = get_specific_results_file_from_drive(selected_file['file_id'])
            
            if not file_result['success']:
                st.error(f"❌ Error leyendo archivo: {file_result.get('error', 'Error desconocido')}")
                return
            
            # Parsear JSON
            try:
                results_data = file_result['results']
            except Exception as e:
                st.error(f"❌ Error parseando JSON: {e}")
                return

            # Corrected: The get_specific_results_file_from_drive returns a dict with 'results' key holding the data
            # So, results_data is already the content. No need to re-parse or access 'data' key here.
            # The previous comment was misleading. The structure is: {'success': True, 'results': actual_json_data}
            # So, results_data is already the actual_json_data. No change needed here for parsing.
            # The issue might be in the content of results_data itself, or how it's used downstream.
            
            # Validar estructura de datos
            if 'results' not in results_data:
                st.error("❌ Estructura de datos inválida: falta 'results'")
                return
            
            processed_results = results_data['results']
            
            # Mostrar información general
            display_results_summary(results_data, processed_results)

            # Generar conclusiones con LLM si se solicitó
            if generate_llm_analysis:
                generative_model_name = results_data.get('config', {}).get(
                    'generative_model_name', 'deepseek-v3-chat'  # Changed default to DeepSeek
                )
                
                # Ensure only supported models are used
                if generative_model_name not in ['deepseek-v3-chat', 'gemini-1.5-flash']:
                    st.warning(f"⚠️ Modelo {generative_model_name} no soportado. Usando DeepSeek por defecto.")
                    generative_model_name = 'deepseek-v3-chat'
                llm_analysis = generate_analysis_with_llm(
                    results_data, generative_model_name
                )
                
                if llm_analysis is not None:
                    model_used = llm_analysis.get('model_used', 'LLM')
                    st.session_state.llm_conclusions = llm_analysis['conclusions']
                    st.session_state.llm_improvements = llm_analysis['improvements']
                    st.session_state.llm_model_used = model_used
                    st.session_state.llm_analysis_prompt = llm_analysis.get('full_prompt', '')
                    st.success(f"✅ Análisis generado por {model_used}.")
                else:
                    st.error("❌ No se pudo generar análisis con ningún modelo LLM disponible.")
                    st.session_state.llm_conclusions = "Error: No se pudo generar análisis automático."
                    st.session_state.llm_improvements = "Verifica la configuración de API keys (DeepSeek vía OpenRouter o Gemini)."
                    st.session_state.llm_model_used = None
            else:
                st.session_state.llm_conclusions = ""
                st.session_state.llm_improvements = ""
                st.session_state.llm_model_used = None

            # Mostrar visualizaciones
            display_results_visualizations(results_data, processed_results, generate_llm_analysis)
            
        except Exception as e:
            st.error(f"❌ Error procesando resultados: {e}")
            st.exception(e)


def display_results_summary(results_data: Dict, processed_results: Dict):
    """Muestra resumen de los resultados"""
    
    st.subheader("📋 Resumen de Evaluación")
    
    # Información de configuración
    config = results_data.get('config', {})
    eval_info = results_data.get('evaluation_info', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Modelos", len(processed_results))
        st.metric("❓ Preguntas", config.get('num_questions', 'N/A'))
    
    with col2:
        st.metric("🔝 Top-K", config.get('top_k', 'N/A'))
        # Mostrar método de reranking desde config
        reranking_method = config.get('reranking_method', 'N/A')
        reranking_display = "✅ " + reranking_method if reranking_method != 'N/A' else "❌"
        st.metric("🔄 Reranking", reranking_display)
    
    with col3:
        st.metric("🚀 GPU Usada", "✅" if eval_info.get('gpu_used') else "❌")
        # Buscar tiempo total en diferentes campos posibles
        total_time = eval_info.get('total_duration_seconds', eval_info.get('total_time_seconds', 0))
        total_minutes = total_time / 60 if total_time else 0
        st.metric("⏱️ Tiempo Total", f"{total_minutes:.1f}m" if total_time else "N/A")
    
    with col4:
        eval_type = eval_info.get('evaluation_type', 'N/A')
        # Truncate long evaluation type names to fit better
        eval_type_short = eval_type[:15] + "..." if len(str(eval_type)) > 15 else eval_type
        st.metric("📊 Tipo", eval_type_short)
        st.metric("✅ Compatible", "✅" if eval_info.get('enhanced_display_compatible') else "⚠️")
    
    # Detalles de configuración
    with st.expander("⚙️ Ver Configuración Completa"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Configuración de Evaluación:**")
            st.json({k: v for k, v in config.items() if k != 'questions_data'})
        
        with col2:
            st.markdown("**📈 Información de Ejecución:**")
            st.json(eval_info)
    
    # Detalles de resultados - FULL JSON STRUCTURE
    with st.expander("📋 Ver Resultados Completos (Full JSON)"):
        st.markdown("**🔍 Archivo JSON Completo:**")
        st.markdown("*Esta es toda la estructura de datos del archivo de resultados, incluyendo configuración y métricas RAG*")
        
        # Show the complete results_data structure (not just processed_results)
        st.json(results_data)
        
        st.markdown("---")
        st.markdown("**📊 Análisis Rápido de Estructura RAG:**")
        
        # Quick analysis of RAG structure
        for model_name, model_data in processed_results.items():
            st.write(f"**Modelo {model_name}:**")
            st.write(f"- Claves disponibles: {list(model_data.keys())}")
            
            if 'rag_metrics' in model_data:
                rag_section = model_data['rag_metrics']
                st.write(f"- ✅ rag_metrics encontrada con claves: {list(rag_section.keys())}")
                
                # Show RAG averages if available
                rag_averages = {}
                for key in ['avg_faithfulness', 'avg_answer_relevance', 'avg_answer_correctness', 'avg_answer_similarity']:
                    if key in rag_section:
                        rag_averages[key] = rag_section[key]
                
                if rag_averages:
                    st.write(f"- 📊 Promedios RAG: {rag_averages}")
                else:
                    st.write(f"- ❌ No se encontraron promedios RAG")
            else:
                st.write(f"- ❌ No se encontró sección rag_metrics")
            
            if 'individual_rag_metrics' in model_data:
                individual_count = len(model_data['individual_rag_metrics'])
                st.write(f"- ✅ individual_rag_metrics: {individual_count} entradas")
                
                if individual_count > 0:
                    first_individual = model_data['individual_rag_metrics'][0]
                    individual_keys = list(first_individual.keys())
                    st.write(f"- 📝 Claves en primera entrada individual: {individual_keys}")
            else:
                st.write(f"- ❌ No se encontró individual_rag_metrics")
            
            st.write("")  # Empty line for spacing


def display_results_visualizations(results_data: Dict, processed_results: Dict, generate_llm_analysis: bool):
    """Muestra las visualizaciones de resultados usando enhanced_metrics_display
    
    Args:
        results_data: Datos completos de los resultados
        processed_results: Resultados procesados por modelo
        generate_llm_analysis: Si True, muestra las secciones de conclusiones y próximos pasos
    """
    
    st.markdown("---")
    
    # Determinar si usar reranker desde config
    # Priorizar reranking_method sobre use_llm_reranker para compatibilidad
    reranking_method = results_data['config'].get('reranking_method', 'crossencoder')
    use_llm_reranker = reranking_method in ['crossencoder', 'llm'] or results_data['config'].get('use_llm_reranker', True)
    
    if len(processed_results) == 1:
        # Para un solo modelo, usar display_enhanced_cumulative_metrics
        model_name = list(processed_results.keys())[0]
        model_results = processed_results[model_name]
        
        # Adaptar formato según la estructura de datos disponible
        # El formato actual tiene avg_before_metrics, avg_after_metrics, all_before_metrics, all_after_metrics
        if 'avg_before_metrics' in model_results and 'avg_after_metrics' in model_results:
            # Formato actual con before/after metrics
            adapted_results = {
                'num_questions_evaluated': model_results.get('num_questions_evaluated', results_data['config']['num_questions']),
                'avg_before_metrics': model_results['avg_before_metrics'],
                'avg_after_metrics': model_results['avg_after_metrics'],
                'individual_before_metrics': model_results.get('all_before_metrics', []),  # Usar all_before_metrics
                'individual_after_metrics': model_results.get('all_after_metrics', []),    # Usar all_after_metrics
                'rag_metrics': model_results.get('rag_metrics', {}),
                'individual_rag_metrics': model_results.get('individual_rag_metrics', [])
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
                'individual_before_metrics': model_results.get('individual_metrics', []),
                'individual_after_metrics': [],
                'rag_metrics': model_results.get('rag_metrics', {}),  # ✅ Add RAG metrics for legacy format too
                'individual_rag_metrics': model_results.get('individual_rag_metrics', [])  # ✅ Add individual RAG metrics
            }
            
            if use_llm_reranker:
                st.warning("⚠️ Reranking LLM estaba habilitado pero estos resultados usan formato antiguo. Mostrando solo m��tricas base.")
                use_llm_reranker = False
        
        # Use enhanced display with cleaner before/after LLM separation
        display_enhanced_cumulative_metrics(adapted_results, model_name, use_llm_reranker, results_data['config'])
        
    else:
        # Para múltiples modelos, usar display_enhanced_models_comparison
        st.markdown("### 🏆 Comparación de Modelos")
        
        # Adaptar formato para múltiples modelos según estructura disponible
        adapted_multi_results = {}
        has_new_format = False
        
        for model_name, model_data in processed_results.items():
            # Verificar si usa el formato actual con before/after metrics
            if 'avg_before_metrics' in model_data and 'avg_after_metrics' in model_data:
                # Formato actual
                adapted_multi_results[model_name] = {
                    'num_questions_evaluated': model_data.get('num_questions_evaluated', results_data['config']['num_questions']),
                    'avg_before_metrics': model_data['avg_before_metrics'],
                    'avg_after_metrics': model_data['avg_after_metrics'],
                    'individual_before_metrics': model_data.get('all_before_metrics', []),  # Usar all_before_metrics
                    'individual_after_metrics': model_data.get('all_after_metrics', []),    # Usar all_after_metrics
                    'rag_metrics': model_data.get('rag_metrics', {}),
                    'individual_rag_metrics': model_data.get('individual_rag_metrics', [])
                }
                has_new_format = True
            else:
                # Formato anterior - compatibilidad
                avg_metrics = model_data.get('avg_metrics', {})
                adapted_multi_results[model_name] = {
                    'num_questions_evaluated': results_data['config']['num_questions'],
                    'avg_before_metrics': avg_metrics,  # Usar métricas como "before"
                    'avg_after_metrics': {},  # Vacío porque no hay reranking
                    'individual_before_metrics': model_data.get('individual_metrics', []),
                    'individual_after_metrics': [],
                    'rag_metrics': model_data.get('rag_metrics', {}),  # ✅ Add RAG metrics for legacy format
                    'individual_rag_metrics': model_data.get('individual_rag_metrics', [])  # ✅ Add individual RAG metrics
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
        display_enhanced_models_comparison(adapted_multi_results, use_llm_reranker, results_data['config'])
    
    st.markdown("---")
    
    # Display methodology section before conclusions
    display_methodology_section()
    
    # Solo mostrar conclusiones y próximos pasos si el checkbox está marcado
    if generate_llm_analysis:
        st.subheader("📝 Conclusiones")
        
        # Add small text showing which model was used
        if 'llm_model_used' in st.session_state and st.session_state.llm_model_used:
            st.markdown(f"<small>(análisis usando {st.session_state.llm_model_used})</small>", unsafe_allow_html=True)
        
        # Check if LLM-generated conclusions are in session state
        if 'llm_conclusions' in st.session_state and st.session_state.llm_conclusions:
            st.markdown(st.session_state.llm_conclusions)
        else:
            st.markdown("""
            Basado en los resultados de la evaluación:
            - **Rendimiento General:** [Insertar conclusión sobre el rendimiento general de los modelos, e.g., qué modelos destacan, si el reranking LLM es efectivo, etc.]
            - **Impacto del Reranking LLM:** [Analizar si el reranking LLM consistentemente mejora las métricas de recuperación y RAG, o si hay casos donde no es beneficioso.]
            - **Métricas Clave:** [Comentar sobre los valores de métricas importantes como F1-Score, Faithfulness, Answer Relevance. ¿Son aceptables? ¿Hay modelos que sobresalen en ciertas métricas?]
            - **Comportamiento por K:** [Observaciones sobre cómo el rendimiento cambia a medida que K (número de documentos recuperados) varía.]
            """)

        # Show improvements if generated by LLM
        if 'llm_improvements' in st.session_state and st.session_state.llm_improvements:
            st.subheader("💡 Posibles Mejoras y Próximos Pasos")
            st.markdown(st.session_state.llm_improvements)
        
        # Show prompt details in an expandable section
        with st.expander("🔍 Ver Detalles del Prompt de Análisis"):
            st.markdown("**Prompt utilizado para generar el análisis académico:**")
            
            # Get the full prompt from the scientific data
            if 'llm_analysis_prompt' in st.session_state:
                full_prompt = st.session_state.llm_analysis_prompt
            else:
                # Create a sample prompt for display
                full_prompt = """You are a senior researcher specializing in information retrieval and RAG systems evaluation. 
Provide a rigorous academic analysis of the experimental results.

EXPERIMENTAL CONTEXT:
- Domain: Microsoft Learn technical documentation retrieval system
- Dataset: ~2,000 technical queries with ground-truth document links
- Corpus: 187,000 document chunks indexed at document level
- Evaluation: Standard IR metrics + semantic similarity assessment
- Reranking: CrossEncoder-based neural reranking applied

EXPECTED BASELINES FOR TECHNICAL DOCUMENTATION:
- Precision@5 > 0.60 indicates good relevance
- NDCG@10 > 0.70 suggests effective ranking
- MRR > 0.50 shows users find answers quickly
- CrossEncoder should improve NDCG/MRR by >10% relative
- RAGAS/BERTScore > 0.80 indicates high semantic quality

ANALYSIS REQUIREMENTS:
1. STATISTICAL SIGNIFICANCE: Focus on improvements >5% relative change
2. MODEL RANKING: Identify best/worst embeddings with quantitative justification
3. RERANKING ANALYSIS: Quantify CrossEncoder impact on ranking metrics
4. SEMANTIC ASSESSMENT: Interpret RAGAS/BERTScore values if available
5. ACADEMIC RIGOR: All claims must be supported by specific metrics

MANDATORY OUTPUT FORMAT IN SPANISH:
## Conclusiones
• [Modelo X logra Y rendimiento, significando Z de forma clara y práctica]
• [CrossEncoder mejora/empeora métrica M en N%, explicando por qué ocurre esto]
• [Comparación de embeddings con números específicos y interpretación simple]
• [Calidad semántica usando RAGAS/BERT explicado en términos entendibles]

## 💡 Mejoras Prioritarias
1. [Acción específica y práctica - explicar QUÉ hacer exactamente y POR QUÉ]
2. [Segunda mejora con pasos concretos - evitar jerga técnica compleja]
3. [Mejora a largo plazo explicada de forma simple y clara]

IMPORTANTE:
- Usa un lenguaje CLARO y DIRECTO, evita jerga técnica innecesaria
- Explica QUÉ significa cada número en términos prácticos
- Las mejoras deben ser ESPECÍFICAS y ACCIONABLES
- Si mencionas conceptos técnicos, explica brevemente qué significan

Write in Spanish with academic precision but clear explanations. Focus on actionable insights supported by data."""
            
            st.code(full_prompt, language="text")
            
            # Add copy to clipboard instructions
            st.markdown("**📋 Para copiar el prompt:**")
            st.info("💡 Selecciona todo el texto del cuadro de arriba con **Ctrl+A** y copia con **Ctrl+C**")
            
            # Additional help
            st.markdown("---")
            st.markdown("**💡 Tip:** También puedes usar este prompt en ChatGPT, Claude, o cualquier otro LLM para generar análisis similares con tus propios datos.")
    else:
        # Mostrar mensaje informativo cuando las conclusiones están ocultas
        st.info("💡 **Sugerencia:** Marca el checkbox '🤖 Generar conclusiones con LLM (DeepSeek/Gemini)' para ver las secciones de conclusiones y próximos pasos.")

    # Sección de descarga (moved to the end)
    st.markdown("---")
    # Prepare data for the download section in the expected format
    config = results_data.get('config', {})
    cached_results = {
        'results': processed_results,
        'evaluation_time': results_data.get('evaluation_info', {}).get('timestamp'),
        'execution_time': results_data.get('evaluation_info', {}).get('total_duration_seconds', 
                                             results_data.get('evaluation_info', {}).get('total_time_seconds')),
        'evaluate_all_models': len(processed_results) > 1,
        'params': {
            'num_questions': config.get('num_questions', 0),
            'selected_models': list(processed_results.keys()),
            'embedding_model_name': list(processed_results.keys())[0] if len(processed_results) == 1 else 'Multi-Model',
            'generative_model_name': config.get('generative_model_name', 'gpt-4'),  # Default if not present
            'top_k': config.get('top_k', 10),
            'use_llm_reranker': config.get('use_llm_reranker', config.get('reranking_method') == 'crossencoder'),
            'batch_size': config.get('batch_size', 32)  # Default if not present
        }
    }
    display_download_section(
        cached_results,
        llm_conclusions=st.session_state.get('llm_conclusions', ''),
        llm_improvements=st.session_state.get('llm_improvements', '')
    )


if __name__ == "__main__":
    show_cumulative_metrics_results_page()
