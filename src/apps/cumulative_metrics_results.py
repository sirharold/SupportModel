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
from src.ui.enhanced_metrics_display import display_enhanced_cumulative_metrics, display_enhanced_models_comparison, generate_analysis_with_llm
from src.data.file_utils import display_download_section
from src.evaluation.metrics import validate_data_integrity
from src.services.storage.real_gdrive_integration import (
    show_gdrive_status, check_evaluation_status_in_drive,
    show_gdrive_authentication_instructions, show_gdrive_debug_info,
    get_all_results_files_from_drive, get_specific_results_file_from_drive
)
from src.apps.cumulative_metrics_page import display_current_colab_status


def get_local_results_files():
    """Busca archivos de resultados locales como fallback."""
    local_results_folders = [
        ".",  # Directorio raíz del proyecto (donde están los archivos reales)
        "simulated_drive/results",
        "results"
    ]
    
    found_files = []
    
    for folder in local_results_folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(folder_path):
            try:
                files = os.listdir(folder_path)
                for file in files:
                    if file.startswith('cumulative_results_') and file.endswith('.json'):
                        file_path = os.path.join(folder_path, file)
                        file_stats = os.stat(file_path)
                        
                        found_files.append({
                            'file_id': file_path,
                            'file_name': file,
                            'modified_time': time.ctime(file_stats.st_mtime),
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
    
    st.title("📊 Métricas Acumulativas - Resultados")
    st.markdown("""
    Esta página permite visualizar y analizar los resultados de evaluaciones completadas en Google Colab.
    """)
    
    # Sección principal de resultados
    st.markdown("---")
    show_available_results_section()


def show_available_results_section():
    """Muestra la sección de resultados disponibles desde Google Drive y archivos locales"""
    
    st.subheader("📂 Resultados Disponibles")
    
    # Intentar obtener archivos de Google Drive primero
    with st.spinner("🔍 Buscando archivos de resultados en Google Drive..."):
        files_result = get_all_results_files_from_drive()
    
    files = []
    gdrive_success = files_result['success'] and files_result.get('files')
    
    if gdrive_success:
        files = files_result['files']
        st.success(f"✅ Encontrados {len(files)} archivos de resultados en Google Drive")
    else:
        # Fallback a archivos locales
        st.warning("⚠️ No se pudieron obtener archivos de Google Drive, buscando archivos locales...")
        with st.spinner("🔍 Buscando archivos de resultados locales..."):
            local_files = get_local_results_files()
        
        if local_files:
            files = local_files
            st.info(f"📁 Encontrados {len(files)} archivos de resultados locales")
        else:
            st.error("❌ No se encontraron archivos de resultados ni en Google Drive ni localmente")
            
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
                    size_mb = int(file_size) / (1024 * 1024)
                    size_str = f"{size_mb:.1f} MB"
                else:
                    size_str = "N/A"
                
                # Formatear fecha
                if modified_time:
                    if is_local:
                        # Para archivos locales, modified_time es ya formateado por time.ctime()
                        date_str = modified_time[:16]
                    else:
                        # Para archivos de Google Drive
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(modified_time.replace('Z', '+00:00'))
                            date_str = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            date_str = modified_time[:16]
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
                    if is_local:
                        size_mb = int(selected_file['size']) / (1024 * 1024)
                        st.write(f"📏 **Tamaño:** {size_mb:.1f} MB")
                    else:
                        size_mb = int(selected_file['size']) / (1024 * 1024)
                        st.write(f"📏 **Tamaño:** {size_mb:.1f} MB")
                        
                if 'modified_time' in selected_file:
                    st.write(f"📅 **Modificado:** {selected_file['modified_time']}")
                elif 'modifiedTime' in selected_file:
                    st.write(f"📅 **Modificado:** {selected_file['modifiedTime'][:16]}")
                    
                if not is_local and 'webViewLink' in selected_file:
                    st.markdown(f"[🔗 Ver en Drive]({selected_file['webViewLink']})")
        
        # Opción para generar conclusiones con ChatGPT
        generate_llm = st.checkbox(
            "🤖 Generar conclusiones con ChatGPT",
            value=False,
            help="Si se marca, se generarán automáticamente las conclusiones y"
                 " posibles mejoras usando un modelo LLM"
        )

        # Botón para mostrar resultados
        if st.button("📊 Mostrar Resultados", type="primary"):
            if selected_file:
                show_selected_results(selected_file, generate_llm)
            else:
                st.error("❌ Por favor selecciona un archivo de resultados")
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
        generate_llm_analysis: Si True, se generarán conclusiones con un modelo LLM.
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
                    'generative_model_name', 'gpt-4'
                )
                llm_analysis = generate_analysis_with_llm(
                    results_data, generative_model_name
                )
                st.session_state.llm_conclusions = llm_analysis['conclusions']
                st.session_state.llm_improvements = llm_analysis['improvements']
                st.success("✅ Análisis generado por LLM.")
            else:
                st.session_state.llm_conclusions = ""
                st.session_state.llm_improvements = ""

            # Mostrar visualizaciones
            display_results_visualizations(results_data, processed_results)
            
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
        st.metric("🤖 LLM Reranking", "✅" if config.get('use_llm_reranker') else "❌")
    
    with col3:
        st.metric("🚀 GPU Usada", "✅" if eval_info.get('gpu_used') else "❌")
        total_time = eval_info.get('total_time_seconds', 0)
        st.metric("⏱️ Tiempo Total", f"{total_time:.1f}s" if total_time else "N/A")
    
    with col4:
        st.metric("📊 Tipo", eval_info.get('evaluation_type', 'N/A'))
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


def display_results_visualizations(results_data: Dict, processed_results: Dict):
    """Muestra las visualizaciones de resultados usando enhanced_metrics_display"""
    
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
                'individual_before_metrics': model_results.get('individual_before_metrics', []),
                'individual_after_metrics': model_results.get('individual_after_metrics', [])
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
                'individual_after_metrics': []
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
            # Verificar si usa el nuevo formato con before/after metrics
            if 'avg_before_metrics' in model_data and 'avg_after_metrics' in model_data:
                # Formato actualizado
                adapted_multi_results[model_name] = {
                    'num_questions_evaluated': model_data.get('num_questions_evaluated', results_data['config']['num_questions']),
                    'avg_before_metrics': model_data['avg_before_metrics'],
                    'avg_after_metrics': model_data['avg_after_metrics'],
                    'individual_before_metrics': model_data.get('individual_before_metrics', []),
                    'individual_after_metrics': model_data.get('individual_after_metrics', [])
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
                    'individual_after_metrics': []
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
    st.subheader("📝 Conclusiones")
    
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

    st.subheader("💡 Posibles Mejoras y Próximos Pasos")
    if 'llm_improvements' in st.session_state and st.session_state.llm_improvements:
        st.markdown(st.session_state.llm_improvements)
    else:
        st.markdown("""
        Para optimizar aún más el sistema RAG y la evaluación:
        - **Análisis de Errores por Pregunta:** Implementar una sección para revisar preguntas individuales donde los modelos tuvieron bajo rendimiento. Esto podría revelar patrones en tipos de preguntas difíciles o problemas en los documentos fuente.
        - **Análisis de Latencia:** Si los datos de tiempo de respuesta por pregunta/modelo están disponibles, visualizarlos para identificar cuellos de botella, especialmente con el reranking LLM.
        - **Diversidad de Contexto:** Evaluar la diversidad de los documentos recuperados para asegurar que no se están obteniendo documentos redundantes o muy similares.
        - **Evaluación Humana (Human-in-the-Loop):** Integrar un mecanismo para que evaluadores humanos revisen una muestra de respuestas generadas y proporcionen feedback cualitativo, especialmente para métricas subjetivas como `answer_relevance` y `answer_correctness`.
        - **Optimización de Modelos:** Experimentar con diferentes modelos de embedding o configuraciones de LLM para el reranking y la generación de respuestas.
        - **Robustez del Reranker:** Analizar el impacto del reranker en casos donde la recuperación inicial es muy pobre. ¿Puede el reranker recuperar una mala recuperación inicial?
        - **Visualización de Distribución de Scores:** Añadir histogramas o box plots para ver la distribución de las métricas individuales (no solo promedios) para cada modelo, lo que daría una idea de la consistencia del rendimiento.
        """)

    # Sección de descarga (moved to the end)
    st.markdown("---")
    # Prepare data for the download section in the expected format
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
    display_download_section(
        cached_results,
        llm_conclusions=st.session_state.get('llm_conclusions', ''),
        llm_improvements=st.session_state.get('llm_improvements', '')
    )


if __name__ == "__main__":
    show_cumulative_metrics_results_page()
