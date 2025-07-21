"""
Página de Visualización de Resultados de Métricas Acumulativas
Permite ver y analizar resultados de evaluaciones completadas en Google Colab
"""

import streamlit as st
import json
import os
from typing import List, Dict, Any

# Importar utilidades
from src.ui.enhanced_metrics_display import display_enhanced_cumulative_metrics, display_enhanced_models_comparison
from src.data.file_utils import display_download_section
from src.evaluation.metrics import validate_data_integrity
from src.services.storage.real_gdrive_integration import (
    show_gdrive_status, check_evaluation_status_in_drive,
    show_gdrive_authentication_instructions, show_gdrive_debug_info,
    get_all_results_files_from_drive, get_specific_results_file_from_drive
)
from src.apps.cumulative_metrics_page import display_current_colab_status


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
    """Muestra la sección de resultados disponibles desde Google Drive"""
    
    st.subheader("📂 Resultados Disponibles")
    
    # Obtener archivos de resultados
    with st.spinner("🔍 Buscando archivos de resultados..."):
        files_result = get_all_results_files_from_drive()
    
    if files_result['success'] and files_result['files']:
        files = files_result['files']
        st.success(f"✅ Encontrados {len(files)} archivos de resultados")
        
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
                
                # Formatear la opción de display
                if file_size:
                    size_mb = int(file_size) / (1024 * 1024)
                    size_str = f"{size_mb:.1f} MB"
                else:
                    size_str = "N/A"
                
                # Formatear fecha
                if modified_time:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(modified_time.replace('Z', '+00:00'))
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = modified_time[:16]
                else:
                    date_str = "N/A"
                
                display_name = f"📄 {file_name} | {size_str} | {date_str}"
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
                st.write(f"📄 **Nombre:** {selected_file['file_name']}")
                if 'size' in selected_file:
                    size_mb = int(selected_file['size']) / (1024 * 1024)
                    st.write(f"📏 **Tamaño:** {size_mb:.1f} MB")
                if 'modifiedTime' in selected_file:
                    st.write(f"📅 **Modificado:** {selected_file['modifiedTime'][:16]}")
                if 'webViewLink' in selected_file:
                    st.markdown(f"[🔗 Ver en Drive]({selected_file['webViewLink']})")
        
        # Botón para mostrar resultados
        if st.button("📊 Mostrar Resultados", type="primary"):
            if selected_file:
                show_selected_results(selected_file)
            else:
                st.error("❌ Por favor selecciona un archivo de resultados")
                
    elif files_result['success'] and not files_result['files']:
        st.warning("⚠️ No se encontraron archivos de resultados en Google Drive")
        st.markdown("""
        **💡 Para generar resultados:**
        1. Ve a la página de **Crear Configuración**
        2. Configura y envía una evaluación a Google Drive
        3. Ejecuta el notebook en Google Colab
        4. Los resultados aparecerán aquí automáticamente
        """)
        
        if st.button("⚙️ Ir a Crear Configuración"):
            st.switch_page("src/apps/cumulative_metrics_create.py")
            
    else:
        st.error(f"❌ Error accediendo a archivos: {files_result.get('error', 'Error desconocido')}")
        st.markdown("**🔧 Posibles soluciones:**")
        st.markdown("- Verifica la autenticación con Google Drive")
        st.markdown("- Revisa que la carpeta `acumulative` existe")
        st.markdown("- Asegúrate de tener permisos de lectura")


def show_selected_results(selected_file: Dict):
    """Muestra los resultados del archivo seleccionado"""
    
    st.markdown("---")
    st.subheader(f"📊 Resultados: {selected_file['file_name']}")
    
    # Descargar y procesar archivo
    with st.spinner("📥 Descargando y procesando resultados..."):
        try:
            # Obtener contenido del archivo
            file_result = get_specific_results_file_from_drive(selected_file['file_id'])
            
            if not file_result['success']:
                st.error(f"❌ Error descargando archivo: {file_result.get('error', 'Error desconocido')}")
                return
            
            # Parsear JSON
            try:
                results_data = file_result['results'] # This should be file_result['data'] from download_json_from_drive
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
                st.warning("⚠️ Reranking LLM estaba habilitado pero estos resultados usan formato antiguo. Mostrando solo m��tricas base.")
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


def check_colab_evaluation_status():
    """Verifica el estado de la evaluación en Google Drive"""
    
    with st.spinner("🔍 Verificando estado de evaluación..."):
        status_result = check_evaluation_status_in_drive()
        
        if status_result['success']:
            status_data = status_result['data'] # Use 'data' key from the check_evaluation_status_in_drive output
            
            st.markdown("### 📊 Estado de Evaluación")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔄 Estado", status_data.get('status', 'Desconocido'))
                st.metric("📅 Última actualización", status_data.get('timestamp', 'N/A')[:19] if status_data.get('timestamp') else 'N/A')
            
            with col2:
                st.metric("🤖 Modelos", status_data.get('models_to_evaluate', 'N/A'))
                st.metric("❓ Preguntas", status_data.get('questions_total', 'N/A'))
            
            with col3:
                st.metric("🚀 GPU", "✅" if status_data.get('gpu_used') else "❌")
                st.metric("⏱️ Tiempo", f"{status_data.get('total_time_seconds', 0):.1f}s" if status_data.get('total_time_seconds') else 'N/A')
            
            # Mostrar detalles adicionales
            if status_data.get('status') == 'completed':
                st.success("🎉 ¡Evaluación completada! Los resultados deberían estar disponibles arriba.")
                
                if st.button("🔄 Actualizar Lista de Resultados"):
                    st.rerun()
                    
            elif status_data.get('status') == 'running':
                st.info("⏳ Evaluación en progreso... Verifica nuevamente en unos minutos.")
                
                if st.button("🔄 Verificar Nuevamente"):
                    st.rerun()
                
            elif status_data.get('status') == 'error':
                st.error("❌ Error en la evaluación. Revisa los logs de Colab.")
                
        else:
            st.warning("⚠️ No se pudo verificar el estado. Asegúrate de que Google Drive esté configurado correctamente.")


if __name__ == "__main__":
    show_cumulative_metrics_results_page()