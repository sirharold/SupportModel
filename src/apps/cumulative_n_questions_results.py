"""
Página de Visualización de Resultados de Análisis Acumulativo N Preguntas
Permite ver y analizar resultados de evaluaciones completadas en Google Colab
"""

import streamlit as st
import json
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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


def get_local_n_questions_results_files():
    """Busca archivos de resultados de N preguntas locales como fallback."""
    import pytz
    from datetime import datetime
    
    local_results_folders = [
        ".",  # Directorio raíz del proyecto
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
                    # Buscar archivos específicos de N preguntas
                    if file.startswith('n_questions_results_') and file.endswith('.json'):
                        file_path = os.path.join(folder_path, file)
                        file_stats = os.stat(file_path)
                        
                        # Convertir timestamp UTC a hora de Chile
                        utc_time = datetime.utcfromtimestamp(file_stats.st_mtime)
                        utc_time = utc_time.replace(tzinfo=pytz.UTC)
                        chile_time = utc_time.astimezone(chile_tz)
                        
                        found_files.append({
                            'filename': file,
                            'full_path': file_path,
                            'last_modified': chile_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'size_mb': file_stats.st_size / (1024 * 1024),
                            'source': 'local'
                        })
                        
            except Exception as e:
                continue
    
    # Ordenar por fecha de modificación (más reciente primero)
    found_files.sort(key=lambda x: x['last_modified'], reverse=True)
    
    return found_files


def show_cumulative_n_questions_results_page():
    """Página principal para visualizar resultados de análisis acumulativo N preguntas."""
    
    st.title("📊 Resultados Análisis Acumulativo N Preguntas")
    st.markdown("""
    Esta página muestra los resultados de las evaluaciones de análisis acumulativo ejecutadas en Google Colab.
    """)
    
    # Estado de Google Drive
    show_gdrive_status()
    
    # Buscar archivos de resultados
    st.subheader("📁 Archivos de Resultados Disponibles")
    
    # Obtener archivos desde Google Drive
    gdrive_files = []
    try:
        with st.spinner("🔍 Buscando archivos en Google Drive..."):
            gdrive_result = get_all_results_files_from_drive()
            if gdrive_result['success']:
                # Filtrar solo los archivos de N preguntas
                gdrive_files = [
                    {
                        'filename': f['file_name'],
                        'full_path': f['file_id'],  # Use file_id as path for Google Drive
                        'last_modified': f['modified_time'],
                        'size_mb': float(f['size']) / (1024 * 1024) if f['size'] != 'N/A' and f['size'] else 0,
                        'source': 'gdrive'
                    }
                    for f in gdrive_result['files']
                    if (f['file_name'].startswith('n_questions_results_') or 
                        f['file_name'].startswith('cumulative_results_')) and 
                       f['file_name'].endswith('.json')
                ]
            else:
                st.warning(f"⚠️ No se pudieron obtener archivos de Google Drive: {gdrive_result.get('error', 'Error desconocido')}")
    except Exception as e:
        st.warning(f"⚠️ No se pudieron obtener archivos de Google Drive: {str(e)}")
    
    # Obtener archivos locales como fallback
    local_files = get_local_n_questions_results_files()
    
    # Combinar archivos
    all_files = gdrive_files + local_files
    
    if not all_files:
        st.warning("📭 No se encontraron archivos de resultados de análisis N preguntas.")
        st.info("""
        🎯 **Para generar resultados:**
        1. Ve a la página de configuración
        2. Crea una nueva configuración de evaluación
        3. Ejecuta el análisis en Google Colab
        4. Los resultados aparecerán aquí automáticamente
        """)
        return
    
    # Mostrar lista de archivos disponibles
    st.success(f"✅ Encontrados {len(all_files)} archivos de resultados")
    
    # Selector de archivo
    file_options = []
    file_mapping = {}
    
    for i, file_info in enumerate(all_files):
        display_name = f"📄 {file_info['filename']} ({file_info['last_modified']}) - {file_info['source'].upper()}"
        file_options.append(display_name)
        file_mapping[display_name] = file_info
    
    selected_file_display = st.selectbox(
        "Selecciona un archivo de resultados:",
        options=file_options,
        index=0,
        help="Archivos ordenados por fecha de modificación (más reciente primero)"
    )
    
    if selected_file_display:
        selected_file_info = file_mapping[selected_file_display]
        
        # Mostrar información del archivo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📅 Fecha", selected_file_info['last_modified'])
        with col2:
            st.metric("💾 Tamaño", f"{selected_file_info['size_mb']:.2f} MB")
        with col3:
            st.metric("🗂️ Fuente", selected_file_info['source'].upper())
        with col4:
            if st.button("🔄 Recargar Lista"):
                st.experimental_rerun()
        
        # Cargar y mostrar resultados
        st.markdown("---")
        load_and_display_n_questions_results(selected_file_info)


def load_and_display_n_questions_results(file_info: Dict):
    """Cargar y mostrar resultados de análisis N preguntas."""
    
    try:
        # Cargar datos
        with st.spinner(f"📥 Cargando resultados desde {file_info['source']}..."):
            if file_info['source'] == 'gdrive':
                gdrive_result = get_specific_results_file_from_drive(file_info['full_path'])  # full_path contains file_id for gdrive
                if gdrive_result['success']:
                    results_data = gdrive_result['results']
                else:
                    st.error(f"❌ Error cargando desde Google Drive: {gdrive_result.get('error', 'Error desconocido')}")
                    return
            else:
                with open(file_info['full_path'], 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
        
        if not results_data:
            st.error("❌ No se pudieron cargar los datos del archivo")
            return
        
        st.success(f"✅ Resultados cargados: {file_info['filename']}")
        
        # Validar estructura de datos
        if not validate_n_questions_results_structure(results_data):
            st.error("❌ La estructura de datos no es válida para análisis N preguntas")
            return
        
        # Mostrar resumen de ejecución
        display_execution_summary(results_data)
        
        # Mostrar métricas consolidadas
        display_n_questions_consolidated_metrics(results_data)
        
        # Mostrar visualizaciones
        display_n_questions_visualizations(results_data)
        
        # Mostrar resultados detallados
        display_n_questions_detailed_results(results_data)
        
        # Opciones de exportación
        display_n_questions_export_options(results_data, file_info)
        
    except Exception as e:
        st.error(f"❌ Error al cargar resultados: {str(e)}")
        st.exception(e)


def validate_n_questions_results_structure(data: Dict) -> bool:
    """Validar que los datos tienen la estructura esperada para análisis N preguntas."""
    
    required_keys = ['individual_results', 'consolidated_metrics', 'execution_stats']
    
    for key in required_keys:
        if key not in data:
            st.warning(f"⚠️ Clave faltante en datos: {key}")
            return False
    
    return True


def display_execution_summary(data: Dict):
    """Mostrar resumen de la ejecución."""
    
    st.subheader("📊 Resumen de Ejecución")
    
    stats = data.get('execution_stats', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Intentar obtener el conteo real de preguntas desde consolidated_metrics
        actual_count = 0
        consolidated = data.get('consolidated_metrics', {})
        if consolidated:
            # Obtener el count de cualquier modelo y métrica
            for model_data in consolidated.values():
                for metric_data in model_data.values():
                    if isinstance(metric_data, dict) and 'count' in metric_data:
                        actual_count = max(actual_count, metric_data['count'])
                        break
                if actual_count > 0:
                    break
        
        display_count = actual_count if actual_count > 0 else stats.get('questions_processed', 0)
        st.metric(
            "Preguntas Procesadas", 
            display_count
        )
        
        # Mostrar advertencia si hay discrepancia
        if actual_count > 0 and stats.get('questions_processed', 0) != actual_count:
            st.caption(f"⚠️ Discrepancia detectada: stats dice {stats.get('questions_processed', 0)}, métricas indican {actual_count}")
    
    with col2:
        st.metric(
            "Preguntas Fallidas", 
            stats.get('questions_failed', 0)
        )
    
    with col3:
        total_time = stats.get('total_time', 0)
        st.metric(
            "Tiempo Total", 
            f"{total_time:.1f}s"
        )
    
    with col4:
        processed = stats.get('questions_processed', 1)
        avg_time = total_time / max(1, processed)
        st.metric(
            "Promedio por Pregunta", 
            f"{avg_time:.1f}s"
        )
    
    # Mostrar configuración si está disponible
    if 'config' in data:
        with st.expander("⚙️ Configuración de la Evaluación"):
            config = data['config']
            
            col5, col6 = st.columns(2)
            
            with col5:
                st.write("**Configuración de datos:**")
                data_config = config.get('data_config', {})
                st.write(f"- Número de preguntas: {data_config.get('num_questions', 'N/A')}")
                st.write(f"- Top-K documentos: {data_config.get('top_k', 'N/A')}")
                st.write(f"- Reranking: {'✅' if data_config.get('use_reranking', False) else '❌'}")
            
            with col6:
                st.write("**Modelos evaluados:**")
                model_config = config.get('model_config', {})
                st.write(f"- Generativo: {model_config.get('generative_model', 'N/A')}")
                embedding_models = model_config.get('embedding_models', {})
                for short_name, full_name in embedding_models.items():
                    st.write(f"- {short_name.upper()}: {full_name}")


def display_n_questions_consolidated_metrics(data: Dict):
    """Mostrar métricas consolidadas."""
    
    st.subheader("📋 Métricas Consolidadas")
    
    consolidated = data.get('consolidated_metrics', {})
    
    if not consolidated:
        st.warning("⚠️ No hay métricas consolidadas disponibles")
        return
    
    # Detectar y advertir sobre el bug de consolidación
    has_consolidation_bug = False
    total_models_with_bug = 0
    
    for model, metrics in consolidated.items():
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                std_val = metric_data.get('std', 0)
                mean_val = metric_data.get('mean', 0)
                median_val = metric_data.get('median', 0)
                min_val = metric_data.get('min', 0)
                max_val = metric_data.get('max', 0)
                
                # Bug detectado si std=0 y todos los valores son iguales al mean
                if (std_val == 0.0 and 
                    median_val == mean_val and 
                    min_val == mean_val and 
                    max_val == mean_val and
                    mean_val > 0):  # Solo si mean > 0 para evitar falsos positivos
                    has_consolidation_bug = True
                    total_models_with_bug += 1
                    break
    
    if has_consolidation_bug:
        st.error(f"""
        🚨 **Bug de Consolidación Detectado** 
        
        {total_models_with_bug} modelo(s) muestran estadísticas incorrectas (std=0.0, todos los valores iguales al promedio).
        
        **Problema**: La lógica de consolidación antigua calculaba incorrectamente las estadísticas.
        
        **Solución**: 
        1. Usa la versión corregida del notebook Colab (`Cumulative_N_Questions_Colab_Fixed.ipynb`)
        2. Ejecuta una nueva evaluación para obtener estadísticas correctas
        3. Los nuevos resultados mostrarán variación real en std, median, min, max
        """)
        
        with st.expander("🔍 Detalles Técnicos del Bug"):
            st.write("""
            **¿Qué pasaba?** 
            - La consolidación antigua establecía std, median, min, max = mean para todas las métricas
            - Esto es matemáticamente imposible cuando hay variación en los datos
            
            **¿Cómo se arregló?**
            - Ahora se calculan estadísticas reales desde los resultados individuales de cada pregunta
            - std = desviación estándar real de los valores individuales  
            - median = mediana real de los valores individuales
            - min/max = valores mínimo/máximo reales
            """)
    else:
        st.success("✅ Estadísticas de consolidación correctas detectadas")
    
    # Crear tabla de métricas
    table_data = []
    
    for model, metrics in consolidated.items():
        if not metrics:
            continue
            
        row = {'Modelo': model.upper()}
        
        # Métricas principales - incluir todas las disponibles
        main_metrics = [
            ('jaccard_similarity', 'Jaccard'),
            ('ndcg_at_10', 'nDCG@10'),
            ('precision_at_5', 'Precision@5'),
            ('composite_score', 'Score Final'),
            ('avg_quality', 'Calidad LLM'),
            ('faithfulness', 'Faithfulness'),
            ('answer_relevance', 'Relevancia'),
            ('answer_correctness', 'Correctness'),
            ('answer_similarity', 'Similarity')
        ]
        
        for metric_key, metric_label in main_metrics:
            if metric_key in metrics:
                metric_data = metrics[metric_key]
                if isinstance(metric_data, dict):
                    mean_val = metric_data.get('mean', 0)
                    std_val = metric_data.get('std', 0)
                    count = metric_data.get('count', 0)
                    row[f'{metric_label} (μ±σ)'] = f"{mean_val:.3f}±{std_val:.3f}"
                else:
                    # Formato antiguo: solo valor simple
                    row[f'{metric_label}'] = f"{metric_data:.3f}"
        
        # Agregar conteo
        if 'composite_score' in metrics:
            row['N'] = metrics['composite_score'].get('count', 0)
        
        table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ No se pudieron procesar las métricas consolidadas")


def display_n_questions_visualizations(data: Dict):
    """Mostrar visualizaciones de los resultados."""
    
    st.subheader("📈 Visualizaciones")
    
    consolidated = data.get('consolidated_metrics', {})
    
    if not consolidated:
        st.warning("⚠️ No hay datos para visualizar")
        return
    
    models = list(consolidated.keys())
    
    # Gráfico de barras - Score Final
    st.markdown("#### Comparación de Score Final")
    
    mean_scores = []
    std_scores = []
    model_names = []
    
    for model in models:
        if 'composite_score' in consolidated[model]:
            mean_scores.append(consolidated[model]['composite_score']['mean'])
            std_scores.append(consolidated[model]['composite_score']['std'])
            model_names.append(model.upper())
    
    if mean_scores:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names,
            y=mean_scores,
            error_y=dict(type='data', array=std_scores),
            marker_color='lightblue',
            name='Score Final'
        ))
        
        fig.update_layout(
            title="Score Final por Modelo (Media ± Desviación Estándar)",
            xaxis_title="Modelo",
            yaxis_title="Score Final",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de radar - Métricas RAG
    st.markdown("#### Comparación de Métricas RAG")
    
    fig = go.Figure()
    
    rag_metrics = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
    rag_labels = ['Faithfulness', 'Relevancia', 'Correctness', 'Similarity']
    
    has_rag_data = False
    for model in models:
        if model in consolidated:
            values = []
            for metric in rag_metrics:
                if metric in consolidated[model]:
                    values.append(consolidated[model][metric]['mean'])
                else:
                    values.append(0)
            
            if any(v > 0 for v in values):  # Solo agregar si tiene datos
                has_rag_data = True
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=rag_labels,
                    fill='toself',
                    name=model.upper()
                ))
    
    if has_rag_data:
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparación de Métricas RAG por Modelo"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No hay métricas RAG disponibles en este archivo de resultados. Asegúrate de usar la versión más reciente del notebook Colab con evaluación RAG habilitada.")
    
    # Gráfico de distribución - Métricas tradicionales
    st.markdown("#### Métricas de Recuperación Tradicionales")
    
    traditional_metrics = ['jaccard_similarity', 'ndcg_at_10', 'precision_at_5']
    traditional_labels = ['Jaccard Similarity', 'nDCG@10', 'Precision@5']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=traditional_labels,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    for i, (metric, label) in enumerate(zip(traditional_metrics, traditional_labels)):
        model_names_metric = []
        means_metric = []
        stds_metric = []
        
        for model in models:
            if metric in consolidated[model]:
                model_names_metric.append(model.upper())
                means_metric.append(consolidated[model][metric]['mean'])
                stds_metric.append(consolidated[model][metric]['std'])
        
        if means_metric:
            fig.add_trace(
                go.Bar(
                    x=model_names_metric,
                    y=means_metric,
                    error_y=dict(type='data', array=stds_metric),
                    name=label,
                    showlegend=False
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title="Métricas de Recuperación Tradicionales por Modelo",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_n_questions_detailed_results(data: Dict):
    """Mostrar resultados detallados por pregunta."""
    
    st.subheader("🔍 Resultados Detallados")
    
    individual_results = data.get('individual_results', {})
    
    if not individual_results:
        st.warning("⚠️ No hay resultados individuales disponibles")
        return
    
    # Mostrar estadísticas básicas
    total_questions = len(individual_results)
    st.info(f"📊 Total de preguntas analizadas: {total_questions}")
    
    # Selector de pregunta para ver detalles
    question_ids = list(individual_results.keys())
    
    if question_ids:
        selected_question_id = st.selectbox(
            "Selecciona una pregunta para ver detalles:",
            options=question_ids,
            format_func=lambda x: f"Pregunta {x}: {individual_results[x]['question']['title'][:50]}..."
        )
        
        if selected_question_id:
            question_data = individual_results[selected_question_id]
            
            # Mostrar información de la pregunta
            with st.expander("📝 Información de la Pregunta", expanded=True):
                question = question_data['question']
                st.write(f"**Título:** {question['title']}")
                st.write(f"**Contenido:** {question['content']}")
                st.write(f"**Respuesta Aceptada:** {question['accepted_answer'][:200]}...")
            
            # Mostrar resultados por modelo
            results = question_data.get('results', {})
            
            if results:
                st.markdown("#### Resultados por Modelo")
                
                for model, model_results in results.items():
                    with st.expander(f"🤖 {model.upper()}", expanded=False):
                        metrics = model_results.get('metrics', {})
                        
                        if metrics:
                            col7, col8, col9 = st.columns(3)
                            
                            with col7:
                                st.write("**Métricas IR:**")
                                st.write(f"- Jaccard: {metrics.get('jaccard_similarity', 0):.3f}")
                                st.write(f"- nDCG@10: {metrics.get('ndcg_at_10', 0):.3f}")
                                st.write(f"- Precision@5: {metrics.get('precision_at_5', 0):.3f}")
                            
                            with col8:
                                st.write("**Métricas RAG:**")
                                st.write(f"- Faithfulness: {metrics.get('faithfulness', 0):.3f}")
                                st.write(f"- Relevancia: {metrics.get('answer_relevance', 0):.3f}")
                                st.write(f"- Correctness: {metrics.get('answer_correctness', 0):.3f}")
                                st.write(f"- Similarity: {metrics.get('answer_similarity', 0):.3f}")
                            
                            with col9:
                                st.write("**Calidad LLM:**")
                                st.write(f"- Pregunta: {metrics.get('question_quality', 0):.3f}")
                                st.write(f"- Respuesta: {metrics.get('answer_quality', 0):.3f}")
                                st.write(f"- Promedio: {metrics.get('avg_quality', 0):.3f}")
                                st.write(f"**Score Final: {metrics.get('composite_score', 0):.3f}**")
                        else:
                            st.warning(f"⚠️ No hay métricas disponibles para {model}")


def display_n_questions_export_options(data: Dict, file_info: Dict):
    """Mostrar opciones de exportación."""
    
    st.subheader("📥 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exportar Métricas Consolidadas"):
            consolidated_df = create_n_questions_consolidated_export_df(data)
            csv_consolidated = consolidated_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV Consolidado",
                data=csv_consolidated,
                file_name=f"n_questions_consolidated_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Exportar Resultados Detallados"):
            detailed_df = create_n_questions_detailed_export_df(data)
            csv_detailed = detailed_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV Detallado",
                data=csv_detailed,
                file_name=f"n_questions_detailed_{int(time.time())}.csv",
                mime="text/csv"
            )


def create_n_questions_consolidated_export_df(data: Dict) -> pd.DataFrame:
    """Crear DataFrame para exportación de métricas consolidadas."""
    
    consolidated = data.get('consolidated_metrics', {})
    export_data = []
    
    for model, metrics in consolidated.items():
        row = {'model': model}
        
        # Aplanar todas las métricas
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):
                row[f'{metric_name}_mean'] = stats.get('mean', 0)
                row[f'{metric_name}_std'] = stats.get('std', 0)
                row[f'{metric_name}_median'] = stats.get('median', 0)
                row[f'{metric_name}_min'] = stats.get('min', 0)
                row[f'{metric_name}_max'] = stats.get('max', 0)
                row[f'{metric_name}_count'] = stats.get('count', 0)
        
        export_data.append(row)
    
    return pd.DataFrame(export_data)


def create_n_questions_detailed_export_df(data: Dict) -> pd.DataFrame:
    """Crear DataFrame para exportación de resultados detallados."""
    
    individual_results = data.get('individual_results', {})
    export_data = []
    
    for question_id, result in individual_results.items():
        question = result['question']
        
        for model, model_results in result.get('results', {}).items():
            metrics = model_results.get('metrics', {})
            
            if metrics:
                row = {
                    'question_id': question_id,
                    'question_title': question['title'],
                    'model': model,
                    **metrics  # Incluir todas las métricas
                }
                
                export_data.append(row)
    
    return pd.DataFrame(export_data)


if __name__ == "__main__":
    show_cumulative_n_questions_results_page()