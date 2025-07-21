"""
Extensión de la página de comparación con métricas de recuperación detalladas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from utils.retrieval_metrics import format_metrics_for_display
from utils.clients import initialize_clients
from config import EMBEDDING_MODELS, MODEL_DESCRIPTIONS


def show_retrieval_metrics_comparison(
    question: str,
    selected_question: Dict,
    top_k: int = 10,
    use_reranker: bool = True
):
    """
    Muestra una comparación detallada de métricas de recuperación entre modelos.
    
    Args:
        question: Pregunta a procesar
        selected_question: Diccionario con información de la pregunta seleccionada
        top_k: Número de documentos a recuperar
        use_reranker: Si usar reranking
    """
    
    st.markdown("---")
    st.markdown("### 📊 Métricas de Recuperación Detalladas")
    st.info("🔍 **Evaluación Before/After Reranking**: Estas métricas muestran el impacto del reranking en la calidad de recuperación de documentos para cada modelo de embedding.")
    
    # Configuración de métricas
    with st.expander("⚙️ Configuración de Métricas", expanded=False):
        st.markdown("**Métricas Calculadas:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Métricas Principales:**
            - **MRR (Mean Reciprocal Rank)**: Posición del primer documento relevante
            - **Recall@k**: Fracción de documentos relevantes recuperados
            - **Precision@k**: Fracción de documentos recuperados que son relevantes
            - **F1@k**: Media armónica de Precision y Recall
            """)
        
        with col2:
            st.markdown("""
            **📈 Valores de k evaluados:**
            - k=1: Solo el primer documento
            - k=3: Top 3 documentos
            - k=5: Top 5 documentos  
            - k=10: Top 10 documentos
            """)
    
    # Botón para calcular métricas
    if st.button("📊 Calcular Métricas de Recuperación", type="primary", use_container_width=True):
        
        if not question:
            st.warning("No hay pregunta para evaluar.")
            return
        
        # Inicializar contenedor para resultados
        st.session_state.retrieval_metrics_results = {}
        
        # Calcular métricas para cada modelo
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_key in enumerate(EMBEDDING_MODELS.keys()):
            status_text.text(f"Calculando métricas para {model_key}...")
            
            try:
                # Inicializar cliente para este modelo
                chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(
                    model_key, 
                    st.session_state.get('generative_model_name', 'llama-4-scout')
                )
                
                # Ejecutar pipeline con métricas
                result = answer_question_with_retrieval_metrics(
                    question=question,
                    chromadb_wrapper=chromadb_wrapper,
                    embedding_client=embedding_client,
                    openai_client=openai_client,
                    gemini_client=gemini_client,
                    local_tinyllama_client=local_tinyllama_client,
                    local_mistral_client=local_mistral_client,
                    openrouter_client=openrouter_client,
                    gemini_client=gemini_client,
                    local_tinyllama_client=local_tinyllama_client,
                    local_mistral_client=local_mistral_client,
                    top_k=top_k,
                    use_llm_reranker=use_reranker,
                    generate_answer=False,  # Solo documentos para métricas
                    calculate_metrics=True,
                    ground_truth_answer=selected_question.get('accepted_answer', ''),
                    ms_links=selected_question.get('ms_links', []),
                    generative_model_name=st.session_state.get('generative_model_name', 'tinyllama-1.1b')
                )
                
                # Extraer métricas del resultado
                if len(result) >= 3:
                    docs, debug_info, retrieval_metrics = result
                    
                    # Añadir información del modelo
                    retrieval_metrics['model_key'] = model_key
                    retrieval_metrics['model_info'] = MODEL_DESCRIPTIONS.get(model_key, {})
                    
                    st.session_state.retrieval_metrics_results[model_key] = retrieval_metrics
                else:
                    st.session_state.retrieval_metrics_results[model_key] = {
                        'error': 'Resultado incompleto del pipeline'
                    }
                
            except Exception as e:
                st.session_state.retrieval_metrics_results[model_key] = {
                    'error': str(e)
                }
                st.error(f"Error calculando métricas para {model_key}: {e}")
            
            # Actualizar progreso
            progress_bar.progress((i + 1) / len(EMBEDDING_MODELS))
        
        # Limpiar indicadores de progreso
        status_text.text("✅ Cálculo completado!")
        progress_bar.empty()
        status_text.empty()
    
    # Mostrar resultados si existen
    if 'retrieval_metrics_results' in st.session_state:
        display_retrieval_metrics_results(st.session_state.retrieval_metrics_results)


def display_retrieval_metrics_results(results: Dict):
    """
    Muestra los resultados de las métricas de recuperación de manera organizada.
    
    Args:
        results: Diccionario con métricas por modelo
    """
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        st.error("No se pudieron calcular métricas válidas para ningún modelo.")
        return
    
    # 1. Tabla de resumen comparativo
    st.markdown("#### 📋 Resumen Comparativo de Métricas")
    
    # Crear DataFrame para comparación
    comparison_data = []
    for model_key, metrics in valid_results.items():
        if 'before_reranking' in metrics and 'after_reranking' in metrics:
            before = metrics['before_reranking']
            after = metrics['after_reranking']
            
            row = {
                'Modelo': model_key,
                'Ground Truth Links': metrics.get('ground_truth_links_count', 0),
                'Docs Before': metrics.get('docs_before_count', 0),
                'Docs After': metrics.get('docs_after_count', 0)
            }
            
            # Métricas principales
            for metric in ['MRR', 'Recall@1', 'Recall@5', 'Precision@1', 'Precision@5', 'F1@1', 'F1@5']:
                before_val = before.get(metric, 0)
                after_val = after.get(metric, 0)
                improvement = after_val - before_val
                
                row[f'{metric}_Before'] = before_val
                row[f'{metric}_After'] = after_val
                row[f'{metric}_Improvement'] = improvement
                row[f'{metric}_Pct_Improvement'] = (improvement / before_val * 100) if before_val > 0 else 0
            
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Mostrar métricas principales
        st.markdown("##### 🎯 Métricas Clave - Comparación Before/After")
        
        # Crear tabs para diferentes métricas
        tab1, tab2, tab3, tab4 = st.tabs(["📊 MRR", "🔍 Recall", "🎯 Precision", "⚖️ F1-Score"])
        
        with tab1:
            display_metric_comparison(comparison_df, 'MRR', 'Mean Reciprocal Rank')
        
        with tab2:
            display_metric_comparison(comparison_df, 'Recall@5', 'Recall@5')
        
        with tab3:
            display_metric_comparison(comparison_df, 'Precision@5', 'Precision@5')
        
        with tab4:
            display_metric_comparison(comparison_df, 'F1@5', 'F1@5')
    
    # 2. Gráficos detallados
    st.markdown("#### 📈 Análisis Visual de Métricas")
    
    # Crear gráficos comparativos
    create_retrieval_metrics_charts(valid_results)
    
    # 3. Detalles por modelo
    st.markdown("#### 🔍 Detalles por Modelo")
    
    for model_key, metrics in valid_results.items():
        with st.expander(f"📊 {model_key} - Métricas Detalladas", expanded=False):
            if 'before_reranking' in metrics and 'after_reranking' in metrics:
                # Formatear métricas para mostrar
                formatted_metrics = format_metrics_for_display(metrics)
                st.text(formatted_metrics)
            else:
                st.error(f"Métricas incompletas para {model_key}")


def display_metric_comparison(df: pd.DataFrame, metric: str, metric_name: str):
    """
    Muestra comparación de una métrica específica.
    
    Args:
        df: DataFrame con datos de comparación
        metric: Nombre de la métrica
        metric_name: Nombre display de la métrica
    """
    
    # Crear columnas para before/after
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{metric_name} - Before Reranking**")
        before_col = f'{metric}_Before'
        if before_col in df.columns:
            fig_before = px.bar(
                df, 
                x='Modelo', 
                y=before_col, 
                title=f'{metric_name} Before Reranking',
                color='Modelo',
                text=before_col
            )
            fig_before.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_before.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_before, use_container_width=True)
    
    with col2:
        st.markdown(f"**{metric_name} - After Reranking**")
        after_col = f'{metric}_After'
        if after_col in df.columns:
            fig_after = px.bar(
                df, 
                x='Modelo', 
                y=after_col, 
                title=f'{metric_name} After Reranking',
                color='Modelo',
                text=after_col
            )
            fig_after.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_after.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_after, use_container_width=True)
    
    # Mostrar mejora
    st.markdown(f"**{metric_name} - Mejora**")
    improvement_col = f'{metric}_Improvement'
    pct_improvement_col = f'{metric}_Pct_Improvement'
    
    if improvement_col in df.columns and pct_improvement_col in df.columns:
        # Crear gráfico de mejora
        fig_improvement = go.Figure()
        
        # Barras de mejora absoluta
        fig_improvement.add_trace(go.Bar(
            name='Mejora Absoluta',
            x=df['Modelo'],
            y=df[improvement_col],
            text=df[improvement_col].apply(lambda x: f'{x:.4f}'),
            textposition='auto',
            yaxis='y1',
            marker_color='green'
        ))
        
        # Línea de mejora porcentual
        fig_improvement.add_trace(go.Scatter(
            name='Mejora %',
            x=df['Modelo'],
            y=df[pct_improvement_col],
            mode='lines+markers+text',
            text=df[pct_improvement_col].apply(lambda x: f'{x:.1f}%'),
            textposition='top center',
            yaxis='y2',
            marker_color='red'
        ))
        
        # Configurar ejes duales
        fig_improvement.update_layout(
            title=f'{metric_name} - Mejora por Modelo',
            xaxis_title='Modelo',
            yaxis=dict(
                title='Mejora Absoluta',
                side='left'
            ),
            yaxis2=dict(
                title='Mejora Porcentual (%)',
                side='right',
                overlaying='y'
            ),
            height=500
        )
        
        st.plotly_chart(fig_improvement, use_container_width=True)
    
    # Tabla de resumen
    st.markdown(f"**{metric_name} - Tabla de Resumen**")
    summary_cols = ['Modelo', before_col, after_col, improvement_col, pct_improvement_col]
    summary_cols = [col for col in summary_cols if col in df.columns]
    
    if summary_cols:
        summary_df = df[summary_cols].copy()
        
        # Formatear columnas numéricas
        for col in summary_cols[1:]:  # Saltar 'Modelo'
            if col in summary_df.columns:
                if 'Pct' in col:
                    summary_df[col] = summary_df[col].apply(lambda x: f'{x:.2f}%')
                else:
                    summary_df[col] = summary_df[col].apply(lambda x: f'{x:.4f}')
        
        st.dataframe(summary_df, use_container_width=True)


def create_retrieval_metrics_charts(results: Dict):
    """
    Crea gráficos comparativos de métricas de recuperación.
    
    Args:
        results: Diccionario con métricas por modelo
    """
    
    # Preparar datos para gráficos
    chart_data = []
    
    for model_key, metrics in results.items():
        if 'before_reranking' in metrics and 'after_reranking' in metrics:
            before = metrics['before_reranking']
            after = metrics['after_reranking']
            
            # Datos para cada k
            for k in [1, 3, 5, 10]:
                for metric_type in ['Recall', 'Precision', 'F1']:
                    metric_key = f'{metric_type}@{k}'
                    
                    if metric_key in before and metric_key in after:
                        chart_data.append({
                            'Modelo': model_key,
                            'Métrica': metric_type,
                            'k': k,
                            'Before': before[metric_key],
                            'After': after[metric_key],
                            'Improvement': after[metric_key] - before[metric_key]
                        })
    
    if not chart_data:
        st.warning("No hay datos suficientes para crear gráficos.")
        return
    
    chart_df = pd.DataFrame(chart_data)
    
    # Gráfico 1: Heatmap de mejoras
    st.markdown("##### 🔥 Heatmap de Mejoras")
    
    # Crear pivot table para heatmap
    pivot_improvement = chart_df.pivot_table(
        values='Improvement', 
        index=['Modelo'], 
        columns=['Métrica', 'k'], 
        aggfunc='first'
    )
    
    if not pivot_improvement.empty:
        fig_heatmap = px.imshow(
            pivot_improvement.values,
            x=[f'{col[0]}@{col[1]}' for col in pivot_improvement.columns],
            y=pivot_improvement.index,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title='Mejora en Métricas por Modelo (After - Before)'
        )
        fig_heatmap.update_layout(
            xaxis_title='Métrica@k',
            yaxis_title='Modelo de Embedding',
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Gráfico 2: Comparación Before/After por métrica
    st.markdown("##### 📊 Comparación Before/After por Métrica")
    
    # Crear subplots para diferentes métricas
    metric_types = ['Recall', 'Precision', 'F1']
    
    fig_comparison = make_subplots(
        rows=1, cols=3,
        subplot_titles=metric_types,
        shared_yaxis=True
    )
    
    colors = ['blue', 'red', 'green']
    
    for i, metric_type in enumerate(metric_types):
        metric_data = chart_df[chart_df['Métrica'] == metric_type]
        
        if not metric_data.empty:
            # Before values
            fig_comparison.add_trace(
                go.Scatter(
                    x=metric_data['k'],
                    y=metric_data['Before'],
                    mode='lines+markers',
                    name=f'{metric_type} Before',
                    line=dict(color=colors[i], dash='dash'),
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
            
            # After values
            fig_comparison.add_trace(
                go.Scatter(
                    x=metric_data['k'],
                    y=metric_data['After'],
                    mode='lines+markers',
                    name=f'{metric_type} After',
                    line=dict(color=colors[i], dash='solid'),
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
    
    fig_comparison.update_layout(
        title='Evolución de Métricas por Valor de k',
        height=500
    )
    
    fig_comparison.update_xaxes(title_text='k (número de documentos)')
    fig_comparison.update_yaxes(title_text='Valor de métrica', col=1)
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Gráfico 3: Ranking de modelos por mejora
    st.markdown("##### 🏆 Ranking de Modelos por Mejora")
    
    # Calcular mejora promedio por modelo
    model_improvements = chart_df.groupby('Modelo')['Improvement'].agg(['mean', 'sum', 'count']).reset_index()
    model_improvements.columns = ['Modelo', 'Mejora_Promedio', 'Mejora_Total', 'Num_Metricas']
    model_improvements = model_improvements.sort_values('Mejora_Promedio', ascending=False)
    
    # Crear gráfico de barras
    fig_ranking = px.bar(
        model_improvements,
        x='Modelo',
        y='Mejora_Promedio',
        title='Mejora Promedio por Modelo (todas las métricas)',
        color='Mejora_Promedio',
        color_continuous_scale='RdYlGn',
        text='Mejora_Promedio'
    )
    
    fig_ranking.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_ranking.update_layout(height=400)
    
    st.plotly_chart(fig_ranking, use_container_width=True)


def export_retrieval_metrics_report(results: Dict) -> bytes:
    """
    Exporta un reporte PDF con las métricas de recuperación.
    
    Args:
        results: Diccionario con métricas por modelo
        
    Returns:
        PDF como bytes
    """
    # Esta función se implementaría similar a generate_pdf_report
    # pero enfocada en métricas de recuperación
    pass