"""
Enhanced metrics display module for cleaner before/after LLM visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List


def display_enhanced_cumulative_metrics(results: Dict[str, Any], model_name: str, use_llm_reranker: bool, config: Dict = None):
    """
    Enhanced display for cumulative metrics with clear before/after LLM sections
    """
    num_questions = results['num_questions_evaluated']
    avg_before = results['avg_before_metrics']
    avg_after = results.get('avg_after_metrics', {})
    
    st.success(f"✅ Evaluación completada para {num_questions} preguntas con modelo {model_name}")
    
    # Main metrics overview
    display_main_metrics_overview(avg_before, avg_after, use_llm_reranker)
    
    # Before/After LLM comparison section
    if use_llm_reranker and avg_after:
        display_before_after_comparison(avg_before, avg_after)
    
    # Metrics by K values section
    display_metrics_by_k_values(avg_before, avg_after, use_llm_reranker)
    
    # Performance visualization
    display_performance_charts(avg_before, avg_after, use_llm_reranker, model_name)

    # New section: RAG Metrics Summary (for single model, adapt results structure)
    # Create a dummy results dict for display_rag_metrics_summary
    single_model_results_for_rag = {model_name: results}
    display_rag_metrics_summary(single_model_results_for_rag, use_llm_reranker, config)


def display_main_metrics_overview(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool):
    """Display main metrics overview with key performance indicators"""
    
    st.subheader("📊 Resumen de Métricas Principales")
    
    # Select key metrics to highlight (updated with new metrics)
    key_metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']
    
    if use_llm_reranker and avg_after:
        # Show before and after side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Antes del LLM Reranking")
            for metric in key_metrics:
                if metric in avg_before:
                    value = avg_before[metric]
                    st.metric(
                        label=metric.upper().replace('@', ' @ '),
                        value=f"{value:.3f}",
                        help=f"Valor promedio de {metric} usando solo embedding retrieval"
                    )
        
        with col2:
            st.markdown("#### 🤖 Después del LLM Reranking")
            for metric in key_metrics:
                if metric in avg_after:
                    after_value = avg_after[metric]
                    before_value = avg_before.get(metric, 0)
                    delta = after_value - before_value
                    
                    st.metric(
                        label=metric.upper().replace('@', ' @ '),
                        value=f"{after_value:.3f}",
                        delta=f"{delta:+.3f}",
                        help=f"Valor después del LLM reranking. Delta vs embedding-only: {delta:+.3f}"
                    )
    else:
        # Show only before metrics
        st.markdown("#### 📊 Métricas de Retrieval por Embeddings")
        
        cols = st.columns(3)
        for i, metric in enumerate(key_metrics):
            if metric in avg_before:
                with cols[i % 3]:
                    value = avg_before[metric]
                    quality = get_metric_quality(value)
                    st.metric(
                        label=metric.upper().replace('@', ' @ '),
                        value=f"{value:.3f}",
                        help=f"Calidad: {quality}"
                    )


def display_before_after_comparison(avg_before: Dict, avg_after: Dict):
    """Display dedicated before/after LLM comparison section"""
    
    st.subheader("🔄 Comparación: Antes vs Después del LLM Reranking")
    
    # Prepare comparison data
    comparison_data = []
    metrics_to_compare = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr', 'ndcg@5']  # MRR is a single value, not per-K
    
    for metric in metrics_to_compare:
        if metric in avg_before and metric in avg_after:
            before_val = avg_before[metric]
            after_val = avg_after[metric]
            improvement = after_val - before_val
            improvement_pct = (improvement / before_val * 100) if before_val > 0 else 0
            
            comparison_data.append({
                'Métrica': metric.upper().replace('@', ' @ '),
                'Antes LLM': f"{before_val:.3f}",
                'Después LLM': f"{after_val:.3f}",
                'Mejora Absoluta': f"{improvement:+.3f}",
                'Mejora %': f"{improvement_pct:+.1f}%",
                'Estado': get_improvement_status(improvement)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        def style_improvement(val):
            if 'Mejora' in val:
                if '+' in str(val):
                    return 'background-color: #c6f5c6'  # Light green
                elif '-' in str(val):
                    return 'background-color: #f7c6c6'  # Light red
            return ''
        
        styled_df = df.style.applymap(style_improvement, subset=['Mejora Absoluta', 'Mejora %'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary insights
        improvements = [float(row['Mejora Absoluta']) for row in comparison_data]
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        
        if positive_improvements > len(improvements) / 2:
            st.success(f"🎯 LLM Reranking mejoró {positive_improvements}/{len(improvements)} métricas")
        else:
            st.warning(f"⚠️ LLM Reranking solo mejoró {positive_improvements}/{len(improvements)} métricas")


def display_metrics_by_k_values(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool):
    """Display metrics organized by K values in 2x3 matrix format with charts and table"""
    
    st.subheader("📈 Métricas por Valores de K")
    
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # All metrics that are calculated in updated Colab notebook (v2.0)
    k_metrics = ['precision', 'recall', 'f1', 'ndcg', 'map']  # Metrics that have @k values
    single_metrics = ['mrr']  # Metrics that are single values (no @k)
    
    # Create 2x3 matrix of charts - one chart per metric type (X: K values, Y: metric values)
    st.markdown("#### 📊 Gráficos por Métrica (X: Valores de K, Y: Valor de la Métrica)")
    
    # First row: Precision, Recall, F1
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_across_k_chart(avg_before, avg_after, 'precision', k_values, use_llm_reranker)
    with col2:
        create_metric_across_k_chart(avg_before, avg_after, 'recall', k_values, use_llm_reranker)
    with col3:
        create_metric_across_k_chart(avg_before, avg_after, 'f1', k_values, use_llm_reranker)
    
    # Second row: MRR, NDCG, MAP
    col1, col2, col3 = st.columns(3)
    with col1:
        create_single_metric_chart(avg_before, avg_after, 'mrr', use_llm_reranker)
    with col2:
        create_metric_across_k_chart(avg_before, avg_after, 'ndcg', k_values, use_llm_reranker)
    with col3:
        create_metric_across_k_chart(avg_before, avg_after, 'map', k_values, use_llm_reranker)
    
    # Table with all metrics (now includes NDCG and MAP)
    st.markdown("#### 📋 Tabla Completa de Métricas")
    all_metric_types = k_metrics + single_metrics  # All metrics including NDCG and MAP
    display_complete_metrics_table(avg_before, avg_after, k_values, all_metric_types, use_llm_reranker)
    
    # Add metrics explanation accordion
    display_retrieval_metrics_explanation()


def display_k_metrics(metrics_dict: Dict, k: int, metric_types: List[str], before_metrics: Dict = None):
    """Display metrics for a specific K value"""
    
    for metric_type in metric_types:
        metric_key = f"{metric_type}@{k}"
        if metric_key in metrics_dict:
            value = metrics_dict[metric_key]
            
            if before_metrics and metric_key in before_metrics:
                # Show delta if we have before metrics
                before_value = before_metrics[metric_key]
                delta = value - before_value
                st.metric(
                    label=metric_type.upper(),
                    value=f"{value:.3f}",
                    delta=f"{delta:+.3f}"
                )
            else:
                # Show just the value
                quality = get_metric_quality(value)
                st.metric(
                    label=metric_type.upper(),
                    value=f"{value:.3f}",
                    help=f"Calidad: {quality}"
                )


def create_k_comparison_chart(avg_before: Dict, avg_after: Dict, k: int, metric_types: List[str]):
    """Create comparison chart for specific K value"""
    
    metrics_data = []
    for metric_type in metric_types:
        metric_key = f"{metric_type}@{k}"
        if metric_key in avg_before and metric_key in avg_after:
            metrics_data.append({
                'Métrica': metric_type.upper(),
                'Antes LLM': avg_before[metric_key],
                'Después LLM': avg_after[metric_key]
            })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Antes LLM',
            x=df['Métrica'],
            y=df['Antes LLM'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Después LLM',
            x=df['Métrica'],
            y=df['Después LLM'],
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title=f'Comparación de Métricas - Top-{k}',
            xaxis_title='Métricas',
            yaxis_title='Valor',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_k_comparison_chart(avg_before: Dict, avg_after: Dict, k: int, metric_types: List[str]):
    """Create comparison chart for specific K value"""
    
    metrics_data = []
    for metric_type in metric_types:
        metric_key = f"{metric_type}@{k}"
        if metric_key in avg_before and metric_key in avg_after:
            metrics_data.append({
                'Métrica': metric_type.upper(),
                'Antes LLM': avg_before[metric_key],
                'Después LLM': avg_after[metric_key]
            })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Antes LLM',
            x=df['Métrica'],
            y=df['Antes LLM'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Después LLM',
            x=df['Métrica'],
            y=df['Después LLM'],
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title=f'Comparación de Métricas - Top-{k}',
            xaxis_title='Métricas',
            yaxis_title='Valor',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_k_metrics_table(avg_before: Dict, avg_after: Dict, k: int, metric_types: List[str], use_llm_reranker: bool):
    """Displays a table with detailed metrics for a specific K value, before and after LLM.
    """
    st.markdown(f"##### Tabla de Datos - Top-{k}")
    table_data = []
    for metric_type in metric_types:
        metric_key = f"{metric_type}@{k}"
        row = {'Métrica': metric_type.upper()}
        
        if metric_key in avg_before:
            row['Antes LLM'] = f"{avg_before[metric_key]:.3f}"
        else:
            row['Antes LLM'] = 'N/A'

        if use_llm_reranker and metric_key in avg_after:
            row['Después LLM'] = f"{avg_after[metric_key]:.3f}"
            if metric_key in avg_before and avg_before[metric_key] > 0:
                improvement = avg_after[metric_key] - avg_before[metric_key]
                improvement_pct = (improvement / avg_before[metric_key]) * 100
                row['Mejora Absoluta'] = f"{improvement:+.3f}"
                row['Mejora %'] = f"{improvement_pct:+.1f}%"
            else:
                row['Mejora Absoluta'] = 'N/A'
                row['Mejora %'] = 'N/A'
        elif use_llm_reranker:
            row['Después LLM'] = 'N/A'
            row['Mejora Absoluta'] = 'N/A'
            row['Mejora %'] = 'N/A'

        table_data.append(row)

    if table_data:
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)


def display_performance_charts(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, model_name: str):
    """Display comprehensive performance visualization"""
    
    st.subheader("📈 Visualización de Rendimiento")
    
    # Performance across K values
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    main_metrics = ['f1'] # Focus on F1-score as the primary 'score'
    
    # Create a single subplot for the F1-score
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['F1-Score'],
        specs=[[{"secondary_y": False}]]
    )
    
    colors = ['blue', 'green'] if use_llm_reranker and avg_after else ['blue']
    names = ['Antes LLM', 'Después LLM'] if use_llm_reranker and avg_after else ['Retrieval']
    metric_dicts = [avg_before, avg_after] if use_llm_reranker and avg_after else [avg_before]
    
    for i, metric in enumerate(main_metrics):
        for j, (metrics_dict, color, name) in enumerate(zip(metric_dicts, colors, names)):
            if metrics_dict:  # Check if dict is not empty
                y_values = []
                for k in k_values:
                    metric_key = f"{metric}@{k}"
                    y_values.append(metrics_dict.get(metric_key, 0))
                
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=y_values,
                        mode='lines+markers',
                        name=f"{name}", # Simplified name for legend
                        line=dict(color=color),
                        showlegend=True # Always show legend for this single plot
                    ),
                    row=1, col=1 # Always target the first (and only) subplot
                )
    
    fig.update_layout(
        title=f'Rendimiento por Valores de K - {model_name}',
        height=400,
        showlegend=True
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Valor de K", row=1, col=1)
    fig.update_yaxes(title_text="F1-Score", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def display_enhanced_models_comparison(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, config: Dict = None):
    """Enhanced comparison visualization for multiple models"""
    
    st.subheader("🏆 Comparación Avanzada Entre Modelos")
    
    # Prepare data for visualization using average scores
    models_data = []
    for model_name, model_results in results.items():
        before_metrics = model_results['avg_before_metrics']
        after_metrics = model_results.get('avg_after_metrics', {})
        
        # Calculate average performance (similar to create_models_summary_table)
        all_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr', 'ndcg@5']  # MRR is a single value
        before_avg = np.mean([before_metrics.get(m, 0) for m in all_metrics if m in before_metrics])
        
        row_data = {
            'Modelo': model_name,
            'Promedio Antes LLM': before_avg,
        }
        
        if use_llm_reranker and after_metrics:
            after_avg = np.mean([after_metrics.get(m, 0) for m in all_metrics if m in after_metrics])
            row_data['Promedio Después LLM'] = after_avg
        
        models_data.append(row_data)
    
    if models_data:
        df = pd.DataFrame(models_data)
        
        # Create comparison visualization
        if use_llm_reranker and 'Promedio Después LLM' in df.columns and df['Promedio Después LLM'].notna().any():
            # Before vs After comparison
            fig = px.bar(
                df, 
                x='Modelo', 
                y=['Promedio Antes LLM', 'Promedio Después LLM'], 
                barmode='group',
                title='Comparación de Modelos: Promedio de Score (Antes vs Después LLM)',
                labels={'value': 'Promedio de Score', 'variable': 'Estado'}
            )
        else:
            # Only before metrics
            fig = px.bar(
                df,
                x='Modelo',
                y='Promedio Antes LLM',
                title='Comparación de Modelos: Promedio de Score (Solo Retrieval)',
                labels={'Promedio Antes LLM': 'Promedio de Score'}
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        create_models_summary_table(results, use_llm_reranker)

        # New section: All metrics by K for all models
        display_all_metrics_by_k_for_all_models(results, use_llm_reranker)

        # Table with data for metrics by K
        create_all_metrics_by_k_table(results, use_llm_reranker)

        # Add metric definitions table before RAG metrics
        display_metric_definitions_table()

        # New section: RAG Metrics Summary
        display_rag_metrics_summary(results, use_llm_reranker, config)


def display_all_metrics_by_k_for_all_models(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool):
    """
    Displays a 2x3 grid of plots, each showing a specific metric across K values
    for all models, with before/after LLM lines.
    """
    st.subheader("📈 Rendimiento Detallado por Métrica y K")

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics_to_plot = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']

    # Create subplots: 2 rows, 3 columns for 6 metrics
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m.upper() for m in metrics_to_plot],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    colors = px.colors.qualitative.Plotly # Use a qualitative color scale for models

    for i, metric in enumerate(metrics_to_plot):
        row = (i // 3) + 1
        col = (i % 3) + 1

        for model_idx, (model_name, model_results) in enumerate(results.items()):
            before_metrics = model_results['avg_before_metrics']
            after_metrics = model_results.get('avg_after_metrics', {})

            # Data for 'Before LLM'
            if metric == 'mrr':
                # MRR is a single value, not per K - show same value for all K points
                y_before = [before_metrics.get('mrr', 0)] * len(k_values)
            else:
                y_before = [before_metrics.get(f"{metric}@{k}", 0) for k in k_values]
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=y_before,
                    mode='lines+markers',
                    name=f"{model_name} (Antes LLM)",
                    line=dict(color=colors[model_idx % len(colors)], dash='solid'),
                    showlegend=(i == 0) # Only show legend for the first subplot to avoid clutter
                ),
                row=row, col=col
            )

            # Data for 'After LLM' if applicable
            if use_llm_reranker and after_metrics:
                if metric == 'mrr':
                    # MRR is a single value, not per K - show same value for all K points
                    y_after = [after_metrics.get('mrr', 0)] * len(k_values)
                else:
                    y_after = [after_metrics.get(f"{metric}@{k}", 0) for k in k_values]
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=y_after,
                        mode='lines+markers',
                        name=f"{model_name} (Después LLM)",
                        line=dict(color=colors[model_idx % len(colors)], dash='dot'),
                        showlegend=(i == 0) # Only show legend for the first subplot
                    ),
                    row=row, col=col
                )
        
        # Update axes for each subplot
        fig.update_xaxes(title_text="Valor de K", row=row, col=col)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=row, col=col) # Ensure y-axis is 0-1 for scores

    fig.update_layout(
        title_text='Rendimiento de Modelos por Métrica y Valor de K',
        height=800, # Adjust height for 2 rows
        showlegend=True,
        legend_title_text="Modelo (Estado LLM)"
    )

    st.plotly_chart(fig, use_container_width=True)


def create_all_metrics_by_k_table(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool):
    """
    Displays a comprehensive table showing all models' metrics across K values,
    with before/after LLM scores.
    """
    st.subheader("📋 Tabla Detallada de Métricas por K (Todos los Modelos)")

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics_to_display = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']

    table_data = []

    for model_name, model_results in results.items():
        before_metrics = model_results['avg_before_metrics']
        after_metrics = model_results.get('avg_after_metrics', {})

        for metric_type in metrics_to_display:
            row_base = {'Modelo': model_name, 'Métrica': metric_type.upper()}

            if metric_type == 'mrr':
                # MRR is a single value, not per K - only process once
                before_val = before_metrics.get('mrr')
                
                # DEBUG: Print MRR value for debugging
                # st.write(f"DEBUG - {model_name} MRR before: {before_val} (type: {type(before_val)})")
                
                # Handle None/NaN values properly
                if before_val is not None and not np.isnan(before_val):
                    row_base['Antes LLM'] = f"{before_val:.3f}"
                else:
                    row_base['Antes LLM'] = '-'
                
                # Add empty columns for K values to match table structure
                for k in k_values:
                    row_base[f'Antes LLM @{k}'] = '-'  # MRR doesn't vary by K
                    row_base[f'Después LLM @{k}'] = '-'
                    row_base[f'Mejora @{k}'] = '-'
                
                # After LLM score and improvement for MRR
                if use_llm_reranker and after_metrics and 'mrr' in after_metrics:
                    after_val = after_metrics.get('mrr')
                    if after_val is not None and not np.isnan(after_val):
                        row_base['Después LLM'] = f"{after_val:.3f}"
                        
                        if before_val is not None and not np.isnan(before_val) and before_val > 0:
                            improvement = after_val - before_val
                            improvement_pct = (improvement / before_val) * 100
                            row_base['Mejora'] = f"{improvement:+.3f} ({improvement_pct:+.1f}%) " + get_improvement_status_icon(improvement)
                        else:
                            row_base['Mejora'] = '-'
                    else:
                        row_base['Después LLM'] = '-'
                        row_base['Mejora'] = '-'
                elif use_llm_reranker:
                    row_base['Después LLM'] = '-'
                    row_base['Mejora'] = '-'
            else:
                # Handle metrics that vary by K
                # Add empty overall columns for K-based metrics
                row_base['Antes LLM'] = '-'  # K-based metrics don't have overall values
                row_base['Después LLM'] = '-'
                row_base['Mejora'] = '-'
                
                for k in k_values:
                    metric_key = f"{metric_type}@{k}"
                    
                    # Before LLM score
                    before_val = before_metrics.get(metric_key, np.nan)
                    row_base[f'Antes LLM @{k}'] = f"{before_val:.3f}" if not np.isnan(before_val) else '-'

                    # After LLM score and improvement
                    if use_llm_reranker and after_metrics and metric_key in after_metrics:
                        after_val = after_metrics.get(metric_key, np.nan)
                        row_base[f'Después LLM @{k}'] = f"{after_val:.3f}" if not np.isnan(after_val) else '-'
                        
                        if not np.isnan(before_val) and before_val > 0:
                            improvement = after_val - before_val
                            improvement_pct = (improvement / before_val) * 100
                            row_base[f'Mejora @{k}'] = f"{improvement:+.3f} ({improvement_pct:+.1f}%) " + get_improvement_status_icon(improvement)
                        else:
                            row_base[f'Mejora @{k}'] = '-'
                    elif use_llm_reranker:
                        row_base[f'Después LLM @{k}'] = '-'
                        row_base[f'Mejora @{k}'] = '-'

            table_data.append(row_base)

    if table_data:
        df_full_metrics = pd.DataFrame(table_data)
        st.dataframe(df_full_metrics, use_container_width=True)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
from typing import Dict, Any, List
from src.services.auth.clients import initialize_clients

def _format_metrics_for_llm(results_data: Dict[str, Any]) -> str:
    """
    Formats the evaluation metrics data into a human-readable string for an LLM prompt.
    """
    formatted_string = ""
    config = results_data.get('config', {})
    results = results_data.get('results', {})
    evaluation_info = results_data.get('evaluation_info', {})

    formatted_string += "## Configuración de la Evaluación\n"
    formatted_string += f"- Número de preguntas: {config.get('num_questions', 'N/A')}\n"
    formatted_string += f"- Modelos de embedding evaluados: {config.get('selected_models', 'N/A')}\n"
    formatted_string += f"- Modelo generativo (reranking/respuesta): {config.get('generative_model_name', 'N/A')}\n"
    formatted_string += f"- Top-K documentos recuperados: {config.get('top_k', 'N/A')}\n"
    formatted_string += f"- Reranking LLM habilitado: {config.get('use_llm_reranker', 'N/A')}\n"
    
    # Document aggregation information
    chunk_to_doc_config = config.get('chunk_to_document_config', {})
    if chunk_to_doc_config and chunk_to_doc_config.get('enabled', False):
        formatted_string += f"- **Agregación de Documentos habilitada**: Sí\n"
        formatted_string += f"  - Multiplicador de chunks: {chunk_to_doc_config.get('chunk_multiplier', 'N/A')}\n"
        formatted_string += f"  - Documentos objetivo: {chunk_to_doc_config.get('target_documents', 'N/A')}\n"
        formatted_string += f"  - Método: Chunks → Documentos completos mediante agrupación por enlace\n"
        formatted_string += f"  - **Límites de Contenido Optimizados**:\n"
        formatted_string += f"    - Generación de respuestas: 2000 chars/doc (mejorado desde 500)\n"
        formatted_string += f"    - Contexto RAGAS: 3000 chars/doc (mejorado desde 1000)\n"
        formatted_string += f"    - Reranking LLM: 4000 chars/doc (mejorado desde 3000)\n"
        formatted_string += f"    - BERTScore: Sin límite (anteriormente limitado)\n"
    else:
        formatted_string += f"- **Agregación de Documentos**: No habilitada (usando chunks directamente)\n"
        formatted_string += f"- **Límites de Contenido**: Optimizados para chunks individuales\n"
    
    formatted_string += f"- Tiempo total de ejecución: {evaluation_info.get('total_time_seconds', 'N/A')} segundos\n"
    formatted_string += f"- GPU utilizada: {'Sí' if evaluation_info.get('gpu_used') else 'No'}\n\n"

    formatted_string += "## Resultados Detallados por Modelo\n"
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics_types = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']
    # Use STANDARD metric names from RAGAS and BERTScore libraries  
    standard_ragas_types = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness', 'answer_similarity', 'semantic_similarity']
    standard_bertscore_types = ['bert_precision', 'bert_recall', 'bert_f1']

    for model_name, model_data in results.items():
        formatted_string += f"### Modelo: {model_name}\n"
        before_metrics = model_data.get('avg_before_metrics', {})
        after_metrics = model_data.get('avg_after_metrics', {})
        individual_before_metrics = model_data.get('individual_before_metrics', [])
        individual_after_metrics = model_data.get('individual_after_metrics', [])

        # Average Retrieval Metrics
        formatted_string += "#### Métricas de Recuperación Promedio\n"
        formatted_string += "| Métrica | Antes LLM | Después LLM | Mejora Absoluta | Mejora % |\n"
        formatted_string += "|---|---|---|---|---|\n"
        for metric_type in metrics_types:
            if metric_type == 'mrr':
                # MRR is a single value, not per K - only process once
                metric_key = 'mrr'
                before_val = before_metrics.get('mrr')
                after_val = after_metrics.get('mrr')
                
                # Handle None values properly
                before_str = f"{before_val:.3f}" if before_val is not None and not np.isnan(before_val) else 'N/A'
                after_str = f"{after_val:.3f}" if after_val is not None and not np.isnan(after_val) else 'N/A'
                
                improvement_abs = 'N/A'
                improvement_pct = 'N/A'
                if (before_val is not None and not np.isnan(before_val) and 
                    after_val is not None and not np.isnan(after_val)):
                    improvement_abs = after_val - before_val
                    if before_val != 0:
                        improvement_pct = (improvement_abs / before_val) * 100
                    else:
                        improvement_pct = 0 if improvement_abs == 0 else float('inf')

                # Format improvement values safely
                improvement_abs_str = f"{improvement_abs:+.3f}" if isinstance(improvement_abs, (int, float)) and not np.isinf(improvement_abs) else str(improvement_abs)
                improvement_pct_str = f"{improvement_pct:+.1f}%" if isinstance(improvement_pct, (int, float)) and not np.isinf(improvement_pct) else str(improvement_pct)
                
                formatted_string += (
                    f"| {metric_key} "
                    f"| {before_str} "
                    f"| {after_str} "
                    f"| {improvement_abs_str} "
                    f"| {improvement_pct_str} |\n"
                )
            else:
                for k in k_values:
                    metric_key = f"{metric_type}@{k}"
                    before_val = before_metrics.get(metric_key, np.nan)
                    after_val = after_metrics.get(metric_key, np.nan)
                    
                    improvement_abs = 'N/A'
                    improvement_pct = 'N/A'
                    if not np.isnan(before_val) and not np.isnan(after_val):
                        improvement_abs = after_val - before_val
                        if before_val != 0:
                            improvement_pct = (improvement_abs / before_val) * 100
                        else:
                            improvement_pct = 0 if improvement_abs == 0 else float('inf') # Handle division by zero

                    # Format improvement values safely
                    improvement_abs_str = f"{improvement_abs:+.3f}" if isinstance(improvement_abs, (int, float)) and not np.isinf(improvement_abs) else str(improvement_abs)
                    improvement_pct_str = f"{improvement_pct:+.1f}%" if isinstance(improvement_pct, (int, float)) and not np.isinf(improvement_pct) else str(improvement_pct)
                    
                    formatted_string += (
                        f"| {metric_key} "
                        f"| {before_val:.3f} "
                        f"| {after_val:.3f} "
                        f"| {improvement_abs_str} "
                        f"| {improvement_pct_str} |\n"
                    )
        formatted_string += "\n"

        # Check if we have any RAG metrics data
        has_rag_data = False
        for metrics in individual_before_metrics + individual_after_metrics:
            if isinstance(metrics, dict) and (
                'rag_metrics' in metrics or 'rag_metrics_after_rerank' in metrics
            ):
                rag_data = metrics.get('rag_metrics') or metrics.get(
                    'rag_metrics_after_rerank', {}
                )
                if rag_data and any(v is not None for v in rag_data.values()):
                    has_rag_data = True
                    break
        
        if has_rag_data:
            # Average RAG Metrics (only if data exists)
            formatted_string += "#### Métricas RAG Promedio\n"
            formatted_string += "| Métrica | Antes LLM | Después LLM | Mejora Absoluta | Mejora % |\n"
            formatted_string += "|---|---|---|---|---|\n"
            for metric_type in standard_ragas_types:
                before_values = [
                    q.get('rag_metrics', {}).get(metric_type, np.nan)
                    for q in individual_before_metrics
                    if q.get('rag_metrics', {}).get(metric_type) is not None
                ]
                before_avg = np.mean(before_values) if before_values else np.nan

                after_values = []
                for q in individual_after_metrics:
                    if not isinstance(q, dict):
                        continue
                    rag_after = q.get('rag_metrics_after_rerank') or q.get('rag_metrics')
                    if rag_after and rag_after.get(metric_type) is not None:
                        after_values.append(rag_after.get(metric_type, np.nan))
                after_avg = np.mean(after_values) if after_values else np.nan

                improvement_abs = 'N/A'
                improvement_pct = 'N/A'
                if not np.isnan(before_avg) and not np.isnan(after_avg):
                    improvement_abs = after_avg - before_avg
                    if before_avg != 0:
                        improvement_pct = (improvement_abs / before_avg) * 100
                    else:
                        improvement_pct = 0 if improvement_abs == 0 else float('inf')

                # Format improvement values safely
                improvement_abs_str = f"{improvement_abs:+.3f}" if isinstance(improvement_abs, (int, float)) and not np.isinf(improvement_abs) else str(improvement_abs)
                improvement_pct_str = f"{improvement_pct:+.1f}%" if isinstance(improvement_pct, (int, float)) and not np.isinf(improvement_pct) else str(improvement_pct)
                before_avg_str = f"{before_avg:.3f}" if not np.isnan(before_avg) else 'N/A'
                after_avg_str = f"{after_avg:.3f}" if not np.isnan(after_avg) else 'N/A'
                
                formatted_string += (
                    f"| {metric_type.replace('answer_', '').replace('context_', '').capitalize()} "
                    f"| {before_avg_str} "
                    f"| {after_avg_str} "
                    f"| {improvement_abs_str} "
                    f"| {improvement_pct_str} |\n"
                )
            
            # Add BERTScore metrics section if available
            has_bertscore_data = False
            for metrics in individual_before_metrics + individual_after_metrics:
                if isinstance(metrics, dict) and (
                    'rag_metrics' in metrics or 'rag_metrics_after_rerank' in metrics
                ):
                    rag_data = metrics.get('rag_metrics') or metrics.get(
                        'rag_metrics_after_rerank', {}
                    )
                    if rag_data and any(k in rag_data for k in standard_bertscore_types):
                        has_bertscore_data = True
                        break
            
            if has_bertscore_data:
                formatted_string += "#### Métricas BERTScore Promedio\n"
                formatted_string += "| Métrica | Antes LLM | Después LLM | Mejora Absoluta | Mejora % |\n"
                formatted_string += "|---|---|---|---|---|\n"
                for metric_type in standard_bertscore_types:
                    before_values = [
                        q.get('rag_metrics', {}).get(metric_type, np.nan)
                        for q in individual_before_metrics
                        if q.get('rag_metrics', {}).get(metric_type) is not None
                    ]
                    before_avg = np.mean(before_values) if before_values else np.nan

                    after_values = []
                    for q in individual_after_metrics:
                        if not isinstance(q, dict):
                            continue
                        rag_after = q.get('rag_metrics_after_rerank') or q.get('rag_metrics')
                        if rag_after and rag_after.get(metric_type) is not None:
                            after_values.append(rag_after.get(metric_type, np.nan))
                    after_avg = np.mean(after_values) if after_values else np.nan

                    improvement_abs = 'N/A'
                    improvement_pct = 'N/A'
                    if not np.isnan(before_avg) and not np.isnan(after_avg):
                        improvement_abs = after_avg - before_avg
                        if before_avg != 0:
                            improvement_pct = (improvement_abs / before_avg) * 100
                        else:
                            improvement_pct = 0 if improvement_abs == 0 else float('inf')

                    # Format improvement values safely
                    improvement_abs_str = f"{improvement_abs:+.3f}" if isinstance(improvement_abs, (int, float)) and not np.isinf(improvement_abs) else str(improvement_abs)
                    improvement_pct_str = f"{improvement_pct:+.1f}%" if isinstance(improvement_pct, (int, float)) and not np.isinf(improvement_pct) else str(improvement_pct)
                    before_avg_str = f"{before_avg:.3f}" if not np.isnan(before_avg) else 'N/A'
                    after_avg_str = f"{after_avg:.3f}" if not np.isnan(after_avg) else 'N/A'
                    
                    # Clean metric name for display
                    display_name = metric_type.replace('bert_', 'BERT ').title()
                    
                    formatted_string += (
                        f"| {display_name} "
                        f"| {before_avg_str} "
                        f"| {after_avg_str} "
                        f"| {improvement_abs_str} "
                        f"| {improvement_pct_str} |\n"
                    )

            formatted_string += "\n"
        else:
            # Add note about missing RAG metrics
            formatted_string += "#### Métricas RAG y BERTScore\n"
            formatted_string += "**Nota:** Las métricas RAG (Faithfulness, Answer Relevance, Answer Correctness, Answer Similarity) y BERTScore (Precision, Recall, F1) no están disponibles porque la evaluación se ejecutó en modo de solo recuperación (sin generación de respuestas).\n\n"

    return formatted_string

def generate_analysis_with_llm(results_data: Dict[str, Any], generative_model_name: str) -> Dict[str, str]:
    """
    Generates conclusions and improvements using an LLM based on evaluation results.
    Returns a dictionary with 'conclusions' and 'improvements'.
    """
    st.info(f"🤖 Generando análisis con {generative_model_name}... Esto puede tomar unos minutos.")

    try:
        # Initialize only the generative clients we need (not ChromaDB or embedding)
        from src.services.storage.chromadb_utils import ChromaDBConfig
        from openai import OpenAI
        import google.generativeai as genai
        from src.services.local_models import get_tinyllama_client, get_mistral_client
        from src.services.auth.openrouter_client import get_cached_llama4_scout_client
        
        config = ChromaDBConfig.from_env()
        
        # Initialize clients based on the generative model
        openai_client = None
        gemini_client = None
        local_tinyllama_client = None
        local_mistral_client = None
        openrouter_client = None
        
        if generative_model_name == "gpt-4" and config.openai_api_key:
            openai_client = OpenAI(api_key=config.openai_api_key)
        elif generative_model_name == "gemini-1.5-flash" and config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            gemini_client = genai.GenerativeModel('gemini-1.5-flash')
        elif generative_model_name == "llama-4-scout":
            openrouter_client = get_cached_llama4_scout_client()
        elif generative_model_name == "tinyllama-1.1b":
            local_tinyllama_client = get_tinyllama_client()
        elif generative_model_name == "mistral-7b":
            local_mistral_client = get_mistral_client()

        llm_client = None
        if generative_model_name == "gpt-4" and openai_client:
            llm_client = openai_client
            model_to_use = "gpt-4"
        elif generative_model_name == "gemini-1.5-flash" and gemini_client:
            llm_client = gemini_client
            model_to_use = "gemini-1.5-flash"
        elif generative_model_name == "llama-4-scout" and openrouter_client:
            llm_client = openrouter_client
            model_to_use = "llama-4-scout"
        elif generative_model_name == "tinyllama-1.1b" and local_tinyllama_client:
            llm_client = local_tinyllama_client
            model_to_use = "tinyllama-1.1b"
        elif generative_model_name == "mistral-7b" and local_mistral_client:
            llm_client = local_mistral_client
            model_to_use = "mistral-7b"
        
        if not llm_client:
            st.error(f"❌ Error: Cliente LLM no disponible para el modelo seleccionado: {generative_model_name}")
            st.info("Asegúrate de que las API keys estén configuradas correctamente o que los modelos locales estén cargados.")
            return {
                'conclusions': "❌ Error: Cliente LLM no disponible para el modelo seleccionado.",
                'improvements': "Verifica la configuración de tu API key o la disponibilidad del modelo local."
            }

        # Format metrics data for the prompt
        formatted_metrics = _format_metrics_for_llm(results_data)

        system_prompt = (
            "Eres un experto en sistemas RAG (Retrieval Augmented Generation) y evaluación de modelos. "
            "Analiza los resultados de evaluación proporcionados y genera conclusiones concisas y "
            "recomendaciones de mejora para el sistema RAG. "
            "Tu respuesta debe estar estructurada en dos secciones: 'Conclusiones' y 'Posibles Mejoras y Próximos Pasos'. "
            "Utiliza un lenguaje claro y técnico, y basa tus afirmaciones estrictamente en los datos proporcionados. "
            "Si el reranking LLM fue utilizado, comenta específicamente sobre su impacto. "
            "Si la agregación de documentos está habilitada, considera su efecto en la recuperación y contexto. "
            "Las conclusiones deben ser en formato de lista de puntos y las mejoras en formato de lista numerada. "
            "No incluyas preámbulos ni postámbulos, solo las dos secciones solicitadas."
        )

        user_prompt = (
            f"Aquí están los resultados detallados de la evaluación de un sistema RAG:\n\n{formatted_metrics}\n\n"
            "Por favor, genera las conclusiones y las posibles mejoras y próximos pasos en español, "
            "siguiendo el formato especificado (Conclusiones: lista de puntos; Mejoras: lista numerada)."
        )

        full_response_content = ""

        try:
            if model_to_use in ["gpt-4", "gemini-1.5-flash", "llama-4-scout"]:
                # For API-based models
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                if model_to_use == "gpt-4":
                    response = llm_client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=0.2, # Keep it low for factual analysis
                        max_tokens=1500
                    )
                    full_response_content = response.choices[0].message.content
                elif model_to_use == "gemini-1.5-flash":
                    response = llm_client.generate_content(messages)
                    full_response_content = response.text
                elif model_to_use == "llama-4-scout":
                    # OpenRouter client might have a different method signature
                    # Assuming it has a similar generate_answer or chat.completions.create
                    full_response_content = llm_client.generate_answer(
                        question=user_prompt, # Pass the full user prompt
                        context="", # No additional context needed, it's in the prompt
                        max_length=1500
                    )
            elif model_to_use in ["tinyllama-1.1b", "mistral-7b"]:
                # For local models, assuming a generate_answer or similar method
                # The local models might not handle complex system/user roles as well as API models
                # So, combine into a single prompt
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                generated_text, _ = llm_client.generate_answer(
                    question=combined_prompt,
                    retrieved_docs=[], # No docs needed, data is in prompt
                    max_length=1500
                )
                full_response_content = generated_text
            else:
                raise ValueError(f"Modelo LLM no reconocido o no soportado: {model_to_use}")

        except Exception as e:
            st.error(f"❌ Error durante la llamada al LLM ({model_to_use}): {e}")
            st.exception(e) # Display full traceback
            return {
                'conclusions': "❌ Error al generar conclusiones con LLM.",
                'improvements': f"❌ Error al generar mejoras con LLM: {e}"
            }

        # Validate response content
        if not full_response_content or full_response_content.strip() == "":
            st.warning("⚠️ El LLM devolvió una respuesta vacía. Usando análisis predeterminado.")
            return {
                'conclusions': "❌ No se pudo generar análisis automático. Revisa las métricas de rendimiento manualmente.",
                'improvements': "1. Verifica la configuración del modelo LLM\n2. Revisa los datos de entrada\n3. Considera usar un modelo diferente"
            }

        # Parse the response content
        conclusions = "No se pudieron extraer conclusiones del análisis generado." 
        improvements = "No se pudieron extraer mejoras del análisis generado."

        # Try multiple parsing strategies
        response_lower = full_response_content.lower()
        
        # Strategy 1: Look for exact headings
        conclusions_start = full_response_content.find("Conclusiones")
        improvements_start = full_response_content.find("Posibles Mejoras y Próximos Pasos")
        
        # Strategy 2: Look for alternative headings
        if conclusions_start == -1:
            conclusions_start = max(
                response_lower.find("conclusiones:"),
                response_lower.find("## conclusiones"),
                response_lower.find("### conclusiones")
            )
        
        if improvements_start == -1:
            improvements_start = max(
                response_lower.find("mejoras:"),
                response_lower.find("## mejoras"),
                response_lower.find("### mejoras"),
                response_lower.find("posibles mejoras"),
                response_lower.find("próximos pasos")
            )

        if conclusions_start != -1 and improvements_start != -1 and improvements_start > conclusions_start:
            conclusions_raw = full_response_content[conclusions_start:improvements_start]
            improvements_raw = full_response_content[improvements_start:]
            
            # Clean up the extracted content
            conclusions_raw = conclusions_raw.replace("Conclusiones", "").replace("conclusiones:", "").replace("## conclusiones", "").replace("### conclusiones", "").strip()
            improvements_raw = improvements_raw.replace("Posibles Mejoras y Próximos Pasos", "").replace("mejoras:", "").replace("## mejoras", "").replace("### mejoras", "").replace("posibles mejoras", "").replace("próximos pasos", "").strip()
            
            if conclusions_raw:
                conclusions = conclusions_raw
            if improvements_raw:
                improvements = improvements_raw
                
        elif conclusions_start != -1:
            conclusions_raw = full_response_content[conclusions_start:].replace("Conclusiones", "").replace("conclusiones:", "").replace("## conclusiones", "").replace("### conclusiones", "").strip()
            if conclusions_raw:
                conclusions = conclusions_raw
                
        elif improvements_start != -1:
            improvements_raw = full_response_content[improvements_start:].replace("Posibles Mejoras y Próximos Pasos", "").replace("mejoras:", "").replace("## mejoras", "").replace("### mejoras", "").replace("posibles mejoras", "").replace("próximos pasos", "").strip()
            if improvements_raw:
                improvements = improvements_raw
        else:
            # If no structured sections found, use the entire response as conclusions
            st.warning("⚠️ No se encontraron secciones estructuradas en la respuesta del LLM. Usando respuesta completa.")
            conclusions = full_response_content.strip()
            improvements = "Revisa el análisis anterior y considera implementar mejoras basadas en las observaciones."

        # Final validation - ensure we don't return empty strings
        if not conclusions or conclusions.strip() == "":
            conclusions = "✅ Análisis completado. Revisa las métricas mostradas para obtener insights sobre el rendimiento del sistema RAG."
        
        if not improvements or improvements.strip() == "":
            improvements = "1. Analiza las métricas individuales para identificar áreas de mejora\n2. Considera ajustar los parámetros del sistema\n3. Evalúa diferentes modelos de embedding o generativos"

        return {'conclusions': conclusions, 'improvements': improvements}

    except Exception as e:
        error_msg = str(e) if e else "Error desconocido"
        st.error(f"❌ Error general al generar análisis con LLM: {error_msg}")
        st.exception(e)
        return {
            'conclusions': f"❌ Error general al generar conclusiones con LLM: {error_msg}. Revisa las métricas manualmente.",
            'improvements': f"❌ Error al generar mejoras con LLM: {error_msg}. Considera:\n1. Verificar la configuración del modelo\n2. Revisar la conectividad\n3. Usar análisis manual de las métricas"
        }

def get_improvement_status_icon(improvement: float) -> str:
    """Return a simple icon for improvement status."""
    if improvement > 0.01: # Small positive change
        return "⬆️"
    elif improvement < -0.01: # Small negative change
        return "⬇️"
    else:
        return "➡️"


def create_models_summary_table(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool):
    """Create summary table for model comparison"""
    
    st.subheader("📋 Tabla Resumen de Modelos")
    
    summary_data = []
    for model_name, model_results in results.items():
        before_metrics = model_results['avg_before_metrics']
        after_metrics = model_results.get('avg_after_metrics', {})
        
        # Calculate average performance - handle MRR properly
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr', 'ndcg@5']
        valid_before_metrics = []
        for m in key_metrics:
            val = before_metrics.get(m)
            if val is not None and not np.isnan(val):
                valid_before_metrics.append(val)
        
        before_avg = np.mean(valid_before_metrics) if valid_before_metrics else 0
        
        row = {
            'Modelo': model_name,
            'Preguntas': model_results.get('num_questions_evaluated', 0),
            'Promedio Antes': f"{before_avg:.3f}",
            'Calidad': get_metric_quality(before_avg)
        }
        
        if use_llm_reranker and after_metrics:
            valid_after_metrics = []
            for m in key_metrics:
                val = after_metrics.get(m)
                if val is not None and not np.isnan(val):
                    valid_after_metrics.append(val)
            
            after_avg = np.mean(valid_after_metrics) if valid_after_metrics else 0
            improvement = after_avg - before_avg
            
            row.update({
                'Promedio Después': f"{after_avg:.3f}",
                'Mejora': f"{improvement:+.3f}",
                'Mejora %': f"{(improvement/before_avg*100):+.1f}%" if before_avg > 0 else "N/A"
            })
        
        summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

def extract_rag_metrics_from_individual(individual_metrics: List[Dict]) -> Dict:
    """
    Extract and calculate average RAG and BERTScore metrics from individual question results.
    Handles both old format (no retrieval/rag separation) and new format.
    """
    rag_metric_keys = ['faithfulness', 'answer_relevancy', 'answer_correctness', 'semantic_similarity', 'context_precision', 'context_recall']
    bertscore_metric_keys = ['bert_precision', 'bert_recall', 'bert_f1']
    all_metric_keys = rag_metric_keys + bertscore_metric_keys
    
    rag_sums = {}
    rag_counts = {}
    
    for individual in individual_metrics:
        # Check if this individual result has a separate rag_metrics section
        if isinstance(individual, dict):
            rag_data = individual.get('rag_metrics', {})
            
            # If no separate rag_metrics section, check direct in individual
            if not rag_data:
                rag_data = individual
            
            # Process both RAG and BERTScore metrics
            for key in all_metric_keys:
                if key in rag_data and rag_data[key] is not None:
                    if key not in rag_sums:
                        rag_sums[key] = 0
                        rag_counts[key] = 0
                    rag_sums[key] += rag_data[key]
                    rag_counts[key] += 1
    
    # Calculate averages
    avg_rag_metrics = {}
    for key in all_metric_keys:
        if key in rag_sums and rag_counts[key] > 0:
            avg_rag_metrics[key] = rag_sums[key] / rag_counts[key]
    
    return avg_rag_metrics


def display_metric_definitions_table():
    """Display accordion table with metric definitions and formulas"""
    
    with st.expander("📚 Definiciones y Fórmulas de Métricas", expanded=False):
        st.markdown("### Métricas de Recuperación de Información")
        
        # Create DataFrame with metric definitions
        metrics_data = [
            {
                "Métrica": "Precision@K",
                "Definición": "Proporción de documentos relevantes entre los K primeros resultados",
                "Fórmula": "P@K = (Documentos relevantes en top-K) / K",
                "Interpretación": "Valores altos indican que los primeros K resultados son muy relevantes"
            },
            {
                "Métrica": "Recall@K", 
                "Definición": "Proporción de documentos relevantes totales que están en los K primeros",
                "Fórmula": "R@K = (Documentos relevantes en top-K) / (Total documentos relevantes)",
                "Interpretación": "Valores altos indican buena cobertura de documentos relevantes"
            },
            {
                "Métrica": "F1@K",
                "Definición": "Media armónica entre Precision@K y Recall@K",
                "Fórmula": "F1@K = 2 × (P@K × R@K) / (P@K + R@K)",
                "Interpretación": "Balance entre precisión y cobertura"
            },
            {
                "Métrica": "MAP@K",
                "Definición": "Media de Average Precision para todas las consultas hasta posición K",
                "Fórmula": "MAP@K = (1/Q) × Σ(AP@K_q) para todas las consultas q",
                "Interpretación": "Considera el orden de los documentos relevantes"
            },
            {
                "Métrica": "MRR",
                "Definición": "Media del recíproco del rango del primer documento relevante",
                "Fórmula": "MRR = (1/Q) × Σ(1/rank_q) donde rank_q es la posición del primer relevante",
                "Interpretación": "Valores altos indican que el primer resultado suele ser relevante"
            },
            {
                "Métrica": "NDCG@K",
                "Definición": "Normalized Discounted Cumulative Gain hasta posición K",
                "Fórmula": "NDCG@K = DCG@K / IDCG@K, donde DCG considera relevancia graduada",
                "Interpretación": "Métrica sofisticada que considera orden y grados de relevancia"
            }
        ]
        
        # Display as table
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        st.markdown("### Notas Importantes")
        st.info("""
        **📊 Interpretación de Valores:**
        - **0.8-1.0**: Excelente rendimiento
        - **0.6-0.8**: Buen rendimiento  
        - **0.4-0.6**: Rendimiento moderado
        - **< 0.4**: Necesita mejoras
        
        **🔍 Consideraciones:**
        - Los valores @K dependen del valor de K seleccionado
        - MAP y NDCG son más sensibles al orden de los resultados
        - MRR es especialmente útil para tareas donde solo importa encontrar UN documento relevante
        """)


def display_rag_metrics_summary(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, config: Dict = None):
    """
    Displays a summary of RAG metrics (faithfulness, answer relevance, etc.),
    with simple table format: Modelo, Faithfulness, Relevance, Correctness, Similarity, BERTScore metrics
    """
    st.subheader("📊 Métricas RAGAS/BERTScore")

    rag_metric_keys = ['faithfulness', 'answer_relevancy', 'answer_correctness', 'semantic_similarity', 'context_precision', 'context_recall']
    bertscore_metric_keys = ['bert_precision', 'bert_recall', 'bert_f1']
    all_metric_keys = rag_metric_keys + bertscore_metric_keys
    table_data = []
    chart_data = []
    
    has_any_rag_metric = False

    for model_name, model_results in results.items():
        # Try to get RAG metrics from multiple sources:
        # 1. Direct RAG averages (newest format)
        rag_before_metrics = model_results.get('avg_rag_before_metrics', {})
        rag_after_metrics = model_results.get('avg_rag_after_metrics', {})
        
        # 2. NEW FORMAT: RAG metrics section (v2.0 Colab format)
        rag_metrics_section = model_results.get('rag_metrics', {})
        
        # 3. Pre-calculated averages with RAG metrics mixed in (older format)
        before_metrics = model_results.get('avg_before_metrics', {})
        after_metrics = model_results.get('avg_after_metrics', {})
        
        # Combine RAG metrics from dedicated sections if available
        if rag_before_metrics:
            before_metrics = {**before_metrics, **rag_before_metrics}
        if rag_after_metrics:
            after_metrics = {**after_metrics, **rag_after_metrics}
        
        # NEW: Extract from rag_metrics section (our new format)
        if rag_metrics_section:
            # RAG metrics are stored as 'avg_faithfulness', 'avg_answer_relevance', etc.
            for key in rag_metric_keys:
                avg_key = f'avg_{key}'
                if avg_key in rag_metrics_section:
                    before_metrics[key] = rag_metrics_section[avg_key]
            
            # BERTScore metrics are stored as 'avg_bert_precision', 'avg_bert_recall', etc.
            for key in bertscore_metric_keys:
                avg_key = f'avg_{key}'
                if avg_key in rag_metrics_section:
                    before_metrics[key] = rag_metrics_section[avg_key]
        
        # 4. Calculate from individual metrics if not in averages
        individual_before = model_results.get('individual_before_metrics', [])
        individual_after = model_results.get('individual_after_metrics', [])
        individual_rag = model_results.get('individual_rag_metrics', [])  # NEW: dedicated RAG metrics
        
        # 5. Fallback to old format individual_metrics
        if not individual_before and not individual_after:
            individual_metrics = model_results.get('individual_metrics', [])
            if individual_metrics:
                individual_before = individual_metrics
        
        # Extract RAG and BERTScore metrics from individual data if not available in averages
        if not any(key in before_metrics for key in all_metric_keys) and individual_before:
            extracted_before = extract_rag_metrics_from_individual(individual_before)
            before_metrics.update(extracted_before)
        
        if not any(key in after_metrics for key in all_metric_keys) and individual_after:
            extracted_after = extract_rag_metrics_from_individual(individual_after)
            after_metrics.update(extracted_after)
        
        # NEW: Extract from dedicated individual RAG metrics (v2.0 format)
        if not any(key in before_metrics for key in all_metric_keys) and individual_rag:
            extracted_rag = extract_rag_metrics_from_individual(individual_rag)
            before_metrics.update(extracted_rag)

        # Check if at least one RAG or BERTScore metric exists for this model
        model_has_rag = any(key in before_metrics or key in after_metrics for key in all_metric_keys)
        
        # If generate_rag_metrics is True in config but no RAG metrics exist, show a message
        config_wants_rag = config and config.get('generate_rag_metrics', False)
        if config_wants_rag and not model_has_rag:
            rag_available = rag_metrics_section.get('rag_available', False) if rag_metrics_section else False
            if not rag_available:
                st.warning("⚠️ Generación de métricas RAG está habilitada en la configuración pero no se pudieron generar. Posibles causas:")
                st.markdown("""
                - **OpenAI API no disponible**: Verifica que `OPENAI_API_KEY` esté configurada en Colab Secrets o .env
                - **Errores en generación**: Revisa los logs del notebook para errores de OpenAI API
                - **Formato de datos**: Los resultados pueden ser de una versión anterior sin RAG metrics
                """)
            else:
                st.info("ℹ️ Generación de métricas RAG está habilitada pero no se encontraron métricas calculadas en los resultados.")
            return
        if not model_has_rag:
            continue
        
        has_any_rag_metric = True

        # Create row for this model with all RAG metrics
        row = {'Modelo': model_name}
        
        # Add each RAG metric as a column
        for key in rag_metric_keys:
            metric_val = before_metrics.get(key)
            if metric_val is not None:
                if key == 'faithfulness':
                    row['Faithfulness'] = f"{metric_val:.3f}"
                elif key == 'answer_relevancy':
                    row['Relevancy'] = f"{metric_val:.3f}"
                elif key == 'answer_correctness':
                    row['Correctness'] = f"{metric_val:.3f}"
                elif key == 'semantic_similarity':
                    row['Similarity'] = f"{metric_val:.3f}"
                elif key == 'context_precision':
                    row['Context Precision'] = f"{metric_val:.3f}"
                elif key == 'context_recall':
                    row['Context Recall'] = f"{metric_val:.3f}"
                
                # Add to chart data for visualization
                metric_name = key.replace('_', ' ').replace('answer', '').replace('semantic', '').strip().title()
                chart_data.append({
                    'Modelo': model_name, 
                    'Métrica RAG': metric_name, 
                    'Valor': metric_val
                })
            else:
                # Add placeholder for missing metrics
                if key == 'faithfulness':
                    row['Faithfulness'] = '-'
                elif key == 'answer_relevancy':
                    row['Relevancy'] = '-'
                elif key == 'answer_correctness':
                    row['Correctness'] = '-'
                elif key == 'semantic_similarity':
                    row['Similarity'] = '-'
                elif key == 'context_precision':
                    row['Context Precision'] = '-'
                elif key == 'context_recall':
                    row['Context Recall'] = '-'
        
        # Add BERTScore metrics as columns
        for key in bertscore_metric_keys:
            metric_val = before_metrics.get(key)
            if metric_val is not None:
                if key == 'bert_precision':
                    row['BERT Precision'] = f"{metric_val:.3f}"
                elif key == 'bert_recall':
                    row['BERT Recall'] = f"{metric_val:.3f}"
                elif key == 'bert_f1':
                    row['BERT F1'] = f"{metric_val:.3f}"
                
                # Add to chart data for visualization
                metric_name = key.replace('bert_', 'BERT ').replace('_', ' ').title()
                chart_data.append({
                    'Modelo': model_name, 
                    'Métrica RAG': metric_name, 
                    'Valor': metric_val
                })
            else:
                # Add placeholder for missing BERTScore metrics
                if key == 'bert_precision':
                    row['BERT Precision'] = '-'
                elif key == 'bert_recall':
                    row['BERT Recall'] = '-'
                elif key == 'bert_f1':
                    row['BERT F1'] = '-'
        
        table_data.append(row)

    if not has_any_rag_metric:
        # Debug: Show what data structure we're working with
        with st.expander("🔍 Debug: Estructura de datos examinada"):
            for model_name, model_results in results.items():
                st.write(f"**Modelo: {model_name}**")
                st.write(f"- Claves disponibles: {list(model_results.keys())}")
                
                # Show RAG metrics section structure if available
                if 'rag_metrics' in model_results:
                    rag_section = model_results['rag_metrics']
                    st.write(f"- Sección rag_metrics encontrada: {list(rag_section.keys())}")
                    if 'avg_faithfulness' in rag_section:
                        st.write(f"  avg_faithfulness: {rag_section['avg_faithfulness']}")
                    if 'avg_answer_relevance' in rag_section:
                        st.write(f"  avg_answer_relevance: {rag_section['avg_answer_relevance']}")
                else:
                    st.write("- No se encontró sección rag_metrics")
                    
                # Show individual RAG metrics structure
                if 'individual_rag_metrics' in model_results:
                    individual_rag = model_results['individual_rag_metrics']
                    if individual_rag:
                        first_rag = individual_rag[0]
                        st.write(f"- Primera métrica individual RAG: {list(first_rag.keys())}")
                        if 'faithfulness' in first_rag:
                            st.write(f"  faithfulness: {first_rag['faithfulness']}")
                    else:
                        st.write("- individual_rag_metrics está vacío")
                else:
                    st.write("- No se encontró individual_rag_metrics")
        
        st.info("""
        **📝 No se encontraron métricas RAG ni BERTScore en los resultados.**
        
        Para generar estas métricas, asegúrate de que la opción
        `📝 Generar Métricas RAG` esté habilitada durante la configuración de la evaluación.
        
        **Métricas RAG esperadas:** faithfulness, answer_relevancy, answer_correctness, semantic_similarity, context_precision, context_recall
        
        **Métricas BERTScore esperadas:** bert_precision, bert_recall, bert_f1
        """)
        return

    if table_data:
        df_rag = pd.DataFrame(table_data)
        
        # Apply color styling to both RAGAS and BERTScore columns
        def get_metrics_color_for_column(value, column_name):
            # Apply colors to RAGAS and BERTScore columns (all use 0-1 scale)
            ragas_columns = ['Faithfulness', 'Relevancy', 'Correctness', 'Similarity', 'Context Precision', 'Context Recall']
            bertscore_columns = ['BERT Precision', 'BERT Recall', 'BERT F1']
            
            if (column_name in ragas_columns or column_name in bertscore_columns) and value != '-':
                try:
                    numeric_value = float(value)
                    if numeric_value > 0.8:
                        return 'background-color: #d4edda; color: #155724'  # Green for Excelente
                    elif numeric_value >= 0.6:
                        return 'background-color: #fff3cd; color: #856404'  # Yellow for Bueno
                    else:
                        return 'background-color: #f8d7da; color: #721c24'  # Red for Necesita mejora
                except:
                    pass
            return ''  # No styling for non-metric columns or invalid values
        
        # Apply styling with apply for the whole dataframe
        styled_df_rag = df_rag.style.apply(lambda x: [get_metrics_color_for_column(val, col) for val, col in zip(x, df_rag.columns)], axis=1)
        
        st.dataframe(styled_df_rag, use_container_width=True)

    if chart_data:
        # Create line graph with metrics on X-axis and different lines for each model
        df_line = pd.DataFrame(chart_data)
        
        # Create line graph
        fig = px.line(
            df_line,
            x='Métrica RAG',
            y='Valor',
            color='Modelo',
            markers=True,
            title='Métricas RAG por Modelo',
            labels={'Valor': 'Puntuación', 'Métrica RAG': 'Métricas RAG'},
            range_y=[0, 1]
        )
        
        # Update layout for better visualization
        fig.update_traces(mode='lines+markers', marker=dict(size=10))
        fig.update_layout(
            height=400,
            xaxis_title="Métricas RAG",
            yaxis_title="Puntuación",
            legend_title="Modelos",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # BERTScore visualization is now handled separately to avoid duplication
    
    # Add RAG metrics explanation accordion
    display_rag_metrics_explanation()

def display_bertscore_detailed_charts(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, config: Dict = None):
    """
    Create specific visualizations for BERTScore metrics (precision, recall, F1)
    """
    bertscore_metric_keys = ['bert_precision', 'bert_recall', 'bert_f1']
    bertscore_data = []
    
    # Check if any model has BERTScore metrics
    has_bertscore = False
    for model_name, model_results in results.items():
        # Check various data sources for BERTScore metrics
        sources = [
            model_results.get('avg_rag_before_metrics', {}),
            model_results.get('avg_before_metrics', {}),
            model_results.get('rag_metrics', {}),
        ]
        
        for source in sources:
            if any(key in source for key in bertscore_metric_keys):
                has_bertscore = True
                for key in bertscore_metric_keys:
                    if key in source and source[key] is not None:
                        metric_name = key.replace('bert_', '').replace('_', ' ').title()
                        bertscore_data.append({
                            'Modelo': model_name,
                            'Métrica': metric_name,
                            'Valor': source[key]
                        })
                break
    
    if not has_bertscore:
        return  # No BERTScore data to display
    
    if bertscore_data:
        st.subheader("📊 Análisis Detallado BERTScore")
        
        # Create DataFrame
        df_bert = pd.DataFrame(bertscore_data)
        
        # Create grouped bar chart
        fig = px.bar(
            df_bert,
            x='Modelo',
            y='Valor',
            color='Métrica',
            barmode='group',
            title='Métricas BERTScore por Modelo',
            labels={'Valor': 'Puntuación BERTScore', 'Modelo': 'Modelos de Embedding'},
            range_y=[0, 1]
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            xaxis_title="Modelos de Embedding",
            yaxis_title="Puntuación BERTScore",
            legend_title="Métricas BERTScore",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a summary table for BERTScore
        st.subheader("📋 Resumen BERTScore")
        
        # Pivot the data for better table display
        df_pivot = df_bert.pivot(index='Modelo', columns='Métrica', values='Valor')
        
        # Format values to 3 decimal places and add color-coding
        df_formatted = df_pivot.round(3)
        
        # Function to determine color based on BERTScore interpretation ranges
        def get_bertscore_color(value):
            if pd.isna(value):
                return 'background-color: #f0f0f0'  # Light gray for missing values
            elif value > 0.8:
                return 'background-color: #d4edda; color: #155724'  # Green for Excelente
            elif value >= 0.6:
                return 'background-color: #fff3cd; color: #856404'  # Yellow for Bueno
            else:
                return 'background-color: #f8d7da; color: #721c24'  # Red for Necesita mejora
        
        # Apply color styling to the dataframe
        styled_df = df_formatted.style.map(get_bertscore_color)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Add interpretation help
        with st.expander("💡 Interpretación de BERTScore", expanded=False):
            st.markdown("""
            **Cómo interpretar BERTScore:**
            
            - **Precision > 0.8**: Excelente - La respuesta generada es muy precisa
            - **Precision 0.6-0.8**: Buena - La respuesta tiene buena precisión semántica
            - **Precision < 0.6**: Necesita mejora - La respuesta puede tener información irrelevante
            
            - **Recall > 0.8**: Excelente - La respuesta cubre muy bien el contenido de referencia
            - **Recall 0.6-0.8**: Buena - La respuesta cubre adecuadamente el contenido
            - **Recall < 0.6**: Necesita mejora - La respuesta puede omitir información importante
            
            - **F1 > 0.8**: Excelente balance entre precisión y cobertura
            - **F1 0.6-0.8**: Buen balance general
            - **F1 < 0.6**: El modelo necesita mejoras en precisión o cobertura
            """)

def get_improvement_status_icon(improvement: float) -> str:
    """Return a simple icon for improvement status."""
    if improvement > 0.01: # Small positive change
        return "⬆️"
    elif improvement < -0.01: # Small negative change
        return "⬇️"
    else:
        return "➡️"

def get_metric_quality(value: float) -> str:
    """Return qualitative assessment of metric value"""
    if value >= 0.7:
        return "🟢 Excelente"
    elif value >= 0.5:
        return "🟡 Bueno"
    elif value >= 0.3:
        return "🟠 Regular"
    else:
        return "🔴 Bajo"


def get_improvement_status(improvement: float) -> str:
    """Return status icon for improvement"""
    if improvement > 0.05:
        return "🚀 Mejora significativa"
    elif improvement > 0:
        return "📈 Mejora leve"
    elif improvement < -0.05:
        return "📉 Empeoramiento significativo"
    else:
        return "➖ Sin cambio significativo"


def create_metric_across_k_chart(avg_before: Dict, avg_after: Dict, metric_type: str, k_values: List[int], use_llm_reranker: bool):
    """Create a chart showing one metric across different K values"""
    
    # Prepare data for the chart - X axis: K values, Y axis: metric values
    k_labels = [str(k) for k in k_values]
    values_before = []
    values_after = []
    
    # Collect metric values for each K
    for k in k_values:
        metric_key = f"{metric_type}@{k}"  # Remove 'avg_' prefix - data should be in format 'precision@5', not 'avg_precision@5'
        
        if metric_key in avg_before:
            values_before.append(avg_before[metric_key])
        else:
            values_before.append(None)
            
        if use_llm_reranker and avg_after and metric_key in avg_after:
            values_after.append(avg_after[metric_key])
        else:
            values_after.append(None)
    
    # Check if we have any data for this metric
    has_before_data = any(v is not None for v in values_before)
    has_after_data = any(v is not None for v in values_after)
    
    if not has_before_data:
        st.write(f"No hay datos para {metric_type.upper()}")
        return
    
    # Create the chart
    fig = go.Figure()
    
    # Add before metrics line
    valid_before_values = [v if v is not None else 0 for v in values_before]
    fig.add_trace(go.Scatter(
        name='Antes LLM' if use_llm_reranker and has_after_data else metric_type.upper(),
        x=k_labels,
        y=valid_before_values,
        mode='lines+markers+text',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        text=[f"{v:.3f}" if v is not None else "" for v in values_before],
        textposition='top center',
        textfont=dict(size=10)
    ))
    
    # Add after metrics line if available
    if use_llm_reranker and has_after_data:
        valid_after_values = [v if v is not None else 0 for v in values_after]
        fig.add_trace(go.Scatter(
            name='Después LLM',
            x=k_labels,
            y=valid_after_values,
            mode='lines+markers+text',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            text=[f"{v:.3f}" if v is not None else "" for v in values_after],
            textposition='bottom center',
            textfont=dict(size=10)
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{metric_type.upper()} por Valor de K",
        xaxis_title="Valor de K",
        yaxis_title=f"Valor de {metric_type.upper()}",
        height=400,
        showlegend=use_llm_reranker and has_after_data,
        xaxis=dict(type='category'),  # Treat K values as categories
        yaxis=dict(range=[0, max(max(valid_before_values), max(valid_after_values) if has_after_data else 0) * 1.1])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_single_metric_chart(avg_before: Dict, avg_after: Dict, metric_type: str, use_llm_reranker: bool):
    """Create a chart for single-value metrics (like MRR)"""
    
    # Get metric values (no @k suffix for single metrics like MRR)
    metric_key = metric_type  # For MRR, the key is just 'mrr', not 'avg_mrr'
    
    before_val = avg_before.get(metric_key)
    after_val = avg_after.get(metric_key) if use_llm_reranker and avg_after else None
    
    if before_val is None:
        st.write(f"No hay datos para {metric_type.upper()}")
        return
    
    # Create bar chart for single value comparison
    fig = go.Figure()
    
    categories = ['Antes LLM'] if not (use_llm_reranker and after_val is not None) else ['Antes LLM', 'Después LLM']
    values = [before_val] if not (use_llm_reranker and after_val is not None) else [before_val, after_val]
    colors = ['blue'] if len(values) == 1 else ['blue', 'green']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition='auto',
        name=metric_type.upper()
    ))
    
    fig.update_layout(
        title=f"{metric_type.upper()} (Valor Único)",
        xaxis_title="Condición",
        yaxis_title=f"Valor de {metric_type.upper()}",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_complete_metrics_table(avg_before: Dict, avg_after: Dict, k_values: List[int], metric_types: List[str], use_llm_reranker: bool):
    """Display complete metrics table for all K values"""
    
    # Prepare table data
    table_data = []
    
    for metric_type in metric_types:
        if metric_type == 'mrr':
            # Handle single-value metrics (no @k) - MRR key should be just 'mrr'
            metric_key = metric_type  # For MRR, use 'mrr' not 'avg_mrr'
            
            before_val = avg_before.get(metric_key)
            if before_val is not None and not np.isnan(before_val):
                row = {
                    'Métrica': metric_type.upper(),
                    'Valor': f"{before_val:.3f}"
                }
                
                if use_llm_reranker and avg_after and metric_key in avg_after:
                    after_val = avg_after.get(metric_key)
                    if after_val is not None and not np.isnan(after_val):
                        improvement = after_val - before_val
                        improvement_pct = (improvement / before_val * 100) if before_val > 0 else 0
                        
                        row.update({
                            'Antes LLM': f"{before_val:.3f}",
                            'Después LLM': f"{after_val:.3f}",
                            'Mejora': f"{improvement:+.3f}",
                            'Mejora %': f"{improvement_pct:+.1f}%"
                        })
                        # Remove the single 'Valor' column since we have before/after
                        del row['Valor']
                    else:
                        # After value is None/NaN
                        row.update({
                            'Antes LLM': f"{before_val:.3f}",
                            'Después LLM': 'N/A',
                            'Mejora': 'N/A',
                            'Mejora %': 'N/A'
                        })
                        del row['Valor']
                
                table_data.append(row)
        else:
            # Handle K-based metrics (precision, recall, f1)
            for k in k_values:
                metric_key = f"{metric_type}@{k}"  # Remove 'avg_' prefix
                
                if metric_key in avg_before:
                    row = {
                        'Métrica': f"{metric_type.upper()}@{k}",
                        'Valor': f"{avg_before[metric_key]:.3f}"
                    }
                    
                    if use_llm_reranker and avg_after and metric_key in avg_after:
                        after_val = avg_after[metric_key]
                        improvement = after_val - avg_before[metric_key]
                        improvement_pct = (improvement / avg_before[metric_key] * 100) if avg_before[metric_key] > 0 else 0
                        
                        row.update({
                            'Antes LLM': f"{avg_before[metric_key]:.3f}",
                            'Después LLM': f"{after_val:.3f}",
                            'Mejora': f"{improvement:+.3f}",
                            'Mejora %': f"{improvement_pct:+.1f}%"
                        })
                        # Remove the single 'Valor' column since we have before/after
                        del row['Valor']
                    
                    table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Style the dataframe for better visualization
        if use_llm_reranker and 'Mejora %' in df.columns:
            def highlight_improvements(val):
                if 'Mejora' in str(val) and '+' in str(val):
                    return 'background-color: #d4edda'  # Light green
                elif 'Mejora' in str(val) and '-' in str(val):
                    return 'background-color: #f8d7da'  # Light red
                return ''
            
            styled_df = df.style.applymap(highlight_improvements)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.write("No hay datos de métricas disponibles.")


def display_retrieval_metrics_explanation():
    """Display accordion with retrieval metrics explanations"""
    with st.expander("📚 Explicación de Métricas de Recuperación", expanded=False):
        st.markdown("""
        **Precision@k**: Fracción de documentos recuperados en el top-k que son relevantes.
        > **Fórmula**: `Precision@k = Documentos relevantes en top-k / k`
        
        **Recall@k**: Fracción de documentos relevantes que fueron recuperados en el top-k.
        > **Fórmula**: `Recall@k = Documentos relevantes en top-k / Total documentos relevantes`
        
        **F1@k**: Media armónica entre Precision@k y Recall@k.
        > **Fórmula**: `F1@k = 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)`
        
        **NDCG@k**: Ganancia acumulada descontada normalizada que considera el orden de los resultados.
        > **Fórmula**: `NDCG@k = DCG@k / IDCG@k`
        
        **MAP@k**: Precisión promedio hasta el corte k para múltiples consultas.
        > **Fórmula**: `MAP@k = Σ(Precision@i × relevancia_i) / Documentos relevantes`
        
        **MRR**: Rango recíproco promedio del primer documento relevante encontrado.
        > **Fórmula**: `MRR = 1 / rank del primer documento relevante`
        """)


def display_methodology_section():
    """Display comprehensive methodology section with evaluation process explanation"""
    with st.expander("🔬 Metodología de Evaluación", expanded=False):
        st.markdown("""
        ## 📋 Metodología Completa del Sistema de Evaluación RAG
        
        ### 🎯 1. Obtención de Scores de Recuperación (Pre y Post Reranking)
        
        **Proceso de Evaluación:**
        - **Ground Truth**: Utilizamos los enlaces de Microsoft Learn contenidos en las respuestas aceptadas de Stack Overflow como referencias de documentos relevantes
        - **Similitud de Enlaces**: Normalizamos los URLs eliminando fragmentos (#) y parámetros de consulta (?) para hacer comparaciones exactas
        - **Métricas de Recuperación**: Calculamos Precision@k, Recall@k, F1@k, NDCG@k, MAP@k y MRR comparando los documentos recuperados vs. los enlaces de referencia
        - **Evaluación por Pregunta**: Cada pregunta se evalúa individualmente y luego se promedian los resultados across todas las preguntas
        
        ### 🤖 2. Estrategia de Reranking con LLM
        
        **Método de Reordenamiento:**
        - **Modelo**: OpenAI GPT-3.5-turbo para reordenar los documentos recuperados
        - **Prompt Engineering**: Se envía la pregunta original junto con hasta 4000 caracteres de cada documento recuperado (con truncación inteligente)
        - **Proceso**: El LLM ordena los documentos del 1 al 10 basándose en relevancia a la pregunta
        - **Manejo de Errores**: Si el LLM no puede generar un ranking válido, se mantiene el orden original de similitud coseno
        - **Temperatura**: 0.1 para generar rankings consistentes y determinísticos
        
        ### 📄 3. Agregación de Chunks a Documentos
        
        **Estrategia de Conversión:**
        - **Problema**: Los embeddings están calculados a nivel de chunks (fragmentos de documentos), pero necesitamos documentos completos para evaluación
        - **Solución**: DocumentAggregator que convierte chunks en documentos mediante agrupación inteligente
        - **Configuración**: chunk_multiplier (por defecto 3.0) determina cuántos chunks recuperar inicialmente (ej: 30 chunks → 10 documentos)
        - **Normalización de Enlaces**: URLs se normalizan eliminando fragmentos (#) y parámetros (?) para agrupación consistente
        - **Agregación de Contenido**: Chunks del mismo documento se concatenan preservando el contexto completo
        - **Deduplicación**: Se eliminan documentos duplicados basándose en enlaces normalizados
        - **Orden Preservado**: Se mantiene el orden de relevancia del chunk con mayor score por documento
        
        **Límites de Contenido Optimizados:**
        - **Generación de Respuestas**: 2000 caracteres por documento (mejorado desde 500)
        - **Contexto RAGAS**: 3000 caracteres por documento (mejorado desde 1000)
        - **Reranking LLM**: 4000 caracteres por documento (mejorado desde 3000)
        - **Evaluación BERTScore**: Sin límite - contenido completo (anteriormente limitado)
        - **Objetivo**: Aprovechar completamente los documentos agregados vs. truncación severa de chunks individuales
        
        ### 🔍 4. Metodología de Evaluación General
        
        **Bibliotecas y Frameworks:**
        - **RAGAS**: Framework oficial para evaluación de sistemas RAG con métricas validadas científicamente
        - **BERTScore**: Evaluación semántica usando representaciones contextuales de BERT
        - **OpenAI API**: Para generación de respuestas y reranking de documentos
        - **scikit-learn**: Para cálculo de similitud coseno en la recuperación inicial
        - **Sentence Transformers**: Para generación de embeddings de consultas
        
        **Proceso de Evaluación:**
        1. **Carga de Datos**: Parquet files con embeddings pre-calculados (~187K chunks de documentos)
        2. **Generación de Query**: Embedding de la pregunta usando el modelo correspondiente
        3. **Recuperación Inicial**: Top-K chunks usando similitud coseno (donde K = target_documents × chunk_multiplier)
        4. **Agregación de Documentos**: Conversión de chunks a documentos completos mediante agrupación por enlace normalizado
        5. **Reranking** (opcional): Reordenamiento con LLM de los documentos agregados
        6. **Generación RAG**: Respuesta usando contexto de documentos completos + OpenAI
        7. **Evaluación RAGAS**: Métricas automáticas usando framework RAGAS
        8. **BERTScore**: Evaluación semántica adicional
        
        ### 📊 4. Cálculo de Métricas Específicas
        
        **Métricas de Recuperación:**
        - **Precision@k**: `Documentos relevantes en top-k / k`
        - **Recall@k**: `Documentos relevantes en top-k / Total documentos relevantes`
        - **F1@k**: `2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)`
        - **NDCG@k**: `DCG@k / IDCG@k` (considera orden de resultados)
        - **MAP@k**: `Σ(Precision@i × relevancia_i) / Documentos relevantes`
        - **MRR**: `1 / rank del primer documento relevante`
        
        **Métricas RAGAS:**
        - **Faithfulness**: Evalúa fidelidad al contexto usando verificación de claims
        - **Answer Relevancy**: Mide relevancia usando similitud de embeddings pregunta-respuesta
        - **Answer Correctness**: Combina exactitud factual y completitud semántica
        - **Semantic Similarity**: Similitud semántica entre respuesta generada y esperada
        
        **Métricas BERTScore:**
        - **BERT Precision**: Proporción de tokens de respuesta presentes en referencia (usando BERT embeddings)
        - **BERT Recall**: Proporción de tokens de referencia presentes en respuesta (usando BERT embeddings)
        - **BERT F1**: Media armónica entre BERT Precision y BERT Recall
        
        ### 🔄 6. Diagrama de Proceso (1 Modelo, 1 Pregunta)
        
        ```
        📝 PREGUNTA + MS LINKS (Ground Truth)
                        ↓
        🔤 GENERACIÓN DE EMBEDDING (Sentence Transformers / OpenAI)
                        ↓
        🔍 RECUPERACIÓN DE CHUNKS (Similitud Coseno Top-K)
               K = target_documents × chunk_multiplier
                        ↓
        📄 AGREGACIÓN DE DOCUMENTOS (DocumentAggregator)
               ├── Normalización de enlaces
               ├── Agrupación por documento
               ├── Concatenación de contenido
               └── Deduplicación → Top-10 documentos
                        ↓
                    📊 EVALUACIÓN PRE-RERANKING
                    (Precision, Recall, F1, NDCG, MAP, MRR)
                        ↓
        🤖 RERANKING LLM (GPT-3.5-turbo + 4000 chars/doc) [OPCIONAL]
                        ↓
                    📈 EVALUACIÓN POST-RERANKING
                    (Mismas métricas de recuperación)
                        ↓
        🎭 GENERACIÓN DE RESPUESTA (GPT-3.5-turbo + 2000 chars/doc)
                        ↓
                    🔬 EVALUACIÓN RAG
                    ├── RAGAS (Faithfulness, Answer Relevancy, etc.) [3000 chars/doc]
                    └── BERTScore (Precision, Recall, F1) [Sin límite]
                        ↓
        📊 RESULTADOS FINALES (Promedios + Individuales)
        
        🔁 REPETIR PARA N PREGUNTAS → PROMEDIAR RESULTADOS
        ```
        
        ### 🎯 7. Garantías de Calidad Científica
        
        **Reproducibilidad:**
        - Evaluación completamente determinística (temperatura 0.1 en LLM)
        - Sin operaciones aleatorias en el cálculo de métricas
        - Selección de preguntas con seed fijo (42) solo durante configuración
        - Mismos datasets y embeddings pre-calculados
        
        **Validación:**
        - Sin valores simulados o aleatorios
        - Verificación de integridad de datos en cada paso
        - Logging detallado de errores y excepciones
        - Validación de estructura de resultados JSON
        
        **Escalabilidad:**
        - Procesamiento batch eficiente
        - Manejo de memoria con garbage collection
        - Paralelización cuando es posible
        - Arquitectura modular para extensibilidad
        """)

def display_rag_metrics_explanation():
    """Display accordion with RAG metrics explanations"""
    with st.expander("🤖 Explicación de Métricas RAG", expanded=False):
        st.markdown("""
        ### Métricas RAGAS
        
        **Faithfulness**: Mide qué tan fiel es la respuesta generada al contexto recuperado.
        > **Descripción**: Evalúa si las afirmaciones en la respuesta están respaldadas por el contexto.
        
        **Answer Relevance**: Evalúa qué tan relevante es la respuesta generada para la pregunta.
        > **Descripción**: Mide si la respuesta aborda directamente lo que se preguntó.
        
        **Answer Correctness**: Combina exactitud factual y completitud de la respuesta.
        > **Descripción**: Evalúa si la respuesta es factualmente correcta y completa.
        
        **Answer Similarity**: Mide la similitud semántica entre la respuesta generada y la esperada.
        > **Descripción**: Compara la respuesta del modelo con una respuesta de referencia usando embeddings.
        
        ### Métricas BERTScore
        
        **BERT Precision**: Mide qué proporción de tokens en la respuesta generada están presentes en la respuesta de referencia.
        > **Descripción**: Evalúa la precisión a nivel de token usando representaciones contextuales de BERT.
        
        **BERT Recall**: Mide qué proporción de tokens en la respuesta de referencia están presentes en la respuesta generada.
        > **Descripción**: Evalúa la cobertura a nivel de token usando representaciones contextuales de BERT.
        
        **BERT F1**: Media armónica entre BERT Precision y BERT Recall.
        > **Descripción**: Combina precisión y recall de BERTScore en una sola métrica balanceada.
        
        **Nota**: BERTScore utiliza embeddings contextuales de BERT para evaluar similitud semántica más allá de coincidencias exactas de texto.
        """)