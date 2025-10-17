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


def get_reranking_labels(reranking_method: str = 'standard'):
    """Get before/after labels based on reranking method"""
    if reranking_method == 'crossencoder':
        return {
            'before': 'Antes CrossEncoder',
            'after': 'Después CrossEncoder',
            'before_short': 'Antes CE',
            'after_short': 'Después CE'
        }
    else:
        return {
            'before': 'Antes LLM',
            'after': 'Después LLM', 
            'before_short': 'Antes LLM',
            'after_short': 'Después LLM'
        }


def display_enhanced_cumulative_metrics(results: Dict[str, Any], model_name: str, use_llm_reranker: bool, config: Dict = None):
    """
    Enhanced display for cumulative metrics with clear before/after reranking sections
    """
    num_questions = results['num_questions_evaluated']
    avg_before = results['avg_before_metrics']
    avg_after = results.get('avg_after_metrics', {})
    
    # Determine reranking method from config
    reranking_method = config.get('reranking_method', 'standard') if config else 'standard'
    
    st.success(f"✅ Evaluación completada para {num_questions} preguntas con modelo {model_name}")
    
    # Main metrics overview
    display_main_metrics_overview(avg_before, avg_after, use_llm_reranker, reranking_method)
    
    # NEW: Enhanced scoring methodology display
    display_enhanced_scoring_section(avg_before, avg_after, use_llm_reranker, model_name, reranking_method)
    
    # Before/After reranking comparison section
    if use_llm_reranker and avg_after:
        display_before_after_comparison(avg_before, avg_after, reranking_method)
    
    # Metrics by K values section
    display_metrics_by_k_values(avg_before, avg_after, use_llm_reranker, reranking_method, config)
    
    # Performance visualization
    display_performance_charts(avg_before, avg_after, use_llm_reranker, model_name, reranking_method, config)

    # NEW: Document-level score analysis
    display_document_score_analysis(results, use_llm_reranker, model_name, reranking_method)

    # New section: RAG Metrics Summary (for single model, adapt results structure)
    # Create a dummy results dict for display_rag_metrics_summary
    single_model_results_for_rag = {model_name: results}
    display_rag_metrics_summary(single_model_results_for_rag, use_llm_reranker, config)


def display_main_metrics_overview(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, reranking_method: str = 'standard'):
    """Display main metrics overview with key performance indicators"""
    
    st.subheader("📊 Resumen de Métricas Principales")
    
    # Select key metrics to highlight (updated with new metrics)
    key_metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']
    
    # Determine display names based on reranking method
    if reranking_method == 'crossencoder':
        before_title = "🔍 Antes de usar CrossEncoder"
        after_title = "🧠 Después de usar CrossEncoder"
        help_suffix = "CrossEncoder"
    else:
        before_title = "🔍 Antes del LLM Reranking"
        after_title = "🤖 Después del LLM Reranking"
        help_suffix = "LLM reranking"
    
    if use_llm_reranker and avg_after:
        # Show before and after side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {before_title}")
            for metric in key_metrics:
                if metric in avg_before:
                    value = avg_before[metric]
                    st.metric(
                        label=metric.upper().replace('@', ' @ '),
                        value=f"{value:.3f}",
                        help=f"Valor promedio de {metric} usando solo embedding retrieval"
                    )
        
        with col2:
            st.markdown(f"#### {after_title}")
            for metric in key_metrics:
                if metric in avg_after:
                    after_value = avg_after[metric]
                    before_value = avg_before.get(metric, 0)
                    delta = after_value - before_value
                    
                    st.metric(
                        label=metric.upper().replace('@', ' @ '),
                        value=f"{after_value:.3f}",
                        delta=f"{delta:+.3f}",
                        help=f"Valor después del {help_suffix}. Delta vs embedding-only: {delta:+.3f}"
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


def display_before_after_comparison(avg_before: Dict, avg_after: Dict, reranking_method: str = 'standard'):
    """Display dedicated before/after reranking comparison section"""
    
    # Determine title based on reranking method
    if reranking_method == 'crossencoder':
        title = "🔄 Comparación: Antes vs Después de usar CrossEncoder"
    else:
        title = "🔄 Comparación: Antes vs Después del LLM Reranking"
    
    st.subheader(title)
    
    # Prepare comparison data with dynamic column names
    comparison_data = []
    metrics_to_compare = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr', 'ndcg@5']  # MRR is a single value, not per-K
    
    # Get labels based on reranking method
    labels = get_reranking_labels(reranking_method)
    before_col = labels['before']
    after_col = labels['after']
    
    for metric in metrics_to_compare:
        if metric in avg_before and metric in avg_after:
            before_val = avg_before[metric]
            after_val = avg_after[metric]
            improvement = after_val - before_val
            improvement_pct = (improvement / before_val * 100) if before_val > 0 else 0
            
            comparison_data.append({
                'Métrica': metric.upper().replace('@', ' @ '),
                before_col: f"{before_val:.3f}",
                after_col: f"{after_val:.3f}",
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


def display_metrics_by_k_values(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, reranking_method: str = 'standard', config: Dict = None):
    """Display metrics organized by K values in 2x3 matrix format with charts and table"""
    
    st.subheader("📈 Métricas por Valores de K")
    
    # Get top_k from config, default to 50 if not provided
    top_k = config.get('top_k', 50) if config else 50
    k_values = list(range(1, top_k + 1))  # Support k values from 1 to top_k
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


def display_performance_charts(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, model_name: str, reranking_method: str = 'standard', config: Dict = None):
    """Display comprehensive performance visualization"""
    
    st.subheader("📈 Visualización de Rendimiento")
    
    # Performance across K values
    # Get top_k from config, default to 50 if not provided
    top_k = config.get('top_k', 50) if config else 50
    k_values = list(range(1, top_k + 1))  # Support k values from 1 to top_k
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
    
    # Determine reranking method from config
    reranking_method = config.get('reranking_method', 'standard') if config else 'standard'

    # Enhanced multi-model scoring analysis section only
    display_multi_model_scoring_analysis(results, use_llm_reranker, reranking_method)

    # New section: All metrics by K for all models
    display_all_metrics_by_k_for_all_models(results, use_llm_reranker, config)

    # Table with data for metrics by K
    create_all_metrics_by_k_table(results, use_llm_reranker, config)

    # Add metric definitions table before RAG metrics
    display_metric_definitions_table()

    # New section: RAG Metrics Summary
    display_rag_metrics_summary(results, use_llm_reranker, config)


def display_all_metrics_by_k_for_all_models(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, config: Dict = None):
    """
    Displays a 2x3 grid of plots, each showing a specific metric across K values
    for all models, with before/after LLM lines.
    """
    st.subheader("📈 Rendimiento Detallado por Métrica y K")

    # Get top_k from config, default to 50 if not provided
    top_k = config.get('top_k', 50) if config else 50
    k_values = list(range(1, top_k + 1))  # Support k values from 1 to top_k
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


def create_all_metrics_by_k_table(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, config: Dict = None):
    """
    Displays a comprehensive table showing all models' metrics across K values,
    with before/after LLM scores.
    """
    st.subheader("📋 Tabla Detallada de Métricas por K (Todos los Modelos)")

    # Get top_k from config, default to 50 if not provided
    top_k = config.get('top_k', 50) if config else 50
    k_values = list(range(1, top_k + 1))  # Support k values from 1 to top_k
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

def _format_metrics_for_llm_simplified(results_data: Dict[str, Any]) -> str:
    """Create a simplified metrics summary for LLM analysis when context is too large"""
    
    metrics_str = "=== RESUMEN SIMPLIFICADO DE MÉTRICAS ===\n\n"
    
    # Basic config info only
    config = results_data.get('config', {})
    metrics_str += f"📊 Configuración:\n"
    metrics_str += f"- Modelos evaluados: {config.get('models_evaluated', 'N/A')}\n"
    metrics_str += f"- Preguntas: {config.get('num_questions', 'N/A')}\n"
    metrics_str += f"- Top-K: {config.get('top_k', 'N/A')}\n"
    metrics_str += f"- Método reranking: {config.get('reranking_method', 'N/A')}\n\n"
    
    # Only key metrics for each model
    results = results_data.get('results', {})
    metrics_str += "📈 Resultados por Modelo (métricas clave):\n\n"
    
    for model_name, model_data in results.items():
        metrics_str += f"🤖 {model_name.upper()}:\n"
        
        # Only the most important metrics
        after_metrics = model_data.get('avg_after_metrics', {})
        rag_metrics = model_data.get('rag_metrics', {})
        
        # Key retrieval metrics
        f1_5 = after_metrics.get('f1@5', 0)
        mrr = after_metrics.get('mrr', 0)
        avg_score = after_metrics.get('model_avg_score', 0)
        
        metrics_str += f"  - F1@5: {f1_5:.3f}\n"
        metrics_str += f"  - MRR: {mrr:.3f}\n"
        metrics_str += f"  - Score Promedio: {avg_score:.3f}\n"
        
        # Key RAG metrics if available
        if rag_metrics.get('rag_available'):
            faith = rag_metrics.get('avg_faithfulness', 0)
            bert_f1 = rag_metrics.get('avg_bert_f1', 0)
            metrics_str += f"  - Faithfulness: {faith:.3f}\n"
            metrics_str += f"  - BERT F1: {bert_f1:.3f}\n"
        
        metrics_str += "\n"
    
    return metrics_str

def extract_scientific_metrics(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only essential metrics for scientific analysis.
    NO truncation, only selection of relevant data points with 2 decimal precision.
    """
    scientific_data = {
        "experimental_context": {
            "project": "RAG System Evaluation - Microsoft Learn Technical Documentation",
            "dataset_size": "~2,000 questions with ground-truth links",
            "corpus_size": "187,000 document chunks",
            "retrieval_method": "Document-level retrieval (not chunk-based)",
            "reranking_method": "CrossEncoder semantic reranking",
            "evaluation_scope": "Information retrieval quality + semantic similarity"
        },
        "models_evaluated": []
    }
    
    results = results_data.get('results', {})
    config = results_data.get('config', {})
    
    # Add configuration context
    scientific_data["configuration"] = {
        "top_k": config.get('top_k', 10),
        "reranking_enabled": config.get('reranking_method', 'none') != 'none',
        "aggregation_enabled": config.get('aggregate_method', 'none') != 'none',
        "questions_evaluated": config.get('num_questions', 0)
    }
    
    # Extract metrics for each model
    for model_name, model_data in results.items():
        before_metrics = model_data.get('avg_before_metrics', {})
        after_metrics = model_data.get('avg_after_metrics', {})
        rag_metrics = model_data.get('rag_metrics', {})
        
        model_metrics = {
            "model_name": model_name,
            "average_scores": {
                "before_reranking": round(before_metrics.get('model_avg_score', 0), 2),
                "after_reranking": round(after_metrics.get('model_avg_score', 0), 2)
            },
            "retrieval_metrics": {
                "precision@5": {
                    "before": round(before_metrics.get('precision@5', 0), 2),
                    "after": round(after_metrics.get('precision@5', 0), 2)
                },
                "recall@5": {
                    "before": round(before_metrics.get('recall@5', 0), 2),
                    "after": round(after_metrics.get('recall@5', 0), 2)
                },
                "f1@5": {
                    "before": round(before_metrics.get('f1@5', 0), 2),
                    "after": round(after_metrics.get('f1@5', 0), 2)
                },
                "map": {
                    "before": round(before_metrics.get('map', 0), 2),
                    "after": round(after_metrics.get('map', 0), 2)
                },
                "mrr": {
                    "before": round(before_metrics.get('mrr', 0), 2),
                    "after": round(after_metrics.get('mrr', 0), 2)
                },
                "ndcg@10": {
                    "before": round(before_metrics.get('ndcg@10', 0), 2),
                    "after": round(after_metrics.get('ndcg@10', 0), 2)
                }
            }
        }
        
        # Add semantic quality metrics if available
        if rag_metrics.get('rag_available', False):
            model_metrics["semantic_quality"] = {
                "ragas_faithfulness": {
                    "before": round(rag_metrics.get('avg_faithfulness', 0), 2),
                    "after": round(rag_metrics.get('avg_faithfulness_after', rag_metrics.get('avg_faithfulness', 0)), 2)
                },
                "ragas_answer_relevance": {
                    "before": round(rag_metrics.get('avg_answer_relevance', 0), 2),
                    "after": round(rag_metrics.get('avg_answer_relevance_after', rag_metrics.get('avg_answer_relevance', 0)), 2)
                },
                "bert_f1": {
                    "before": round(rag_metrics.get('avg_bert_f1', 0), 2),
                    "after": round(rag_metrics.get('avg_bert_f1_after', rag_metrics.get('avg_bert_f1', 0)), 2)
                }
            }
        else:
            model_metrics["semantic_quality"] = {
                "note": "RAGAS and BERTScore metrics not available (retrieval-only mode)"
            }
        
        scientific_data["models_evaluated"].append(model_metrics)
    
    return scientific_data

def _generate_fallback_analysis(results_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate basic analysis without LLM when all else fails"""
    
    results = results_data.get('results', {})
    config = results_data.get('config', {})
    
    # Find best performing model
    best_model = ""
    best_f1 = 0
    total_models = len(results)
    
    for model_name, model_data in results.items():
        f1_5 = model_data.get('avg_after_metrics', {}).get('f1@5', 0)
        if f1_5 > best_f1:
            best_f1 = f1_5
            best_model = model_name
    
    conclusions = f"""
    📊 **Análisis Automático de Resultados**
    
    Se evaluaron {total_models} modelos con {config.get('num_questions', 'N/A')} preguntas usando top-k={config.get('top_k', 'N/A')}.
    
    🏆 **Mejor Modelo:** {best_model} (F1@5: {best_f1:.3f})
    
    📈 **Observaciones:**
    - Método de reranking: {config.get('reranking_method', 'N/A')}
    - Se aplicaron métricas RAG completas incluyendo RAGAS y BERTScore
    - Los resultados están disponibles para análisis detallado en las visualizaciones
    """
    
    improvements = """
    💡 **Posibles Mejoras:**
    
    1. **Análisis Detallado:** Revisa las métricas individuales por modelo en las tablas de rendimiento
    2. **Comparación Visual:** Utiliza los gráficos de rendimiento para identificar patrones
    3. **Optimización:** Considera ajustar el valor de top-k basado en los resultados
    4. **RAG Metrics:** Analiza las métricas de faithfulness y BERT F1 para evaluar calidad de respuestas
    5. **Reranking:** Evalúa si el método de reranking está mejorando los resultados
    """
    
    return {
        'conclusions': conclusions,
        'improvements': improvements
    }

def _format_metrics_for_llm(results_data: Dict[str, Any]) -> str:
    """
    Formats the evaluation metrics data into a human-readable string for an LLM prompt.
    Includes size limits to prevent context length errors.
    """
    formatted_string = ""
    config = results_data.get('config', {})
    results = results_data.get('results', {})
    evaluation_info = results_data.get('evaluation_info', {})
    
    # Set a soft limit for the formatted string size
    SOFT_LIMIT = 8000  # Leave buffer for truncation handling

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
    
    # Get top_k from config and create a subset for LLM analysis to prevent context overflow
    top_k = config.get('top_k', 50)
    # Include key values up to the actual top_k value
    predefined_k_values = [1, 3, 5, 10, 20, 30, 40, 50]
    k_values = [k for k in predefined_k_values if k <= top_k]
    # Ensure we always include the actual top_k value if it's not in our predefined list
    if top_k not in k_values:
        k_values.append(top_k)
        k_values.sort()
    
    formatted_string += f"*Nota: Para el análisis LLM, se muestran valores k selectos hasta {top_k} para optimizar el uso del contexto.*\n\n"
    metrics_types = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']
    # Use STANDARD metric names from RAGAS and BERTScore libraries  
    standard_ragas_types = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness', 'answer_similarity', 'semantic_similarity']
    standard_bertscore_types = ['bert_precision', 'bert_recall', 'bert_f1']

    models_processed = 0
    total_models = len(results)
    
    for model_name, model_data in results.items():
        # Check size before adding each model
        if len(formatted_string) > SOFT_LIMIT and models_processed > 0:
            formatted_string += f"\n### ... y {total_models - models_processed} modelos más (omitidos para evitar exceder límites) ...\n"
            break
            
        formatted_string += f"### Modelo: {model_name}\n"
        before_metrics = model_data.get('avg_before_metrics', {})
        after_metrics = model_data.get('avg_after_metrics', {})
        individual_before_metrics = model_data.get('individual_before_metrics', [])
        individual_after_metrics = model_data.get('individual_after_metrics', [])
        models_processed += 1

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

def _parse_llm_response(full_response_content: str) -> Dict[str, str]:
    """
    Parse the LLM response and extract conclusions and improvements.
    
    Args:
        full_response_content: The full response text from the LLM
        
    Returns:
        Dictionary with 'conclusions' and 'improvements' keys
    """
    # Validate response content
    if not full_response_content or full_response_content.strip() == "":
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
        conclusions = full_response_content.strip()
        improvements = "Revisa el análisis anterior y considera implementar mejoras basadas en las observaciones."

    # Final validation - ensure we don't return empty strings
    if not conclusions or conclusions.strip() == "":
        conclusions = "✅ Análisis completado. Revisa las métricas mostradas para obtener insights sobre el rendimiento del sistema RAG."
    
    if not improvements or improvements.strip() == "":
        improvements = "1. Analiza las métricas individuales para identificar áreas de mejora\n2. Considera ajustar los parámetros del sistema\n3. Evalúa diferentes modelos de embedding o generativos"

    return {'conclusions': conclusions, 'improvements': improvements}

def _generate_with_deepseek(deepseek_client, formatted_metrics: str) -> Dict[str, str]:
    """Generate analysis using DeepSeek model"""
    # Usar el mismo prompt mejorado que Claude
    system_prompt = (
        "Eres un investigador senior especializado en evaluación de sistemas de recuperación de información y RAG. "
        "Tu tarea es proporcionar un análisis académico riguroso y MUY DETALLADO de los resultados experimentales. "

        "CONTEXTO EXPERIMENTAL: "
        "- Dominio: Sistema de recuperación de documentación técnica de Microsoft Learn "
        "- Dataset: ~2,000 consultas técnicas con enlaces ground-truth validados "
        "- Corpus: 187,000 chunks de documentos indexados a nivel de documento "
        "- Evaluación: Métricas IR estándar + métricas RAG (RAGAS) + BERTScore "
        "- Reranking: CrossEncoder neural (ms-marco-MiniLM-L-6-v2) con normalización Min-Max "
        "- Optimizaciones: Cache OpenAI, modelo semantic similarity global, gpt-3.5-turbo-0125 "

        "BASELINES ESPERADOS (incluye rangos de interpretación): "
        "- Precision@5: >0.60=buena | 0.40-0.60=moderada | <0.40=baja "
        "- Recall@5: >0.50=buena | 0.30-0.50=moderada | <0.30=baja "
        "- F1@5: >0.50=bueno | 0.35-0.50=moderado | <0.35=bajo "
        "- NDCG@10: >0.70=efectivo | 0.50-0.70=aceptable | <0.50=deficiente "
        "- MRR: >0.50=rápido | 0.30-0.50=aceptable | <0.30=lento "
        "- MAP@5: >0.50=buena | 0.35-0.50=moderada | <0.35=baja "
        "- RAGAS Faithfulness: >0.80=alta | 0.60-0.80=aceptable | <0.60=preocupante "
        "- RAGAS Answer Relevancy: >0.75=alta | 0.55-0.75=moderada | <0.55=baja "
        "- RAGAS Answer Correctness: >0.70=correcta | 0.50-0.70=parcial | <0.50=incorrecta "
        "- RAGAS Context Precision: >0.70=relevante | 0.50-0.70=ruido moderado | <0.50=mucho ruido "
        "- RAGAS Context Recall: >0.70=buena | 0.50-0.70=parcial | <0.50=incompleta "
        "- Semantic Similarity: >0.75=muy similar | 0.60-0.75=moderada | <0.60=diferente "
        "- BERTScore F1: >0.80=alta | 0.65-0.80=buena | <0.65=necesita mejoras "

        "REQUISITOS MANDATORY: "
        "1. ANÁLISIS POR MÉTRICA INDIVIDUAL: Cada métrica debe tener valor exacto, clasificación vs baseline, interpretación práctica, y comparación entre modelos "
        "2. COMPARACIÓN ENTRE MODELOS: Rankear modelos, cuantificar diferencias (X% más que Y), identificar mejor modelo "
        "3. IMPACTO CROSSENCODER: Calcular mejora relativa ((after-before)/before)*100%, explicar POR QUÉ mejora/empeora "
        "4. ANÁLISIS RAGAS/BERTSCORE: Evaluar cada métrica vs baseline, determinar si calidad es aceptable "
        "5. TRADE-OFFS Y PATRONES: Identificar trade-offs, comportamiento por K, anomalías "
        "6. RIGOR: Números exactos, cálculos de mejora, comparaciones cuantitativas "

        "FORMATO EN ESPAÑOL: "
        "## Conclusiones "
        "### 📊 Métricas IR "
        "[Analizar Precision, Recall, F1, NDCG, MAP, MRR con valores exactos, clasificación, interpretación, comparación modelos, impacto CrossEncoder] "
        "### 🤖 Métricas RAG "
        "[Analizar Faithfulness, Answer Relevancy, Correctness, Context Precision/Recall, Semantic Similarity, BERTScore con valores y clasificaciones] "
        "### 🏆 Ranking Modelos "
        "[Tabla con ranking y justificación numérica] "
        "### 🔄 Impacto CrossEncoder "
        "[Mejoras before/after con porcentajes] "
        "### ⚠️ Problemas "
        "[Métricas bajo baseline, trade-offs, anomalías] "
        ""
        "## 💡 Mejoras Prioritarias "
        "### 1. [Problema crítico] "
        "**Evidencia:** [Métrica con valor]. **Impacto:** [Consecuencia]. **Acción:** [Paso técnico]. **Resultado:** [Mejora esperada]. "
        "### 2-4. [Otros problemas con mismo formato] "
        ""
        "CRÍTICO: Números exactos, porcentajes de mejora, comparaciones cuantitativas, interpretaciones prácticas, exhaustividad. "
    )
    
    user_prompt = (
        f"Aquí están los resultados experimentales del sistema RAG en formato JSON:\n\n{formatted_metrics}\n\n"
        "Por favor, genera las conclusiones y mejoras siguiendo el formato especificado."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = deepseek_client.client.client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=messages,
        temperature=0.2,  # Low temperature for factual analysis
        max_tokens=2000
    )
    
    result = _parse_llm_response(response.choices[0].message.content)
    result['model_used'] = 'DeepSeek R1'
    result['full_prompt'] = f"{system_prompt}\n\nDatos experimentales:\n{formatted_metrics}\n\n{user_prompt}"
    return result

def _generate_with_claude(claude_client, formatted_metrics: str) -> Dict[str, str]:
    """Generate analysis using Claude model via OpenRouter"""
    system_prompt = (
        "Eres un investigador senior especializado en evaluación de sistemas de recuperación de información y RAG. "
        "Tu tarea es proporcionar un análisis académico riguroso y MUY DETALLADO de los resultados experimentales. "

        "CONTEXTO EXPERIMENTAL: "
        "- Dominio: Sistema de recuperación de documentación técnica de Microsoft Learn "
        "- Dataset: ~2,000 consultas técnicas con enlaces ground-truth validados "
        "- Corpus: 187,000 chunks de documentos indexados a nivel de documento "
        "- Evaluación: Métricas IR estándar + métricas RAG (RAGAS) + BERTScore "
        "- Reranking: CrossEncoder neural (ms-marco-MiniLM-L-6-v2) con normalización Min-Max "
        "- Optimizaciones: Cache OpenAI, modelo semantic similarity global, gpt-3.5-turbo-0125 "

        "BASELINES ESPERADOS PARA DOCUMENTACIÓN TÉCNICA: "
        "- Precision@5 > 0.60 = buena relevancia | 0.40-0.60 = moderada | < 0.40 = necesita mejoras "
        "- Recall@5 > 0.50 = buena cobertura | 0.30-0.50 = moderada | < 0.30 = baja cobertura "
        "- F1@5 > 0.50 = buen balance | 0.35-0.50 = moderado | < 0.35 = necesita ajustes "
        "- NDCG@10 > 0.70 = ranking efectivo | 0.50-0.70 = aceptable | < 0.50 = ranking deficiente "
        "- MRR > 0.50 = usuarios encuentran respuestas rápido | 0.30-0.50 = aceptable | < 0.30 = lento "
        "- MAP@5 > 0.50 = buena precisión promedio | 0.35-0.50 = moderada | < 0.35 = baja "
        "- CrossEncoder debería mejorar NDCG/MRR en >10% relativo "
        "- RAGAS Faithfulness > 0.80 = alta fidelidad | 0.60-0.80 = aceptable | < 0.60 = preocupante "
        "- RAGAS Answer Relevancy > 0.75 = muy relevante | 0.55-0.75 = moderada | < 0.55 = irrelevante "
        "- RAGAS Answer Correctness > 0.70 = correcta | 0.50-0.70 = parcial | < 0.50 = incorrecta "
        "- RAGAS Context Precision > 0.70 = contexto relevante | 0.50-0.70 = algo ruido | < 0.50 = mucho ruido "
        "- RAGAS Context Recall > 0.70 = buena cobertura | 0.50-0.70 = parcial | < 0.50 = incompleta "
        "- Semantic Similarity > 0.75 = muy similar | 0.60-0.75 = moderada | < 0.60 = diferente "
        "- BERTScore F1 > 0.80 = alta calidad | 0.65-0.80 = buena | < 0.65 = necesita mejoras "

        "REQUISITOS DE ANÁLISIS (MANDATORY - DEBES CUMPLIRLOS TODOS): "
        "1. ANÁLISIS POR MÉTRICA INDIVIDUAL: Para CADA métrica (Precision, Recall, F1, NDCG, MAP, MRR, Faithfulness, "
        "   Answer Relevancy, Context Precision, etc.), proporciona: "
        "   a) Valor numérico exacto encontrado "
        "   b) Clasificación según baselines (excelente/bueno/moderado/bajo) "
        "   c) Interpretación práctica (¿qué significa ese número para el usuario final?) "
        "   d) Comparación con baseline esperado "
        "   e) Si hay before/after, calcular mejora relativa: ((after-before)/before)*100% "

        "2. COMPARACIÓN ENTRE MODELOS: Si hay múltiples modelos (ada, mpnet, e5-large, minilm): "
        "   a) Rankear modelos por cada métrica importante (top-3 al menos) "
        "   b) Cuantificar diferencias: \"Modelo X supera a Y en Z% en métrica M\" "
        "   c) Identificar el mejor modelo general y justificar por qué "
        "   d) Identificar debilidades específicas de cada modelo "

        "3. IMPACTO DE CROSSENCODER: Analizar mejora/deterioro por reranking: "
        "   a) Para métricas de ranking (NDCG, MRR, MAP): calcular mejora relativa "
        "   b) Para métricas de relevancia (Precision, Recall): evaluar impacto "
        "   c) Determinar si CrossEncoder es estadísticamente significativo (>5% mejora) "
        "   d) Explicar POR QUÉ el CrossEncoder mejora/empeora ciertas métricas "

        "4. ANÁLISIS DE CALIDAD SEMÁNTICA: Para métricas RAGAS y BERTScore: "
        "   a) Evaluar cada métrica RAGAS individualmente vs baseline "
        "   b) Identificar si hay problemas de faithfulness, relevancia o correctness "
        "   c) Analizar BERTScore (precision, recall, F1) y semantic similarity "
        "   d) Determinar si la calidad de respuestas es aceptable para producción "

        "5. ANÁLISIS DE TRADE-OFFS Y PATRONES: "
        "   a) Identificar trade-offs (ej: alta precision pero bajo recall) "
        "   b) Analizar comportamiento a diferentes valores de K (K=1,5,10,20,50) "
        "   c) Detectar patrones inesperados o anomalías en los datos "
        "   d) Evaluar consistencia entre métricas relacionadas "

        "6. RIGOR ACADÉMICO: TODOS los hallazgos deben tener: "
        "   a) Números exactos de las métricas "
        "   b) Cálculos de mejora relativa cuando aplique "
        "   c) Comparaciones cuantitativas entre modelos/métodos "
        "   d) Ninguna afirmación sin soporte numérico "

        "FORMATO DE SALIDA OBLIGATORIO EN ESPAÑOL: "
        "## Conclusiones "
        ""
        "### 📊 Análisis de Métricas de Recuperación (IR) "
        "**Precision@K:** [Valor exacto]. [Clasificación vs baseline]. [Interpretación práctica]. [Comparación entre modelos con números]. [Impacto CrossEncoder con %]. "
        ""
        "**Recall@K:** [Valor exacto]. [Clasificación vs baseline]. [Interpretación práctica]. [Análisis de cobertura]. [Comparación entre modelos]. "
        ""
        "**F1@K:** [Valor exacto]. [Balance precision-recall]. [¿Es aceptable? ¿Por qué?]. [Mejores/peores modelos]. "
        ""
        "**NDCG@K:** [Valor exacto]. [Calidad del ranking]. [Impacto CrossEncoder: +X%]. [Comparación entre modelos]. "
        ""
        "**MAP@K:** [Valor exacto]. [Precisión promedio]. [Interpretación]. [Modelo ganador]. "
        ""
        "**MRR:** [Valor exacto]. [Velocidad encontrar primer relevante]. [¿Usuarios satisfechos?]. [CrossEncoder mejora: +X%]. "
        ""
        "### 🤖 Análisis de Métricas RAG (RAGAS + BERTScore) "
        "**Faithfulness:** [Valor]. [¿Respuestas fieles al contexto?]. [¿Hay alucinaciones?]. [Comparación modelos]. "
        ""
        "**Answer Relevancy:** [Valor]. [¿Respuestas relevantes?]. [Clasificación vs baseline]. [Análisis por modelo]. "
        ""
        "**Answer Correctness:** [Valor]. [¿Respuestas correctas?]. [Nivel de exactitud]. [Modelos más/menos correctos]. "
        ""
        "**Context Precision:** [Valor]. [¿Contexto recuperado relevante?]. [¿Hay ruido?]. [Impacto en calidad]. "
        ""
        "**Context Recall:** [Valor]. [¿Contexto cubre información necesaria?]. [¿Información faltante?]. "
        ""
        "**Semantic Similarity:** [Valor]. [Similitud respuesta-ground truth]. [Interpretación]. "
        ""
        "**BERTScore (P/R/F1):** [Valores]. [Calidad token-level]. [Comparación vs baselines]. "
        ""
        "### 🏆 Ranking de Modelos "
        "[Tabla o lista con ranking por métrica clave]. [Justificación numérica del mejor modelo]. [Análisis de fortalezas/debilidades]. "
        ""
        "### 🔄 Impacto del CrossEncoder "
        "[Tabla con mejoras before/after]. [Métricas que más mejoran]. [Métricas que empeoran (si aplica)]. [¿Vale la pena el costo computacional?]. "
        ""
        "### ⚠️ Problemas Identificados "
        "[Listar métricas por debajo de baseline]. [Trade-offs problemáticos]. [Anomalías detectadas]. [Métricas inconsistentes]. "
        ""
        "## 💡 Mejoras Prioritarias "
        ""
        "### 1. [Problema más crítico identificado] "
        "**Evidencia:** [Métrica específica con valor]. "
        "**Impacto:** [Consecuencia práctica para usuarios]. "
        "**Acción:** [Paso específico y técnico]. "
        "**Resultado Esperado:** [Mejora cuantificable esperada]. "
        ""
        "### 2. [Segundo problema en prioridad] "
        "[Mismo formato detallado]. "
        ""
        "### 3. [Optimización a largo plazo] "
        "[Mismo formato detallado]. "
        ""
        "### 4. [Mejora de eficiencia/costos] "
        "[Análisis costo-beneficio]. "
        ""
        "INSTRUCCIONES FINALES CRÍTICAS: "
        "- USA NÚMEROS EXACTOS EN TODO MOMENTO "
        "- CALCULA PORCENTAJES DE MEJORA/DETERIORO "
        "- COMPARA MODELOS CUANTITATIVAMENTE "
        "- INTERPRETA CADA MÉTRICA DE FORMA PRÁCTICA "
        "- SÉ EXHAUSTIVO - CUBRE TODAS LAS MÉTRICAS "
        "- ESCRIBE EN ESPAÑOL CLARO Y TÉCNICO "
        "- NO OMITAS NINGUNA MÉTRICA DISPONIBLE EN LOS DATOS "
    )
    
    user_prompt = f"""
Analiza los siguientes resultados experimentales de evaluación de modelos de embedding para documentación técnica de Microsoft Azure:

{formatted_metrics}

Proporciona conclusiones académicas rigurosas en español basadas únicamente en estos datos, siguiendo el formato especificado.
"""

    try:
        response = claude_client.client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet:beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            extra_headers={
                "HTTP-Referer": "https://azure-qa-support.app",
                "X-Title": "Azure Q&A Support System",
            }
        )
        
        full_response = response.choices[0].message.content
        
        # Parse the response to extract conclusions and improvements
        result = _parse_llm_response(full_response)
        result['model_used'] = 'Claude 3.5 Sonnet'
        result['full_prompt'] = f"{system_prompt}\n\nDatos experimentales:\n{formatted_metrics}\n\n{user_prompt}"
        return result
        
    except Exception as e:
        st.error(f"❌ Error generando análisis con Claude: {str(e)}")
        return None


def _generate_with_gemini(gemini_client, formatted_metrics: str) -> Dict[str, str]:
    """Generate analysis using Gemini model"""
    # Usar el mismo prompt mejorado que Claude y DeepSeek
    system_prompt = (
        "Eres un investigador senior especializado en evaluación de sistemas de recuperación de información y RAG. "
        "Tu tarea es proporcionar un análisis académico riguroso y MUY DETALLADO de los resultados experimentales. "

        "CONTEXTO EXPERIMENTAL: "
        "- Dominio: Sistema de recuperación de documentación técnica de Microsoft Learn "
        "- Dataset: ~2,000 consultas técnicas con enlaces ground-truth validados "
        "- Corpus: 187,000 chunks de documentos indexados a nivel de documento "
        "- Evaluación: Métricas IR estándar + métricas RAG (RAGAS) + BERTScore "
        "- Reranking: CrossEncoder neural (ms-marco-MiniLM-L-6-v2) con normalización Min-Max "
        "- Optimizaciones: Cache OpenAI, modelo semantic similarity global, gpt-3.5-turbo-0125 "

        "BASELINES ESPERADOS (incluye rangos de interpretación): "
        "- Precision@5: >0.60=buena | 0.40-0.60=moderada | <0.40=baja "
        "- Recall@5: >0.50=buena | 0.30-0.50=moderada | <0.30=baja "
        "- F1@5: >0.50=bueno | 0.35-0.50=moderado | <0.35=bajo "
        "- NDCG@10: >0.70=efectivo | 0.50-0.70=aceptable | <0.50=deficiente "
        "- MRR: >0.50=rápido | 0.30-0.50=aceptable | <0.30=lento "
        "- MAP@5: >0.50=buena | 0.35-0.50=moderada | <0.35=baja "
        "- RAGAS Faithfulness: >0.80=alta | 0.60-0.80=aceptable | <0.60=preocupante "
        "- RAGAS Answer Relevancy: >0.75=alta | 0.55-0.75=moderada | <0.55=baja "
        "- RAGAS Answer Correctness: >0.70=correcta | 0.50-0.70=parcial | <0.50=incorrecta "
        "- RAGAS Context Precision: >0.70=relevante | 0.50-0.70=ruido moderado | <0.50=mucho ruido "
        "- RAGAS Context Recall: >0.70=buena | 0.50-0.70=parcial | <0.50=incompleta "
        "- Semantic Similarity: >0.75=muy similar | 0.60-0.75=moderada | <0.60=diferente "
        "- BERTScore F1: >0.80=alta | 0.65-0.80=buena | <0.65=necesita mejoras "

        "REQUISITOS MANDATORY: "
        "1. ANÁLISIS POR MÉTRICA INDIVIDUAL: Cada métrica debe tener valor exacto, clasificación vs baseline, interpretación práctica, y comparación entre modelos "
        "2. COMPARACIÓN ENTRE MODELOS: Rankear modelos, cuantificar diferencias (X% más que Y), identificar mejor modelo "
        "3. IMPACTO CROSSENCODER: Calcular mejora relativa ((after-before)/before)*100%, explicar POR QUÉ mejora/empeora "
        "4. ANÁLISIS RAGAS/BERTSCORE: Evaluar cada métrica vs baseline, determinar si calidad es aceptable "
        "5. TRADE-OFFS Y PATRONES: Identificar trade-offs, comportamiento por K, anomalías "
        "6. RIGOR: Números exactos, cálculos de mejora, comparaciones cuantitativas "

        "FORMATO EN ESPAÑOL: "
        "## Conclusiones "
        "### 📊 Métricas IR "
        "[Analizar Precision, Recall, F1, NDCG, MAP, MRR con valores exactos, clasificación, interpretación, comparación modelos, impacto CrossEncoder] "
        "### 🤖 Métricas RAG "
        "[Analizar Faithfulness, Answer Relevancy, Correctness, Context Precision/Recall, Semantic Similarity, BERTScore con valores y clasificaciones] "
        "### 🏆 Ranking Modelos "
        "[Tabla con ranking y justificación numérica] "
        "### 🔄 Impacto CrossEncoder "
        "[Mejoras before/after con porcentajes] "
        "### ⚠️ Problemas "
        "[Métricas bajo baseline, trade-offs, anomalías] "
        ""
        "## 💡 Mejoras Prioritarias "
        "### 1. [Problema crítico] "
        "**Evidencia:** [Métrica con valor]. **Impacto:** [Consecuencia]. **Acción:** [Paso técnico]. **Resultado:** [Mejora esperada]. "
        "### 2-4. [Otros problemas con mismo formato] "
        ""
        "CRÍTICO: Números exactos, porcentajes de mejora, comparaciones cuantitativas, interpretaciones prácticas, exhaustividad. "
    )
    
    combined_prompt = (
        f"{system_prompt}\n\n"
        f"Aquí están los resultados experimentales del sistema RAG en formato JSON:\n\n{formatted_metrics}\n\n"
        "Por favor, genera las conclusiones y mejoras siguiendo el formato especificado."
    )
    
    response = gemini_client.generate_content(
        combined_prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 2000,
        }
    )
    
    result = _parse_llm_response(response.text)
    result['model_used'] = 'Gemini 1.5 Flash'
    result['full_prompt'] = combined_prompt
    return result

def generate_analysis_with_llm(results_data: Dict[str, Any], generative_model_name: str) -> Dict[str, str]:
    """
    Generates conclusions and improvements using an LLM based on evaluation results.
    Uses Claude, DeepSeek and Gemini models for scientific analysis.
    Returns a dictionary with 'conclusions' and 'improvements'.
    """
    
    # Validate supported models
    if generative_model_name not in ['claude-3.5-sonnet', 'deepseek-v3-chat', 'gemini-1.5-flash']:
        st.error(f"❌ Modelo {generative_model_name} no soportado. Solo se admiten: claude-3.5-sonnet, deepseek-v3-chat, gemini-1.5-flash")
        return None

    # Extract scientific metrics for analysis
    scientific_data = extract_scientific_metrics(results_data)
    
    # Convert scientific data to formatted string for LLM
    formatted_metrics = json.dumps(scientific_data, indent=2, ensure_ascii=False)
    
    
    try:
        # Initialize only DeepSeek and Gemini clients
        from src.services.storage.chromadb_utils import ChromaDBConfig
        import google.generativeai as genai
        from src.services.auth.openrouter_client import get_cached_deepseek_openrouter_client
        
        config = ChromaDBConfig.from_env()
        
        # Try with the selected model first
        first_error = None
        
        if generative_model_name == "claude-3.5-sonnet":
            try:
                # Use Claude via OpenRouter
                from src.services.auth.openrouter_client import OpenRouterClient
                claude_client = OpenRouterClient()
                if claude_client:
                    return _generate_with_claude(claude_client, formatted_metrics)
                else:
                    raise Exception("Cliente Claude no disponible - verificar API key de OpenRouter")
            except Exception as e:
                first_error = f"Claude: {str(e)}"
                st.warning(f"⚠️ Claude falló: {str(e)}. Intentando con Gemini...")
                
        elif generative_model_name == "deepseek-v3-chat":
            try:
                # Use DeepSeek via OpenRouter
                deepseek_client = get_cached_deepseek_openrouter_client()
                if deepseek_client:
                    return _generate_with_deepseek(deepseek_client, formatted_metrics)
                else:
                    raise Exception("Cliente DeepSeek no disponible - verificar API key de OpenRouter")
            except Exception as e:
                first_error = f"DeepSeek: {str(e)}"
                st.warning(f"⚠️ DeepSeek falló: {str(e)}. Intentando con Gemini...")
                
        elif generative_model_name == "gemini-1.5-flash":
            try:
                if config.gemini_api_key:
                    genai.configure(api_key=config.gemini_api_key)
                    gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                    return _generate_with_gemini(gemini_client, formatted_metrics)
                else:
                    raise Exception("API key de Gemini no configurada en variables de entorno")
            except Exception as e:
                first_error = f"Gemini: {str(e)}"
                st.warning(f"⚠️ Gemini falló: {str(e)}. Intentando con DeepSeek...")
        
        # If the primary model failed, try the fallback models
        if generative_model_name == "claude-3.5-sonnet" and config.gemini_api_key:
            # Claude failed, try Gemini
            try:
                genai.configure(api_key=config.gemini_api_key)
                gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                return _generate_with_gemini(gemini_client, formatted_metrics)
            except Exception as e:
                st.error(f"❌ Ambos modelos fallaron.")
                st.error(f"- {first_error}")
                st.error(f"- Gemini: {str(e)}")
                return None
                
        elif generative_model_name == "deepseek-v3-chat" and config.gemini_api_key:
            # DeepSeek failed, try Gemini
            try:
                genai.configure(api_key=config.gemini_api_key)
                gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                return _generate_with_gemini(gemini_client, formatted_metrics)
            except Exception as e:
                st.error(f"❌ Ambos modelos fallaron.")
                st.error(f"- {first_error}")
                st.error(f"- Gemini: {str(e)}")
                return None
                
        elif generative_model_name == "gemini-1.5-flash":
            # Gemini failed, try Claude first, then DeepSeek
            try:
                from src.services.auth.openrouter_client import OpenRouterClient
                claude_client = OpenRouterClient()
                return _generate_with_claude(claude_client, formatted_metrics)
            except Exception as e:
                # Claude failed, try DeepSeek
                try:
                    deepseek_client = get_cached_deepseek_openrouter_client()
                    if deepseek_client:
                        return _generate_with_deepseek(deepseek_client, formatted_metrics)
                    else:
                        raise Exception("Cliente DeepSeek no disponible")
                except Exception as e2:
                    st.error(f"❌ Todos los modelos fallaron.")
                    st.error(f"- {first_error}")
                    st.error(f"- Claude: {str(e)}")
                    st.error(f"- DeepSeek: {str(e2)}")
                    return None
        
        # If we get here, both models failed
        st.error("❌ No se pudo generar el análisis con ningún modelo disponible.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error inesperado al generar análisis: {str(e)}")
        return None



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

        > **Nota Importante**: Este sistema evalúa un motor de búsqueda semántica sobre la documentación oficial de Microsoft Learn.
        > No utilizamos Stack Overflow ni otras fuentes externas. El objetivo es encontrar la documentación técnica más relevante
        > directamente desde las fuentes oficiales de Microsoft Azure.

        ### 🎯 1. Obtención de Scores de Recuperación (Pre y Post Reranking)

        **Sistema de Recuperación de Información (Implementación Real del Colab):**
        - **Fuente de Datos**: Documentación oficial de Microsoft Learn sobre Azure (187,031 documentos)
        - **Embeddings Reales**: Se generan embeddings verdaderos usando cada modelo específico:
          - **Ada**: OpenAI API `text-embedding-ada-002` (1536D)
          - **E5-Large**: `intfloat/e5-large-v2` (1024D)
          - **MPNet**: `sentence-transformers/all-mpnet-base-v2` (768D) con prefijo "query:"
          - **MiniLM**: `sentence-transformers/all-MiniLM-L6-v2` (384D)
        - **Búsqueda por Similitud**: Similitud coseno entre embedding de pregunta y embeddings de documentos pre-calculados
        - **Ground Truth con Enlaces**: Utiliza enlaces validados del ground truth para cálculo de métricas de recuperación tradicionales
        - **Normalización URL**: Implementa `normalize_url()` que elimina parámetros de consulta (?query) y fragmentos (#anchor)
        - **Evaluación Acumulativa**: Cada pregunta se evalúa individualmente y se promedian resultados finales
        - **💾 Sistema de Cache OpenAI**: Cache persistente en Google Drive para almacenar respuestas de OpenAI API y evitar re-procesamiento
          - Cache basado en hash MD5 de pregunta + contexto + tipo de prompt
          - Estadísticas de hits/misses para monitoreo de eficiencia
          - Ahorro de costos: ~$0.05 por consulta cacheada (6-7 llamadas por pregunta)
          - Guardado incremental cada 50 preguntas para prevenir pérdida de datos
        
        ### 🤖 2. Estrategia de Reranking con CrossEncoder
        
        **Método de Reordenamiento (Implementación Real del Colab):**
        - **Modelo**: CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` especializado en ranking MS-MARCO
        - **Entrada**: Pares [pregunta, contenido_documento] con hasta 500 caracteres por documento
        - **Proceso de Scoring**: 
          1. El modelo genera logits de relevancia para cada par pregunta-documento
          2. Se aplica normalización Min-Max: `(scores - scores.min()) / (scores.max() - scores.min())`
          3. Scores normalizados quedan en rango [0, 1] interpretable
          4. Documentos se reordenan por score descendente
        - **Configuración**: Funciona sin temperatura (determinístico)
        - **Impacto**: Reordena documentos basándose en relevancia semántica contextual vs. solo similitud coseno
        
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
        
        **Proceso de Evaluación Real del Colab:**
        1. **Inicialización**:
           - Carga de archivos Parquet con embeddings pre-calculados (187,031 documentos por modelo)
           - Inicialización del cache de OpenAI (carga entradas previas si existen)
           - Carga única del modelo semantic similarity (`all-mpnet-base-v2`) para BERTScore
        2. **Generación de Query Embedding**: Embedding real de la pregunta usando el modelo específico (Ada/E5/MPNet/MiniLM)
        3. **Recuperación por Similitud Coseno**: `cosine_similarity(query_embedding, document_embeddings)` para encontrar top-k documentos
        4. **Cálculo de Métricas Pre-Reranking**: Métricas tradicionales IR usando ground truth links validados
        5. **Reranking CrossEncoder**: Reordenamiento opcional usando scores normalizados Min-Max
        6. **Cálculo de Métricas Post-Reranking**: Métricas IR recalculadas después del reordenamiento
        7. **Generación RAG con Cache**:
           - Verifica cache OpenAI antes de llamar API
           - Respuesta con GPT-3.5-turbo-0125 (50% más barato) usando contexto de top-3 documentos (800 chars/doc)
           - Guarda respuesta en cache para futuras ejecuciones
        8. **Evaluación RAGAS Completa con Cache**:
           - Verifica cache OpenAI para métricas RAGAS completas
           - 6 métricas usando GPT-3.5-turbo-0125 para evaluación (3000 chars/doc)
           - Guarda métricas en cache (cacheando ~$0.30 por pregunta en llamadas API)
        9. **Evaluación BERTScore Optimizada**:
           - 3 métricas usando `microsoft/deberta-base-mnli` sin límite de contenido
           - Usa modelo semantic similarity global (cargado una sola vez, no 2067 veces)
           - Limpieza de memoria GPU después de BERTScore
        10. **Checkpoint Incremental**: Guarda cache cada 50 preguntas para prevención de pérdida de datos
        11. **Agregación de Resultados**: Promedios y resultados individuales guardados en JSON compatible con Streamlit
        
        ### 📊 4. Cálculo de Métricas Específicas (Implementación Real del Colab)
        
        **Métricas de Recuperación Tradicionales (IR):**
        - **Precision@k**: `sum(relevantes_en_top_k) / k` donde relevancia se determina por matching de URLs normalizadas
        - **Recall@k**: `sum(relevantes_en_top_k) / total_relevantes_ground_truth`
        - **F1@k**: Media armónica de Precision@k y Recall@k
        - **NDCG@k**: `DCG@k / IDCG@k` usando relevancia binaria (1 si URL match, 0 si no)
        - **MAP@k**: Mean Average Precision calculada como promedio de precisions en posiciones relevantes
        - **MRR@k**: `1 / rank_primer_relevante` o 0 si no hay relevantes en top-k
        - **MRR global**: MRR sin límite de k
        
        **Métricas RAGAS (Usando OpenAI GPT-3.5-turbo-0125):**
        - **Faithfulness**: Escala 1-5 normalizada a [0,1] - evalúa si respuesta contradice contexto
        - **Answer Relevancy**: Escala 1-5 normalizada a [0,1] - evalúa relevancia respuesta-pregunta
        - **Answer Correctness**: Escala 1-5 normalizada a [0,1] - compara exactitud vs ground truth
        - **Context Precision**: Escala 1-5 normalizada a [0,1] - evalúa relevancia del contexto recuperado
        - **Context Recall**: Escala 1-5 normalizada a [0,1] - evalúa cobertura de información necesaria
        - **Semantic Similarity**: Similitud coseno entre embeddings `all-mpnet-base-v2` de respuesta y ground truth
        - **Modelo LLM**: `gpt-3.5-turbo-0125` (50% más económico que `gpt-3.5-turbo` estándar)
        - **Costo Optimizado**: ~6 llamadas API por pregunta, cacheables para ejecuciones futuras

        **Métricas BERTScore (Usando microsoft/deberta-base-mnli):**
        - **BERT Precision**: Token-level semantic similarity (precision)
        - **BERT Recall**: Token-level semantic similarity (recall)
        - **BERT F1**: Media armónica de BERT Precision y Recall
        - **Semantic Similarity (separada)**: Cosine similarity usando `all-mpnet-base-v2` global (optimizado)
        
        ### 🔄 5. Diagrama de Proceso Real del Colab (1 Modelo, 1 Pregunta)
        
        ```
        📝 PREGUNTA TÉCNICA SOBRE AZURE (ejemplo: 15 preguntas de evaluación)
                        ↓
        🔤 GENERACIÓN DE EMBEDDING REAL
               ├── Ada: OpenAI API text-embedding-ada-002 → 1536D
               ├── E5-Large: intfloat/e5-large-v2 → 1024D  
               ├── MPNet: all-mpnet-base-v2 + "query:" prefix → 768D
               └── MiniLM: all-MiniLM-L6-v2 → 384D
                        ↓
        🔍 RECUPERACIÓN POR SIMILITUD COSENO
               cosine_similarity(query_embedding, parquet_embeddings)
               → Top-K documentos ordenados por score descendente
                        ↓
                    📊 EVALUACIÓN PRE-RERANKING
                    calculate_real_retrieval_metrics() usando URLs normalizadas
                    (Precision@k, Recall@k, F1@k, NDCG@k, MAP@k, MRR@k, MRR global)
                        ↓
        🤖 RERANKING CROSSENCODER [SI HABILITADO]
               rerank_with_cross_encoder() usando ms-marco-MiniLM-L-6-v2
               ├── Genera logits para pares [pregunta, doc_content[:500]]
               ├── Normalización Min-Max: (scores-min)/(max-min)
               └── Reordena documentos por score CrossEncoder descendente
                        ↓
                    📈 EVALUACIÓN POST-RERANKING  
                    calculate_real_retrieval_metrics() en documentos reordenados
                    (Mismas métricas IR pero con nuevo orden)
                        ↓
        🎭 GENERACIÓN DE RESPUESTA RAG
               generate_rag_answer() usando GPT-3.5-turbo
               Contexto: Top-3 documentos, 800 chars/documento
                        ↓
                    🔬 EVALUACIÓN RAG COMPLETA
                    calculate_rag_metrics_real() usando OpenAI API
                    ├── 6 RAGAS metrics (GPT-3.5 evalúa con prompts, 3000 chars/doc)
                    └── 3 BERTScore metrics (DistilUSE, sin límite contenido)
                        ↓
        📊 RESULTADOS INDIVIDUALES (por pregunta y modelo)
        
        🔁 REPETIR PARA 15 PREGUNTAS → PROMEDIAR RESULTADOS → GUARDAR JSON
        ```
        
        ### 💰 6. Optimizaciones de Costo y Rendimiento

        **Sistema de Cache OpenAI (Reducción de Costos):**
        - **Implementación**: Clase `OpenAICache` con almacenamiento persistente en Google Drive
        - **Hash de Consultas**: MD5(pregunta + contexto + tipo_prompt) para identificación única
        - **Ahorro Primera Ejecución**: ~50% por uso de gpt-3.5-turbo-0125 vs gpt-3.5-turbo
        - **Ahorro Re-ejecuciones**: ~100% (0% nuevas llamadas API si cache completo)
        - **Costo Estimado**:
          - Sin optimizaciones: ~$200-250 por evaluación completa (2067 preguntas)
          - Con gpt-3.5-turbo-0125: ~$100-125 por primera ejecución
          - Con cache completo: ~$0 por re-ejecuciones (solo infraestructura GPU)
        - **Estadísticas en Tiempo Real**: Hit rate, cache hits/misses cada 50 preguntas

        **Optimización de Memoria GPU:**
        - **Problema Original**: Modelo semantic similarity cargado 2067 veces → GPU OOM ~16% progreso
        - **Solución**: Variable global `semantic_similarity_model` cargada una sola vez al inicio
        - **Impacto**: Reduce uso de GPU de ~11.90 GiB acumulado a ~2-3 GiB estable
        - **Limpieza Proactiva**: `torch.cuda.empty_cache()` después de BERTScore y cada 100 preguntas
        - **Variables Eliminadas**: `del P, R, F1` después de cálculo BERTScore

        **Checkpoints Incrementales:**
        - **Frecuencia**: Cache guardado cada 50 preguntas procesadas
        - **Protección**: Previene pérdida total en caso de error CUDA OOM o interrupción
        - **Reanudación**: Re-ejecuciones aprovechan cache existente automáticamente
        - **Metadata**: Timestamp, hits/misses, hit_rate en archivo JSON de cache

        ### 🎯 7. Garantías de Calidad Científica (Implementación Real del Colab)

        **Datos Reales y Verificables:**
        - **Embeddings Reales**: Sin simulación - usa APIs y modelos reales para generar embeddings
        - **Ground Truth Validado**: URLs del ground truth verificadas contra colección de documentos
        - **Métricas No Aleatorias**: Todos los valores calculados usando algoritmos determinísticos
        - **Verificación de Datos**: `data_verification.is_real_data = True` en resultados JSON
        - **Cache Verificable**: Cada entrada de cache incluye timestamp y metadata de generación

        **Reproducibilidad Completa:**
        - **CrossEncoder Determinístico**: Sin temperatura, mismos scores para mismas entradas
        - **Normalización Consistente**: Min-Max aplicada uniformemente: `(scores - min) / (max - min)`
        - **Datasets Fijos**: Mismos archivos Parquet de embeddings para todas las evaluaciones
        - **URLs Normalizadas**: Función `normalize_url()` consistente elimina parámetros y fragmentos
        - **Cache Determinístico**: Mismo hash MD5 para mismas entradas garantiza respuestas idénticas

        **Validación Técnica:**
        - **Framework RAGAS Oficial**: Uso de biblioteca científicamente validada
        - **BERTScore Estándar**: Implementación usando modelos reconocidos (deberta-base-mnli, all-mpnet-base-v2)
        - **Logging Completo**: Errores y excepciones registrados para debugging
        - **Estructura JSON Validada**: Formato compatible con visualización Streamlit
        - **Optimización Verificable**: Logs de cache hits/misses y memoria GPU disponibles

        **Metodología Científica:**
        - **Evaluación Individual**: Cada pregunta evaluada independientemente, luego promediada
        - **Métricas Before/After**: Comparación pre y post reranking para medir impacto
        - **Contexto Controlado**: Límites de caracteres definidos y consistentes por tipo de evaluación
        - **Múltiples Métricas**: IR tradicionales + RAGAS + BERTScore para evaluación comprehensiva
        - **Eficiencia Energética**: Cache reduce consumo GPU y API calls en 50-100% para re-ejecuciones
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


# =============================================================================
# 🆕 ENHANCED SCORING METHODOLOGY DISPLAY FUNCTIONS
# =============================================================================

def display_enhanced_scoring_section(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, model_name: str, reranking_method: str = 'standard'):
    """Display enhanced scoring methodology information matching individual search page"""
    
    st.subheader("🎯 Metodología de Scoring Enhanced")
    
    # Explain scoring methodology alignment
    with st.expander("📖 Metodología de Scoring (Alineada con Búsqueda Individual)", expanded=False):
        st.markdown("""
        ### 🔄 Pipeline de Scoring Implementado
        
        El sistema de scoring implementado en este análisis **replica exactamente** la metodología 
        utilizada en la página de búsqueda individual para garantizar consistencia:
        
        #### 1. **Cosine Similarity Score (Inicial)**
        ```python
        score = max(0, min(1, 1 - distance))
        ```
        - **Fuente**: Similitud coseno entre query embedding y document embedding
        - **Rango**: [0, 1] donde 1 = máxima similitud
        - **Uso**: Score base antes de reranking
        
        #### 2. **CrossEncoder Reranking**
        - **Modelo**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
        - **Input**: Pares [query, document_content]
        - **Output**: Logits raw del modelo
        
        #### 3. **Sigmoid Normalization**
        ```python
        final_score = 1 / (1 + exp(-raw_logits))
        ```
        - **Propósito**: Mapear logits a probabilidades [0, 1]
        - **Beneficio**: Scores comparables entre diferentes embeddings
        
        #### 4. **Multi-Level Score Preservation**
        - **📄 Document Level**: Scores individuales por documento
        - **❓ Question Level**: Estadísticas por pregunta (avg, max, min)
        - **🏷️ Model Level**: Agregaciones a nivel de modelo
        """)
    
    # Display model-level score statistics if available
    display_model_level_score_stats(avg_before, avg_after, use_llm_reranker, model_name, reranking_method)


def display_model_level_score_stats(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, model_name: str, reranking_method: str):
    """Display model-level score statistics from enhanced Colab results"""
    
    st.markdown("#### 📊 Estadísticas de Scores a Nivel de Modelo")
    
    # Check for model-level score metrics
    model_score_keys = [
        'model_avg_score', 'model_max_score', 'model_min_score',
        'model_all_documents_avg_score', 'model_all_documents_max_score', 
        'model_all_documents_min_score', 'model_all_documents_std_score',
        'model_total_documents_evaluated'
    ]
    
    crossencoder_keys = [
        'model_avg_crossencoder_score', 'model_max_crossencoder_score', 
        'model_min_crossencoder_score', 'model_all_documents_avg_crossencoder_score',
        'model_all_documents_max_crossencoder_score', 'model_all_documents_min_crossencoder_score'
    ]
    
    # Check if we have model-level statistics - using both old and new key names
    basic_score_keys = ['model_avg_score', 'model_max_score', 'model_min_score']
    has_before_scores = any(key in avg_before for key in model_score_keys + basic_score_keys)
    has_after_scores = any(key in avg_after for key in crossencoder_keys + basic_score_keys) if avg_after else False
    
    if has_before_scores or has_after_scores:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔍 Cosine Similarity Scores")
            if has_before_scores:
                # Display cosine similarity statistics using correct key names
                total_docs = avg_before.get('model_total_documents_evaluated', 0)
                avg_score = avg_before.get('model_avg_score', avg_before.get('model_all_documents_avg_score', 0))
                max_score = avg_before.get('model_max_score', avg_before.get('model_all_documents_max_score', 0))
                min_score = avg_before.get('model_min_score', avg_before.get('model_all_documents_min_score', 0))
                std_score = avg_before.get('model_std_score', avg_before.get('model_all_documents_std_score', 0))
                
                st.metric("📊 Documentos Evaluados", f"{total_docs:,}" if total_docs else "N/A")
                st.metric("📈 Score Promedio", f"{avg_score:.3f}" if avg_score is not None and avg_score >= 0 else "N/A")
                st.metric("🔝 Score Máximo", f"{max_score:.3f}" if max_score is not None and max_score >= 0 else "N/A")
                st.metric("🔻 Score Mínimo", f"{min_score:.3f}" if min_score is not None and min_score >= 0 else "N/A")
                st.metric("📏 Desviación Estándar", f"{std_score:.3f}" if std_score is not None and std_score >= 0 else "N/A")
            else:
                st.info("ℹ️ Estadísticas de cosine similarity no disponibles en este resultado")
        
        with col2:
            st.markdown("##### 🧠 CrossEncoder Scores")
            if has_after_scores and use_llm_reranker:
                # Display CrossEncoder statistics using correct key names
                total_reranked = avg_after.get('model_total_documents_reranked', 0)
                avg_ce_score = avg_after.get('model_avg_crossencoder_score', avg_after.get('model_avg_score', 0))
                max_ce_score = avg_after.get('model_max_crossencoder_score', avg_after.get('model_max_score', 0))
                min_ce_score = avg_after.get('model_min_crossencoder_score', avg_after.get('model_min_score', 0))
                avg_per_question = avg_after.get('model_avg_documents_reranked_per_question', 0)
                
                st.metric("🔄 Documentos Rerankeados", f"{total_reranked:,}" if total_reranked else "N/A")
                st.metric("📈 CE Score Promedio", f"{avg_ce_score:.3f}" if avg_ce_score is not None and avg_ce_score >= 0 else "N/A")
                st.metric("🔝 CE Score Máximo", f"{max_ce_score:.3f}" if max_ce_score is not None and max_ce_score >= 0 else "N/A")
                st.metric("🔻 CE Score Mínimo", f"{min_ce_score:.3f}" if min_ce_score is not None and min_ce_score >= 0 else "N/A")
                st.metric("📊 Promedio por Pregunta", f"{avg_per_question:.1f}" if avg_per_question is not None and avg_per_question >= 0 else "N/A")
            else:
                if use_llm_reranker:
                    st.info("ℹ️ Estadísticas de CrossEncoder no disponibles en este resultado")
                else:
                    st.info("ℹ️ CrossEncoder no se utilizó en esta evaluación")
    else:
        st.info("ℹ️ Las estadísticas de scores a nivel de modelo no están disponibles en este resultado. Esto indica que se usó una versión anterior del notebook de evaluación.")


def display_document_score_analysis(results: Dict[str, Any], use_llm_reranker: bool, model_name: str, reranking_method: str):
    """Display detailed document-level score analysis from individual metrics"""
    
    st.subheader("📄 Análisis de Scores por Documento")
    
    # Get individual metrics
    individual_before = results.get('individual_before_metrics', [])
    individual_after = results.get('individual_after_metrics', []) if use_llm_reranker else []
    
    # Check if we have document scores in the individual metrics
    has_document_scores_before = any('document_scores' in metrics for metrics in individual_before)
    has_document_scores_after = any('document_scores' in metrics for metrics in individual_after)
    
    if has_document_scores_before or has_document_scores_after:
        st.success("✅ Datos de scores por documento encontrados en los resultados")
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Distribución de Scores", "📈 Análisis por Pregunta", "🔍 Top Documentos"])
        
        with tab1:
            display_score_distribution_analysis(individual_before, individual_after, use_llm_reranker, reranking_method)
        
        with tab2:
            display_question_level_score_analysis(individual_before, individual_after, use_llm_reranker, reranking_method)
        
        with tab3:
            display_top_documents_analysis(individual_before, individual_after, use_llm_reranker, model_name)
            
    else:
        st.info("ℹ️ Los scores por documento no están disponibles en este resultado. Para obtener esta información detallada, utiliza la versión enhanced del notebook de evaluación.")


def display_score_distribution_analysis(individual_before: List[Dict], individual_after: List[Dict], use_llm_reranker: bool, reranking_method: str):
    """Display score distribution analysis"""
    
    st.markdown("#### 📊 Distribución de Scores")
    
    # NEW: Add candlestick chart for document scores across all questions
    display_document_scores_candlestick(individual_before, individual_after, use_llm_reranker, reranking_method)
    
    # Extract all document scores
    all_cosine_scores = []
    all_crossencoder_scores = []
    
    # Extract scores from before metrics
    for question_metrics in individual_before:
        if 'document_scores' in question_metrics:
            doc_scores = question_metrics['document_scores']
            cosine_scores = [doc['cosine_similarity'] for doc in doc_scores]
            all_cosine_scores.extend(cosine_scores)
    
    # Extract CrossEncoder scores from after metrics
    if use_llm_reranker and individual_after:
        for question_metrics in individual_after:
            if 'document_scores' in question_metrics:
                doc_scores = question_metrics['document_scores']
                ce_scores = [doc.get('crossencoder_score') for doc in doc_scores if 'crossencoder_score' in doc]
                all_crossencoder_scores.extend([s for s in ce_scores if s is not None])
    
    if all_cosine_scores:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔍 Cosine Similarity Distribution")
            
            # Create histogram
            fig = px.histogram(
                x=all_cosine_scores,
                title="Distribución de Cosine Similarity Scores",
                labels={'x': 'Cosine Similarity Score', 'y': 'Frecuencia'},
                nbins=20
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("**📈 Estadísticas:**")
            st.write(f"• Total documentos: {len(all_cosine_scores):,}")
            st.write(f"• Score promedio: {np.mean(all_cosine_scores):.3f}")
            st.write(f"• Score máximo: {np.max(all_cosine_scores):.3f}")
            st.write(f"• Score mínimo: {np.min(all_cosine_scores):.3f}")
            st.write(f"• Desviación estándar: {np.std(all_cosine_scores):.3f}")
        
        with col2:
            if all_crossencoder_scores and use_llm_reranker:
                st.markdown("##### 🧠 CrossEncoder Score Distribution")
                
                # Create histogram
                fig = px.histogram(
                    x=all_crossencoder_scores,
                    title="Distribución de CrossEncoder Scores",
                    labels={'x': 'CrossEncoder Score', 'y': 'Frecuencia'},
                    nbins=20
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("**📈 Estadísticas:**")
                st.write(f"• Total documentos rerankeados: {len(all_crossencoder_scores):,}")
                st.write(f"• Score promedio: {np.mean(all_crossencoder_scores):.3f}")
                st.write(f"• Score máximo: {np.max(all_crossencoder_scores):.3f}")
                st.write(f"• Score mínimo: {np.min(all_crossencoder_scores):.3f}")
                st.write(f"• Desviación estándar: {np.std(all_crossencoder_scores):.3f}")
            else:
                st.info("ℹ️ CrossEncoder scores no disponibles o reranking no utilizado")


def display_question_level_score_analysis(individual_before: List[Dict], individual_after: List[Dict], use_llm_reranker: bool, reranking_method: str):
    """Display question-level score analysis"""
    
    st.markdown("#### 📈 Análisis por Pregunta")
    
    # Prepare question-level data
    question_data = []
    
    for i, question_metrics in enumerate(individual_before):
        question_text = question_metrics.get('original_question', f'Pregunta {i+1}')
        
        # Get question-level score statistics
        row = {
            'Pregunta': i + 1,
            'Texto': question_text[:100] + "..." if len(question_text) > 100 else question_text,
        }
        
        # Before metrics (cosine similarity)
        if 'question_avg_score' in question_metrics:
            row['Avg Score (Cosine)'] = f"{question_metrics['question_avg_score']:.3f}"
            row['Max Score (Cosine)'] = f"{question_metrics.get('question_max_score', 0):.3f}"
            row['Min Score (Cosine)'] = f"{question_metrics.get('question_min_score', 0):.3f}"
        
        # After metrics (CrossEncoder) if available
        if use_llm_reranker and i < len(individual_after):
            after_metrics = individual_after[i]
            if 'question_avg_crossencoder_score' in after_metrics:
                row['Avg Score (CrossEncoder)'] = f"{after_metrics['question_avg_crossencoder_score']:.3f}"
                row['Max Score (CrossEncoder)'] = f"{after_metrics.get('question_max_crossencoder_score', 0):.3f}"
                row['Min Score (CrossEncoder)'] = f"{after_metrics.get('question_min_crossencoder_score', 0):.3f}"
        
        # Reranking info
        docs_reranked = question_metrics.get('documents_reranked', 0)
        row['Docs Rerankeados'] = docs_reranked
        
        question_data.append(row)
    
    if question_data:
        # Display as table
        df = pd.DataFrame(question_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Question-level score trends
        if len(question_data) > 1:
            st.markdown("##### 📊 Tendencias de Scores por Pregunta")
            
            # Extract numeric scores for plotting
            question_numbers = [row['Pregunta'] for row in question_data]
            cosine_avgs = []
            crossencoder_avgs = []
            
            for row in question_data:
                if 'Avg Score (Cosine)' in row:
                    cosine_avgs.append(float(row['Avg Score (Cosine)']))
                if 'Avg Score (CrossEncoder)' in row:
                    crossencoder_avgs.append(float(row['Avg Score (CrossEncoder)']))
            
            # Create line chart
            fig = go.Figure()
            
            if cosine_avgs:
                fig.add_trace(go.Scatter(
                    x=question_numbers,
                    y=cosine_avgs,
                    mode='lines+markers',
                    name='Cosine Similarity Avg',
                    line=dict(color='blue')
                ))
            
            if crossencoder_avgs:
                fig.add_trace(go.Scatter(
                    x=question_numbers[:len(crossencoder_avgs)],
                    y=crossencoder_avgs,
                    mode='lines+markers',
                    name='CrossEncoder Avg',
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                title="Scores Promedio por Pregunta",
                xaxis_title="Número de Pregunta",
                yaxis_title="Score Promedio",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No se encontraron estadísticas por pregunta en los resultados")


def display_top_documents_analysis(individual_before: List[Dict], individual_after: List[Dict], use_llm_reranker: bool, model_name: str):
    """Display analysis of top-scoring documents"""
    
    st.markdown("#### 🔍 Análisis de Top Documentos")
    
    # Collect all documents with their scores
    all_docs_with_scores = []
    
    for q_idx, question_metrics in enumerate(individual_before):
        if 'document_scores' in question_metrics:
            doc_scores = question_metrics['document_scores']
            question_text = question_metrics.get('original_question', f'Pregunta {q_idx+1}')
            
            for doc in doc_scores:
                doc_info = {
                    'question_number': q_idx + 1,
                    'question_text': question_text,
                    'rank': doc.get('rank', 0),
                    'cosine_similarity': doc.get('cosine_similarity', 0),
                    'title': doc.get('title', 'Sin título'),
                    'link': doc.get('link', ''),
                    'relevant': doc.get('relevant', False),
                    'reranked': doc.get('reranked', False)
                }
                
                # Add CrossEncoder score if available
                if 'crossencoder_score' in doc:
                    doc_info['crossencoder_score'] = doc['crossencoder_score']
                
                all_docs_with_scores.append(doc_info)
    
    if all_docs_with_scores:
        # Top documents by cosine similarity - show all available documents
        st.markdown(f"##### 🔝 Todos los Documentos por Cosine Similarity ({len(all_docs_with_scores)} documentos)")
        
        top_cosine_docs = sorted(all_docs_with_scores, key=lambda x: x['cosine_similarity'], reverse=True)
        
        cosine_table_data = []
        for i, doc in enumerate(top_cosine_docs, 1):
            cosine_table_data.append({
                'Rank': i,
                'Pregunta': doc['question_number'],
                'Cosine Score': f"{doc['cosine_similarity']:.3f}",
                'Título': doc['title'][:80] + "..." if len(doc['title']) > 80 else doc['title'],
                'Relevante': "✅" if doc['relevant'] else "❌",
                'Rerankeado': "🔄" if doc['reranked'] else "➖"
            })
        
        df_cosine = pd.DataFrame(cosine_table_data)
        st.dataframe(df_cosine, use_container_width=True)
        
        # Top documents by CrossEncoder if available
        if use_llm_reranker:
            crossencoder_docs = [doc for doc in all_docs_with_scores if 'crossencoder_score' in doc]
            if crossencoder_docs:
                st.markdown(f"##### 🧠 Todos los Documentos por CrossEncoder Score ({len(crossencoder_docs)} documentos)")
                
                top_ce_docs = sorted(crossencoder_docs, key=lambda x: x['crossencoder_score'], reverse=True)
                
                ce_table_data = []
                for i, doc in enumerate(top_ce_docs, 1):
                    ce_table_data.append({
                        'Rank': i,
                        'Pregunta': doc['question_number'],
                        'CrossEncoder Score': f"{doc['crossencoder_score']:.3f}",
                        'Cosine Score': f"{doc['cosine_similarity']:.3f}",
                        'Título': doc['title'][:80] + "..." if len(doc['title']) > 80 else doc['title'],
                        'Relevante': "✅" if doc['relevant'] else "❌"
                    })
                
                df_ce = pd.DataFrame(ce_table_data)
                st.dataframe(df_ce, use_container_width=True)
        
        # Score correlation analysis
        if use_llm_reranker:
            ce_docs = [doc for doc in all_docs_with_scores if 'crossencoder_score' in doc]
            if len(ce_docs) > 5:
                st.markdown("##### 📊 Correlación entre Cosine Similarity y CrossEncoder")
                
                cosine_scores = [doc['cosine_similarity'] for doc in ce_docs]
                ce_scores = [doc['crossencoder_score'] for doc in ce_docs]
                
                # Create scatter plot
                fig = px.scatter(
                    x=cosine_scores,
                    y=ce_scores,
                    title="Correlación: Cosine Similarity vs CrossEncoder Score",
                    labels={'x': 'Cosine Similarity', 'y': 'CrossEncoder Score'},
                    hover_data={'Documento': [doc['title'][:50] for doc in ce_docs]}
                )
                
                # Add trend line
                import numpy as np
                z = np.polyfit(cosine_scores, ce_scores, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(x=cosine_scores, y=p(cosine_scores), mode='lines', name='Tendencia'))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(cosine_scores, ce_scores)[0, 1]
                st.metric("📈 Coeficiente de Correlación", f"{correlation:.3f}")
                
                if correlation > 0.7:
                    st.success("🎯 Alta correlación positiva - Los métodos de scoring son consistentes")
                elif correlation > 0.4:
                    st.info("📊 Correlación moderada - Los métodos de scoring se complementan")
                else:
                    st.warning("⚠️ Baja correlación - Los métodos de scoring evalúan aspectos diferentes")
    else:
        st.info("ℹ️ No se encontraron datos de documentos individuales en los resultados")


def display_multi_model_scoring_analysis(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, reranking_method: str):
    """Display scoring analysis across multiple models"""
    
    st.subheader("🏆 Análisis de Scoring Multi-Modelo")
    
    # Collect model-level scoring statistics
    model_stats = []
    
    for model_name, model_results in results.items():
        before_metrics = model_results.get('avg_before_metrics', {})
        after_metrics = model_results.get('avg_after_metrics', {})
        
        stats = {'Modelo': model_name}
        
        # Cosine similarity statistics - only include available data
        stats['Avg Cosine Score'] = before_metrics.get('model_avg_score', before_metrics.get('model_all_documents_avg_score', 0))
        
        # Only add optional statistics if they exist
        if before_metrics.get('model_total_documents_evaluated') is not None:
            stats['Docs Evaluados'] = before_metrics.get('model_total_documents_evaluated')
        if before_metrics.get('model_max_score') is not None:
            stats['Max Cosine Score'] = before_metrics.get('model_max_score', before_metrics.get('model_all_documents_max_score'))
        if before_metrics.get('model_std_score') is not None:
            stats['Std Cosine Score'] = before_metrics.get('model_std_score', before_metrics.get('model_all_documents_std_score'))
        
        # CrossEncoder statistics if available - only include available data
        if use_llm_reranker and after_metrics:
            stats['Avg CE Score'] = after_metrics.get('model_avg_crossencoder_score', after_metrics.get('model_avg_score', 0))
            
            # Only add optional statistics if they exist
            if after_metrics.get('model_total_documents_reranked') is not None:
                stats['Docs Rerankeados'] = after_metrics.get('model_total_documents_reranked')
            if after_metrics.get('model_max_crossencoder_score') is not None:
                stats['Max CE Score'] = after_metrics.get('model_max_crossencoder_score', after_metrics.get('model_max_score'))
        
        model_stats.append(stats)
    
    # Check if we have scoring statistics (check for basic scoring data)
    has_scoring_stats = any(
        stat.get('Avg Cosine Score', 0) > 0 or 
        stat.get('Avg CE Score', 0) != 0 
        for stat in model_stats
    )
    
    if has_scoring_stats:
        st.success("✅ Estadísticas de scoring disponibles para comparación multi-modelo")
        
        # Check what specific stats are available
        has_detailed_stats = any(stat.get('Docs Evaluados') != "N/A" for stat in model_stats)
        if not has_detailed_stats:
            st.info("ℹ️ Mostrando scores básicos disponibles. Para estadísticas detalladas (min, max, std), usa la versión enhanced del notebook.")
        
        # Model comparison table
        st.markdown("#### 📊 Comparación de Estadísticas de Scoring")
        
        df_stats = pd.DataFrame(model_stats)
        
        # Format numeric columns - only include columns that have real data
        available_cols = df_stats.columns.tolist()
        numeric_cols = []
        
        # Add available cosine similarity columns
        cosine_cols = ['Avg Cosine Score']
        for col in cosine_cols:
            if col in available_cols:
                numeric_cols.append(col)
        
        # Add available CrossEncoder columns if reranking is used
        if use_llm_reranker:
            ce_cols = ['Avg CE Score']
            for col in ce_cols:
                if col in available_cols:
                    numeric_cols.append(col)
        
        for col in numeric_cols:
            if col in df_stats.columns:
                if 'Score' in col:
                    df_stats[col] = df_stats[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and x >= 0 else "N/A")
                elif 'Docs' in col:
                    df_stats[col] = df_stats[col].apply(lambda x: f"{x:,}" if isinstance(x, (int, float)) and x > 0 else "N/A")
        
        st.dataframe(df_stats, use_container_width=True)
        
        # Single unified score comparison chart
        if len(model_stats) > 1:
            st.markdown("#### 📈 Comparación de Scores por Modelo")
            
            # Prepare data for unified comparison
            models = [stat['Modelo'] for stat in model_stats]
            cosine_scores = [float(stat['Avg Cosine Score']) if isinstance(stat['Avg Cosine Score'], (int, float)) and stat['Avg Cosine Score'] > 0 else 0 for stat in model_stats]
            
            chart_data = []
            for i, model in enumerate(models):
                chart_data.append({
                    'Modelo': model,
                    'Score': cosine_scores[i],
                    'Tipo': 'Pre-Reranking (Cosine)'
                })
            
            # Add CrossEncoder scores if available
            if use_llm_reranker:
                for stat in model_stats:
                    ce_score = stat.get('Avg CE Score', 0)
                    if isinstance(ce_score, (int, float)) and ce_score != 0:
                        chart_data.append({
                            'Modelo': stat['Modelo'],
                            'Score': float(ce_score),
                            'Tipo': 'Post-Reranking (CrossEncoder)'
                        })
            
            if chart_data:
                df_chart = pd.DataFrame(chart_data)
                
                # Create side-by-side comparison with different colors
                fig = px.bar(
                    df_chart,
                    x='Modelo',
                    y='Score',
                    color='Tipo',
                    barmode='group',
                    title="Comparación de Scores: Pre vs Post Reranking por Modelo",
                    labels={'Score': 'Score Value', 'Tipo': 'Método de Scoring'},
                    color_discrete_map={
                        'Pre-Reranking (Cosine)': '#3498db',      # Blue for before
                        'Post-Reranking (CrossEncoder)': '#e74c3c' # Red for after
                    }
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No hay datos de scores disponibles para comparación")
    else:
        st.info("ℹ️ Las estadísticas de scoring detalladas no están disponibles en estos resultados. Utiliza la versión enhanced del notebook para obtener esta información.")


def display_document_scores_candlestick(individual_before: List[Dict], individual_after: List[Dict], use_llm_reranker: bool, reranking_method: str):
    """Display candlestick chart showing document score ranges across all questions"""
    
    st.markdown("##### 🕯️ Gráfico de Velas: Rangos de Scores por Pregunta")
    
    # Collect score statistics for each question
    question_stats = []
    
    for i, question_metrics in enumerate(individual_before):
        if 'document_scores' in question_metrics:
            doc_scores = question_metrics['document_scores']
            cosine_scores = [doc['cosine_similarity'] for doc in doc_scores]
            
            if cosine_scores:
                stats = {
                    'Pregunta': i + 1,
                    'Open': cosine_scores[0],     # First document score
                    'High': max(cosine_scores),   # Highest score
                    'Low': min(cosine_scores),    # Lowest score
                    'Close': cosine_scores[-1],  # Last document score
                    'Tipo': 'Cosine Similarity'
                }
                question_stats.append(stats)
    
    # Add CrossEncoder stats if available
    if use_llm_reranker and individual_after:
        for i, question_metrics in enumerate(individual_after):
            if 'document_scores' in question_metrics and i < len(question_stats):
                doc_scores = question_metrics['document_scores']
                ce_scores = [doc.get('crossencoder_score') for doc in doc_scores if 'crossencoder_score' in doc]
                ce_scores = [s for s in ce_scores if s is not None]
                
                if ce_scores:
                    stats = {
                        'Pregunta': i + 1,
                        'Open': ce_scores[0],      # First document score
                        'High': max(ce_scores),    # Highest score
                        'Low': min(ce_scores),     # Lowest score
                        'Close': ce_scores[-1],   # Last document score
                        'Tipo': 'CrossEncoder'
                    }
                    question_stats.append(stats)
    
    if question_stats:
        df_candlestick = pd.DataFrame(question_stats)
        
        # Create candlestick chart using go.Candlestick
        fig = go.Figure()
        
        # Add cosine similarity candlesticks
        cosine_data = df_candlestick[df_candlestick['Tipo'] == 'Cosine Similarity']
        if not cosine_data.empty:
            fig.add_trace(go.Candlestick(
                x=cosine_data['Pregunta'],
                open=cosine_data['Open'],
                high=cosine_data['High'],
                low=cosine_data['Low'],
                close=cosine_data['Close'],
                name='Cosine Similarity',
                increasing_line_color='#2E86AB',  # Blue for increasing
                decreasing_line_color='#A23B72',  # Purple for decreasing
                showlegend=True
            ))
        
        # Add CrossEncoder candlesticks if available
        if use_llm_reranker:
            ce_data = df_candlestick[df_candlestick['Tipo'] == 'CrossEncoder']
            if not ce_data.empty:
                fig.add_trace(go.Candlestick(
                    x=ce_data['Pregunta'] + 0.3,  # Offset slightly for visibility
                    open=ce_data['Open'],
                    high=ce_data['High'],
                    low=ce_data['Low'],
                    close=ce_data['Close'],
                    name='CrossEncoder',
                    increasing_line_color='#F18F01',  # Orange for increasing
                    decreasing_line_color='#C73E1D',  # Red for decreasing
                    showlegend=True
                ))
        
        fig.update_layout(
            title="Distribución de Scores por Pregunta (Gráfico de Velas)",
            xaxis_title="Número de Pregunta",
            yaxis_title="Score Range",
            height=500,
            xaxis=dict(type='category'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("📖 Explicación del Gráfico de Velas"):
            st.markdown("""
            **Interpretación del Gráfico de Velas:**
            
            - **Apertura (Open)**: Score del primer documento en el ranking
            - **Máximo (High)**: Score más alto entre todos los documentos de la pregunta
            - **Mínimo (Low)**: Score más bajo entre todos los documentos de la pregunta  
            - **Cierre (Close)**: Score del último documento en el ranking
            
            **Colores:**
            - **Azul/Naranja**: Cuando el score de cierre > score de apertura (mejora en el ranking)
            - **Púrpura/Rojo**: Cuando el score de cierre < score de apertura (declive en el ranking)
            
            **Interpretación:**
            - **Velas altas**: Gran variabilidad de scores entre documentos
            - **Velas cortas**: Scores similares entre documentos
            - **Comparación**: Cosine Similarity vs CrossEncoder para cada pregunta
            """)
    else:
        st.info("ℹ️ No se encontraron datos de scores por documento para crear el gráfico de velas")