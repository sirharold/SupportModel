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
    
    # Select key metrics to highlight
    key_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr@5', 'ndcg@5']
    
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
    metrics_to_compare = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr@5', 'ndcg@5']
    
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
    """Display metrics organized by K values in separate sections"""
    
    st.subheader("📈 Métricas por Valores de K")
    
    k_values = [1, 3, 5, 10]
    metric_types = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']
    
    # Create tabs for each K value
    tabs = st.tabs([f"📊 Top-{k}" for k in k_values])
    
    for i, k in enumerate(k_values):
        with tabs[i]:
            st.markdown(f"#### Métricas para Top-{k} documentos")
            
            if use_llm_reranker and avg_after:
                # Show before and after for this K
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Antes del LLM**")
                    display_k_metrics(avg_before, k, metric_types)
                
                with col2:
                    st.markdown("**🤖 Después del LLM**") 
                    display_k_metrics(avg_after, k, metric_types, avg_before)
            else:
                # Show only before metrics
                st.markdown("**📊 Métricas de Retrieval**")
                display_k_metrics(avg_before, k, metric_types)
            
            # Visualization for this K
            if use_llm_reranker and avg_after:
                create_k_comparison_chart(avg_before, avg_after, k, metric_types)
            
            # Table for this K
            display_k_metrics_table(avg_before, avg_after, k, metric_types, use_llm_reranker)


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
    k_values = [1, 3, 5, 10]
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
        all_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr@5', 'ndcg@5']
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

        # New section: RAG Metrics Summary
        display_rag_metrics_summary(results, use_llm_reranker, config)


def display_all_metrics_by_k_for_all_models(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool):
    """
    Displays a 2x3 grid of plots, each showing a specific metric across K values
    for all models, with before/after LLM lines.
    """
    st.subheader("📈 Rendimiento Detallado por Métrica y K")

    k_values = [1, 3, 5, 10]
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

    k_values = [1, 3, 5, 10]
    metrics_to_display = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']

    table_data = []

    for model_name, model_results in results.items():
        before_metrics = model_results['avg_before_metrics']
        after_metrics = model_results.get('avg_after_metrics', {})

        for metric_type in metrics_to_display:
            row_base = {'Modelo': model_name, 'Métrica': metric_type.upper()}

            for k in k_values:
                metric_key = f"{metric_type}@{k}"
                
                # Before LLM score
                before_val = before_metrics.get(metric_key, np.nan)
                row_base[f'Antes LLM @{k}'] = f"{before_val:.3f}" if not np.isnan(before_val) else 'N/A'

                # After LLM score and improvement
                if use_llm_reranker and after_metrics and metric_key in after_metrics:
                    after_val = after_metrics.get(metric_key, np.nan)
                    row_base[f'Después LLM @{k}'] = f"{after_val:.3f}" if not np.isnan(after_val) else 'N/A'
                    
                    if not np.isnan(before_val) and before_val > 0:
                        improvement = after_val - before_val
                        improvement_pct = (improvement / before_val) * 100
                        row_base[f'Mejora @{k}'] = f"{improvement:+.3f} ({improvement_pct:+.1f}%) " + get_improvement_status_icon(improvement)
                    else:
                        row_base[f'Mejora @{k}'] = 'N/A'
                elif use_llm_reranker:
                    row_base[f'Después LLM @{k}'] = 'N/A'
                    row_base[f'Mejora @{k}'] = 'N/A'

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
    formatted_string += f"- Tiempo total de ejecución: {evaluation_info.get('total_time_seconds', 'N/A')} segundos\n"
    formatted_string += f"- GPU utilizada: {'Sí' if evaluation_info.get('gpu_used') else 'No'}\n\n"

    formatted_string += "## Resultados Detallados por Modelo\n"
    k_values = [1, 3, 5, 10]
    metrics_types = ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']
    rag_metrics_types = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']

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
            for metric_type in rag_metrics_types:
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
            formatted_string += "\n"
        else:
            # Add note about missing RAG metrics
            formatted_string += "#### Métricas RAG\n"
            formatted_string += "**Nota:** Las métricas RAG (Faithfulness, Answer Relevance, Answer Correctness, Answer Similarity) no están disponibles porque la evaluación se ejecutó en modo de solo recuperación (sin generación de respuestas).\n\n"

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
        elif generative_model_name == "gemini-pro" and config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            gemini_client = genai.GenerativeModel('gemini-pro')
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
        elif generative_model_name == "gemini-pro" and gemini_client:
            llm_client = gemini_client
            model_to_use = "gemini-pro"
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
            if model_to_use in ["gpt-4", "gemini-pro", "llama-4-scout"]:
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
                elif model_to_use == "gemini-pro":
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
        
        # Calculate average performance
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr@5', 'ndcg@5']
        before_avg = np.mean([before_metrics.get(m, 0) for m in key_metrics if m in before_metrics])
        
        row = {
            'Modelo': model_name,
            'Preguntas': model_results.get('num_questions_evaluated', 0),
            'Promedio Antes': f"{before_avg:.3f}",
            'Calidad': get_metric_quality(before_avg)
        }
        
        if use_llm_reranker and after_metrics:
            after_avg = np.mean([after_metrics.get(m, 0) for m in key_metrics if m in after_metrics])
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


def create_models_summary_table(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool):
    """Create summary table for model comparison"""
    
    st.subheader("📋 Tabla Resumen de Modelos")
    
    summary_data = []
    for model_name, model_results in results.items():
        before_metrics = model_results['avg_before_metrics']
        after_metrics = model_results.get('avg_after_metrics', {})
        
        # Calculate average performance
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr@5', 'ndcg@5']
        before_avg = np.mean([before_metrics.get(m, 0) for m in key_metrics if m in before_metrics])
        
        row = {
            'Modelo': model_name,
            'Preguntas': model_results.get('num_questions_evaluated', 0),
            'Promedio Antes': f"{before_avg:.3f}",
            'Calidad': get_metric_quality(before_avg)
        }
        
        if use_llm_reranker and after_metrics:
            after_avg = np.mean([after_metrics.get(m, 0) for m in key_metrics if m in after_metrics])
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

def display_rag_metrics_summary(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool, config: Dict = None):
    """
    Displays RAG metrics summary. Checks configuration to determine if RAG metrics were supposed to be generated.
    """
    st.subheader("📊 Métricas RAG (Generación de Respuesta)")
    
    # Check if any model has RAG metrics data
    has_rag_metrics = False
    for model_name, model_results in results.items():
        individual_metrics = model_results.get('individual_before_metrics', [])
        individual_after_metrics = model_results.get('individual_after_metrics', [])
        
        # Check if any question has rag_metrics
        for metrics in individual_metrics + individual_after_metrics:
            if isinstance(metrics, dict) and (
                'rag_metrics' in metrics or 'rag_metrics_after_rerank' in metrics
            ):
                rag_data = metrics.get('rag_metrics') or metrics.get(
                    'rag_metrics_after_rerank', {}
                )
                if rag_data and any(v is not None for v in rag_data.values()):
                    has_rag_metrics = True
                    break
        if has_rag_metrics:
            break
    
    # Check configuration to see if RAG metrics were supposed to be generated
    rag_metrics_enabled = False
    if config:
        rag_metrics_enabled = config.get('generate_rag_metrics', False)
    
    if not has_rag_metrics:
        if rag_metrics_enabled:
            # RAG metrics were enabled but not found - possible error or processing issue
            st.warning("""
            **⚠️ Métricas RAG Faltantes:**
            
            Las métricas RAG (Faithfulness, Answer Relevance, Answer Correctness, Answer Similarity) 
            estaban **habilitadas** en la configuración (`generate_rag_metrics=True`) pero no se 
            encuentran en los resultados.
            
            **Posibles causas:**
            - Error durante la generación de respuestas en Colab
            - Problemas con la integración de OpenAI API 
            - Fallo en el cálculo de métricas RAG durante la evaluación
            - Datos filtrados o perdidos durante el procesamiento
            
            **Solución recomendada:**
            - Verificar los logs de Google Colab para errores
            - Confirmar que la API key de OpenAI está configurada correctamente
            - Volver a ejecutar la evaluación si es necesario
            
            **Métricas disponibles actualmente:**
            - ✅ Métricas de Recuperación: Precision, Recall, F1, MAP, MRR, NDCG
            - ❌ Métricas RAG: Faltantes (pero habilitadas en configuración)
            """)
        else:
            # RAG metrics were not enabled - normal behavior
            st.info("""
            **📝 Nota sobre Métricas RAG:**
            
            Las métricas RAG (Faithfulness, Answer Relevance, Answer Correctness, Answer Similarity) 
            no están disponibles porque la evaluación se ejecutó en modo de solo recuperación 
            (`generate_rag_metrics=False`).
            
            **Para obtener métricas RAG completas:**
            - Habilita "📝 Generar Métricas RAG" en la configuración
            - Las métricas RAG requieren generar respuestas para cada pregunta
            - Esto incluye evaluar la fidelidad, relevancia y corrección de las respuestas generadas
            - La evaluación se enfoca actualmente en métricas de recuperación (precision, recall, F1, etc.)
            
            **Métricas disponibles actualmente:**
            - ✅ Métricas de Recuperación: Precision, Recall, F1, MAP, MRR, NDCG
            - ❌ Métricas RAG: Deshabilitadas en configuración
            """)
        return
    
    # If we have RAG metrics, display them (this is for future compatibility)
    rag_metrics_types = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
    table_data = []

    for model_name, model_results in results.items():
        individual_metrics = model_results.get('individual_after_metrics', model_results.get('individual_before_metrics', []))
        row_base = {'Modelo': model_name}

        for metric_type in rag_metrics_types:
            # Look for RAG metrics in the most recent available data
            values = []
            for metrics in individual_metrics:
                if isinstance(metrics, dict):
                    rag_data = metrics.get('rag_metrics', {})
                    if rag_data and metric_type in rag_data and rag_data[metric_type] is not None:
                        values.append(rag_data[metric_type])
            
            avg_value = np.mean(values) if values else np.nan
            metric_display_name = metric_type.replace('answer_', '').replace('context_', '').capitalize()
            row_base[f'{metric_display_name}'] = f"{avg_value:.3f}" if not np.isnan(avg_value) else 'N/A'

        table_data.append(row_base)

    if table_data and any(any(v != 'N/A' for k, v in row.items() if k != 'Modelo') for row in table_data):
        df_rag_metrics = pd.DataFrame(table_data)
        st.dataframe(df_rag_metrics, use_container_width=True)

        # Create a simple bar chart for available RAG metrics
        chart_data = []
        for row in table_data:
            model = row['Modelo']
            for key, value in row.items():
                if key != 'Modelo' and value != 'N/A':
                    chart_data.append({
                        'Modelo': model,
                        'Métrica RAG': key,
                        'Valor': float(value)
                    })
        
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig = px.bar(
                df_chart,
                x='Métrica RAG',
                y='Valor',
                color='Modelo',
                barmode='group',
                title='Métricas RAG por Modelo',
                range_y=[0, 1]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

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