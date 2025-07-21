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


def display_enhanced_cumulative_metrics(results: Dict[str, Any], model_name: str, use_llm_reranker: bool):
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


def display_enhanced_models_comparison(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool):
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