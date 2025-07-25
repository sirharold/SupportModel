"""
Utilidades para mostrar métricas en la interfaz.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, Any, List


def grade_metric(value: float) -> str:
    """Return a qualitative label for a metric value."""
    if value >= 0.7:
        return "✓ Muy bueno"
    elif value >= 0.4:
        return "~ Bueno"
    else:
        return "✗ Malo"


def color_metric(val: float) -> str:
    """Return background color for dataframe styling."""
    if pd.isna(val):
        return ""
    if val >= 0.7:
        return "background-color: #c6f5c6"  # light green
    elif val >= 0.4:
        return "background-color: #fff4b3"  # light yellow
    else:
        return "background-color: #f7c6c6"  # light red


def style_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply coloring to before/after metric columns."""
    return df.style.map(color_metric, subset=["Before", "After"])


def display_rag_statistics(results: Dict[str, Any]):
    """Muestra estadísticas de RAG si están disponibles."""
    if 'avg_rag_stats' in results and results['avg_rag_stats']:
        st.subheader("🔍 Estadísticas de Recuperación RAG")
        rag_stats = results['avg_rag_stats']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Enlaces Ground Truth Promedio",
                value=f"{rag_stats.get('ground_truth_links_count', 0):.1f}",
                help="Número promedio de enlaces de referencia por pregunta"
            )
        with col2:
            st.metric(
                label="Documentos Antes Reranking",
                value=f"{rag_stats.get('docs_before_count', 0):.1f}",
                help="Número promedio de documentos antes del reranking"
            )
        with col3:
            st.metric(
                label="Documentos Después Reranking",
                value=f"{rag_stats.get('docs_after_count', 0):.1f}",
                help="Número promedio de documentos después del reranking"
            )
        
        st.divider()


def display_metrics_tabs(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool):
    """Muestra métricas organizadas en tabs por valores de k."""
    k_values = [1, 3, 5, 10]
    tab1, tab2, tab3, tab4 = st.tabs([f"📊 Top-{k}" for k in k_values])
    
    for i, k in enumerate(k_values):
        with [tab1, tab2, tab3, tab4][i]:
            k_metrics = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}', 
                        f'BinaryAccuracy@{k}', f'RankingAccuracy@{k}']
            if k == 5:  # Agregar MRR y nDCG solo para k=5
                k_metrics.extend(['MRR', 'nDCG@5'])  # MRR is a single value, not per-K
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔍 Antes del Reranking**")
                for metric in k_metrics:
                    if metric in avg_before:
                        st.metric(
                            label=metric,
                            value=f"{avg_before[metric]:.3f}",
                            help=f"Promedio de {metric} antes del reranking LLM"
                        )
            
            with col2:
                if use_llm_reranker:
                    st.markdown("**🤖 Después del Reranking LLM**")
                    for metric in k_metrics:
                        if metric in avg_after:
                            # Calcular delta
                            delta = avg_after[metric] - avg_before.get(metric, 0)
                            st.metric(
                                label=metric,
                                value=f"{avg_after[metric]:.3f}",
                                delta=f"{delta:+.3f}",
                                help=f"Promedio de {metric} después del reranking LLM"
                            )
                else:
                    st.info("ℹ️ Reranking LLM deshabilitado")


def display_visual_comparison(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool, 
                            num_questions: int, model_name: str):
    """Muestra gráfico de comparación visual."""
    if use_llm_reranker and avg_before and avg_after:
        st.subheader("📈 Comparación Visual")
        
        # Preparar datos para el gráfico
        metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'Accuracy@5', 'BinaryAccuracy@5', 'RankingAccuracy@5', 'MRR', 'nDCG@5']  # MRR is a single value
        before_values = [avg_before.get(m, 0) for m in metrics_to_plot]
        after_values = [avg_after.get(m, 0) for m in metrics_to_plot]
        
        # Crear gráfico de barras comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Antes del Reranking',
            x=metrics_to_plot,
            y=before_values,
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Después del Reranking',
            x=metrics_to_plot,
            y=after_values,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title=f'Comparación de Métricas Promedio ({num_questions} preguntas)',
            xaxis_title='Métricas',
            yaxis_title='Valor Promedio',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"individual_comparison_{model_name}")


def display_summary_table(avg_before: Dict, avg_after: Dict, use_llm_reranker: bool):
    """Muestra tabla resumen con interpretación."""
    summary_metrics = ['MRR', 'Precision@5', 'Recall@5', 'F1@5', 'nDCG@5']
    
    if use_llm_reranker:
        st.subheader("📋 Resumen Comparativo")
        
        summary_data = []
        for metric in summary_metrics:
            if metric in avg_before or metric in avg_after:
                before_val = avg_before.get(metric, 0)
                after_val = avg_after.get(metric, 0)
                improvement = after_val - before_val
                
                summary_data.append({
                    'Métrica': metric,
                    'Antes': f"{before_val:.3f}",
                    'Después': f"{after_val:.3f}",
                    'Mejora': f"{improvement:+.3f}",
                    'Evaluación Antes': grade_metric(before_val),
                    'Evaluación Después': grade_metric(after_val)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.subheader("📋 Resumen de Métricas")
        
        summary_data = []
        for metric in summary_metrics:
            if metric in avg_before:
                val = avg_before[metric]
                summary_data.append({
                    'Métrica': metric,
                    'Valor': f"{val:.3f}",
                    'Evaluación': grade_metric(val)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)


def display_cumulative_metrics(results: Dict[str, Any], model_name: str, use_llm_reranker: bool):
    """
    Muestra los resultados de métricas acumulativas en la interfaz.
    
    Args:
        results: Resultados de la evaluación
        model_name: Nombre del modelo usado
        use_llm_reranker: Si se usó LLM reranking
    """
    num_questions = results['num_questions_evaluated']
    avg_before = results['avg_before_metrics']
    avg_after = results['avg_after_metrics']
    
    st.success(f"✅ Evaluación completada para {num_questions} preguntas")
    
    # Mostrar estadísticas de RAG
    display_rag_statistics(results)
    
    # Métricas principales en columnas
    st.subheader("📊 Métricas Promedio")
    
    # Mostrar métricas en tabs
    display_metrics_tabs(avg_before, avg_after, use_llm_reranker)
    
    # Gráfico de comparación
    display_visual_comparison(avg_before, avg_after, use_llm_reranker, num_questions, model_name)
    
    # Tabla resumen
    display_summary_table(avg_before, avg_after, use_llm_reranker)


def display_models_comparison(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool) -> None:
    """Display comprehensive side-by-side comparison of multiple models."""
    st.subheader("📈 Comparación Completa Entre Modelos")

    # Métricas principales para visualización (enfoque en k=5 para simplicidad en comparación)
    main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR', 'nDCG@5']  # MRR is a single value, not per-K
    
    # Métricas completas para análisis detallado
    all_k_metrics = []
    for k in [1, 3, 5, 10]:
        all_k_metrics.extend([f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}'])
    all_k_metrics.extend(['MRR', 'nDCG@5'])  # MRR is a single value, not per-K
    
    # Preparar datos para múltiples visualizaciones
    models_data = {}
    comparison_data = []
    improvement_data = []
    
    for model_name, res in results.items():
        before_metrics = res['avg_before_metrics']
        after_metrics = res['avg_after_metrics']
        
        models_data[model_name] = {
            'before': before_metrics,
            'after': after_metrics,
            'num_questions': res['num_questions_evaluated']
        }
        
        # Datos para gráfico de barras comparativo
        metrics = after_metrics if use_llm_reranker else before_metrics
        for m in main_metrics:
            comparison_data.append({
                'Modelo': model_name, 
                'Métrica': m, 
                'Valor': metrics.get(m, 0),
                'Tipo': 'Después del Reranking' if use_llm_reranker else 'Antes del Reranking'
            })
        
        # Datos para gráfico de mejora
        if use_llm_reranker:
            for m in main_metrics:
                before_val = before_metrics.get(m, 0)
                after_val = after_metrics.get(m, 0)
                improvement = ((after_val - before_val) / before_val * 100) if before_val > 0 else 0
                improvement_data.append({
                    'Modelo': model_name,
                    'Métrica': m,
                    'Mejora (%)': improvement,
                    'Delta': after_val - before_val
                })

    # Tab layout para múltiples visualizaciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Comparación General", "🎯 Gráfico Radar", "🔥 Mapa de Calor", "📈 Mejoras", "🔢 Métricas por K"])
    
    with tab1:
        st.markdown("#### Comparación de Métricas Principales")
        
        # Gráfico de barras mejorado
        df_comparison = pd.DataFrame(comparison_data)
        fig_bar = px.bar(
            df_comparison, 
            x='Métrica', 
            y='Valor', 
            color='Modelo',
            barmode='group',
            title=f"Comparación de Métricas {'Después del Reranking' if use_llm_reranker else 'Antes del Reranking'}",
            height=500
        )
        
        fig_bar.update_layout(
            xaxis_title="Métrica",
            yaxis_title="Valor",
            legend_title="Modelo de Embedding",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Agregar líneas de referencia
        fig_bar.add_hline(y=0.7, line_dash="dash", line_color="green", 
                         annotation_text="Muy Bueno (≥0.7)")
        fig_bar.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                         annotation_text="Bueno (≥0.4)")
        
        st.plotly_chart(fig_bar, use_container_width=True, key="comparison_bar_chart")
        
        # Tabla resumen
        st.markdown("#### Resumen Numérico")
        summary_table = []
        for model_name, data in models_data.items():
            metrics = data['after'] if use_llm_reranker else data['before']
            summary_table.append({
                'Modelo': model_name,
                'Precision@5': f"{metrics.get('Precision@5', 0):.3f}",
                'Recall@5': f"{metrics.get('Recall@5', 0):.3f}",
                'F1@5': f"{metrics.get('F1@5', 0):.3f}",
                'MRR': f"{metrics.get('MRR', 0):.3f}",  # MRR is a single value
                'nDCG@5': f"{metrics.get('nDCG@5', 0):.3f}",
                'Preguntas': data['num_questions']
            })
        
        summary_df = pd.DataFrame(summary_table)
        st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### Gráfico Radar - Vista Multidimensional")
        
        # Crear gráfico radar
        fig_radar = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (model_name, data) in enumerate(models_data.items()):
            metrics = data['after'] if use_llm_reranker else data['before']
            
            values = [metrics.get(m, 0) for m in main_metrics]
            values.append(values[0])  # Cerrar el radar
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=main_metrics + [main_metrics[0]],
                fill='toself',
                name=model_name,
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                )
            ),
            showlegend=True,
            title="Comparación Multidimensional de Modelos",
            height=600
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
        
        # Interpretación del radar
        st.markdown("##### 🎯 Interpretación:")
        st.markdown("""
        - **Área más grande**: Mejor rendimiento general
        - **Forma regular**: Rendimiento equilibrado entre métricas
        - **Picos específicos**: Fortalezas particulares del modelo
        """)
    
    with tab3:
        st.markdown("#### Mapa de Calor - Rendimiento Detallado")
        
        # Preparar datos para heatmap
        all_metrics = set()
        for data in models_data.values():
            metrics = data['after'] if use_llm_reranker else data['before']
            all_metrics.update(metrics.keys())
        
        # Usar métricas detalladas para el heatmap
        relevant_metrics = [m for m in all_k_metrics if m in all_metrics]
        relevant_metrics.sort()
        
        heatmap_data = []
        for model_name, data in models_data.items():
            metrics = data['after'] if use_llm_reranker else data['before']
            model_values = [metrics.get(m, 0) for m in relevant_metrics]
            heatmap_data.append(model_values)
        
        # Crear heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=relevant_metrics,
            y=list(models_data.keys()),
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Valor de Métrica"),
            hoverongaps=False,
            text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title="Mapa de Calor - Todas las Métricas",
            xaxis_title="Métrica",
            yaxis_title="Modelo",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart")
        
        # Ranking de modelos
        st.markdown("##### 🏆 Ranking por Métrica:")
        for metric in main_metrics:
            metric_values = [(name, data['after' if use_llm_reranker else 'before'].get(metric, 0)) 
                           for name, data in models_data.items()]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            
            ranking_text = " > ".join([f"**{name}** ({val:.3f})" for name, val in metric_values])
            st.markdown(f"**{metric}:** {ranking_text}")
    
    with tab4:
        if use_llm_reranker and improvement_data:
            st.markdown("#### Mejoras con Reranking LLM")
            
            # Gráfico de mejoras porcentuales
            df_improvement = pd.DataFrame(improvement_data)
            
            fig_improvement = px.bar(
                df_improvement,
                x='Métrica',
                y='Mejora (%)',
                color='Modelo',
                barmode='group',
                title="Mejora Porcentual por Modelo y Métrica",
                height=500
            )
            
            fig_improvement.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig_improvement.update_layout(
                xaxis_title="Métrica",
                yaxis_title="Mejora (%)",
                legend_title="Modelo"
            )
            
            st.plotly_chart(fig_improvement, use_container_width=True, key="improvement_chart")
            
            # Gráfico de deltas absolutas
            fig_delta = px.bar(
                df_improvement,
                x='Métrica',
                y='Delta',
                color='Modelo',
                barmode='group',
                title="Mejora Absoluta por Modelo y Métrica",
                height=400
            )
            
            fig_delta.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig_delta.update_layout(
                xaxis_title="Métrica",
                yaxis_title="Delta (Después - Antes)",
                legend_title="Modelo"
            )
            
            st.plotly_chart(fig_delta, use_container_width=True, key="delta_chart")
            
            # Estadísticas de mejora
            st.markdown("##### 📊 Estadísticas de Mejora:")
            for model_name in models_data.keys():
                model_improvements = [item for item in improvement_data if item['Modelo'] == model_name]
                avg_improvement = np.mean([item['Mejora (%)'] for item in model_improvements])
                positive_improvements = sum(1 for item in model_improvements if item['Mejora (%)'] > 0)
                
                st.markdown(f"**{model_name}:** {avg_improvement:+.1f}% mejora promedio, {positive_improvements}/{len(model_improvements)} métricas mejoradas")
        else:
            st.info("ℹ️ Las mejoras solo se muestran cuando el reranking LLM está habilitado")
    
    with tab5:
        st.markdown("#### Comparación Detallada por Valores de K")
        
        # Crear subtabs para cada valor de k
        k_tab1, k_tab2, k_tab3, k_tab4 = st.tabs([f"📊 K={k}" for k in [1, 3, 5, 10]])
        
        for i, k in enumerate([1, 3, 5, 10]):
            with [k_tab1, k_tab2, k_tab3, k_tab4][i]:
                k_metrics = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}']
                
                # Datos para gráfico de barras por k
                k_data = []
                for model_name, res in results.items():
                    metrics = res['avg_after_metrics'] if use_llm_reranker else res['avg_before_metrics']
                    for metric in k_metrics:
                        if metric in metrics:
                            k_data.append({
                                'Modelo': model_name,
                                'Métrica': metric,
                                'Valor': metrics[metric]
                            })
                
                if k_data:
                    df_k = pd.DataFrame(k_data)
                    fig_k = px.bar(
                        df_k,
                        x='Métrica',
                        y='Valor',
                        color='Modelo',
                        barmode='group',
                        title=f"Comparación de Métricas para K={k}",
                        height=400
                    )
                    
                    fig_k.update_layout(
                        xaxis_title="Métrica",
                        yaxis_title="Valor",
                        legend_title="Modelo"
                    )
                    
                    st.plotly_chart(fig_k, use_container_width=True, key=f"k_comparison_{k}")
                    
                    # Tabla resumen para este k
                    st.markdown(f"##### Tabla Resumen para K={k}")
                    k_summary = []
                    for model_name, res in results.items():
                        metrics = res['avg_after_metrics'] if use_llm_reranker else res['avg_before_metrics']
                        row = {'Modelo': model_name}
                        for metric in k_metrics:
                            row[metric] = f"{metrics.get(metric, 0):.3f}"
                        k_summary.append(row)
                    
                    k_summary_df = pd.DataFrame(k_summary)
                    st.dataframe(k_summary_df, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para K={k}")
    
    # Resumen general al final
    st.markdown("---")
    st.markdown("### 🏆 Resumen General de Rendimiento")
    
    # Calcular puntaje general por modelo
    model_scores = {}
    for model_name, data in models_data.items():
        metrics = data['after'] if use_llm_reranker else data['before']
        
        # Calcular puntaje ponderado (puedes ajustar los pesos según importancia)
        weights = {
            'Precision@5': 0.25,
            'Recall@5': 0.25,
            'F1@5': 0.25,
            'MRR': 0.15,  # MRR is a single value, not per-K
            'nDCG@5': 0.10
        }
        
        score = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
        model_scores[model_name] = {
            'score': score,
            'num_questions': data['num_questions']
        }
    
    # Ordenar modelos por puntaje
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Mostrar ranking
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Ranking Global")
        for i, (model_name, data) in enumerate(sorted_models):
            medal = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}."
            score_pct = data['score'] * 100
            st.markdown(f"{medal} **{model_name}**: {score_pct:.1f}% puntaje general")
    
    with col2:
        st.markdown("#### 📈 Métricas del Ganador")
        if sorted_models:
            winner_name = sorted_models[0][0]
            winner_metrics = models_data[winner_name]['after' if use_llm_reranker else 'before']
            
            for metric in main_metrics:
                value = winner_metrics.get(metric, 0)
                quality = grade_metric(value)
                st.metric(
                    label=metric,
                    value=f"{value:.3f}",
                    help=f"Calidad: {quality}"
                )
    
    # Gráfico de puntaje general
    st.markdown("#### 📊 Puntaje General Comparativo")
    score_data = []
    for model_name, data in model_scores.items():
        score_data.append({
            'Modelo': model_name,
            'Puntaje General': data['score'],
            'Puntaje (%)': data['score'] * 100
        })
    
    score_df = pd.DataFrame(score_data)
    fig_score = px.bar(
        score_df,
        x='Modelo',
        y='Puntaje (%)',
        title="Puntaje General por Modelo (Ponderado)",
        color='Puntaje (%)',
        color_continuous_scale='RdYlGn',
        height=400
    )
    
    fig_score.update_layout(
        xaxis_title="Modelo de Embedding",
        yaxis_title="Puntaje General (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig_score, use_container_width=True, key="score_chart")
    
    # Recomendaciones
    st.markdown("#### 💡 Recomendaciones")
    
    if sorted_models:
        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]
        
        best_metrics = models_data[best_model]['after' if use_llm_reranker else 'before']
        worst_metrics = models_data[worst_model]['after' if use_llm_reranker else 'before']
        
        st.success(f"🏆 **Mejor modelo general**: {best_model}")
        st.info(f"📊 **Uso recomendado**: {best_model} para aplicaciones que requieren balance entre precisión y recall")
        
        # Encontrar fortalezas específicas
        for metric in main_metrics:
            metric_ranking = [(name, data['after' if use_llm_reranker else 'before'].get(metric, 0)) 
                            for name, data in models_data.items()]
            metric_ranking.sort(key=lambda x: x[1], reverse=True)
            
            if metric_ranking[0][0] != best_model:
                st.info(f"🎯 **Especialista en {metric}**: {metric_ranking[0][0]} ({metric_ranking[0][1]:.3f})")
    
    # Mostrar estadísticas adicionales
    with st.expander("📋 Estadísticas Detalladas"):
        st.markdown("##### Variabilidad entre modelos:")
        for metric in main_metrics:
            values = [data['after' if use_llm_reranker else 'before'].get(metric, 0) 
                     for data in models_data.values()]
            std_dev = np.std(values)
            mean_val = np.mean(values)
            cv = (std_dev / mean_val) * 100 if mean_val > 0 else 0
            
            st.markdown(f"**{metric}**: μ={mean_val:.3f}, σ={std_dev:.3f}, CV={cv:.1f}%")