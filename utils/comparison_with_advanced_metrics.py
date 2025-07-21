"""
Enhanced comparison functionality with advanced RAG metrics.
This can be integrated into the comparison page.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.enhanced_evaluation import (
    evaluate_rag_with_advanced_metrics,
    create_advanced_metrics_summary,
    get_advanced_metrics_interpretation
)

def show_advanced_metrics_comparison(
    question: str,
    models_to_compare: list,
    chromadb_wrapper,
    embedding_client,
    openai_client,
    gemini_client=None,
    local_tinyllama_client=None,
    local_mistral_client=None,
    top_k: int = 10
):
    """
    Show comparison with advanced RAG metrics.
    
    Args:
        question: Question to evaluate
        models_to_compare: List of model names to compare
        ... (other clients)
        top_k: Number of documents to retrieve
    """
    if st.button("🧪 Ejecutar Evaluación Avanzada", type="primary"):
        
        results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(models_to_compare):
            status_text.text(f"Evaluando modelo: {model_name}")
            
            # Determine which clients to use based on model
            eval_result = evaluate_rag_with_advanced_metrics(
                question=question,
                chromadb_wrapper=chromadb_wrapper,
                embedding_client=embedding_client,
                openai_client=openai_client,
                gemini_client=gemini_client,
                local_tinyllama_client=local_tinyllama_client,
                local_mistral_client=local_mistral_client,
                generative_model_name=model_name,
                top_k=top_k
            )
            
            results[model_name] = eval_result
            progress_bar.progress((i + 1) / len(models_to_compare))
        
        status_text.text("Evaluación completada!")
        
        # Display results
        display_advanced_metrics_results(results)

def display_advanced_metrics_results(results: dict):
    """Display advanced metrics results in Streamlit."""
    
    st.markdown("## 🔬 Resultados de Evaluación Avanzada")
    
    # 1. Summary table
    st.markdown("### 📊 Resumen de Métricas Avanzadas")
    
    summary_data = []
    for model_name, result in results.items():
        if result.get('status') == 'success' and 'advanced_metrics' in result:
            adv_metrics = result['advanced_metrics']
            
            row = {
                "Modelo": model_name,
                "Respuesta Generada": "✅" if result.get('generated_answer') else "❌",
                "Tiempo (s)": f"{result.get('response_time', 0):.2f}",
            }
            
            # Add advanced metrics if available
            if 'hallucination' in adv_metrics:
                row["🚫 Alucinación"] = f"{adv_metrics['hallucination']['hallucination_score']:.3f}"
            
            if 'context_utilization' in adv_metrics:
                row["🎯 Utilización"] = f"{adv_metrics['context_utilization']['utilization_score']:.3f}"
            
            if 'completeness' in adv_metrics:
                row["✅ Completitud"] = f"{adv_metrics['completeness']['completeness_score']:.3f}"
            
            if 'satisfaction' in adv_metrics:
                row["😊 Satisfacción"] = f"{adv_metrics['satisfaction']['satisfaction_score']:.3f}"
            
            summary_data.append(row)
        else:
            summary_data.append({
                "Modelo": model_name,
                "Respuesta Generada": "❌",
                "Error": result.get('error', 'Unknown error')
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Apply color coding
        if len(summary_df.columns) > 3:  # Has metrics
            styled_summary = style_advanced_metrics_table(summary_df)
            st.dataframe(styled_summary, use_container_width=True)
        else:
            st.dataframe(summary_df, use_container_width=True)
    
    # 2. Detailed metrics visualization
    create_advanced_metrics_visualizations(results)
    
    # 3. Generated answers comparison
    st.markdown("### 💬 Comparación de Respuestas Generadas")
    
    for model_name, result in results.items():
        if result.get('generated_answer'):
            with st.expander(f"Respuesta de {model_name}"):
                st.markdown(result['generated_answer'])
                
                # Show advanced metrics details
                if 'advanced_metrics' in result:
                    show_detailed_metrics(result['advanced_metrics'])
    
    # 4. Interpretation guide
    with st.expander("📖 Guía de Interpretación de Métricas Avanzadas"):
        interpretation = get_advanced_metrics_interpretation()
        
        for metric_name, metric_info in interpretation['metrics'].items():
            st.markdown(f"**{metric_name.replace('_', ' ').title()}**")
            st.markdown(f"- {metric_info['description']}")
            st.markdown(f"- Rango: {metric_info['range']}")
            st.markdown(f"- Interpretación: {metric_info['interpretation']}")
            
            # Show thresholds
            thresholds = metric_info['thresholds']
            st.markdown(f"- 🟢 Excelente: {thresholds['excellent']}")
            st.markdown(f"- 🟡 Bueno: {thresholds['good']}")
            st.markdown(f"- 🟠 Aceptable: {thresholds['acceptable']}")
            st.markdown(f"- 🔴 Pobre: {thresholds['poor']}")
            st.markdown("---")

def style_advanced_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply color styling to advanced metrics table."""
    
    def color_advanced_metric(val, column_name):
        """Color cells based on advanced metric thresholds."""
        if pd.isna(val) or not isinstance(val, str):
            return ''
        
        try:
            numeric_val = float(val)
        except:
            return ''
        
        # Define thresholds for each advanced metric
        thresholds = {
            "🚫 Alucinación": {"excellent": 0.1, "good": 0.2, "reverse": True},  # Lower is better
            "🎯 Utilización": {"excellent": 0.8, "good": 0.6, "reverse": False},  # Higher is better
            "✅ Completitud": {"excellent": 0.9, "good": 0.7, "reverse": False},  # Higher is better
            "😊 Satisfacción": {"excellent": 0.8, "good": 0.6, "reverse": False}  # Higher is better
        }
        
        if column_name not in thresholds:
            return ''
        
        threshold = thresholds[column_name]
        excellent_thresh = threshold["excellent"]
        good_thresh = threshold["good"]
        is_reverse = threshold["reverse"]
        
        if is_reverse:
            # For metrics where lower is better
            if numeric_val <= excellent_thresh:
                return 'background-color: #90EE90'  # Light green
            elif numeric_val <= good_thresh:
                return 'background-color: #FFFFE0'  # Light yellow
            else:
                return 'background-color: #FFB6C1'  # Light red
        else:
            # For metrics where higher is better
            if numeric_val >= excellent_thresh:
                return 'background-color: #90EE90'  # Light green
            elif numeric_val >= good_thresh:
                return 'background-color: #FFFFE0'  # Light yellow
            else:
                return 'background-color: #FFB6C1'  # Light red
    
    # Apply styling
    styled_df = df.style
    
    for col in ["🚫 Alucinación", "🎯 Utilización", "✅ Completitud", "😊 Satisfacción"]:
        if col in df.columns:
            styled_df = styled_df.apply(
                lambda x: [color_advanced_metric(val, col) for val in x], 
                subset=[col]
            )
    
    return styled_df

def create_advanced_metrics_visualizations(results: dict):
    """Create visualizations for advanced metrics."""
    
    st.markdown("### 📈 Visualizaciones de Métricas Avanzadas")
    
    # Prepare data for visualization
    viz_data = []
    for model_name, result in results.items():
        if result.get('status') == 'success' and 'advanced_metrics' in result:
            adv_metrics = result['advanced_metrics']
            
            viz_row = {"Modelo": model_name}
            
            if 'hallucination' in adv_metrics:
                viz_row["Alucinación"] = adv_metrics['hallucination']['hallucination_score']
            
            if 'context_utilization' in adv_metrics:
                viz_row["Utilización"] = adv_metrics['context_utilization']['utilization_score']
            
            if 'completeness' in adv_metrics:
                viz_row["Completitud"] = adv_metrics['completeness']['completeness_score']
            
            if 'satisfaction' in adv_metrics:
                viz_row["Satisfacción"] = adv_metrics['satisfaction']['satisfaction_score']
            
            viz_data.append(viz_row)
    
    if not viz_data:
        st.warning("No hay datos suficientes para generar visualizaciones.")
        return
    
    viz_df = pd.DataFrame(viz_data)
    
    # 1. Radar chart for each model
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Radar Chart - Métricas por Modelo")
        create_radar_chart(viz_df)
    
    with col2:
        st.markdown("#### Comparación de Métricas")
        create_metrics_comparison_chart(viz_df)
    
    # 2. Detailed breakdown
    st.markdown("#### Distribución de Métricas")
    create_metrics_distribution_chart(viz_df)

def create_radar_chart(df: pd.DataFrame):
    """Create radar chart for metrics comparison."""
    
    metrics = ['Alucinación', 'Utilización', 'Completitud', 'Satisfacción']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 3:
        st.warning("Necesitamos al menos 3 métricas para crear un radar chart.")
        return
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        model_name = row['Modelo']
        
        # Invert hallucination for radar chart (so higher is better for all)
        values = []
        labels = []
        
        for metric in available_metrics:
            if metric in row and pd.notna(row[metric]):
                if metric == 'Alucinación':
                    values.append(1 - row[metric])  # Invert so higher is better
                    labels.append('Anti-Alucinación')
                else:
                    values.append(row[metric])
                    labels.append(metric)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=model_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_metrics_comparison_chart(df: pd.DataFrame):
    """Create bar chart comparing metrics across models."""
    
    metrics = ['Alucinación', 'Utilización', 'Completitud', 'Satisfacción']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        st.warning("No hay métricas disponibles para comparar.")
        return
    
    fig = make_subplots(
        rows=len(available_metrics), 
        cols=1,
        subplot_titles=available_metrics,
        vertical_spacing=0.1
    )
    
    for i, metric in enumerate(available_metrics):
        fig.add_trace(
            go.Bar(
                x=df['Modelo'],
                y=df[metric],
                name=metric,
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        # Add threshold lines
        if metric == 'Alucinación':
            fig.add_hline(y=0.1, line_dash="dash", line_color="green", 
                         annotation_text="Excelente", row=i+1, col=1)
            fig.add_hline(y=0.2, line_dash="dash", line_color="orange", 
                         annotation_text="Bueno", row=i+1, col=1)
        else:
            fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                         annotation_text="Excelente", row=i+1, col=1)
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                         annotation_text="Bueno", row=i+1, col=1)
    
    fig.update_layout(height=200 * len(available_metrics))
    st.plotly_chart(fig, use_container_width=True)

def create_metrics_distribution_chart(df: pd.DataFrame):
    """Create distribution chart for all metrics."""
    
    metrics = ['Alucinación', 'Utilización', 'Completitud', 'Satisfacción']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return
    
    # Melt dataframe for easier plotting
    df_melted = pd.melt(
        df, 
        id_vars=['Modelo'], 
        value_vars=available_metrics,
        var_name='Métrica', 
        value_name='Valor'
    )
    
    fig = px.box(
        df_melted, 
        x='Métrica', 
        y='Valor', 
        color='Modelo',
        title="Distribución de Métricas Avanzadas por Modelo"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_detailed_metrics(advanced_metrics: dict):
    """Show detailed breakdown of advanced metrics."""
    
    tabs = st.tabs(["🚫 Alucinación", "🎯 Utilización", "✅ Completitud", "😊 Satisfacción"])
    
    with tabs[0]:  # Hallucination
        if 'hallucination' in advanced_metrics:
            hall_metrics = advanced_metrics['hallucination']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score de Alucinación", f"{hall_metrics['hallucination_score']:.3f}")
            with col2:
                st.metric("Claims Totales", hall_metrics['total_claims'])
            with col3:
                st.metric("Claims No Soportadas", hall_metrics['unsupported_claims'])
            
            if hall_metrics.get('details'):
                st.markdown("**Ejemplos de Claims No Soportadas:**")
                for claim in hall_metrics['details'].get('unsupported', []):
                    st.markdown(f"- {claim}")
    
    with tabs[1]:  # Context Utilization
        if 'context_utilization' in advanced_metrics:
            util_metrics = advanced_metrics['context_utilization']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score de Utilización", f"{util_metrics['utilization_score']:.3f}")
            with col2:
                st.metric("Docs Utilizados", f"{util_metrics['docs_utilized']}/{util_metrics['total_docs']}")
            with col3:
                st.metric("Ratio de Compresión", f"{util_metrics['compression_ratio']:.3f}")
    
    with tabs[2]:  # Completeness
        if 'completeness' in advanced_metrics:
            comp_metrics = advanced_metrics['completeness']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score de Completitud", f"{comp_metrics['completeness_score']:.3f}")
            with col2:
                st.metric("Tipo de Pregunta", comp_metrics['question_type'])
            
            st.markdown("**Componentes Presentes:**")
            for component in comp_metrics.get('present_components', []):
                st.markdown(f"✅ {component}")
            
            if comp_metrics.get('missing_components'):
                st.markdown("**Componentes Faltantes:**")
                for component in comp_metrics['missing_components']:
                    st.markdown(f"❌ {component}")
    
    with tabs[3]:  # Satisfaction
        if 'satisfaction' in advanced_metrics:
            sat_metrics = advanced_metrics['satisfaction']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Claridad", f"{sat_metrics['clarity_score']:.3f}")
            with col2:
                st.metric("Directness", f"{sat_metrics['directness_score']:.3f}")
            with col3:
                st.metric("Actionabilidad", f"{sat_metrics['actionability_score']:.3f}")
            with col4:
                st.metric("Confianza", f"{sat_metrics['confidence_score']:.3f}")

# Example integration function
def add_advanced_metrics_to_comparison_page():
    """
    Function to integrate advanced metrics into existing comparison page.
    This would be called from the main comparison page.
    """
    
    # Add toggle for advanced evaluation
    with st.sidebar:
        st.markdown("### 🧪 Evaluación Avanzada")
        enable_advanced = st.checkbox(
            "Habilitar Métricas Avanzadas",
            help="Incluye detección de alucinaciones, utilización de contexto, completitud y satisfacción del usuario"
        )
        
        if enable_advanced:
            st.info("⚠️ Las métricas avanzadas pueden aumentar significativamente el tiempo de procesamiento.")
    
    return enable_advanced