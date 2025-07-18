# utils/pdf_generator.py
import base64
import html
from io import BytesIO
from weasyprint import HTML
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Any

def generate_pdf_report(
    selected_question: dict,
    comparison_results: dict,
    perf_df: pd.DataFrame
) -> bytes:
    """Generates a PDF report from the comparison results."""
    
    # --- 1. Generate Graph Images ---
    fig_latency = px.bar(
        perf_df, x="Modelo", y="Latencia (s)", color="Modelo",
        title="Latencia de Consulta (End-to-End)",
        labels={"Latencia (s)": "Tiempo (segundos)"}
    )
    latency_img = fig_latency.to_image(format="png")
    latency_img_b64 = base64.b64encode(latency_img).decode('utf-8')

    fig_throughput = px.bar(
        perf_df, x="Modelo", y="Throughput Est. (QPS)", color="Modelo",
        title="Throughput Estimado (Consultas por Segundo)",
        labels={"Throughput Est. (QPS)": "Consultas / Segundo"}
    )
    throughput_img = fig_throughput.to_image(format="png")
    throughput_img_b64 = base64.b64encode(throughput_img).decode('utf-8')

    # --- 2. Build HTML Content ---
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; }}
            h1, h2, h3 {{ color: #333; }}
            .question-details, .model-column, .metrics-summary {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                page-break-inside: avoid;
            }}
            .doc-card {{
                border-left: 4px solid #ccc;
                margin-bottom: 5px;
                padding: 0.5rem;
                background-color: #f9f9f9;
            }}
            .ground-truth {{ background-color: #e6ffed; }}
            .graph-container img {{ width: 100%; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Comparación de Modelos</h1>
        
        <div class="question-details">
            <h2>Pregunta Seleccionada</h2>
            <p><b>Título:</b> {html.escape(selected_question.get('title', 'N/A'))}</p>
            <div><b>Contenido:</b> {selected_question.get('question_content', 'N/A')}</div>
        </div>

        <h2>Resultados por Modelo</h2>
    """

    for model_key, data in comparison_results.items():
        html_content += f'<div class="model-column">'
        html_content += f"<h3>Modelo: {model_key}</h3>"
        if data.get("error"):
            html_content += f'<p style="color: red;">Error: {data["error"]}</p>'
        elif not data.get("results"):
            html_content += "<p>No se encontraron documentos.</p>"
        else:
            for rank, doc in enumerate(data["results"], 1):
                title = html.escape(doc.get('title', 'Sin título'))
                link = html.escape(doc.get('link', ''))
                score = doc.get('score', 0)
                is_gt = link in selected_question.get("ms_links", [])
                card_class = "doc-card ground-truth" if is_gt else "doc-card"
                
                html_content += (
                    f'<div class="{card_class}">'
                    f'<p><b>#{rank}</b>: {title} {"✅" if is_gt else ""}</p>'
                    f'<p><b>Score:</b> {score:.4f}</p>'
                    f'<p><a href="{link}">{link}</a></p>'
                    f'</div>'
                )
        html_content += "</div>"

    html_content += f"""
        <div class="metrics-summary">
            <h2>Métricas de Rendimiento</h2>
            <h3>Tabla de Métricas</h3>
            {perf_df.to_html(classes='table', float_format='{:.2f}'.format)}
            
            <h3>Gráficos de Comparación</h3>
            <div class="graph-container">
                <img src="data:image/png;base64,{latency_img_b64}">
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{throughput_img_b64}">
            </div>
        </div>
    </body>
    </html>
    """

    # --- 3. Convert to PDF ---
    pdf_file = BytesIO()
    HTML(string=html_content).write_pdf(pdf_file)
    pdf_file.seek(0)
    return pdf_file.getvalue()


def get_color_palette():
    """Retorna una paleta de colores personalizada."""
    return [
        '#1f77b4',  # azul
        '#ff7f0e',  # naranja
        '#2ca02c',  # verde
        '#d62728',  # rojo
        '#9467bd',  # púrpura
        '#8c564b',  # marrón
        '#e377c2',  # rosa
        '#7f7f7f',  # gris
        '#bcbd22',  # oliva
        '#17becf'   # cian
    ]


def add_metric_definitions_page(pdf, ax):
    """Añade una página con definiciones de métricas al PDF."""
    ax.text(0.5, 0.95, 'Definiciones de Métricas', 
           fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    definitions = [
        ("Precision@k", "Proporción de documentos relevantes entre los k primeros recuperados"),
        ("Recall@k", "Proporción de documentos relevantes recuperados entre todos los relevantes"),
        ("F1@k", "Media armónica entre Precision@k y Recall@k"),
        ("Accuracy@k", "Proporción de documentos correctamente clasificados"),
        ("BinaryAccuracy@k", "Accuracy binaria considerando relevancia/no relevancia"),
        ("RankingAccuracy@k", "Accuracy considerando el orden de ranking"),
        ("MRR (Mean Reciprocal Rank)", "Promedio del recíproco de la posición del primer documento relevante"),
        ("nDCG@k", "Ganancia Cumulativa Descontada Normalizada para los k primeros documentos"),
        ("Reranking LLM", "Reordenamiento de documentos usando un modelo de lenguaje grande")
    ]
    
    y_position = 0.85
    for metric, definition in definitions:
        ax.text(0.05, y_position, f"• {metric}:", 
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, y_position - 0.03, f"  {definition}", 
               fontsize=10, transform=ax.transAxes, wrap=True)
        y_position -= 0.08
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def generate_cumulative_pdf_report(results: Dict[str, Any], model_name: str, use_llm_reranker: bool, 
                                  generative_model_name: str, top_k: int, evaluation_time: str) -> bytes:
    """
    Genera un reporte PDF completo de métricas cumulativas.
    
    Args:
        results: Resultados de la evaluación
        model_name: Nombre del modelo de embedding
        use_llm_reranker: Si se usó reranking LLM
        generative_model_name: Nombre del modelo generativo
        top_k: Valor de k usado
        evaluation_time: Tiempo de evaluación
        
    Returns:
        Datos del PDF en bytes
    """
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Página 1: Resumen ejecutivo
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        # Título principal
        ax.text(0.5, 0.95, 'Reporte de Métricas Cumulativas', 
               fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Información del modelo
        ax.text(0.5, 0.88, f'Modelo: {model_name}', 
               fontsize=14, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.84, f'Generativo: {generative_model_name}', 
               fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.80, f'Top-K: {top_k} | Reranking LLM: {"Sí" if use_llm_reranker else "No"}', 
               fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.76, f'Fecha: {evaluation_time}', 
               fontsize=10, ha='center', transform=ax.transAxes)
        
        # Línea separadora
        ax.axhline(y=0.72, xmin=0.1, xmax=0.9, color='black', linewidth=1, transform=ax.transAxes)
        
        # Resumen de resultados
        num_questions = results['num_questions_evaluated']
        avg_before = results['avg_before_metrics']
        avg_after = results.get('avg_after_metrics', {})
        
        ax.text(0.5, 0.68, f'Evaluación completada para {num_questions} preguntas', 
               fontsize=14, ha='center', fontweight='bold', transform=ax.transAxes)
        
        # Métricas principales
        y_pos = 0.60
        main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR']
        
        ax.text(0.5, y_pos, 'Métricas Principales', 
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        y_pos -= 0.05
        
        for metric in main_metrics:
            if metric in avg_before:
                before_val = avg_before[metric]
                after_val = avg_after.get(metric, 0) if use_llm_reranker else before_val
                
                if use_llm_reranker and metric in avg_after:
                    improvement = after_val - before_val
                    ax.text(0.2, y_pos, f'{metric}:', 
                           fontsize=12, fontweight='bold', transform=ax.transAxes)
                    ax.text(0.4, y_pos, f'{before_val:.3f} → {after_val:.3f} ({improvement:+.3f})', 
                           fontsize=12, transform=ax.transAxes)
                else:
                    ax.text(0.2, y_pos, f'{metric}:', 
                           fontsize=12, fontweight='bold', transform=ax.transAxes)
                    ax.text(0.4, y_pos, f'{before_val:.3f}', 
                           fontsize=12, transform=ax.transAxes)
                y_pos -= 0.04
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Página 2: Definiciones de métricas
        fig, ax = plt.subplots(figsize=(8.5, 11))
        add_metric_definitions_page(pdf, ax)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Página 3: Gráfico de comparación (si hay reranking)
        if use_llm_reranker and avg_before and avg_after:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            
            metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'MRR', 'nDCG@5']
            metrics_available = [m for m in metrics_to_plot if m in avg_before and m in avg_after]
            
            if metrics_available:
                before_values = [avg_before[m] for m in metrics_available]
                after_values = [avg_after[m] for m in metrics_available]
                
                x = range(len(metrics_available))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], before_values, width, 
                      label='Antes del Reranking', color='lightblue')
                ax.bar([i + width/2 for i in x], after_values, width, 
                      label='Después del Reranking', color='darkblue')
                
                ax.set_xlabel('Métricas')
                ax.set_ylabel('Valor')
                ax.set_title(f'Comparación de Métricas Promedio\n({num_questions} preguntas)')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_available, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    buffer.seek(0)
    return buffer.getvalue()


def generate_multi_model_pdf_report(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool,
                                   generative_model_name: str, top_k: int, evaluation_time: str) -> bytes:
    """
    Genera un reporte PDF de comparación multi-modelo.
    
    Args:
        results: Resultados de múltiples modelos
        use_llm_reranker: Si se usó reranking LLM
        generative_model_name: Nombre del modelo generativo
        top_k: Valor de k usado
        evaluation_time: Tiempo de evaluación
        
    Returns:
        Datos del PDF en bytes
    """
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Página 1: Resumen ejecutivo
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        ax.text(0.5, 0.95, 'Reporte Comparativo Multi-Modelo', 
               fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        ax.text(0.5, 0.88, f'Generativo: {generative_model_name}', 
               fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.84, f'Top-K: {top_k} | Reranking LLM: {"Sí" if use_llm_reranker else "No"}', 
               fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.80, f'Fecha: {evaluation_time}', 
               fontsize=10, ha='center', transform=ax.transAxes)
        
        # Resumen por modelo
        y_pos = 0.70
        ax.text(0.5, y_pos, 'Resumen por Modelo', 
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        y_pos -= 0.06
        
        for model_name, model_results in results.items():
            num_questions = model_results['num_questions_evaluated']
            ax.text(0.1, y_pos, f'• {model_name}:', 
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
            ax.text(0.1, y_pos - 0.025, f'  {num_questions} preguntas evaluadas', 
                   fontsize=10, transform=ax.transAxes)
            y_pos -= 0.06
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Página 2: Definiciones
        fig, ax = plt.subplots(figsize=(8.5, 11))
        add_metric_definitions_page(pdf, ax)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Página 3: Gráfico comparativo
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        metrics_to_compare = ['Precision@5', 'Recall@5', 'F1@5', 'MRR']
        model_names = list(results.keys())
        colors = get_color_palette()[:len(model_names)]
        
        metric_data = {}
        for metric in metrics_to_compare:
            if use_llm_reranker:
                metric_data[metric] = [results[model]['avg_after_metrics'].get(metric, 0) for model in model_names]
            else:
                metric_data[metric] = [results[model]['avg_before_metrics'].get(metric, 0) for model in model_names]
        
        x = range(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_compare):
            offset = (i - len(metrics_to_compare)/2 + 0.5) * width
            ax.bar([j + offset for j in x], metric_data[metric], width, 
                  label=metric, color=colors[i % len(colors)])
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Valor de Métrica')
        ax.set_title(f'Comparación de Métricas entre Modelos\n{"Después del Reranking" if use_llm_reranker else "Antes del Reranking"}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    buffer.seek(0)
    return buffer.getvalue()
