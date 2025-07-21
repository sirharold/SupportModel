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
                                  generative_model_name: str, top_k: int, evaluation_time: str,
                                  llm_conclusions: str = '', llm_improvements: str = '') -> bytes:
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
        ax.axhline(y=0.72, xmin=0.1, xmax=0.9, color='black', linewidth=1)
        
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

        # Page for LLM Conclusions and Improvements - MEJORADO
        if llm_conclusions or llm_improvements:
            def add_text_with_wrapping(ax, text, x, y_start, max_width=80, font_size=10):
                """Helper function to add text with proper line wrapping"""
                lines = []
                for line in text.split('\n'):
                    if line.strip():
                        if len(line) > max_width:
                            words = line.split()
                            current_line = ""
                            for word in words:
                                if len(current_line + word + " ") <= max_width:
                                    current_line += word + " "
                                else:
                                    if current_line.strip():
                                        lines.append(current_line.strip())
                                    current_line = word + " "
                            if current_line.strip():
                                lines.append(current_line.strip())
                        else:
                            lines.append(line.strip())
                
                y_pos = y_start
                for line in lines:
                    if y_pos < 0.1:  # Nueva página si se acaba el espacio
                        return y_pos, True  # Indica que necesita nueva página
                    ax.text(x, y_pos, line, fontsize=font_size, transform=ax.transAxes)
                    y_pos -= 0.025
                return y_pos, False
            
            # Primera página de conclusiones
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.95, 'Conclusiones y Mejoras (Generado por LLM)', 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
            
            y_pos = 0.88
            need_new_page = False
            
            if llm_conclusions:
                ax.text(0.05, y_pos, '📝 Conclusiones:', fontsize=12, fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.03
                y_pos, need_new_page = add_text_with_wrapping(ax, llm_conclusions, 0.07, y_pos)
                y_pos -= 0.04  # Espacio extra entre secciones
                
                if need_new_page and llm_improvements:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Nueva página para mejoras
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(0.5, 0.95, 'Posibles Mejoras (Continuación)', 
                           fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
                    ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
                    y_pos = 0.88

            if llm_improvements:
                if y_pos < 0.3:  # Si queda poco espacio, nueva página
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(0.5, 0.95, 'Posibles Mejoras (Continuación)', 
                           fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
                    ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
                    y_pos = 0.88
                
                ax.text(0.05, y_pos, '💡 Posibles Mejoras y Próximos Pasos:', fontsize=12, fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.03
                add_text_with_wrapping(ax, llm_improvements, 0.07, y_pos)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Page 2: Definiciones de métricas
        fig, ax = plt.subplots(figsize=(8.5, 11))
        add_metric_definitions_page(pdf, ax)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Gráfico de comparación (si hay reranking)
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
            
        # Page 4: Tabla detallada de todas las métricas
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.95, f'Tabla Detallada de Métricas - {model_name}', 
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
        
        # Preparar datos de tabla
        all_metrics = ['precision@1', 'precision@3', 'precision@5', 'precision@10',
                      'recall@1', 'recall@3', 'recall@5', 'recall@10',
                      'f1@1', 'f1@3', 'f1@5', 'f1@10',
                      'map@1', 'map@3', 'map@5', 'map@10',
                      'mrr@1', 'mrr@3', 'mrr@5', 'mrr@10',
                      'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']
        
        table_data = []
        if use_llm_reranker and avg_before and avg_after:
            # Mostrar antes y después
            headers = ['Métrica', 'Antes LLM', 'Después LLM', 'Mejora']
            for metric in all_metrics:
                if metric in avg_before:
                    before_val = avg_before[metric]
                    after_val = avg_after.get(metric, before_val)
                    improvement = after_val - before_val
                    table_data.append([
                        metric.replace('@', ' @').upper(),
                        f"{before_val:.3f}",
                        f"{after_val:.3f}",
                        f"{improvement:+.3f}"
                    ])
        else:
            # Solo mostrar métricas disponibles
            headers = ['Métrica', 'Valor']
            for metric in all_metrics:
                if metric in avg_before:
                    table_data.append([
                        metric.replace('@', ' @').upper(),
                        f"{avg_before[metric]:.3f}"
                    ])
        
        # Crear tabla si hay datos
        if table_data:
            # Dividir en dos columnas si hay muchas métricas
            mid_point = len(table_data) // 2
            left_data = table_data[:mid_point]
            right_data = table_data[mid_point:]
            
            # Tabla izquierda
            table_left = ax.table(cellText=left_data, colLabels=headers,
                                cellLoc='center', loc='center',
                                bbox=[0.05, 0.3, 0.4, 0.6])
            table_left.auto_set_font_size(False)
            table_left.set_fontsize(8)
            table_left.scale(1, 1.5)
            
            # Colorear headers
            for i in range(len(headers)):
                table_left[(0, i)].set_facecolor('#4CAF50')
                table_left[(0, i)].set_text_props(weight='bold', color='white')
            
            # Tabla derecha si hay datos
            if right_data:
                table_right = ax.table(cellText=right_data, colLabels=headers,
                                     cellLoc='center', loc='center',
                                     bbox=[0.55, 0.3, 0.4, 0.6])
                table_right.auto_set_font_size(False)
                table_right.set_fontsize(8)
                table_right.scale(1, 1.5)
                
                # Colorear headers
                for i in range(len(headers)):
                    table_right[(0, i)].set_facecolor('#4CAF50')
                    table_right[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Métricas RAG (si están disponibles)
        rag_metrics = {}
        
        # Buscar en avg_rag_before_metrics o avg_rag_after_metrics
        if use_llm_reranker:
            rag_metrics = results.get('avg_rag_after_metrics', 
                         results.get('avg_rag_before_metrics', {}))
        else:
            rag_metrics = results.get('avg_rag_before_metrics', {})
        
        # También buscar en avg_before_metrics/avg_after_metrics
        if not rag_metrics:
            if use_llm_reranker and avg_after:
                all_metrics = avg_after
            else:
                all_metrics = avg_before
            
            # Extraer métricas RAG
            rag_keys = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
            rag_metrics = {k: v for k, v in all_metrics.items() if k in rag_keys}
        
        if rag_metrics and any(v > 0 for v in rag_metrics.values()):
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.95, f'Métricas RAG - {model_name}', 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
            
            # Crear visualización de métricas RAG
            rag_names = ['Faithfulness', 'Answer Relevance', 'Answer Correctness', 'Answer Similarity']
            rag_keys = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
            rag_values = [rag_metrics.get(key, 0) for key in rag_keys]
            
            # Crear gráfico de barras
            y_positions = [0.75, 0.65, 0.55, 0.45]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (name, value) in enumerate(zip(rag_names, rag_values)):
                # Barra de fondo
                from matplotlib.patches import Rectangle
                bar_bg = Rectangle((0.3, y_positions[i] - 0.02), 0.4, 0.04, 
                                  facecolor='#E0E0E0', transform=ax.transAxes)
                ax.add_patch(bar_bg)
                
                # Barra de valor
                bar_val = Rectangle((0.3, y_positions[i] - 0.02), 0.4 * value, 0.04,
                                   facecolor=colors[i], transform=ax.transAxes)
                ax.add_patch(bar_val)
                
                # Etiquetas
                ax.text(0.05, y_positions[i], name + ':', fontsize=12, fontweight='bold',
                       va='center', transform=ax.transAxes)
                ax.text(0.75, y_positions[i], f'{value:.3f}', fontsize=12,
                       va='center', transform=ax.transAxes)
            
            # Descripción de métricas RAG
            y_pos = 0.35
            ax.text(0.05, y_pos, 'Descripción de Métricas RAG:', 
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.03
            
            descriptions = [
                '• Faithfulness: ¿La respuesta es fiel a los documentos recuperados? (0-1)',
                '• Answer Relevance: ¿La respuesta responde directamente la pregunta? (0-1)', 
                '• Answer Correctness: ¿La respuesta es factualmente correcta? (0-1)',
                '• Answer Similarity: ¿La respuesta es similar a la respuesta esperada? (0-1)'
            ]
            
            for desc in descriptions:
                ax.text(0.07, y_pos, desc, fontsize=10, transform=ax.transAxes)
                y_pos -= 0.025
                
            # Interpretación
            y_pos -= 0.02
            ax.text(0.05, y_pos, 'Interpretación:', 
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.025
            
            interpretations = [
                '• Valores > 0.8: Excelente rendimiento',
                '• Valores 0.6-0.8: Buen rendimiento',
                '• Valores 0.4-0.6: Rendimiento moderado',
                '• Valores < 0.4: Necesita mejora'
            ]
            
            for interp in interpretations:
                ax.text(0.07, y_pos, interp, fontsize=9, transform=ax.transAxes)
                y_pos -= 0.02
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    buffer.seek(0)
    return buffer.getvalue()


def generate_multi_model_pdf_report(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool,
                                   generative_model_name: str, top_k: int, evaluation_time: str,
                                   llm_conclusions: str = '', llm_improvements: str = '') -> bytes:
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

        # Page for LLM Conclusions and Improvements - MEJORADO (Multi-modelo)
        if llm_conclusions or llm_improvements:
            def add_text_with_wrapping_multi(ax, text, x, y_start, max_width=80, font_size=10):
                """Helper function to add text with proper line wrapping for multi-model"""
                lines = []
                for line in text.split('\n'):
                    if line.strip():
                        if len(line) > max_width:
                            words = line.split()
                            current_line = ""
                            for word in words:
                                if len(current_line + word + " ") <= max_width:
                                    current_line += word + " "
                                else:
                                    if current_line.strip():
                                        lines.append(current_line.strip())
                                    current_line = word + " "
                            if current_line.strip():
                                lines.append(current_line.strip())
                        else:
                            lines.append(line.strip())
                
                y_pos = y_start
                for line in lines:
                    if y_pos < 0.1:
                        return y_pos, True
                    ax.text(x, y_pos, line, fontsize=font_size, transform=ax.transAxes)
                    y_pos -= 0.025
                return y_pos, False
            
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.95, 'Conclusiones y Mejoras Multi-Modelo (Generado por LLM)', 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
            
            y_pos = 0.88
            need_new_page = False
            
            if llm_conclusions:
                ax.text(0.05, y_pos, '📝 Conclusiones:', fontsize=12, fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.03
                y_pos, need_new_page = add_text_with_wrapping_multi(ax, llm_conclusions, 0.07, y_pos)
                y_pos -= 0.04
                
                if need_new_page and llm_improvements:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(0.5, 0.95, 'Mejoras Multi-Modelo (Continuación)', 
                           fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
                    ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
                    y_pos = 0.88

            if llm_improvements:
                if y_pos < 0.3:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(0.5, 0.95, 'Mejoras Multi-Modelo (Continuación)', 
                           fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
                    ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
                    y_pos = 0.88
                
                ax.text(0.05, y_pos, '💡 Posibles Mejoras y Próximos Pasos:', fontsize=12, fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.03
                add_text_with_wrapping_multi(ax, llm_improvements, 0.07, y_pos)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Page 2: Definiciones
        fig, ax = plt.subplots(figsize=(8.5, 11))
        add_metric_definitions_page(pdf, ax)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Gráfico comparativo
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
        
        # Page 4: Tabla detallada de métricas por modelo
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.95, 'Tabla Detallada de Métricas por Modelo', 
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
        
        # Crear tabla de métricas
        model_names = list(results.keys())
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'mrr@5', 'ndcg@5']
        
        # Preparar datos para la tabla
        table_data = []
        for model in model_names:
            if use_llm_reranker:
                metrics = results[model].get('avg_after_metrics', results[model].get('avg_before_metrics', {}))
            else:
                metrics = results[model].get('avg_before_metrics', {})
            
            row = [model[:20]]  # Truncar nombres largos
            for metric in key_metrics:
                value = metrics.get(metric, 0)
                row.append(f"{value:.3f}")
            table_data.append(row)
        
        # Crear tabla
        headers = ['Modelo'] + [m.replace('@', ' @').upper() for m in key_metrics]
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        bbox=[0.1, 0.3, 0.8, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Colorear headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Análisis por valores de K
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.95, 'Rendimiento por Valores de K', 
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
        
        # Gráfico de rendimiento por K
        k_values = [1, 3, 5, 10]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        y_pos = 0.85
        ax.text(0.5, y_pos, 'Comparación de F1-Score por K', 
               fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Crear subplot para gráfico
        from matplotlib.patches import Rectangle
        plot_box = Rectangle((0.1, 0.1), 0.8, 0.7, facecolor='white', edgecolor='black')
        ax.add_patch(plot_box)
        
        # Simular gráfico de líneas para cada modelo
        for i, model in enumerate(model_names[:4]):  # Máximo 4 modelos por claridad
            if use_llm_reranker:
                metrics = results[model].get('avg_after_metrics', results[model].get('avg_before_metrics', {}))
            else:
                metrics = results[model].get('avg_before_metrics', {})
            
            f1_values = [metrics.get(f'f1@{k}', 0) for k in k_values]
            
            # Crear líneas manualmente
            for j in range(len(k_values) - 1):
                x1 = 0.2 + j * 0.2
                x2 = 0.2 + (j + 1) * 0.2
                y1 = 0.2 + f1_values[j] * 0.5
                y2 = 0.2 + f1_values[j + 1] * 0.5
                ax.plot([x1, x2], [y1, y2], color=colors[i], linewidth=2, 
                       label=model[:15] if j == 0 else "", transform=ax.transAxes)
                ax.plot(x1, y1, 'o', color=colors[i], markersize=6, transform=ax.transAxes)
        
        # Leyenda y etiquetas
        ax.text(0.1, 0.05, 'K values: 1, 3, 5, 10', fontsize=10, transform=ax.transAxes)
        
        # Agregar leyenda manualmente
        legend_y = 0.75
        for i, model in enumerate(model_names[:4]):
            ax.plot(0.75, legend_y - i*0.04, 'o-', color=colors[i], linewidth=2, 
                   markersize=4, transform=ax.transAxes)
            ax.text(0.78, legend_y - i*0.04, model[:15], fontsize=9, 
                   va='center', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Métricas RAG (si están disponibles)
        has_rag_metrics = False
        rag_data = {}
        
        for model_name, model_results in results.items():
            # Buscar métricas RAG
            rag_metrics = {}
            
            # Buscar en avg_rag_before_metrics o avg_rag_after_metrics
            if use_llm_reranker:
                rag_metrics = model_results.get('avg_rag_after_metrics', 
                             model_results.get('avg_rag_before_metrics', {}))
            else:
                rag_metrics = model_results.get('avg_rag_before_metrics', {})
            
            # También buscar en avg_before_metrics/avg_after_metrics
            if not rag_metrics:
                if use_llm_reranker:
                    all_metrics = model_results.get('avg_after_metrics', 
                                 model_results.get('avg_before_metrics', {}))
                else:
                    all_metrics = model_results.get('avg_before_metrics', {})
                
                # Extraer métricas RAG
                rag_keys = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
                rag_metrics = {k: v for k, v in all_metrics.items() if k in rag_keys}
            
            if rag_metrics and any(v > 0 for v in rag_metrics.values()):
                has_rag_metrics = True
                rag_data[model_name] = rag_metrics
        
        if has_rag_metrics:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.95, 'Métricas RAG (Generación de Respuestas)', 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
            
            # Crear tabla de métricas RAG
            rag_metrics_names = ['faithfulness', 'answer_relevance', 'answer_correctness', 'answer_similarity']
            table_data = []
            
            for model in rag_data.keys():
                row = [model[:20]]
                for metric in rag_metrics_names:
                    value = rag_data[model].get(metric, 0)
                    row.append(f"{value:.3f}")
                table_data.append(row)
            
            headers = ['Modelo', 'Faithfulness', 'Relevance', 'Correctness', 'Similarity']
            table = ax.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            bbox=[0.1, 0.4, 0.8, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Colorear headers
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Añadir descripción de métricas RAG
            y_pos = 0.3
            ax.text(0.05, y_pos, 'Descripción de Métricas RAG:', 
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.03
            
            descriptions = [
                '• Faithfulness: ¿La respuesta es fiel a los documentos recuperados?',
                '• Answer Relevance: ¿La respuesta responde directamente la pregunta?',
                '• Answer Correctness: ¿La respuesta es factualmente correcta?',
                '• Answer Similarity: ¿La respuesta es similar a la respuesta esperada?'
            ]
            
            for desc in descriptions:
                ax.text(0.07, y_pos, desc, fontsize=10, transform=ax.transAxes)
                y_pos -= 0.025
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Page 7: Comparación avanzada (Before vs After si hay reranking)
        if use_llm_reranker:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.95, 'Comparación Avanzada: Antes vs Después del Reranking LLM', 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='black', linewidth=1)
            
            # Crear tabla comparativa
            comparison_data = []
            key_metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr@5']
            
            for model in model_names:
                before_metrics = results[model].get('avg_before_metrics', {})
                after_metrics = results[model].get('avg_after_metrics', {})
                
                for metric in key_metrics:
                    before_val = before_metrics.get(metric, 0)
                    after_val = after_metrics.get(metric, 0)
                    improvement = after_val - before_val
                    improvement_pct = (improvement / before_val * 100) if before_val > 0 else 0
                    
                    comparison_data.append([
                        model[:15], 
                        metric.replace('@', ' @').upper(),
                        f"{before_val:.3f}",
                        f"{after_val:.3f}",
                        f"{improvement:+.3f}",
                        f"{improvement_pct:+.1f}%"
                    ])
            
            headers = ['Modelo', 'Métrica', 'Antes', 'Después', 'Mejora', '% Mejora']
            table = ax.table(cellText=comparison_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            bbox=[0.05, 0.2, 0.9, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Colorear headers
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#9C27B0')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear filas de mejora/empeoramiento
            for i, row in enumerate(comparison_data, 1):
                improvement = float(row[4])
                if improvement > 0.01:
                    table[(i, 5)].set_facecolor('#C8E6C9')  # Verde claro para mejoras
                elif improvement < -0.01:
                    table[(i, 5)].set_facecolor('#FFCDD2')  # Rojo claro para empeoramientos
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    buffer.seek(0)
    return buffer.getvalue()
