# utils/pdf_generator.py
import base64
import html
from io import BytesIO
from weasyprint import HTML
import pandas as pd
import plotly.express as px

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
