# comparison_page.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import html
import time
from utils.clients import initialize_clients
from utils.qa_pipeline import answer_question_documents_only
from config import EMBEDDING_MODELS, MODEL_DESCRIPTIONS
from utils.weaviate_utils_improved import WeaviateConfig
from utils.extract_links import extract_urls_from_answer
from utils.metrics import calculate_content_metrics
from utils.pdf_generator import generate_pdf_report

def generate_summary(question: str, results: list, gemini_client) -> str:
    """Generates a brief summary of why the results are relevant using Gemini."""
    if not results:
        return "No se encontraron documentos para generar un resumen."

    titles = [f"#{i+1}: {doc.get('title', 'Sin título')}" for i, doc in enumerate(results)]
    
    prompt = (
        f"User Question: {question}\n\n"
        f"The following documents were returned:\n"
        f"{str.join(' | ', titles)}\n\n"
        "Based on the document titles, briefly explain in one paragraph why these results are relevant to the user's question. "
        "Focus on the main themes and technologies covered."
    )
    
    try:
        response = gemini_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error al generar el resumen: {e}"


@st.cache_data(ttl=600)
def get_and_filter_questions(_weaviate_wrapper, num_questions: int = 100):
    """Gets random questions and filters them for valid links."""
    random_questions = _weaviate_wrapper.get_sample_questions(limit=num_questions, random_sample=True)
    
    filtered_questions = []
    for q in random_questions:
        answer = q.get("accepted_answer", "")
        links = extract_urls_from_answer(answer)
        ms_links = [link for link in links if "learn.microsoft.com" in link]
        
        if ms_links:
            q["ms_links"] = ms_links
            filtered_questions.append(q)
    
    return filtered_questions

def show_comparison_page():
    """Displays the model comparison page."""
    st.subheader("🔬 Comparación de Modelos de Embedding")
    st.markdown("Ejecuta una consulta en los tres modelos de embedding y compara los resultados y métricas lado a lado.")

    # --- 1. User Input ---
    weaviate_wrapper, _, _, _, _ = initialize_clients("multi-qa-mpnet-base-dot-v1")
    
    if 'questions_for_dropdown' not in st.session_state:
        with st.spinner("Cargando preguntas de ejemplo..."):
            st.session_state.questions_for_dropdown = get_and_filter_questions(weaviate_wrapper)

    questions = st.session_state.questions_for_dropdown
    question_options = {f"#{i+1}: {q.get('title', 'Sin título')}": q for i, q in enumerate(questions)}
    
    selected_question_title = st.selectbox(
        "Selecciona una pregunta de prueba:",
        options=question_options.keys()
    )
    
    selected_question = question_options[selected_question_title]
    
    with st.expander("Detalles de la Pregunta Seleccionada", expanded=True):
        st.markdown(f"**Título:** {selected_question.get('title', 'N/A')}")
        st.markdown(f"**Contenido:**")
        st.markdown(selected_question.get('question_content', 'N/A'), unsafe_allow_html=True)
        with st.expander("Respuesta Aceptada (Ground Truth)", expanded=False):
            st.markdown(selected_question.get('accepted_answer', 'N/A'), unsafe_allow_html=True)
        st.markdown(f"**Enlaces de Microsoft Learn Extraídos ({len(selected_question.get('ms_links', []))}):**")
        for link in selected_question.get("ms_links", []):
            st.markdown(f"- {link}")

    top_k = st.slider("Documentos a retornar por modelo", 5, 20, 10, key="comparison_top_k")
    use_reranker = st.toggle("Habilitar Reranking con LLM", value=True, help="Usa un LLM para reordenar los resultados iniciales y mejorar la relevancia. Aumentará la latencia.")

    if st.button("🔍 Comparar Modelos", type="primary", use_container_width=True):
        question_to_ask = f"{selected_question.get('title', '')} {selected_question.get('question_content', '')}".strip()

        if not question_to_ask:
            st.warning("La pregunta seleccionada está vacía.")
            return

        st.session_state.comparison_results = {}
        
        for model_key in EMBEDDING_MODELS.keys():
            with st.spinner(f"Consultando con el modelo: {model_key}..."):
                try:
                    start_time = time.time()
                    weaviate_wrapper, embedding_client, openai_client, gemini_client, _ = initialize_clients(model_key, st.session_state.get('generative_model_name', 'gemini-pro'))
                    
                    results, debug_info = answer_question_documents_only(
                        question_to_ask, weaviate_wrapper, embedding_client, openai_client,
                        top_k=top_k, use_llm_reranker=use_reranker, use_questions_collection=True
                    )
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    summary = generate_summary(question_to_ask, results, gemini_client)
                    content_metrics = calculate_content_metrics(results, selected_question.get("accepted_answer", ""))
                    
                    st.session_state.comparison_results[model_key] = {
                        "results": results, "debug_info": debug_info, "summary": summary, "time": elapsed_time, "content_metrics": content_metrics
                    }
                except Exception as e:
                    st.session_state.comparison_results[model_key] = {"results": [], "error": str(e)}

    if 'comparison_results' in st.session_state:
        # --- Performance Data Calculation ---
        perf_data = []
        for model_key, data in st.session_state.comparison_results.items():
            if data and not data.get("error"):
                latency = data.get("time", 0)
                throughput = 1 / latency if latency > 0 else 0
                perf_data.append({
                    "Modelo": model_key,
                    "Latencia (s)": latency,
                    "Throughput (QPS)": throughput
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            # --- Download Button ---
            pdf_bytes = generate_pdf_report(selected_question, st.session_state.comparison_results, perf_df)
            
            st.download_button(
                label="📄 Descargar Reporte en PDF",
                data=pdf_bytes,
                file_name=f"reporte_comparacion_{selected_question.get('title', 'pregunta')[:20].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("### 📊 Resultados de la Comparación")

        ground_truth_links = set(selected_question.get("ms_links", []))
        all_links = {}
        for model_key, model_results in st.session_state.comparison_results.items():
            for doc in model_results.get('results', []):
                link = doc.get('link')
                if link:
                    if link not in all_links: all_links[link] = []
                    all_links[link].append(model_key)
        
        duplicate_links = {link: models for link, models in all_links.items() if len(models) > 1}

        cols = st.columns(len(EMBEDDING_MODELS))
        
        for i, model_key in enumerate(EMBEDDING_MODELS.keys()):
            with cols[i]:
                st.markdown(f"#### Modelo: `{model_key}`")
                model_info = MODEL_DESCRIPTIONS.get(model_key, {})
                st.markdown(f"*{model_info.get('description', 'N/A')}*")
                st.markdown(f"**Dim:** {model_info.get('dimensions', 'N/A')} | **Prov:** {model_info.get('provider', 'N/A')}")
                st.markdown("---")

                result_data = st.session_state.comparison_results.get(model_key)

                if not result_data or result_data.get("error"):
                    st.error(f"Error: {result_data.get('error', 'Error desconocido')}")
                    continue

                if not result_data["results"]:
                    st.warning("No se encontraron documentos.")
                    continue

                scores = [doc.get('score', 0) for doc in result_data["results"]]
                st.metric("Documentos Encontrados", len(result_data["results"]))
                st.metric("Latencia Total", f"{result_data.get('time', 0):.2f}s")
                st.metric("Score Promedio", f"{np.mean(scores):.4f}" if scores else "N/A")

                st.markdown("##### Resumen de Relevancia (IA)")
                st.info(result_data.get("summary", "No se pudo generar el resumen."))
                
                for rank, doc in enumerate(result_data["results"], 1):
                    score = doc.get('score', 0)
                    link = doc.get('link', '')
                    title = doc.get('title', 'Sin título')

                    safe_title = html.escape(title)
                    safe_link = html.escape(link)
                    is_ground_truth = link in ground_truth_links
                    is_duplicate = link in duplicate_links
                    
                    card_style = f"border-left: 4px solid {'#28a745' if score > 0.8 else '#ffc107' if score > 0.6 else '#dc3545'}; margin-bottom: 5px; padding: 0.5rem;"
                    if is_ground_truth: card_style += " background-color: #e6ffed;"

                    tooltip_text = ""
                    if is_duplicate:
                        other_models = [m for m in duplicate_links.get(link, []) if m != model_key]
                        if other_models: tooltip_text = f"También en: {', '.join(other_models)}"
                    
                    html_card = (
                        f'<div class="doc-card" style="{card_style}">'
                        f'<p style="font-size: 0.9em; margin-bottom: 0.2rem; color: #000000;">'
                        f'<b>#{rank}</b>: {safe_title} '
                        f'<span title="{html.escape(tooltip_text)}">{"🔄" if is_duplicate else ""}</span>'
                        f'{"✅" if is_ground_truth else ""}'
                        f'</p>'
                        f'<p style="font-size: 0.8em; margin-bottom: 0.2rem; color: #000000;">'
                        f'<b>Score:</b> <span style="font-weight: bold;">{score:.3f}</span>'
                        f'</p>'
                        f'<p style="font-size: 0.7em; word-wrap: break-word; margin-bottom: 0;">'
                        f'<a href="{safe_link}" target="_blank">{safe_link}</a>'
                        f'</p>'
                        f'</div>'
                    )
                    
                    st.markdown(html_card, unsafe_allow_html=True)

        # --- 4. Performance and Quality Metrics ---
        st.markdown("---")
        st.markdown("### 📈 Métricas de Rendimiento y Calidad")

        if perf_data:
            # Create a single section for all metrics
            st.markdown("#### Tabla de Métricas de Rendimiento")
            st.dataframe(perf_df.style.format({
                "Latencia (s)": "{:.2f}",
                "Throughput (QPS)": "{:.2f}"
            }))

            p_col1, p_col2 = st.columns(2)

            with p_col1:
                st.markdown("#### Latencia por Modelo")
                fig_latency = px.bar(
                    perf_df, x="Modelo", y="Latencia (s)", color="Modelo",
                    title="Latencia de Consulta (End-to-End)",
                    labels={"Latencia (s)": "Tiempo (segundos)"}
                )
                st.plotly_chart(fig_latency, use_container_width=True)

            with p_col2:
                st.markdown("#### Throughput por Modelo")
                fig_throughput = px.bar(
                    perf_df, x="Modelo", y="Throughput (QPS)", color="Modelo",
                    title="Throughput Estimado (Consultas por Segundo)",
                    labels={"Throughput (QPS)": "Consultas / Segundo"}
                )
                st.plotly_chart(fig_throughput, use_container_width=True)
            
            st.markdown("#### Distribución de Scores de Relevancia")
            st.info("Este gráfico muestra la distribución de los scores de similitud para los documentos recuperados por cada modelo. Un buen modelo debería tener scores altos (cercanos a 1.0) y una distribución compacta en la parte superior.")
            score_data = []
            for model, data in st.session_state.comparison_results.items():
                if data and not data.get("error"):
                    for doc in data["results"]:
                        score_data.append({"Modelo": model, "Score": doc.get("score", 0)})
            
            if score_data:
                score_df = pd.DataFrame(score_data)
                fig_box = px.box(score_df, x="Modelo", y="Score", color="Modelo", title="Distribución de Scores por Modelo")
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("#### Métricas de Calidad de Contenido (BERTScore y ROUGE)")
            st.info("Estas métricas comparan el contenido de los documentos recuperados con la **Respuesta Aceptada (Ground Truth)**. Miden la superposición semántica (BERTScore) y de palabras clave (ROUGE).")
            content_metrics_data = []
            for model, data in st.session_state.comparison_results.items():
                if data and data.get("content_metrics"):
                    metrics = data["content_metrics"]
                    metrics["Modelo"] = model
                    content_metrics_data.append(metrics)
            
            if content_metrics_data:
                content_df = pd.DataFrame(content_metrics_data).set_index("Modelo")
                st.dataframe(content_df)

                st.markdown("##### Leyenda de Métricas de Contenido")
                legend_data = {
                    "Métrica": ["BERT_P", "BERT_R", "BERT_F1", "ROUGE1", "ROUGE2", "ROUGE-L"],
                    "Significado": [
                        "Precisión semántica (palabras coincidentes en la respuesta)",
                        "Recall semántico (palabras coincidentes en la referencia)",
                        "Balance de Precisión y Recall semántico",
                        "Superposición de palabras individuales (unigramas)",
                        "Superposición de pares de palabras (bigramas)",
                        "Superposición de la subsecuencia más larga"
                    ],
                    "Interpretación": [
                        "Bueno > 0.9", "Bueno > 0.9", "Bueno > 0.9",
                        "Bueno > 0.4", "Bueno > 0.2", "Bueno > 0.3"
                    ]
                }
                st.table(pd.DataFrame(legend_data))
            else:
                st.info("No hay métricas de calidad de contenido para mostrar.")
        else:
            st.info("No hay suficientes datos para generar gráficos de rendimiento.")

        # --- 5. Process Flow Diagram ---
        st.markdown("---")
        st.markdown("### 🗺️ Diagrama del Proceso RAG")
        st.graphviz_chart('''
            digraph {
                rankdir="LR";
                node [shape=box, style="rounded,filled", fillcolor="#e6f2ff"];
                
                subgraph cluster_user {
                    label="Usuario";
                    color=blue;
                    pregunta [label="Pregunta del Usuario"];
                }
                
                subgraph cluster_system {
                    label="Sistema RAG";
                    color=green;
                    refinar [label="1. Refinar Consulta (Limpiar y Destilar)"];
                    embedding [label="2. Generar Embedding"];
                    busqueda [label="3. Búsqueda Vectorial"];
                    reranking [label="4. Reranking Local (Cross-Encoder)"];
                    documentos [label="5. Documentos Relevantes"];
                    resumen [label="6. Generación de Resumen (Gemini/GPT)"];
                }
                
                pregunta -> refinar;
                refinar -> embedding [label="  Consulta Refinada"];
                embedding -> busqueda [label="  Vector"];
                busqueda -> reranking [label="  Top-K Inicial"];
                reranking -> documentos [label="  Top-K Final"];
                documentos -> resumen;
            }
        ''')

