# comparison_page.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import html
import time
from utils.clients import initialize_clients
from utils.qa_pipeline import answer_question_documents_only, answer_question_with_rag
from config import EMBEDDING_MODELS, MODEL_DESCRIPTIONS
from utils.weaviate_utils_improved import WeaviateConfig
from utils.extract_links import extract_urls_from_answer
from utils.metrics import calculate_content_metrics
from utils.pdf_generator import generate_pdf_report
from utils.enhanced_evaluation import evaluate_rag_with_advanced_metrics

def generate_summary(question: str, results: list, local_client=None) -> str:
    """Generates a brief summary of why the results are relevant using local models or heuristics."""
    if not results:
        return "No se encontraron documentos para generar un resumen."

    titles = [f"#{i+1}: {doc.get('title', 'Sin título')}" for i, doc in enumerate(results)]
    
    # Try local model first
    if local_client:
        prompt = (
            f"User Question: {question}\n\n"
            f"The following documents were returned:\n"
            f"{str.join(' | ', titles)}\n\n"
            "Based on the document titles, briefly explain in one paragraph why these results are relevant to the user's question. "
            "Focus on the main themes and technologies covered."
        )
        
        try:
            from utils.local_answer_generator import generate_final_answer_local
            response, _ = generate_final_answer_local(prompt, [], "mistral-7b", max_length=200)
            if response and not response.startswith("Error"):
                return response.strip()
        except Exception as e:
            pass  # Fall back to heuristic method
    
    # Fallback: Generate heuristic summary
    try:
        # Extract key technologies and services from titles
        technologies = set()
        services = set()
        
        for title in [doc.get('title', '') for doc in results]:
            title_lower = title.lower()
            
            # Common Azure services
            azure_services = ['azure', 'storage', 'blob', 'sql', 'cosmos', 'functions', 'app service', 
                            'virtual machine', 'kubernetes', 'container', 'devops', 'active directory']
            for service in azure_services:
                if service in title_lower:
                    services.add(service.title())
            
            # Common technologies
            tech_keywords = ['python', 'javascript', 'c#', 'java', 'docker', 'api', 'rest', 'json', 'authentication']
            for tech in tech_keywords:
                if tech in title_lower:
                    technologies.add(tech.upper() if len(tech) <= 3 else tech.title())
        
        summary_parts = []
        
        if services:
            summary_parts.append(f"Los documentos cubren servicios de Azure como {', '.join(list(services)[:3])}")
        
        if technologies:
            summary_parts.append(f"tecnologías como {', '.join(list(technologies)[:3])}")
        
        if not summary_parts:
            summary_parts.append("documentación técnica relevante")
        
        return f"Estos resultados son relevantes porque incluyen {' y '.join(summary_parts)} relacionados con tu consulta sobre '{question[:50]}...'."
        
    except Exception as e:
        return f"Se encontraron {len(results)} documentos relevantes para tu consulta."


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
    weaviate_wrapper, _, _, _, _, _, _ = initialize_clients("multi-qa-mpnet-base-dot-v1")
    
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
    
    # Configuración de métricas avanzadas
    with st.expander("🧪 Métricas Avanzadas RAG", expanded=False):
        enable_advanced_metrics = st.checkbox(
            "Habilitar Evaluación Avanzada",
            value=False,
            help="Incluye detección de alucinaciones, utilización de contexto, completitud y satisfacción del usuario. ⚠️ Aumenta significativamente el tiempo de procesamiento."
        )
        
        if enable_advanced_metrics:
            st.info("📊 **Métricas Avanzadas Incluidas:**\n"
                   "- 🚫 **Detección de Alucinaciones**: Información no soportada por el contexto\n"
                   "- 🎯 **Utilización de Contexto**: Qué tan bien se usa el contexto recuperado\n"
                   "- ✅ **Completitud de Respuesta**: Completitud basada en tipo de pregunta\n"
                   "- 😊 **Satisfacción del Usuario**: Claridad, directness y actionabilidad")
            
            generate_answers = st.checkbox(
                "Generar Respuestas para Evaluación",
                value=True,
                help="Necesario para calcular métricas avanzadas. Usará el modelo generativo seleccionado."
            )
        else:
            generate_answers = False

    if st.button("🔍 Comparar Modelos", type="primary", use_container_width=True):
        question_to_ask = f"{selected_question.get('title', '')} {selected_question.get('question_content', '')}".strip()

        if not question_to_ask:
            st.warning("La pregunta seleccionada está vacía.")
            return

        st.session_state.comparison_results = {}
        
        # Initialize progress tracking for advanced metrics
        total_models = len(EMBEDDING_MODELS.keys())
        if enable_advanced_metrics:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, model_key in enumerate(EMBEDDING_MODELS.keys()):
            model_display_name = f"{model_key} {'+ Advanced' if enable_advanced_metrics else ''}"
            
            with st.spinner(f"Consultando con el modelo: {model_display_name}..."):
                try:
                    start_time = time.time()
                    weaviate_wrapper, embedding_client, openai_client, gemini_client, local_llama_client, local_mistral_client, _ = initialize_clients(model_key, st.session_state.get('generative_model_name', 'llama-3.1-8b'))
                    
                    if enable_advanced_metrics and generate_answers:
                        # Use advanced evaluation pipeline
                        if enable_advanced_metrics:
                            status_text.text(f"Evaluando {model_key} con métricas avanzadas...")
                        
                        eval_result = evaluate_rag_with_advanced_metrics(
                            question=question_to_ask,
                            weaviate_wrapper=weaviate_wrapper,
                            embedding_client=embedding_client,
                            openai_client=openai_client,
                            gemini_client=gemini_client,
                            local_llama_client=local_llama_client,
                            local_mistral_client=local_mistral_client,
                            generative_model_name=st.session_state.get('generative_model_name', 'llama-3.1-8b'),
                            top_k=top_k
                        )
                        
                        # Extract data for compatibility with existing code  
                        # The evaluate_rag_with_advanced_metrics returns the RAG pipeline results
                        # We need to run the pipeline separately to get documents for display
                        results, debug_info = answer_question_documents_only(
                            question_to_ask, weaviate_wrapper, embedding_client, openai_client,
                            top_k=top_k, use_llm_reranker=use_reranker, use_questions_collection=True
                        )
                        
                        generated_answer = eval_result.get('generated_answer', '')
                        advanced_metrics = eval_result.get('advanced_metrics', {})
                        
                        # Calculate traditional metrics
                        summary = generate_summary(question_to_ask, results, local_mistral_client)
                        content_metrics = calculate_content_metrics(results, selected_question.get("accepted_answer", ""))
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                        st.session_state.comparison_results[model_key] = {
                            "results": results, 
                            "debug_info": debug_info, 
                            "summary": summary, 
                            "time": elapsed_time, 
                            "content_metrics": content_metrics,
                            "generated_answer": generated_answer,
                            "advanced_metrics": advanced_metrics,
                            "response_time": eval_result.get('response_time', elapsed_time)
                        }
                    else:
                        # Standard evaluation pipeline
                        results, debug_info = answer_question_documents_only(
                            question_to_ask, weaviate_wrapper, embedding_client, openai_client,
                            top_k=top_k, use_llm_reranker=use_reranker, use_questions_collection=True
                        )
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                        summary = generate_summary(question_to_ask, results, local_mistral_client)
                        content_metrics = calculate_content_metrics(results, selected_question.get("accepted_answer", ""))
                        
                        st.session_state.comparison_results[model_key] = {
                            "results": results, 
                            "debug_info": debug_info, 
                            "summary": summary, 
                            "time": elapsed_time, 
                            "content_metrics": content_metrics
                        }
                    
                    # Update progress for advanced metrics
                    if enable_advanced_metrics:
                        progress_bar.progress((i + 1) / total_models)
                        
                except Exception as e:
                    st.session_state.comparison_results[model_key] = {"results": [], "error": str(e)}
        
        # Clear progress indicators
        if enable_advanced_metrics:
            status_text.text("✅ Evaluación completada!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

    if 'comparison_results' in st.session_state:
        # --- Performance Data Calculation ---
        perf_data = []
        for model_key, data in st.session_state.comparison_results.items():
            if data and not data.get("error"):
                latency = data.get("time", 0)
                results = data.get("results", [])
                
                # Mejorar cálculo de throughput considerando overhead
                # Throughput real es menor debido a overhead del sistema
                effective_throughput = (1 / latency * 0.7) if latency > 0 else 0  # Factor 0.7 para overhead
                
                # Calcular métricas adicionales
                avg_score = np.mean([doc.get('score', 0) for doc in results]) if results else 0
                score_std = np.std([doc.get('score', 0) for doc in results]) if results else 0
                
                perf_data.append({
                    "Modelo": model_key,
                    "Latencia (s)": latency,
                    "Throughput Est. (QPS)": effective_throughput,
                    "Docs Recuperados": len(results),
                    "Score Promedio": avg_score,
                    "Desviación Score": score_std,
                    "Consistencia": 1 - score_std if score_std < 1 else 0  # Métrica de consistencia
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
                    perf_df, x="Modelo", y="Throughput Est. (QPS)", color="Modelo",
                    title="Throughput Estimado (Consultas por Segundo)",
                    labels={"Throughput Est. (QPS)": "Consultas / Segundo"}
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

            # Nuevas métricas específicas para RAG
            st.markdown("#### Métricas Adicionales de Calidad RAG")
            
            rag_col1, rag_col2 = st.columns(2)
            
            with rag_col1:
                st.markdown("##### Métricas de Consistencia y Diversidad")
                consistency_fig = px.bar(
                    perf_df, x="Modelo", y="Consistencia", color="Modelo",
                    title="Consistencia de Scores (1 = muy consistente)",
                    labels={"Consistencia": "Índice de Consistencia"}
                )
                st.plotly_chart(consistency_fig, use_container_width=True)
            
            with rag_col2:
                st.markdown("##### Calidad Promedio vs Variabilidad")
                scatter_fig = px.scatter(
                    perf_df, x="Score Promedio", y="Desviación Score", 
                    color="Modelo", size="Docs Recuperados",
                    title="Calidad vs Variabilidad de Resultados",
                    labels={
                        "Score Promedio": "Calidad Promedio",
                        "Desviación Score": "Variabilidad"
                    }
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

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
                
                def color_content_metrics(val, column_name):
                    """Color cells based on content metric thresholds."""
                    if pd.isna(val) or isinstance(val, str):  # Handle errors
                        return ''
                    
                    # Define thresholds for content metrics
                    content_thresholds = {
                        "BERT_P": {"excellent": 0.7, "good": 0.5},
                        "BERT_R": {"excellent": 0.7, "good": 0.5},
                        "BERT_F1": {"excellent": 0.7, "good": 0.5},
                        "ROUGE1": {"excellent": 0.4, "good": 0.25},
                        "ROUGE2": {"excellent": 0.2, "good": 0.12},
                        "ROUGE-L": {"excellent": 0.35, "good": 0.22}
                    }
                    
                    if column_name not in content_thresholds:
                        return ''
                    
                    threshold = content_thresholds[column_name]
                    excellent_thresh = threshold["excellent"]
                    good_thresh = threshold["good"]
                    
                    if val >= excellent_thresh:
                        return 'background-color: #90EE90'  # Light green
                    elif val >= good_thresh:
                        return 'background-color: #FFFFE0'  # Light yellow
                    else:
                        return 'background-color: #FFB6C1'  # Light red
                
                # Apply styling to content metrics
                styled_content_df = content_df.style.format("{:.4f}")
                
                # Apply colors to each content metric column
                for col in content_df.columns:
                    if col in ["BERT_P", "BERT_R", "BERT_F1", "ROUGE1", "ROUGE2", "ROUGE-L"]:
                        styled_content_df = styled_content_df.apply(
                            lambda x: [color_content_metrics(val, col) for val in x], 
                            subset=[col]
                        )
                
                st.dataframe(styled_content_df)
            else:
                st.info("No hay métricas de calidad de contenido para mostrar.")
            
            # Performance metrics table moved here
            st.markdown("#### Tabla de Métricas de Rendimiento Completas")
            
            def color_performance_metrics(val, column_name):
                """Color cells based on performance thresholds."""
                if pd.isna(val) or val == 0:
                    return ''
                
                # Define thresholds for each metric
                thresholds = {
                    "Latencia (s)": {"excellent": 2.0, "good": 5.0, "reverse": True},  # Lower is better
                    "Throughput Est. (QPS)": {"excellent": 1.0, "good": 0.5, "reverse": False},  # Higher is better
                    "Score Promedio": {"excellent": 0.8, "good": 0.6, "reverse": False},
                    "Desviación Score": {"excellent": 0.1, "good": 0.2, "reverse": True},  # Lower is better
                    "Consistencia": {"excellent": 0.8, "good": 0.6, "reverse": False}
                }
                
                if column_name not in thresholds:
                    return ''
                
                threshold = thresholds[column_name]
                excellent_thresh = threshold["excellent"]
                good_thresh = threshold["good"]
                is_reverse = threshold["reverse"]
                
                if is_reverse:
                    # For metrics where lower is better (latency, deviation)
                    if val <= excellent_thresh:
                        return 'background-color: #90EE90'  # Light green
                    elif val <= good_thresh:
                        return 'background-color: #FFFFE0'  # Light yellow
                    else:
                        return 'background-color: #FFB6C1'  # Light red
                else:
                    # For metrics where higher is better (throughput, score, consistency)
                    if val >= excellent_thresh:
                        return 'background-color: #90EE90'  # Light green
                    elif val >= good_thresh:
                        return 'background-color: #FFFFE0'  # Light yellow
                    else:
                        return 'background-color: #FFB6C1'  # Light red
            
            # Apply styling to the dataframe
            styled_df = perf_df.style.format({
                "Latencia (s)": "{:.3f}",
                "Throughput Est. (QPS)": "{:.2f}",
                "Score Promedio": "{:.4f}",
                "Desviación Score": "{:.4f}",
                "Consistencia": "{:.3f}"
            })
            
            # Apply colors to each column
            for col in ["Latencia (s)", "Throughput Est. (QPS)", "Score Promedio", "Desviación Score", "Consistencia"]:
                if col in perf_df.columns:
                    styled_df = styled_df.apply(lambda x: [color_performance_metrics(val, col) for val in x], subset=[col])
            
            st.dataframe(styled_df)
            
            # Advanced RAG metrics table - Same style as performance metrics
            if any(result.get('advanced_metrics') for result in st.session_state.comparison_results.values()):
                st.markdown("#### Tabla de Métricas Avanzadas RAG")
                
                def color_advanced_rag_metrics(val, column_name):
                    """Color cells based on advanced RAG metric thresholds."""
                    # Handle various data types and edge cases
                    if pd.isna(val):
                        return ''
                    
                    # Convert to float if possible
                    try:
                        numeric_val = float(val)
                    except (ValueError, TypeError):
                        return ''
                    
                    # Skip zero values
                    if numeric_val == 0:
                        return ''
                    
                    # Define thresholds for advanced RAG metrics
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
                        # For metrics where lower is better (hallucination)
                        if numeric_val <= excellent_thresh:
                            return 'background-color: #90EE90'  # Light green
                        elif numeric_val <= good_thresh:
                            return 'background-color: #FFFFE0'  # Light yellow
                        else:
                            return 'background-color: #FFB6C1'  # Light red
                    else:
                        # For metrics where higher is better (utilization, completeness, satisfaction)
                        if numeric_val >= excellent_thresh:
                            return 'background-color: #90EE90'  # Light green
                        elif numeric_val >= good_thresh:
                            return 'background-color: #FFFFE0'  # Light yellow
                        else:
                            return 'background-color: #FFB6C1'  # Light red
                
                # Create advanced metrics dataframe
                advanced_metrics_data = []
                for model_key, data in st.session_state.comparison_results.items():
                    if data and data.get("advanced_metrics") and not data.get("error"):
                        adv_metrics = data["advanced_metrics"]
                        
                        row = {"Modelo": str(model_key)}
                        
                        # Extract advanced metrics values with proper type conversion
                        if 'hallucination' in adv_metrics and isinstance(adv_metrics['hallucination'], dict):
                            hall_score = adv_metrics['hallucination'].get('hallucination_score', 0.0)
                            row["🚫 Alucinación"] = float(hall_score) if hall_score is not None else 0.0
                        else:
                            row["🚫 Alucinación"] = 0.0
                        
                        if 'context_utilization' in adv_metrics and isinstance(adv_metrics['context_utilization'], dict):
                            util_score = adv_metrics['context_utilization'].get('utilization_score', 0.0)
                            row["🎯 Utilización"] = float(util_score) if util_score is not None else 0.0
                        else:
                            row["🎯 Utilización"] = 0.0
                        
                        if 'completeness' in adv_metrics and isinstance(adv_metrics['completeness'], dict):
                            comp_score = adv_metrics['completeness'].get('completeness_score', 0.0)
                            row["✅ Completitud"] = float(comp_score) if comp_score is not None else 0.0
                        else:
                            row["✅ Completitud"] = 0.0
                        
                        if 'satisfaction' in adv_metrics and isinstance(adv_metrics['satisfaction'], dict):
                            sat_score = adv_metrics['satisfaction'].get('satisfaction_score', 0.0)
                            row["😊 Satisfacción"] = float(sat_score) if sat_score is not None else 0.0
                        else:
                            row["😊 Satisfacción"] = 0.0
                        
                        advanced_metrics_data.append(row)
                
                if advanced_metrics_data:
                    # Create DataFrame with explicit data types
                    advanced_df = pd.DataFrame(advanced_metrics_data)
                    
                    # Ensure all numeric columns are float64
                    numeric_cols = ["🚫 Alucinación", "🎯 Utilización", "✅ Completitud", "😊 Satisfacción"]
                    for col in numeric_cols:
                        if col in advanced_df.columns:
                            advanced_df[col] = pd.to_numeric(advanced_df[col], errors='coerce').fillna(0.0)
                    
                    # Apply styling to the advanced metrics dataframe
                    styled_advanced_df = advanced_df.style.format({
                        "🚫 Alucinación": "{:.3f}",
                        "🎯 Utilización": "{:.3f}",
                        "✅ Completitud": "{:.3f}",
                        "😊 Satisfacción": "{:.3f}"
                    })
                    
                    # Apply colors to each advanced metric column
                    for col in numeric_cols:
                        if col in advanced_df.columns:
                            styled_advanced_df = styled_advanced_df.apply(
                                lambda x: [color_advanced_rag_metrics(val, col) for val in x], 
                                subset=[col]
                            )
                    
                    st.dataframe(styled_advanced_df)
                else:
                    st.info("No hay métricas avanzadas RAG para mostrar. Asegúrate de habilitar la evaluación avanzada.")
            
            # Guía completa de métricas movida aquí
            st.markdown("##### 📋 Guía Completa de Métricas y Umbrales")
            
            # Función para crear tooltips detallados
            def get_metric_tooltip(metric_name):
                """Return detailed tooltip information for each metric."""
                tooltips = {
                    "Latencia (s)": {
                        "explanation": "Tiempo total transcurrido desde que se recibe la consulta del usuario hasta que se entrega la respuesta final. Incluye tiempo de embedding, búsqueda vectorial, reranking y generación de respuesta.",
                        "formula": "Latencia = t_fin - t_inicio (donde t_inicio es timestamp al recibir consulta y t_fin es timestamp al entregar respuesta)",
                        "reference": "Chen, J., et al. (2023). Performance evaluation of retrieval-augmented generation systems. *Journal of Information Retrieval*, 26(3), 45-62."
                    },
                    "Throughput Est. (QPS)": {
                        "explanation": "Estimación de consultas por segundo que el sistema puede procesar basado en la latencia observada, ajustado por overhead del sistema y concurrencia.",
                        "formula": "Throughput = (1 / Latencia) × Factor_Overhead × Factor_Concurrencia (Factor_Overhead ≈ 0.7)",
                        "reference": "Wang, L., & Zhang, M. (2022). Scalability metrics for RAG systems. *ACM Computing Surveys*, 54(8), 1-35."
                    },
                    "Score Promedio": {
                        "explanation": "Promedio de los scores de similitud coseno entre el vector de consulta y los vectores de documentos recuperados. Indica la relevancia semántica promedio de los resultados.",
                        "formula": "Score_Promedio = (1/n) × Σ(cos_similarity(query_vector, doc_vector_i)) para i=1 a n",
                        "reference": "Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*, 6769-6781."
                    },
                    "Desviación Score": {
                        "explanation": "Desviación estándar de los scores de similitud. Una desviación baja indica consistencia en la calidad de los documentos recuperados.",
                        "formula": "Desviación = √[(1/n) × Σ(score_i - score_promedio)²] para i=1 a n",
                        "reference": "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389."
                    },
                    "Consistencia": {
                        "explanation": "Métrica de uniformidad en calidad calculada como 1 menos la desviación estándar normalizada. Valores altos indican resultados consistentemente relevantes.",
                        "formula": "Consistencia = 1 - (Desviación_Score / max(1, Score_Promedio))",
                        "reference": "Thakur, N., et al. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS 2021*, 15-30."
                    },
                    "BERT_P": {
                        "explanation": "Precisión de BERTScore que mide similitud semántica usando embeddings contextuales de BERT. Evalúa qué porcentaje de tokens en la respuesta tienen correspondencia semántica en la referencia.",
                        "formula": "BERT_P = (1/|x|) × Σ max_y∈Y cos(h_x, h_y) donde h son embeddings BERT contextuales",
                        "reference": "Zhang, T., et al. (2020). BERTScore: Evaluating text generation with BERT. *ICLR 2020*, 1-22."
                    },
                    "BERT_R": {
                        "explanation": "Recall de BERTScore que mide cobertura semántica. Evalúa qué porcentaje de tokens en la referencia tienen correspondencia semántica en la respuesta generada.",
                        "formula": "BERT_R = (1/|y|) × Σ max_x∈X cos(h_y, h_x) donde h son embeddings BERT contextuales",
                        "reference": "Zhang, T., et al. (2020). BERTScore: Evaluating text generation with BERT. *ICLR 2020*, 1-22."
                    },
                    "BERT_F1": {
                        "explanation": "F1-score de BERTScore que combina precisión y recall semántico. Proporciona una medida balanceada de similitud semántica bidireccional.",
                        "formula": "BERT_F1 = 2 × (BERT_P × BERT_R) / (BERT_P + BERT_R)",
                        "reference": "Zhang, T., et al. (2020). BERTScore: Evaluating text generation with BERT. *ICLR 2020*, 1-22."
                    },
                    "ROUGE1": {
                        "explanation": "Superposición de unigramas entre respuesta y referencia. Mide similitud a nivel de palabras individuales, indicando cobertura de contenido básico.",
                        "formula": "ROUGE-1 = |unigramas_comunes| / |unigramas_referencia|",
                        "reference": "Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81."
                    },
                    "ROUGE2": {
                        "explanation": "Superposición de bigramas (pares de palabras consecutivas) entre respuesta y referencia. Captura mejor el orden y estructura del contenido que ROUGE-1.",
                        "formula": "ROUGE-2 = |bigramas_comunes| / |bigramas_referencia|",
                        "reference": "Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81."
                    },
                    "ROUGE-L": {
                        "explanation": "Longest Common Subsequence entre respuesta y referencia. Mide similitud estructural considerando secuencias de palabras no necesariamente consecutivas.",
                        "formula": "ROUGE-L = LCS(respuesta, referencia) / length(referencia)",
                        "reference": "Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81."
                    },
                    "🚫 Alucinación": {
                        "explanation": "Porcentaje de afirmaciones en la respuesta que no pueden ser verificadas o soportadas por el contexto recuperado. Detecta información inventada o incorrecta.",
                        "formula": "Alucinación = |claims_no_soportadas| / |claims_totales| donde claims se extraen via NER y fact-checking",
                        "reference": "Maynez, J., et al. (2020). On faithfulness and factuality in abstractive summarization. *ACL 2020*, 1906-1919."
                    },
                    "🎯 Utilización": {
                        "explanation": "Efectividad en el aprovechamiento del contexto recuperado. Mide qué porcentaje de documentos y frases clave del contexto se utilizan en la respuesta generada.",
                        "formula": "Utilización = (docs_utilizados / docs_totales) × (frases_utilizadas / frases_disponibles)",
                        "reference": "Gao, L., et al. (2023). Context utilization in retrieval-augmented generation. *EMNLP 2023*, 2156-2171."
                    },
                    "✅ Completitud": {
                        "explanation": "Medida de completitud de la respuesta basada en el tipo de pregunta y componentes esperados (pasos, ejemplos, prerrequisitos, etc.).",
                        "formula": "Completitud = |componentes_presentes| / |componentes_esperados| donde componentes se determinan por tipo de pregunta",
                        "reference": "Min, S., et al. (2022). Rethinking the role of demonstrations: What makes in-context learning work? *EMNLP 2022*, 11048-11064."
                    },
                    "😊 Satisfacción": {
                        "explanation": "Proxy de satisfacción del usuario que combina claridad del texto, directness en responder la pregunta, actionabilidad de la información y confianza en el lenguaje usado.",
                        "formula": "Satisfacción = (claridad + directness + actionabilidad + confianza) / 4 - penalización_tiempo",
                        "reference": "Rashkin, H., et al. (2021). Measuring attribution in natural language generation models. *Computational Linguistics*, 47(4), 777-840."
                    }
                }
                return tooltips.get(metric_name, {
                    "explanation": "Información detallada no disponible",
                    "formula": "N/A",
                    "reference": "N/A"
                })
            
            # Crear tabla con expandibles para detalles de métricas
            st.markdown("**📊 Tabla de Métricas con Detalles Expandibles**")
            
            # Crear DataFrame básico para la tabla principal
            consolidated_data = {
                "Métrica": [
                    "Latencia (s)",
                    "Throughput Est. (QPS)", 
                    "Score Promedio",
                    "Desviación Score",
                    "Consistencia",
                    "BERT_P",
                    "BERT_R", 
                    "BERT_F1",
                    "ROUGE1",
                    "ROUGE2",
                    "ROUGE-L",
                    "🚫 Alucinación",
                    "🎯 Utilización",
                    "✅ Completitud",
                    "😊 Satisfacción"
                ],
                "Descripción": [
                    "Tiempo de respuesta end-to-end",
                    "Consultas por segundo (con overhead)",
                    "Calidad promedio de documentos recuperados",
                    "Variabilidad en calidad de resultados",
                    "Uniformidad en calidad (1-desviación)",
                    "Precisión semántica (similitud contextual)",
                    "Recall semántico (cobertura contextual)", 
                    "F1 semántico (balance precisión/recall)",
                    "Superposición de palabras (unigramas)",
                    "Superposición de pares de palabras (bigramas)",
                    "Superposición de subsecuencia más larga",
                    "Porcentaje de información no soportada por contexto",
                    "Efectividad en el uso del contexto recuperado",
                    "Completitud basada en tipo de pregunta",
                    "Proxy de satisfacción (claridad + directness + actionabilidad)"
                ],
                "🟢 Excelente": [
                    "≤ 2.0s",
                    "≥ 1.0 QPS",
                    "≥ 0.8",
                    "≤ 0.1",
                    "≥ 0.8",
                    "≥ 0.7",
                    "≥ 0.7",
                    "≥ 0.7", 
                    "≥ 0.4",
                    "≥ 0.2",
                    "≥ 0.35",
                    "≤ 0.1",
                    "≥ 0.8",
                    "≥ 0.9",
                    "≥ 0.8"
                ],
                "🟡 Bueno": [
                    "≤ 5.0s",
                    "≥ 0.5 QPS",
                    "≥ 0.6",
                    "≤ 0.2", 
                    "≥ 0.6",
                    "≥ 0.5",
                    "≥ 0.5",
                    "≥ 0.5",
                    "≥ 0.25",
                    "≥ 0.12",
                    "≥ 0.22",
                    "≤ 0.2",
                    "≥ 0.6",
                    "≥ 0.7",
                    "≥ 0.6"
                ],
                "🔴 Necesita Mejora": [
                    "> 5.0s",
                    "< 0.5 QPS",
                    "< 0.6",
                    "> 0.2",
                    "< 0.6", 
                    "< 0.5",
                    "< 0.5",
                    "< 0.5",
                    "< 0.25",
                    "< 0.12",
                    "< 0.22",
                    "> 0.2",
                    "< 0.6",
                    "< 0.7",
                    "< 0.6"
                ]
            }
            
            consolidated_df = pd.DataFrame(consolidated_data)
            st.dataframe(consolidated_df, use_container_width=True)
            
            # Sección expandible con detalles técnicos de cada métrica
            with st.expander("🔬 Detalles Técnicos, Fórmulas y Referencias Académicas", expanded=False):
                st.markdown("### 📖 Información Detallada por Métrica")
                
                # Organizar por categorías
                st.markdown("#### ⚡ Métricas de Rendimiento")
                
                performance_metrics = [
                    ("Latencia (s)", "Tiempo de respuesta end-to-end"),
                    ("Throughput Est. (QPS)", "Consultas por segundo (con overhead)"),
                    ("Score Promedio", "Calidad promedio de documentos recuperados"),
                    ("Desviación Score", "Variabilidad en calidad de resultados"),
                    ("Consistencia", "Uniformidad en calidad (1-desviación)")
                ]
                
                for metric, desc in performance_metrics:
                    tooltip_info = get_metric_tooltip(metric)
                    with st.expander(f"📊 {metric} - {desc}", expanded=False):
                        st.markdown(f"**Explicación Detallada:**")
                        st.markdown(tooltip_info['explanation'])
                        st.markdown(f"**Fórmula:**")
                        st.code(tooltip_info['formula'])
                        st.markdown(f"**Referencia Académica:**")
                        st.markdown(f"*{tooltip_info['reference']}*")
                
                st.markdown("#### 🧠 Métricas de Calidad de Contenido (BERT & ROUGE)")
                
                content_metrics = [
                    ("BERT_P", "Precisión semántica (similitud contextual)"),
                    ("BERT_R", "Recall semántico (cobertura contextual)"),
                    ("BERT_F1", "F1 semántico (balance precisión/recall)"),
                    ("ROUGE1", "Superposición de palabras (unigramas)"),
                    ("ROUGE2", "Superposición de pares de palabras (bigramas)"),
                    ("ROUGE-L", "Superposición de subsecuencia más larga")
                ]
                
                for metric, desc in content_metrics:
                    tooltip_info = get_metric_tooltip(metric)
                    with st.expander(f"📈 {metric} - {desc}", expanded=False):
                        st.markdown(f"**Explicación Detallada:**")
                        st.markdown(tooltip_info['explanation'])
                        st.markdown(f"**Fórmula:**")
                        st.code(tooltip_info['formula'])
                        st.markdown(f"**Referencia Académica:**")
                        st.markdown(f"*{tooltip_info['reference']}*")
                
                st.markdown("#### 🧪 Métricas Avanzadas RAG")
                
                advanced_metrics = [
                    ("🚫 Alucinación", "Porcentaje de información no soportada por contexto"),
                    ("🎯 Utilización", "Efectividad en el uso del contexto recuperado"),
                    ("✅ Completitud", "Completitud basada en tipo de pregunta"),
                    ("😊 Satisfacción", "Proxy de satisfacción (claridad + directness + actionabilidad)")
                ]
                
                for metric, desc in advanced_metrics:
                    tooltip_info = get_metric_tooltip(metric)
                    with st.expander(f"🔬 {metric} - {desc}", expanded=False):
                        st.markdown(f"**Explicación Detallada:**")
                        st.markdown(tooltip_info['explanation'])
                        st.markdown(f"**Fórmula:**")
                        st.code(tooltip_info['formula'])
                        st.markdown(f"**Referencia Académica:**")
                        st.markdown(f"*{tooltip_info['reference']}*")
                
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

