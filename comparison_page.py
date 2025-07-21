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
from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from utils.retrieval_metrics import format_metrics_for_display
from config import EMBEDDING_MODELS, MODEL_DESCRIPTIONS
from utils.chromadb_utils import ChromaDBConfig
from utils.extract_links import extract_urls_from_answer
from utils.metrics import calculate_content_metrics
from utils.pdf_generator import generate_pdf_report
from utils.enhanced_evaluation import evaluate_rag_with_advanced_metrics
from utils.gdrive_integration import (
    show_colab_integration_ui, 
    export_comparison_batch_to_drive,
    check_colab_results,
    create_colab_instructions
)

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
def get_and_filter_questions(_chromadb_wrapper, num_questions: int = 100):
    """Gets random questions and filters them for valid links."""
    random_questions = _chromadb_wrapper.get_sample_questions(limit=num_questions, random_sample=True)
    
    filtered_questions = []
    for q in random_questions:
        answer = q.get("accepted_answer", "")
        links = extract_urls_from_answer(answer)
        ms_links = [link for link in links if "learn.microsoft.com" in link]
        
        if ms_links:
            q["ms_links"] = ms_links
            filtered_questions.append(q)
    
    return filtered_questions


def generate_retrieval_metrics_analysis(retrieval_comparison_data):
    """
    Genera un análisis automático de los resultados de las métricas de recuperación.
    
    Args:
        retrieval_comparison_data: Lista de diccionarios con métricas por modelo
        
    Returns:
        String con el análisis en formato markdown
    """
    if not retrieval_comparison_data:
        return ""
    
    try:
        analysis_parts = []
        
        # 1. Resumen general
        num_models = len(retrieval_comparison_data)
        avg_ground_truth = sum(row.get('Ground Truth', 0) for row in retrieval_comparison_data) / num_models
        
        analysis_parts.append(f"**📊 Resumen General:**")
        analysis_parts.append(f"- **{num_models} modelos** comparados con **{avg_ground_truth:.1f} enlaces de referencia** promedio")
        
        # 2. Análisis de MRR
        mrr_improvements = []
        mrr_after_values = []
        for row in retrieval_comparison_data:
            mrr_improvement = row.get('MRR_Δ', 0)
            mrr_after = row.get('MRR_After', 0)
            mrr_improvements.append(mrr_improvement)
            mrr_after_values.append(mrr_after)
        
        avg_mrr_improvement = sum(mrr_improvements) / len(mrr_improvements)
        avg_mrr_after = sum(mrr_after_values) / len(mrr_after_values)
        best_mrr_model = max(retrieval_comparison_data, key=lambda x: x.get('MRR_After', 0))
        
        analysis_parts.append(f"")
        analysis_parts.append(f"**🎯 Análisis de MRR (Mean Reciprocal Rank):**")
        analysis_parts.append(f"- **Mejora promedio:** {avg_mrr_improvement:+.3f} ({avg_mrr_improvement*100:+.1f}%)")
        analysis_parts.append(f"- **MRR promedio post-reranking:** {avg_mrr_after:.3f}")
        analysis_parts.append(f"- **Mejor modelo:** {best_mrr_model.get('Modelo', 'N/A')} (MRR: {best_mrr_model.get('MRR_After', 0):.3f})")
        
        # 3. Análisis de Recall por k
        analysis_parts.append(f"")
        analysis_parts.append(f"**🔍 Análisis de Recall (Cobertura):**")
        
        for k in [1, 5, 10]:
            recall_col = f'Recall@{k}_After'
            recall_delta_col = f'Recall@{k}_Δ'
            
            if any(recall_col in row for row in retrieval_comparison_data):
                recall_values = [row.get(recall_col, 0) for row in retrieval_comparison_data if recall_col in row]
                recall_deltas = [row.get(recall_delta_col, 0) for row in retrieval_comparison_data if recall_delta_col in row]
                
                if recall_values:
                    avg_recall = sum(recall_values) / len(recall_values)
                    avg_recall_improvement = sum(recall_deltas) / len(recall_deltas) if recall_deltas else 0
                    
                    analysis_parts.append(f"- **Recall@{k}:** {avg_recall:.3f} promedio (mejora: {avg_recall_improvement:+.3f})")
        
        # 4. Análisis de Precision por k
        analysis_parts.append(f"")
        analysis_parts.append(f"**🎯 Análisis de Precision (Precisión):**")
        
        for k in [1, 5, 10]:
            precision_col = f'Precision@{k}_After'
            precision_delta_col = f'Precision@{k}_Δ'
            
            if any(precision_col in row for row in retrieval_comparison_data):
                precision_values = [row.get(precision_col, 0) for row in retrieval_comparison_data if precision_col in row]
                precision_deltas = [row.get(precision_delta_col, 0) for row in retrieval_comparison_data if precision_delta_col in row]
                
                if precision_values:
                    avg_precision = sum(precision_values) / len(precision_values)
                    avg_precision_improvement = sum(precision_deltas) / len(precision_deltas) if precision_deltas else 0
                    
                    analysis_parts.append(f"- **Precision@{k}:** {avg_precision:.3f} promedio (mejora: {avg_precision_improvement:+.3f})")
        
        # 5. Análisis de impacto del reranking
        analysis_parts.append(f"")
        analysis_parts.append(f"**⚡ Impacto del Reranking:**")
        
        # Contar cuántos modelos mejoraron
        models_improved = 0
        models_worsened = 0
        models_unchanged = 0
        
        for row in retrieval_comparison_data:
            mrr_delta = row.get('MRR_Δ', 0)
            if mrr_delta > 0.01:  # Mejora significativa
                models_improved += 1
            elif mrr_delta < -0.01:  # Empeoramiento significativo
                models_worsened += 1
            else:
                models_unchanged += 1
        
        analysis_parts.append(f"- **{models_improved} modelos mejoraron** significativamente")
        if models_worsened > 0:
            analysis_parts.append(f"- **{models_worsened} modelos empeoraron**")
        if models_unchanged > 0:
            analysis_parts.append(f"- **{models_unchanged} modelos sin cambios** significativos")
        
        # 6. Recomendaciones
        analysis_parts.append(f"")
        analysis_parts.append(f"**💡 Recomendaciones:**")
        
        if avg_mrr_improvement > 0.1:
            analysis_parts.append(f"- ✅ **El reranking es muy efectivo** para esta consulta (mejora promedio: {avg_mrr_improvement*100:.1f}%)")
        elif avg_mrr_improvement > 0.05:
            analysis_parts.append(f"- ⚡ **El reranking es moderadamente efectivo** (mejora promedio: {avg_mrr_improvement*100:.1f}%)")
        else:
            analysis_parts.append(f"- ⚠️ **El reranking tiene poco impacto** en esta consulta (mejora promedio: {avg_mrr_improvement*100:.1f}%)")
        
        if avg_mrr_after >= 0.8:
            analysis_parts.append(f"- 🎯 **Calidad excelente:** Los documentos relevantes aparecen en las primeras posiciones")
        elif avg_mrr_after >= 0.5:
            analysis_parts.append(f"- 📈 **Calidad buena:** Los documentos relevantes aparecen en posiciones medias")
        else:
            analysis_parts.append(f"- 📉 **Calidad mejorable:** Los documentos relevantes aparecen en posiciones bajas")
        
        # Recomendación de modelo
        if best_mrr_model:
            best_model_name = best_mrr_model.get('Modelo', 'N/A')
            best_mrr_value = best_mrr_model.get('MRR_After', 0)
            analysis_parts.append(f"- 🏆 **Modelo recomendado:** {best_model_name} (MRR: {best_mrr_value:.3f})")
        
        return "\n".join(analysis_parts)
        
    except Exception as e:
        return f"**⚠️ Error generando análisis:** {str(e)}"

def show_comparison_page():
    """Displays the model comparison page."""
    st.subheader("🔬 Comparación de Modelos de Embedding")
    st.markdown("Ejecuta una consulta en los tres modelos de embedding y compara los resultados y métricas lado a lado.")

    # --- 1. User Input ---
    chromadb_wrapper, _, _, _, _, _, _, _ = initialize_clients("multi-qa-mpnet-base-dot-v1")
    
    if 'questions_for_dropdown' not in st.session_state:
        with st.spinner("Cargando preguntas de ejemplo..."):
            st.session_state.questions_for_dropdown = get_and_filter_questions(chromadb_wrapper)

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
    
    # Configuración de métricas de recuperación
    with st.expander("📊 Métricas de Recuperación", expanded=False):
        enable_retrieval_metrics = st.checkbox(
            "Habilitar Métricas de Recuperación",
            value=True,
            help="Calcula Recall@k, Precision@k, F1@k y MRR antes y después del reranking."
        )
        
        if enable_retrieval_metrics:
            st.info("📈 **Métricas de Recuperación Incluidas:**\n"
                   "- 🎯 **MRR (Mean Reciprocal Rank)**: Posición del primer documento relevante\n"
                   "- 🔍 **Recall@k**: Fracción de documentos relevantes recuperados\n"
                   "- 🎯 **Precision@k**: Fracción de documentos recuperados que son relevantes\n"
                   "- ⚖️ **F1@k**: Media armónica de Precision y Recall\n"
                   "- 📊 **Evaluación para k=1,3,5,10**: Before/After reranking")
    
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

    # Google Colab GPU Acceleration
    with st.expander("🚀 Aceleración GPU con Google Colab", expanded=False):
        use_colab_acceleration = st.checkbox(
            "Usar Google Colab para Comparación",
            value=False,
            help="Exporta la comparación a Google Drive y procesa con GPU en Colab (10-50x más rápido)"
        )
        
        if use_colab_acceleration:
            st.info("🚀 **Beneficios del Procesamiento en Colab:**\n"
                   "- ⚡ **10-50x más rápido** con GPU T4/V100\n"
                   "- 💰 **Completamente gratis** (no APIs pagas)\n"
                   "- 🔄 **Automático**: Export → Colab → Import\n"
                   "- 📊 **Métricas completas** en segundos")
            
            colab_integration_available = show_colab_integration_ui()
        else:
            colab_integration_available = False

    # Button logic: Colab export or local processing
    if use_colab_acceleration and colab_integration_available:
        button_text = "🚀 Exportar a Google Colab"
        button_help = "Exporta los datos a Google Drive para procesamiento GPU acelerado"
    else:
        button_text = "🔍 Comparar Modelos (Local)"
        button_help = "Ejecuta la comparación localmente en tu máquina"

    if st.button(button_text, type="primary", use_container_width=True, help=button_help):
        question_to_ask = f"{selected_question.get('title', '')} {selected_question.get('question_content', '')}".strip()

        if not question_to_ask:
            st.warning("La pregunta seleccionada está vacía.")
            return

        # Handle Colab export workflow
        if use_colab_acceleration and colab_integration_available:
            st.markdown("### 🚀 Exportando a Google Colab...")
            
            # Prepare comparison configuration
            comparison_config = {
                "top_k": top_k,
                "use_reranker": use_reranker,
                "enable_retrieval_metrics": enable_retrieval_metrics,
                "enable_advanced_metrics": enable_advanced_metrics,
                "generate_answers": generate_answers if enable_advanced_metrics else False,
                "generative_model_name": st.session_state.get('generative_model_name', 'llama-3.3-70b')
            }
            
            # Prepare questions data (single question for now, can be extended to batch)
            questions_data = [{
                "question_id": "current_comparison",
                "title": selected_question.get('title', ''),
                "question_content": selected_question.get('question_content', ''),
                "question_to_ask": question_to_ask,
                "accepted_answer": selected_question.get('accepted_answer', ''),
                "ms_links": selected_question.get('ms_links', [])
            }]
            
            # Export to Google Drive
            success, file_url = export_comparison_batch_to_drive(
                questions=questions_data,
                comparison_config=comparison_config,
                embedding_models=list(EMBEDDING_MODELS.keys())
            )
            
            if success:
                st.success("✅ Datos exportados exitosamente a Google Drive!")
                
                # Show Colab instructions
                st.markdown(create_colab_instructions(file_url))
                
                # Add button to check for results
                if st.button("📥 Verificar Resultados de Colab", key="check_results"):
                    colab_results = check_colab_results()
                    if colab_results:
                        st.session_state.comparison_results = colab_results.get('results', {})
                        st.session_state.colab_metadata = colab_results.get('metadata', {})
                        st.experimental_rerun()
                    else:
                        st.info("⏳ Resultados aún no disponibles. Asegúrate de completar el procesamiento en Colab.")
                
                return  # Exit early for Colab workflow
            else:
                st.error("❌ Error exportando a Google Drive. Continuando con procesamiento local...")
                # Fall through to local processing

        # Local processing workflow
        st.session_state.comparison_results = {}
        
        # Initialize progress tracking
        total_models = len(EMBEDDING_MODELS.keys())
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_key in enumerate(EMBEDDING_MODELS.keys()):
            model_display_name = f"{model_key}"
            if enable_retrieval_metrics:
                model_display_name += " + Retrieval Metrics"
            if enable_advanced_metrics:
                model_display_name += " + Advanced Metrics"
            
            with st.spinner(f"Consultando con el modelo: {model_display_name}..."):
                try:
                    start_time = time.time()
                    chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(model_key, st.session_state.get('generative_model_name', 'llama-4-scout'))
                    
                    # Use unified pipeline with retrieval metrics
                    if enable_retrieval_metrics:
                        status_text.text(f"Evaluando {model_key} con métricas de recuperación...")
                        
                        # Calculate retrieval metrics using the specialized pipeline
                        result = answer_question_with_retrieval_metrics(
                            question=question_to_ask,
                            chromadb_wrapper=chromadb_wrapper,
                            embedding_client=embedding_client,
                            openai_client=openai_client,
                            gemini_client=gemini_client,
                            local_tinyllama_client=local_tinyllama_client,
                            local_mistral_client=local_mistral_client,
                            openrouter_client=openrouter_client,
                            top_k=top_k,
                            use_llm_reranker=use_reranker,
                            generate_answer=enable_advanced_metrics and generate_answers,
                            calculate_metrics=True,
                            ground_truth_answer=selected_question.get('accepted_answer', ''),
                            ms_links=selected_question.get('ms_links', []),
                            generative_model_name=st.session_state.get('generative_model_name', 'llama-4-scout')
                        )
                        
                        # Extract results based on whether answer generation was enabled
                        if enable_advanced_metrics and generate_answers:
                            results, debug_info, generated_answer, rag_metrics, retrieval_metrics = result
                        else:
                            results, debug_info, retrieval_metrics = result
                            generated_answer = ""
                            rag_metrics = {}
                        
                        # Calculate additional advanced metrics if enabled
                        advanced_metrics = {}
                        if enable_advanced_metrics and generate_answers:
                            try:
                                eval_result = evaluate_rag_with_advanced_metrics(
                                    question=question_to_ask,
                                    chromadb_wrapper=chromadb_wrapper,
                                    embedding_client=embedding_client,
                                    openai_client=openai_client,
                                    gemini_client=gemini_client,
                                    local_tinyllama_client=local_tinyllama_client,
                                    local_mistral_client=local_mistral_client,
                                    generative_model_name=st.session_state.get('generative_model_name', 'llama-4-scout'),
                                    top_k=top_k
                                )
                                advanced_metrics = eval_result.get('advanced_metrics', {})
                            except Exception as e:
                                advanced_metrics = {"error": str(e)}
                        
                    elif enable_advanced_metrics and generate_answers:
                        # Use advanced evaluation pipeline only
                        status_text.text(f"Evaluando {model_key} con métricas avanzadas...")
                        
                        eval_result = evaluate_rag_with_advanced_metrics(
                            question=question_to_ask,
                            chromadb_wrapper=chromadb_wrapper,
                            embedding_client=embedding_client,
                            openai_client=openai_client,
                            gemini_client=gemini_client,
                            local_tinyllama_client=local_tinyllama_client,
                            local_mistral_client=local_mistral_client,
                            generative_model_name=st.session_state.get('generative_model_name', 'llama-4-scout'),
                            top_k=top_k
                        )
                        
                        # Get documents for display
                        results, debug_info = answer_question_documents_only(
                            question_to_ask, chromadb_wrapper, embedding_client, openai_client,
                            top_k=top_k, use_llm_reranker=use_reranker, use_questions_collection=True
                        )
                        
                        generated_answer = eval_result.get('generated_answer', '')
                        advanced_metrics = eval_result.get('advanced_metrics', {})
                        retrieval_metrics = {}
                        
                    else:
                        # Standard evaluation pipeline
                        results, debug_info = answer_question_documents_only(
                            question_to_ask, chromadb_wrapper, embedding_client, openai_client,
                            top_k=top_k, use_llm_reranker=use_reranker, use_questions_collection=True
                        )
                        generated_answer = ""
                        advanced_metrics = {}
                        retrieval_metrics = {}
                    
                    # Common processing for all paths
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # Calculate summary and content metrics
                    summary = generate_summary(question_to_ask, results, local_mistral_client)
                    content_metrics = calculate_content_metrics(results, selected_question.get("accepted_answer", ""))
                    
                    # Store all results
                    st.session_state.comparison_results[model_key] = {
                        "results": results, 
                        "debug_info": debug_info, 
                        "summary": summary, 
                        "time": elapsed_time, 
                        "content_metrics": content_metrics,
                        "generated_answer": generated_answer,
                        "advanced_metrics": advanced_metrics,
                        "retrieval_metrics": retrieval_metrics
                    }
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_models)
                        
                except Exception as e:
                    st.session_state.comparison_results[model_key] = {"results": [], "error": str(e)}
        
        # Clear progress indicators
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

        # --- 4. Retrieval Metrics (Before/After Reranking) ---
        st.markdown("---")
        st.markdown("### 📊 Métricas de Recuperación (Before/After Reranking)")
        
        # Check if retrieval metrics are available
        has_retrieval_metrics = any(
            data.get("retrieval_metrics") and "before_reranking" in data.get("retrieval_metrics", {})
            for data in st.session_state.comparison_results.values()
            if data and not data.get("error")
        )
        
        if has_retrieval_metrics:
            st.info("🔍 **Evaluación del Impacto del Reranking**: Estas métricas muestran cómo el reranking mejora la calidad de recuperación de documentos relevantes para cada modelo de embedding.")
            
            # Create tabs for comprehensive metrics analysis
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🎯 Resumen Ejecutivo", 
                "⏱️ Tiempo y Rendimiento", 
                "🔍 Métricas de Recuperación", 
                "📊 Calidad de Contenido",
                "📈 Visualizaciones",
                "📚 Guía e Interpretación"
            ])
            
            # Prepare data for all tabs
            retrieval_comparison_data = []
            
            for model_key, data in st.session_state.comparison_results.items():
                if data and not data.get("error") and data.get("retrieval_metrics"):
                    metrics = data["retrieval_metrics"]
                    if "before_reranking" in metrics and "after_reranking" in metrics:
                        before = metrics["before_reranking"]
                        after = metrics["after_reranking"]
                        
                        row = {
                            'Modelo': model_key,
                            'Ground Truth': metrics.get('ground_truth_links_count', 0),
                            'Docs Before': metrics.get('docs_before_count', 0),
                            'Docs After': metrics.get('docs_after_count', 0)
                        }
                        
                        # Key metrics (including accuracy metrics for all k values)
                        metrics_list = ['MRR']
                        
                        # Add all metrics for k=1,3,5,10
                        for k in [1, 3, 5, 10]:
                            metrics_list.extend([
                                f'Recall@{k}', f'Precision@{k}', f'F1@{k}',
                                f'Accuracy@{k}', f'BinaryAccuracy@{k}', f'RankingAccuracy@{k}'
                            ])
                        
                        for metric in metrics_list:
                            before_val = before.get(metric, 0)
                            after_val = after.get(metric, 0)
                            improvement = after_val - before_val
                            pct_improvement = (improvement / before_val * 100) if before_val > 0 else 0
                            
                            row[f'{metric}_Before'] = before_val
                            row[f'{metric}_After'] = after_val
                            row[f'{metric}_Δ'] = improvement
                            row[f'{metric}_%'] = pct_improvement
                        
                        retrieval_comparison_data.append(row)
            
            # TAB 1: 🎯 Resumen Ejecutivo
            with tab1:
                if retrieval_comparison_data:
                    retrieval_df = pd.DataFrame(retrieval_comparison_data)
                    
                    st.markdown("### 📊 Resumen Ejecutivo de la Comparación")
                    
                    # Key metrics cards
                    key_metrics = ['MRR', 'Recall@5', 'Precision@5', 'Accuracy@5']
                    summary_cols = st.columns(len(key_metrics))
                    
                    for i, metric in enumerate(key_metrics):
                        with summary_cols[i]:
                            before_col = f'{metric}_Before'
                            after_col = f'{metric}_After'
                            delta_col = f'{metric}_Δ'
                            
                            if before_col in retrieval_df.columns and after_col in retrieval_df.columns:
                                avg_before = retrieval_df[before_col].mean()
                                avg_after = retrieval_df[after_col].mean()
                                avg_improvement = retrieval_df[delta_col].mean()
                                
                                st.metric(
                                    label=f"📊 {metric}",
                                    value=f"{avg_after:.3f}",
                                    delta=f"{avg_improvement:+.3f}"
                                )
                    
                    # Model ranking
                    st.markdown("#### 🏆 Ranking de Modelos")
                    
                    if 'MRR_After' in retrieval_df.columns:
                        ranking_df = retrieval_df[['Modelo', 'MRR_After', 'Recall@5_After', 'Precision@5_After']].sort_values('MRR_After', ascending=False)
                        
                        for idx, (_, row) in enumerate(ranking_df.iterrows(), 1):
                            icon = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}️⃣"
                            st.write(f"{icon} **{row['Modelo']}** - MRR: {row['MRR_After']:.3f}")
                    
                    # Automated analysis
                    st.markdown("#### 🔍 Análisis Automático")
                    analysis_text = generate_retrieval_metrics_analysis(retrieval_comparison_data)
                    if analysis_text:
                        st.markdown(analysis_text)
                else:
                    st.warning("No hay datos de métricas disponibles para el resumen ejecutivo.")
            
            # TAB 2: ⏱️ Tiempo y Rendimiento
            with tab2:
                st.markdown("### ⏱️ Métricas de Tiempo y Rendimiento")
                
                # Performance data from the main comparison results
                perf_data = []
                for model, data in st.session_state.comparison_results.items():
                    if data and not data.get("error"):
                        scores = [doc.get('score', 0) for doc in data.get("results", [])]
                        if scores:
                            perf_data.append({
                                "Modelo": model,
                                "Latencia (s)": data.get('time', 0),
                                "Throughput Est. (QPS)": 1 / data.get('time', 1) if data.get('time', 0) > 0 else 0,
                                "Docs Recuperados": len(data.get("results", [])),
                                "Score Promedio": np.mean(scores),
                                "Score Máximo": np.max(scores),
                                "Score Mínimo": np.min(scores),
                                "Desviación Score": np.std(scores),
                                "Consistencia": 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                            })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    
                    # Performance metrics cards
                    perf_cols = st.columns(4)
                    with perf_cols[0]:
                        st.metric("Latencia Promedio", f"{perf_df['Latencia (s)'].mean():.3f}s")
                    with perf_cols[1]:
                        st.metric("Throughput Promedio", f"{perf_df['Throughput Est. (QPS)'].mean():.2f} QPS")
                    with perf_cols[2]:
                        st.metric("Score Promedio", f"{perf_df['Score Promedio'].mean():.3f}")
                    with perf_cols[3]:
                        st.metric("Consistencia Promedio", f"{perf_df['Consistencia'].mean():.3f}")
                    
                    # Performance comparison table
                    st.markdown("#### 📊 Comparación de Rendimiento")
                    st.dataframe(perf_df, use_container_width=True)
                    
                    # Performance insights
                    fastest_model = perf_df.loc[perf_df['Latencia (s)'].idxmin(), 'Modelo']
                    most_consistent = perf_df.loc[perf_df['Consistencia'].idxmax(), 'Modelo']
                    highest_score = perf_df.loc[perf_df['Score Promedio'].idxmax(), 'Modelo']
                    
                    st.markdown("#### 🏆 Insights de Rendimiento")
                    st.write(f"🚀 **Modelo más rápido:** {fastest_model}")
                    st.write(f"📊 **Modelo más consistente:** {most_consistent}")
                    st.write(f"⭐ **Mejor score promedio:** {highest_score}")
                else:
                    st.warning("No hay datos de rendimiento disponibles.")
            
            # TAB 3: 🔍 Métricas de Recuperación
            with tab3:
                st.markdown("### 🔍 Métricas de Recuperación (Information Retrieval)")
                
                if retrieval_comparison_data:
                    retrieval_df = pd.DataFrame(retrieval_comparison_data)
                    
                    # Create sub-tabs for different metric types
                    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📊 MRR", "🔍 Recall@k", "🎯 Precision@k", "⚖️ F1@k & Accuracy@k"])
                    
                    with sub_tab1:
                        st.markdown("#### 📊 Mean Reciprocal Rank (MRR)")
                        st.info("MRR mide la posición del primer documento relevante. Valores más altos = mejor rendimiento.")
                        
                        mrr_cols = ['Modelo', 'MRR_Before', 'MRR_After', 'MRR_Δ', 'MRR_%']
                        if all(col in retrieval_df.columns for col in mrr_cols):
                            mrr_df = retrieval_df[mrr_cols].copy()
                            for col in ['MRR_Before', 'MRR_After']:
                                mrr_df[col] = mrr_df[col].apply(lambda x: f"{x:.4f}")
                            for col in ['MRR_Δ']:
                                mrr_df[col] = mrr_df[col].apply(lambda x: f"{x:+.4f}")
                            for col in ['MRR_%']:
                                mrr_df[col] = mrr_df[col].apply(lambda x: f"{x:.1f}%")
                            st.dataframe(mrr_df, use_container_width=True)
                    
                    with sub_tab2:
                        st.markdown("#### 🔍 Recall@k - Cobertura de Documentos Relevantes")
                        st.info("Recall@k mide qué proporción de documentos relevantes se recuperaron en los top k.")
                        
                        recall_cols = ['Modelo']
                        for k in [1, 3, 5, 10]:
                            recall_cols.extend([f'Recall@{k}_Before', f'Recall@{k}_After', f'Recall@{k}_Δ', f'Recall@{k}_%'])
                        
                        available_recall_cols = [col for col in recall_cols if col in retrieval_df.columns]
                        if len(available_recall_cols) > 1:
                            recall_df = retrieval_df[available_recall_cols].copy()
                            for col in recall_df.columns:
                                if col != 'Modelo':
                                    if '%' in col:
                                        recall_df[col] = recall_df[col].apply(lambda x: f"{x:.1f}%")
                                    elif 'Δ' in col:
                                        recall_df[col] = recall_df[col].apply(lambda x: f"{x:+.4f}")
                                    else:
                                        recall_df[col] = recall_df[col].apply(lambda x: f"{x:.4f}")
                            st.dataframe(recall_df, use_container_width=True)
                    
                    with sub_tab3:
                        st.markdown("#### 🎯 Precision@k - Precisión de Recuperación")
                        st.info("Precision@k mide qué proporción de los documentos recuperados son relevantes.")
                        
                        precision_cols = ['Modelo']
                        for k in [1, 3, 5, 10]:
                            precision_cols.extend([f'Precision@{k}_Before', f'Precision@{k}_After', f'Precision@{k}_Δ', f'Precision@{k}_%'])
                        
                        available_precision_cols = [col for col in precision_cols if col in retrieval_df.columns]
                        if len(available_precision_cols) > 1:
                            precision_df = retrieval_df[available_precision_cols].copy()
                            for col in precision_df.columns:
                                if col != 'Modelo':
                                    if '%' in col:
                                        precision_df[col] = precision_df[col].apply(lambda x: f"{x:.1f}%")
                                    elif 'Δ' in col:
                                        precision_df[col] = precision_df[col].apply(lambda x: f"{x:+.4f}")
                                    else:
                                        precision_df[col] = precision_df[col].apply(lambda x: f"{x:.4f}")
                            st.dataframe(precision_df, use_container_width=True)
                    
                    with sub_tab4:
                        st.markdown("#### ⚖️ F1@k y Accuracy@k")
                        st.info("F1@k combina Precision y Recall. Accuracy@k mide la proporción de documentos correctamente clasificados.")
                        
                        # F1 metrics section
                        st.markdown("##### 🎯 F1@k - Balance entre Precision y Recall")
                        f1_cols = ['Modelo']
                        for k in [1, 3, 5, 10]:
                            f1_cols.extend([f'F1@{k}_Before', f'F1@{k}_After', f'F1@{k}_Δ', f'F1@{k}_%'])
                        
                        available_f1_cols = [col for col in f1_cols if col in retrieval_df.columns]
                        if len(available_f1_cols) > 1:
                            f1_df = retrieval_df[available_f1_cols].copy()
                            for col in f1_df.columns:
                                if col != 'Modelo':
                                    if '%' in col:
                                        f1_df[col] = f1_df[col].apply(lambda x: f"{x:.1f}%")
                                    elif 'Δ' in col:
                                        f1_df[col] = f1_df[col].apply(lambda x: f"{x:+.4f}")
                                    else:
                                        f1_df[col] = f1_df[col].apply(lambda x: f"{x:.4f}")
                            st.dataframe(f1_df, use_container_width=True)
                        
                        st.markdown("##### ⚖️ Accuracy@k - Exactitud de Clasificación")
                        accuracy_cols = ['Modelo']
                        for k in [1, 3, 5, 10]:
                            accuracy_cols.extend([f'Accuracy@{k}_Before', f'Accuracy@{k}_After', f'Accuracy@{k}_Δ', f'Accuracy@{k}_%'])
                        
                        available_accuracy_cols = [col for col in accuracy_cols if col in retrieval_df.columns]
                        if len(available_accuracy_cols) > 1:
                            accuracy_df = retrieval_df[available_accuracy_cols].copy()
                            for col in accuracy_df.columns:
                                if col != 'Modelo':
                                    if '%' in col:
                                        accuracy_df[col] = accuracy_df[col].apply(lambda x: f"{x:.1f}%")
                                    elif 'Δ' in col:
                                        accuracy_df[col] = accuracy_df[col].apply(lambda x: f"{x:+.4f}")
                                    else:
                                        accuracy_df[col] = accuracy_df[col].apply(lambda x: f"{x:.4f}")
                            st.dataframe(accuracy_df, use_container_width=True)
                else:
                    st.warning("No hay datos de métricas de recuperación disponibles.")
            
            # TAB 4: 📊 Calidad de Contenido  
            with tab4:
                st.markdown("### 📊 Métricas de Calidad de Contenido")
                st.info("Estas métricas comparan el contenido de los documentos recuperados con la **Respuesta Aceptada (Ground Truth)**.")
                
                content_metrics_data = []
                for model, data in st.session_state.comparison_results.items():
                    if data and data.get("content_metrics"):
                        metrics = data["content_metrics"]
                        metrics["Modelo"] = model
                        content_metrics_data.append(metrics)
                
                if content_metrics_data:
                    content_df = pd.DataFrame(content_metrics_data).set_index("Modelo")
                    
                    # Content quality cards
                    if not content_df.empty:
                        content_cols = st.columns(len(content_df.columns))
                        for i, metric in enumerate(content_df.columns):
                            with content_cols[i]:
                                avg_value = content_df[metric].mean()
                                st.metric(f"📊 {metric}", f"{avg_value:.3f}")
                    
                    # Content metrics table
                    st.markdown("#### 📊 Métricas de Calidad por Modelo")
                    st.dataframe(content_df, use_container_width=True)
                    
                    # Content insights
                    if 'BERT_F1' in content_df.columns:
                        best_bert = content_df['BERT_F1'].idxmax()
                        st.write(f"🏆 **Mejor BERTScore F1:** {best_bert} ({content_df.loc[best_bert, 'BERT_F1']:.3f})")
                    
                    if 'ROUGE1' in content_df.columns:
                        best_rouge = content_df['ROUGE1'].idxmax()
                        st.write(f"📝 **Mejor ROUGE-1:** {best_rouge} ({content_df.loc[best_rouge, 'ROUGE1']:.3f})")
                else:
                    st.warning("No hay datos de métricas de calidad de contenido disponibles.")
            
            # TAB 5: 📈 Visualizaciones
            with tab5:
                st.markdown("### 📈 Visualizaciones Interactivas")
                
                if retrieval_comparison_data:
                    retrieval_df = pd.DataFrame(retrieval_comparison_data)
                    
                    # MRR Improvement Chart
                    st.markdown("#### 📊 MRR: Before vs After Reranking")
                    mrr_fig = go.Figure()
                    
                    models = retrieval_df['Modelo'].tolist()
                    mrr_before = retrieval_df['MRR_Before'].tolist()
                    mrr_after = retrieval_df['MRR_After'].tolist()
                    
                    mrr_fig.add_trace(go.Bar(
                        name='Before Reranking',
                        x=models,
                        y=mrr_before,
                        marker_color='lightcoral'
                    ))
                    
                    mrr_fig.add_trace(go.Bar(
                        name='After Reranking',
                        x=models,
                        y=mrr_after,
                        marker_color='lightgreen'
                    ))
                    
                    mrr_fig.update_layout(
                        title='MRR: Before vs After Reranking',
                        barmode='group',
                        xaxis_title='Modelo de Embedding',
                        yaxis_title='MRR Value'
                    )
                    
                    st.plotly_chart(mrr_fig, use_container_width=True)
                    
                    # Improvement heatmap
                    st.markdown("#### 🔥 Heatmap de Mejoras por Métrica")
                    
                    heatmap_data = []
                    for _, row in retrieval_df.iterrows():
                        model = row['Modelo']
                        heatmap_metrics = ['MRR', 'Recall@1', 'Recall@5', 'Precision@1', 'Precision@5', 'Accuracy@1', 'Accuracy@5']
                        for metric in heatmap_metrics:
                            improvement = row.get(f'{metric}_Δ', 0)
                            heatmap_data.append({
                                'Modelo': model,
                                'Métrica': metric,
                                'Mejora': improvement
                            })
                    
                    if heatmap_data:
                        heatmap_df = pd.DataFrame(heatmap_data)
                        heatmap_pivot = heatmap_df.pivot(index='Modelo', columns='Métrica', values='Mejora')
                        
                        heatmap_fig = px.imshow(
                            heatmap_pivot.values,
                            x=heatmap_pivot.columns,
                            y=heatmap_pivot.index,
                            color_continuous_scale='RdYlGn',
                            aspect='auto',
                            title='Mejora por Modelo y Métrica (After - Before)'
                        )
                        
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # Performance scatter plot (if performance data available)
                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        st.markdown("#### ⚡ Rendimiento vs Calidad")
                        
                        # Merge with retrieval data for scatter plot
                        merged_df = pd.merge(perf_df, retrieval_df[['Modelo', 'MRR_After']], on='Modelo', how='inner')
                        
                        scatter_fig = px.scatter(
                            merged_df, 
                            x='Latencia (s)', 
                            y='MRR_After',
                            color='Modelo',
                            size='Score Promedio',
                            hover_data=['Throughput Est. (QPS)', 'Consistencia'],
                            title='Trade-off: Latencia vs Calidad (MRR)',
                            labels={'MRR_After': 'MRR (Post-Reranking)', 'Latencia (s)': 'Latencia (segundos)'}
                        )
                        
                        st.plotly_chart(scatter_fig, use_container_width=True)
                        
                        # Additional performance visualizations
                        st.markdown("#### 📊 Métricas de Rendimiento Detalladas")
                        
                        perf_viz_col1, perf_viz_col2 = st.columns(2)
                        
                        with perf_viz_col1:
                            # Latency chart
                            latency_fig = px.bar(
                                perf_df, x="Modelo", y="Latencia (s)", color="Modelo",
                                title="Latencia de Consulta (End-to-End)",
                                labels={"Latencia (s)": "Tiempo (segundos)"}
                            )
                            st.plotly_chart(latency_fig, use_container_width=True)
                            
                            # Consistency chart
                            consistency_fig = px.bar(
                                perf_df, x="Modelo", y="Consistencia", color="Modelo",
                                title="Consistencia de Scores (1 = muy consistente)",
                                labels={"Consistencia": "Índice de Consistencia"}
                            )
                            st.plotly_chart(consistency_fig, use_container_width=True)
                        
                        with perf_viz_col2:
                            # Throughput chart
                            throughput_fig = px.bar(
                                perf_df, x="Modelo", y="Throughput Est. (QPS)", color="Modelo",
                                title="Throughput Estimado (Consultas por Segundo)",
                                labels={"Throughput Est. (QPS)": "Consultas / Segundo"}
                            )
                            st.plotly_chart(throughput_fig, use_container_width=True)
                            
                            # Quality vs Variability scatter
                            quality_scatter_fig = px.scatter(
                                perf_df, x="Score Promedio", y="Desviación Score", 
                                color="Modelo", size="Docs Recuperados",
                                title="Calidad vs Variabilidad de Resultados",
                                labels={
                                    "Score Promedio": "Calidad Promedio",
                                    "Desviación Score": "Variabilidad"
                                }
                            )
                            st.plotly_chart(quality_scatter_fig, use_container_width=True)
                        
                        # Score distribution
                        st.markdown("#### 📈 Distribución de Scores de Relevancia")
                        st.info("Este gráfico muestra la distribución de los scores de similitud para los documentos recuperados por cada modelo. Un buen modelo debería tener scores altos (cercanos a 1.0) y una distribución compacta en la parte superior.")
                        
                        # Prepare score distribution data
                        score_data = []
                        for model, data in st.session_state.comparison_results.items():
                            if data and not data.get("error"):
                                for doc in data["results"]:
                                    score_data.append({"Modelo": model, "Score": doc.get("score", 0)})
                        
                        if score_data:
                            score_df = pd.DataFrame(score_data)
                            score_box_fig = px.box(score_df, x="Modelo", y="Score", color="Modelo", title="Distribución de Scores por Modelo")
                            st.plotly_chart(score_box_fig, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para visualizaciones.")
            
            # TAB 6: 📚 Guía e Interpretación
            with tab6:
                st.markdown("### 📚 Guía de Interpretación de Métricas")
                
                # Create sub-sections for the guide
                guide_tab1, guide_tab2, guide_tab3, guide_tab4 = st.tabs([
                    "📊 Definiciones", "🎯 Umbrales", "🔄 Proceso RAG", "💡 Tips"
                ])
                
                with guide_tab1:
                    st.markdown("#### 📊 Definiciones de Métricas")
                    
                    st.markdown("""
                    **🔍 Métricas de Recuperación:**
                    - **MRR (Mean Reciprocal Rank)**: Posición promedio del primer documento relevante (1/rank). Valores más altos = mejor.
                    - **Recall@k**: Proporción de documentos relevantes encontrados en top k. Mide "cobertura".
                    - **Precision@k**: Proporción de documentos recuperados que son relevantes. Mide "precisión".
                    - **F1@k**: Media armónica de Precision y Recall. Balance entre ambos.
                    - **Accuracy@k**: Proporción de documentos correctamente clasificados.
                    
                    **⏱️ Métricas de Rendimiento:**
                    - **Latencia**: Tiempo total de procesamiento por consulta.
                    - **Throughput**: Consultas procesadas por segundo (estimado).
                    - **Consistencia**: Uniformidad en los scores (1 - coef. variación).
                    
                    **📊 Métricas de Calidad:**
                    - **BERTScore F1**: Similitud semántica usando embeddings BERT.
                    - **ROUGE-1**: Superposición de unigrams con texto de referencia.
                    """)
                
                with guide_tab2:
                    st.markdown("#### 🎯 Umbrales de Calidad")
                    
                    st.markdown("""
                    **📊 Umbrales Recomendados:**
                    
                    | Métrica | Excelente | Bueno | Aceptable | Mejorable |
                    |---------|-----------|-------|-----------|-----------|
                    | **MRR** | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
                    | **Recall@5** | > 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
                    | **Precision@5** | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
                    | **BERTScore F1** | > 0.85 | 0.75-0.85 | 0.65-0.75 | < 0.65 |
                    | **Latencia** | < 1s | 1-2s | 2-3s | > 3s |
                    
                    **🎯 Objetivos por Caso de Uso:**
                    - **Sistemas de producción**: MRR > 0.7, Latencia < 2s
                    - **Investigación**: MRR > 0.8, Precision@5 > 0.7
                    - **Tiempo real**: Latencia < 500ms, Recall@3 > 0.6
                    """)
                
                with guide_tab3:
                    st.markdown("#### 🔄 Diagrama del Proceso RAG")
                    
                    st.markdown("""
                    ```
                    📝 Pregunta Usuario
                            ↓
                    🔍 Embedding de la Pregunta
                            ↓
                    📊 Búsqueda Vectorial (Weaviate)
                            ↓
                    📋 Documentos Candidatos (Before)
                            ↓
                    🤖 Reranking con LLM (CrossEncoder)
                            ↓
                    📋 Documentos Rerankeados (After)
                            ↓
                    📊 Cálculo de Métricas
                            ↓
                    📈 Comparación Before/After
                    ```
                    
                    **🔄 Proceso de Evaluación:**
                    1. **Extracción Ground Truth**: Enlaces de Microsoft Learn de respuestas aceptadas
                    2. **Normalización URLs**: Eliminación de parámetros y anclajes
                    3. **Cálculo Métricas Before**: Evaluación después de búsqueda vectorial
                    4. **Cálculo Métricas After**: Evaluación después de reranking
                    5. **Análisis de Mejora**: Comparación y recomendaciones
                    """)
                
                with guide_tab4:
                    st.markdown("#### 💡 Tips de Interpretación")
                    
                    st.markdown("""
                    **🎯 Cómo Interpretar Resultados:**
                    
                    **📊 MRR = 1.0**: El primer documento siempre es relevante (ideal)
                    **📊 MRR = 0.5**: El primer documento relevante está en promedio en posición 2
                    **📊 MRR = 0.33**: El primer documento relevante está en promedio en posición 3
                    
                    **🔍 Recall@5 = 0.8**: Se encuentran 80% de documentos relevantes en top 5
                    **🎯 Precision@5 = 0.6**: 60% de los top 5 documentos son relevantes
                    
                    **⚡ Recomendaciones por Escenario:**
                    
                    **🟢 MRR mejora mucho (>50%)**: Reranking muy efectivo, mantener configuración
                    **🟡 MRR mejora poco (<20%)**: Evaluar si vale la pena el costo computacional
                    **🔴 MRR empeora**: Revisar configuración del reranker o modelo base
                    
                    **🏆 Selección de Modelo:**
                    1. **Priorizar calidad**: Elegir mayor MRR y Precision@5
                    2. **Priorizar velocidad**: Elegir menor latencia con calidad aceptable
                    3. **Balanceado**: Usar scatter plot Latencia vs MRR
                    
                    **🔍 Debugging Tips:**
                    - Si Recall bajo: Verificar diversidad en base de documentos
                    - Si Precision bajo: Ajustar threshold de similitud
                    - Si latencia alta: Considerar modelos más ligeros
                    """)
                
                st.markdown("---")
                st.markdown("*💡 Para más detalles técnicos, consulta la documentación completa del sistema.*")
        
        else:
            st.info("💡 **Habilita las Métricas de Recuperación** en la configuración para ver el análisis detallado del impacto del reranking en la calidad de recuperación.")
            
            if not enable_retrieval_metrics:
                st.warning("🔧 Las métricas de recuperación están deshabilitadas. Habilítalas en la sección de configuración y ejecuta la comparación nuevamente.")


        
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


