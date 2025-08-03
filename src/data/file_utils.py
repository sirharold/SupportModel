"""
Utilidades para manejo de archivos.
"""

import json
import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any
from src.ui.pdf_generator import generate_multi_model_pdf_report, generate_cumulative_pdf_report


def load_questions_from_json(file_path: str) -> List[Dict]:
    """
    Carga preguntas desde archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Lista de preguntas y respuestas
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo '{file_path}'")
        return []
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al leer el archivo JSON: {e}")
        return []


def display_download_section(cached_results, llm_conclusions: str = '', llm_improvements: str = ''):
    """Display download section for cached results without causing interface resets."""
    results = cached_results['results']
    evaluation_time = cached_results['evaluation_time']
    evaluate_all_models = cached_results['evaluate_all_models']
    params = cached_results['params']
    
    st.subheader("📥 Descargar Resultados Cached")
    
    if evaluate_all_models:
        st.markdown("**Comparación Multi-Modelo**")
        col1, = st.columns(1)
        
        with col1:
            try:
                # Multi-model PDF Report
                pdf_data = generate_multi_model_pdf_report(
                    results, 
                    params['use_llm_reranker'],
                    params['generative_model_name'], 
                    params['top_k'],
                    evaluation_time,
                    llm_conclusions=llm_conclusions,
                    llm_improvements=llm_improvements
                )
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"multi_model_report_{timestamp}.pdf"
                
                st.download_button(
                    label="📄 Descargar PDF Reporte",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    help="Descarga el reporte completo de comparación entre modelos en PDF"
                )
            except Exception as e:
                st.error(f"Error generando PDF multi-modelo: {e}")
    
    else:
        # Single model results
        model_name = params['embedding_model_name']
        single_results = results[model_name]
        
        st.markdown(f"**Resultados para {model_name}**")
        col1, = st.columns(1)
        
        with col1:
            try:
                # Single model PDF
                pdf_data = generate_cumulative_pdf_report(
                    single_results, 
                    model_name, 
                    params['use_llm_reranker'], 
                    params['generative_model_name'], 
                    params['top_k'],
                    evaluation_time,
                    llm_conclusions=llm_conclusions,
                    llm_improvements=llm_improvements
                )
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"{model_name}_report_{timestamp}.pdf"
                
                st.download_button(
                    label="📄 Descargar PDF",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    help=f"Descarga el reporte completo de {model_name} en PDF"
                )
            except Exception as e:
                st.error(f"Error generando PDF: {e}")