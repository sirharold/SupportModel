"""
Utilidades para manejo de archivos.
"""

import json
import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any
from utils.pdf_generator import generate_multi_model_pdf_report, generate_cumulative_pdf_report


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


def display_download_section(cached_results):
    """Display download section for cached results without causing interface resets."""
    results = cached_results['results']
    evaluation_time = cached_results['evaluation_time']
    evaluate_all_models = cached_results['evaluate_all_models']
    params = cached_results['params']
    
    st.subheader("📥 Descargar Resultados Cached")
    
    if evaluate_all_models:
        st.markdown("**Comparación Multi-Modelo**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                # Multi-model CSV comparison
                comparison_data = []
                for model_name_iter, res in results.items():
                    metrics_before = res['avg_before_metrics']
                    metrics_after = res['avg_after_metrics']
                    
                    all_metrics = set(metrics_before.keys()) | set(metrics_after.keys())
                    
                    for metric in all_metrics:
                        before_val = metrics_before.get(metric, 0)
                        after_val = metrics_after.get(metric, 0)
                        delta = after_val - before_val
                        improvement = (delta / before_val * 100) if before_val > 0 else 0
                        
                        comparison_data.append({
                            'Model': model_name_iter,
                            'Metric': metric,
                            'Before_Reranking': before_val,
                            'After_Reranking': after_val,
                            'Delta': delta,
                            'Improvement_Percent': improvement
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                csv_filename = f"multi_model_comparison_{timestamp}.csv"
                
                st.download_button(
                    label="📊 Descargar CSV Comparación",
                    data=comparison_df.to_csv(index=False).encode('utf-8'),
                    file_name=csv_filename,
                    mime="text/csv",
                    help="Descarga los resultados de comparación entre modelos en formato CSV"
                )
            except Exception as e:
                st.error(f"Error generando CSV: {e}")
        
        with col2:
            try:
                # Multi-model PDF Report
                pdf_data = generate_multi_model_pdf_report(
                    results, 
                    params['use_llm_reranker'],
                    params['generative_model_name'], 
                    params['top_k'],
                    evaluation_time
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
        
        with col3:
            try:
                # Multi-model JSON export
                export_data = {
                    'evaluation_date': evaluation_time,
                    'parameters': params,
                    'results': results
                }
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                json_filename = f"multi_model_data_{timestamp}.json"
                
                st.download_button(
                    label="📋 Descargar JSON Datos",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8'),
                    file_name=json_filename,
                    mime="application/json",
                    help="Descarga todos los datos de evaluación en formato JSON"
                )
            except Exception as e:
                st.error(f"Error generando JSON: {e}")
    
    else:
        # Single model results
        model_name = params['embedding_model_name']
        single_results = results[model_name]
        
        st.markdown(f"**Resultados para {model_name}**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                # Single model CSV
                single_df = pd.DataFrame(single_results['all_questions_data'])
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                csv_filename = f"{model_name}_results_{timestamp}.csv"
                
                st.download_button(
                    label="📊 Descargar CSV",
                    data=single_df.to_csv(index=False).encode('utf-8'),
                    file_name=csv_filename,
                    mime="text/csv",
                    help=f"Descarga los resultados detallados de {model_name} en CSV"
                )
            except Exception as e:
                st.error(f"Error generando CSV: {e}")
        
        with col2:
            try:
                # Single model PDF
                pdf_data = generate_cumulative_pdf_report(
                    single_results, 
                    model_name, 
                    params['use_llm_reranker'], 
                    params['generative_model_name'], 
                    params['top_k'],
                    evaluation_time
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
        
        with col3:
            try:
                # Single model JSON
                export_data = {
                    'model_name': model_name,
                    'evaluation_date': evaluation_time,
                    'parameters': params,
                    'results': single_results
                }
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                json_filename = f"{model_name}_data_{timestamp}.json"
                
                st.download_button(
                    label="📋 Descargar JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8'),
                    file_name=json_filename,
                    mime="application/json",
                    help=f"Descarga todos los datos de {model_name} en JSON"
                )
            except Exception as e:
                st.error(f"Error generando JSON: {e}")