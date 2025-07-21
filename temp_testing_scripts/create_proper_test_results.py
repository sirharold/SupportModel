#!/usr/bin/env python3
"""
Create properly structured test results file for the cumulative evaluation system
"""

from utils.real_gdrive_integration import authenticate_gdrive, load_gdrive_config, find_file_in_drive, upload_json_to_drive
from datetime import datetime
import json

def create_proper_results():
    """Create results with the structure expected by display functions"""
    
    # Structure that matches what display_cumulative_metrics expects
    results = {
        "config": {
            "num_questions": 50,
            "selected_models": ["multi-qa-mpnet-base-dot-v1", "all-MiniLM-L6-v2"],
            "generative_model_name": "llama-3.3-70b",
            "top_k": 10,
            "use_llm_reranker": True,
            "batch_size": 50,
            "evaluation_type": "cumulative",
            "timestamp": datetime.now().isoformat()
        },
        "results": {
            "multi-qa-mpnet-base-dot-v1": {
                "model_name": "multi-qa-mpnet-base-dot-v1",
                "num_questions_evaluated": 50,
                # Before reranking metrics
                "avg_before_metrics": {
                    "avg_precision": 0.72,
                    "avg_recall": 0.68,
                    "avg_f1": 0.70,
                    "avg_map": 0.65,
                    "avg_mrr": 0.71,
                    "avg_ndcg": 0.69
                },
                # After reranking metrics
                "avg_after_metrics": {
                    "avg_precision": 0.85,
                    "avg_recall": 0.78,
                    "avg_f1": 0.81,
                    "avg_map": 0.73,
                    "avg_mrr": 0.82,
                    "avg_ndcg": 0.79
                },
                # Detailed question data
                "all_questions_data": [
                    {
                        "question_id": f"q_{i}",
                        "question": f"Test question {i}",
                        "precision_before": 0.72 + (i * 0.01),
                        "recall_before": 0.68 + (i * 0.01),
                        "f1_before": 0.70 + (i * 0.01),
                        "map_before": 0.65 + (i * 0.01),
                        "mrr_before": 0.71 + (i * 0.01),
                        "ndcg_before": 0.69 + (i * 0.01),
                        "precision_after": 0.85 + (i * 0.005),
                        "recall_after": 0.78 + (i * 0.005),
                        "f1_after": 0.81 + (i * 0.005),
                        "map_after": 0.73 + (i * 0.005),
                        "mrr_after": 0.82 + (i * 0.005),
                        "ndcg_after": 0.79 + (i * 0.005),
                        "improvement_precision": 0.13,
                        "improvement_recall": 0.10,
                        "improvement_f1": 0.11,
                        "improvement_map": 0.08,
                        "improvement_mrr": 0.11,
                        "improvement_ndcg": 0.10
                    } for i in range(50)
                ],
                "evaluation_time": 65.4,
                "questions_processed": 50
            },
            "all-MiniLM-L6-v2": {
                "model_name": "all-MiniLM-L6-v2",
                "num_questions_evaluated": 50,
                "avg_before_metrics": {
                    "avg_precision": 0.69,
                    "avg_recall": 0.65,
                    "avg_f1": 0.67,
                    "avg_map": 0.62,
                    "avg_mrr": 0.68,
                    "avg_ndcg": 0.66
                },
                "avg_after_metrics": {
                    "avg_precision": 0.82,
                    "avg_recall": 0.75,
                    "avg_f1": 0.78,
                    "avg_map": 0.70,
                    "avg_mrr": 0.79,
                    "avg_ndcg": 0.76
                },
                "all_questions_data": [
                    {
                        "question_id": f"q_{i}",
                        "question": f"Test question {i}",
                        "precision_before": 0.69 + (i * 0.01),
                        "recall_before": 0.65 + (i * 0.01),
                        "f1_before": 0.67 + (i * 0.01),
                        "map_before": 0.62 + (i * 0.01),
                        "mrr_before": 0.68 + (i * 0.01),
                        "ndcg_before": 0.66 + (i * 0.01),
                        "precision_after": 0.82 + (i * 0.005),
                        "recall_after": 0.75 + (i * 0.005),
                        "f1_after": 0.78 + (i * 0.005),
                        "map_after": 0.70 + (i * 0.005),
                        "mrr_after": 0.79 + (i * 0.005),
                        "ndcg_after": 0.76 + (i * 0.005),
                        "improvement_precision": 0.13,
                        "improvement_recall": 0.10,
                        "improvement_f1": 0.11,
                        "improvement_map": 0.08,
                        "improvement_mrr": 0.11,
                        "improvement_ndcg": 0.10
                    } for i in range(50)
                ],
                "evaluation_time": 58.2,
                "questions_processed": 50
            }
        },
        "evaluation_info": {
            "total_time_seconds": 123.6,
            "models_evaluated": 2,
            "questions_processed": 50,
            "gpu_used": True,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "cumulative_multi_model",
            "colab_session": "simulated_session_001"
        }
    }
    
    return results

def main():
    print("🔧 Creando archivo de resultados con estructura correcta...")
    
    # Autenticar
    service = authenticate_gdrive()
    folder_id = load_gdrive_config()
    
    if not service or not folder_id:
        print("❌ Error de autenticación")
        return
    
    # Encontrar carpeta results
    results_folder_result = find_file_in_drive(service, folder_id, 'results')
    if not results_folder_result['success'] or not results_folder_result['found']:
        print("❌ No se encontró carpeta results")
        return
    
    results_folder_id = results_folder_result['file_id']
    
    # Crear datos correctos
    proper_results = create_proper_results()
    
    # Subir archivo corregido
    upload_result = upload_json_to_drive(
        service, 
        results_folder_id, 
        'cumulative_results_test.json', 
        proper_results
    )
    
    if upload_result['success']:
        print("✅ Archivo de resultados actualizado con estructura correcta")
        print("📊 Estructura incluye:")
        print("   - avg_before_metrics y avg_after_metrics")
        print("   - all_questions_data con métricas detalladas")
        print("   - Datos para 2 modelos con 50 preguntas cada uno")
        print("\n🎯 Ahora puedes probar el botón 'Mostrar Resultados' en Streamlit")
    else:
        print(f"❌ Error subiendo archivo: {upload_result['error']}")

if __name__ == "__main__":
    main()