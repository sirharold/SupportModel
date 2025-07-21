from typing import List, Dict, Any
from bert_score import score as bert_scorer
from rouge_score import rouge_scorer
import numpy as np
from collections import defaultdict
from src.services.auth.auth import ensure_huggingface_login

def compute_ndcg(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@k)."""
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    # Get the top k documents
    top_k_docs = retrieved_docs[:k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc in enumerate(top_k_docs):
        doc_link = doc.get('link', '')
        if doc_link in relevant_docs:
            # Use log2(i+2) for discounting (i+1 for position, +1 to avoid log(1)=0)
            dcg += 1.0 / np.log2(i + 2)
    
    # Calculate IDCG (Ideal DCG) - assuming all relevant docs at the top
    idcg = 0.0
    for i in range(min(k, len(relevant_docs))):
        idcg += 1.0 / np.log2(i + 2)
    
    # nDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def compute_mrr(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
    """Computes Mean Reciprocal Rank (MRR@k)."""
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    # Get the top k documents
    top_k_docs = retrieved_docs[:k]
    
    # Find the rank of the first relevant document
    for i, doc in enumerate(top_k_docs):
        doc_link = doc.get('link', '')
        if doc_link in relevant_docs:
            # Return reciprocal rank (1-based indexing)
            return 1.0 / (i + 1)
    
    # No relevant document found in top k
    return 0.0

def compute_precision_recall_f1(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> tuple[float, float, float]:
    """Computes Precision, Recall, and F1-score @k."""
    if not retrieved_docs or not relevant_docs:
        return 0.0, 0.0, 0.0
    
    # Get the top k documents
    top_k_docs = retrieved_docs[:k]
    
    # Count relevant documents in top k
    relevant_retrieved = 0
    for doc in top_k_docs:
        doc_link = doc.get('link', '')
        if doc_link in relevant_docs:
            relevant_retrieved += 1
    
    # Calculate precision: relevant_retrieved / k
    precision = relevant_retrieved / k if k > 0 else 0.0
    
    # Calculate recall: relevant_retrieved / total_relevant
    recall = relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
    
    # Calculate F1-score: 2 * (precision * recall) / (precision + recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1

from src.services.storage.weaviate_utils import WeaviateConfig

def calculate_content_metrics(retrieved_docs: List[Dict], ground_truth_answer: str, top_n: int = 3) -> Dict:
    """
    Calculates BERTScore and ROUGE scores based on retrieved content.
    """
    if not retrieved_docs or not ground_truth_answer:
        return {}

    # Concatenate the content of the top N documents
    candidate_text = " ".join([doc.get('content', '') for doc in retrieved_docs[:top_n]])
    
    if not candidate_text.strip():
        return {"error": "No content found in retrieved documents."}

    # BERTScore
    bert_scores = {}
    try:
        # Load config to get the key
        config = WeaviateConfig.from_env()
        hf_api_key = config.huggingface_api_key
        
        print("[DEBUG METRICS] Ensuring Hugging Face login...")
        ensure_huggingface_login(token=hf_api_key)
        
        print(f"[DEBUG METRICS] Calculating BERTScore...")
        P, R, F1 = bert_scorer([candidate_text], [ground_truth_answer], lang="en", verbose=False, model_type='roberta-large-mnli')
        bert_scores = {
            "BERT_P": P.mean().item(),
            "BERT_R": R.mean().item(),
            "BERT_F1": F1.mean().item(),
        }
        print(f"[DEBUG METRICS] BERTScore calculated successfully: {bert_scores}")
    except Exception as e:
        error_message = f"Error calculating BERTScore: {e}"
        print(f"[DEBUG METRICS] {error_message}")
        bert_scores = {"BERT_P": "Error", "BERT_R": "Error", "BERT_F1": error_message} # Make error visible in UI

    # ROUGE Score
    rouge_scores = {}
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth_answer, candidate_text)
        rouge_scores = {
            "ROUGE1": scores['rouge1'].fmeasure,
            "ROUGE2": scores['rouge2'].fmeasure,
            "ROUGE-L": scores['rougeL'].fmeasure,
        }
    except Exception as e:
        print(f"Error calculating ROUGE score: {e}")
        rouge_scores = {}

    return {**bert_scores, **rouge_scores}


def calculate_average_metrics(all_metrics: List[Dict]) -> Dict[str, float]:
    """
    Calcula métricas promedio de una lista de métricas de forma eficiente.
    
    Args:
        all_metrics: Lista de diccionarios con métricas
        
    Returns:
        Diccionario con métricas promedio
    """
    if not all_metrics:
        return {}
    
    # Usar defaultdict para evitar verificaciones repetidas
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    
    # Sumar todas las métricas de forma eficiente
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_sums[key] += value
                metric_counts[key] += 1
    
    # Calcular promedios usando comprehension
    average_metrics = {
        key: metric_sums[key] / metric_counts[key] 
        for key in metric_sums 
        if metric_counts[key] > 0
    }
    
    return average_metrics


def create_question_data_record(question_num: int, question: str, ms_links: List[str], 
                               docs_count: int, before_metrics: Dict, after_metrics: Dict) -> Dict:
    """
    Crea un registro optimizado de datos de pregunta para reducir uso de memoria.
    
    Args:
        question_num: Número de pregunta
        question: Texto de la pregunta
        ms_links: Links de Microsoft Learn
        docs_count: Número de documentos recuperados
        before_metrics: Métricas antes del reranking
        after_metrics: Métricas después del reranking
        
    Returns:
        Diccionario con datos esenciales de la pregunta
    """
    return {
        'question_num': question_num,
        'question': question[:100] + '...' if len(question) > 100 else question,
        'ground_truth_links': len(ms_links),
        'docs_retrieved': docs_count,
        'before_precision_5': before_metrics.get('Precision@5', 0),
        'after_precision_5': after_metrics.get('Precision@5', 0),
        'before_recall_5': before_metrics.get('Recall@5', 0),
        'after_recall_5': after_metrics.get('Recall@5', 0),
        'before_f1_5': before_metrics.get('F1@5', 0),
        'after_f1_5': after_metrics.get('F1@5', 0)
    }


def validate_data_integrity(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida integridad de datos de resultados y corrige inconsistencias.
    
    Args:
        results: Diccionario de resultados a validar
        
    Returns:
        Diccionario de resultados validado y corregido
    """
    validation_report = {
        'status': 'valid',
        'issues': [],
        'corrections': []
    }
    
    # Validar estructura básica
    required_keys = ['num_questions_evaluated', 'avg_before_metrics', 'avg_after_metrics']
    for key in required_keys:
        if key not in results:
            validation_report['issues'].append(f"Missing required key: {key}")
            validation_report['status'] = 'invalid'
    
    # Validar métricas promedio
    if 'avg_before_metrics' in results:
        before_metrics = results['avg_before_metrics']
        if not isinstance(before_metrics, dict):
            validation_report['issues'].append("avg_before_metrics is not a dictionary")
            validation_report['status'] = 'invalid'
        else:
            # Verificar valores NaN o infinitos
            for metric, value in before_metrics.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        validation_report['issues'].append(f"Invalid value for {metric}: {value}")
                        before_metrics[metric] = 0.0
                        validation_report['corrections'].append(f"Set {metric} to 0.0")
    
    if 'avg_after_metrics' in results:
        after_metrics = results['avg_after_metrics']
        if not isinstance(after_metrics, dict):
            validation_report['issues'].append("avg_after_metrics is not a dictionary")
            validation_report['status'] = 'invalid'
        else:
            # Verificar valores NaN o infinitos
            for metric, value in after_metrics.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        validation_report['issues'].append(f"Invalid value for {metric}: {value}")
                        after_metrics[metric] = 0.0
                        validation_report['corrections'].append(f"Set {metric} to 0.0")
    
    # Validar número de preguntas
    if 'num_questions_evaluated' in results:
        num_questions = results['num_questions_evaluated']
        if not isinstance(num_questions, int) or num_questions <= 0:
            validation_report['issues'].append(f"Invalid num_questions_evaluated: {num_questions}")
            validation_report['status'] = 'invalid'
    
    # Añadir reporte de validación a resultados
    results['validation_report'] = validation_report
    
    return results