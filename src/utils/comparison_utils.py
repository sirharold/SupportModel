"""
Utilidades para el comparador de recuperación pregunta vs respuesta
"""

import numpy as np
from typing import Set, List, Dict, Tuple


def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calcula la similitud de Jaccard entre dos conjuntos
    
    Args:
        set1: Primer conjunto de IDs de documentos
        set2: Segundo conjunto de IDs de documentos
        
    Returns:
        float: Similitud de Jaccard (0-1)
    """
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_ndcg_at_k(retrieved_docs: List[Dict], ground_truth_docs: List[Dict], k: int = 10) -> float:
    """
    Calcula nDCG@k usando los documentos de respuesta como ground truth
    
    Args:
        retrieved_docs: Documentos recuperados con la pregunta
        ground_truth_docs: Documentos recuperados con la respuesta (ground truth)
        k: Número de documentos a considerar
        
    Returns:
        float: nDCG@k score (0-1)
    """
    # Create relevance scores based on ground truth
    gt_ids = {f"{doc['title']}_{doc['chunk_index']}": 1.0 / (i + 1) 
              for i, doc in enumerate(ground_truth_docs[:k])}
    
    # Calculate DCG for retrieved documents
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        doc_id = f"{doc['title']}_{doc['chunk_index']}"
        relevance = gt_ids.get(doc_id, 0.0)
        dcg += relevance / np.log2(i + 2)  # i+2 because positions start at 1
    
    # Calculate ideal DCG (sorted by relevance)
    ideal_relevances = sorted(gt_ids.values(), reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances[:k]))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_precision_at_k(retrieved_docs: List[Dict], ground_truth_docs: List[Dict], k: int = 5) -> float:
    """
    Calcula Precision@k respecto a los documentos de ground truth
    
    Args:
        retrieved_docs: Documentos recuperados con la pregunta
        ground_truth_docs: Documentos recuperados con la respuesta (ground truth)
        k: Número de documentos a considerar
        
    Returns:
        float: Precision@k (0-1)
    """
    # Get ground truth IDs
    gt_ids = {f"{doc['title']}_{doc['chunk_index']}" for doc in ground_truth_docs}
    
    # Count relevant documents in top-k retrieved
    relevant_count = 0
    for doc in retrieved_docs[:k]:
        doc_id = f"{doc['title']}_{doc['chunk_index']}"
        if doc_id in gt_ids:
            relevant_count += 1
    
    return relevant_count / k if k > 0 else 0.0


def calculate_composite_score(jaccard: float, ndcg: float, precision: float) -> float:
    """
    Calcula el score compuesto combinando las métricas
    
    Args:
        jaccard: Similitud de Jaccard
        ndcg: nDCG@10
        precision: Precision@5
        
    Returns:
        float: Score compuesto (0-1)
    """
    return 0.5 * jaccard + 0.3 * ndcg + 0.2 * precision


def calculate_recall_at_k(retrieved_docs: List[Dict], ground_truth_docs: List[Dict], k: int = 10) -> float:
    """
    Calcula Recall@k respecto a los documentos de ground truth
    
    Args:
        retrieved_docs: Documentos recuperados con la pregunta
        ground_truth_docs: Documentos recuperados con la respuesta (ground truth)
        k: Número de documentos a considerar
        
    Returns:
        float: Recall@k (0-1)
    """
    # Get ground truth IDs
    gt_ids = {f"{doc['title']}_{doc['chunk_index']}" for doc in ground_truth_docs}
    
    if not gt_ids:
        return 0.0
    
    # Count relevant documents in top-k retrieved
    relevant_count = 0
    for doc in retrieved_docs[:k]:
        doc_id = f"{doc['title']}_{doc['chunk_index']}"
        if doc_id in gt_ids:
            relevant_count += 1
    
    return relevant_count / len(gt_ids)


def calculate_f1_at_k(retrieved_docs: List[Dict], ground_truth_docs: List[Dict], k: int = 10) -> float:
    """
    Calcula F1@k respecto a los documentos de ground truth
    
    Args:
        retrieved_docs: Documentos recuperados con la pregunta
        ground_truth_docs: Documentos recuperados con la respuesta (ground truth)
        k: Número de documentos a considerar
        
    Returns:
        float: F1@k (0-1)
    """
    precision = calculate_precision_at_k(retrieved_docs, ground_truth_docs, k)
    recall = calculate_recall_at_k(retrieved_docs, ground_truth_docs, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def rank_models_by_composite_score(comparison_results: Dict[str, Dict]) -> List[Tuple[str, float]]:
    """
    Rankea los modelos por su score compuesto promedio
    
    Args:
        comparison_results: Resultados de comparación para múltiples preguntas
        
    Returns:
        Lista de tuplas (modelo, score_promedio) ordenada descendentemente
    """
    model_scores = {}
    model_counts = {}
    
    for question_results in comparison_results.values():
        for model_name, model_data in question_results.items():
            if model_data.get('metrics') and 'composite_score' in model_data['metrics']:
                score = model_data['metrics']['composite_score']
                
                if model_name not in model_scores:
                    model_scores[model_name] = 0
                    model_counts[model_name] = 0
                
                model_scores[model_name] += score
                model_counts[model_name] += 1
    
    # Calculate averages
    avg_scores = []
    for model_name, total_score in model_scores.items():
        count = model_counts[model_name]
        if count > 0:
            avg_scores.append((model_name, total_score / count))
    
    # Sort by average score descending
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    
    return avg_scores