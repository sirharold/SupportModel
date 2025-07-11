"""
Métricas de evaluación para sistemas de recuperación de información.
Calcula Recall@k, Precision@k, F1-score y MRR antes y después del reranking.
"""

from typing import List, Dict, Set, Tuple
import numpy as np
from collections import defaultdict
from utils.extract_links import normalize_url


def extract_ground_truth_links(ground_truth_answer: str, ms_links: List[str] = None) -> Set[str]:
    """
    Extrae los enlaces de referencia (ground truth) de una respuesta.
    
    Args:
        ground_truth_answer: Respuesta aceptada que contiene los enlaces de referencia
        ms_links: Lista de enlaces de Microsoft Learn extraídos previamente
        
    Returns:
        Set de enlaces de referencia únicos (normalizados)
    """
    if ms_links:
        # Normalize the provided ms_links
        normalized_links = [normalize_url(link) for link in ms_links]
        return set(normalized_links)
    
    # Si no hay ms_links, extraer de la respuesta
    from utils.extract_links import extract_urls_from_answer
    extracted_links = extract_urls_from_answer(ground_truth_answer)
    # Filtrar solo enlaces de Microsoft Learn y normalizar
    ms_links = [normalize_url(link) for link in extracted_links if "learn.microsoft.com" in link]
    return set(ms_links)


def calculate_recall_at_k(retrieved_docs: List[Dict], ground_truth_links: Set[str], k: int) -> float:
    """
    Calcula Recall@k: ¿Cuántos documentos relevantes se recuperaron en los top k?
    
    Recall@k = |{documentos relevantes en top k}| / |{todos los documentos relevantes}|
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        k: Número de documentos top a considerar
        
    Returns:
        Recall@k como float entre 0 y 1
    """
    if not ground_truth_links:
        return 0.0
    
    # Obtener los top k documentos
    top_k_docs = retrieved_docs[:k]
    
    # Extraer enlaces de los documentos recuperados (normalizados)
    retrieved_links = set()
    for doc in top_k_docs:
        link = doc.get('link', '').strip()
        if link:
            normalized_link = normalize_url(link)
            if normalized_link:
                retrieved_links.add(normalized_link)
    
    # Calcular intersección con ground truth
    relevant_retrieved = retrieved_links.intersection(ground_truth_links)
    
    # Recall = documentos relevantes recuperados / total documentos relevantes
    recall = len(relevant_retrieved) / len(ground_truth_links)
    return recall


def calculate_precision_at_k(retrieved_docs: List[Dict], ground_truth_links: Set[str], k: int) -> float:
    """
    Calcula Precision@k: ¿Qué proporción de los top k documentos son relevantes?
    
    Precision@k = |{documentos relevantes en top k}| / k
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        k: Número de documentos top a considerar
        
    Returns:
        Precision@k como float entre 0 y 1
    """
    if k == 0:
        return 0.0
    
    # Obtener los top k documentos
    top_k_docs = retrieved_docs[:k]
    
    # Extraer enlaces de los documentos recuperados (normalizados)
    retrieved_links = set()
    for doc in top_k_docs:
        link = doc.get('link', '').strip()
        if link:
            normalized_link = normalize_url(link)
            if normalized_link:
                retrieved_links.add(normalized_link)
    
    # Calcular intersección con ground truth
    relevant_retrieved = retrieved_links.intersection(ground_truth_links)
    
    # Precision = documentos relevantes recuperados / total documentos recuperados
    precision = len(relevant_retrieved) / k
    return precision


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calcula F1-score como media armónica de precisión y recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Valor de precisión
        recall: Valor de recall
        
    Returns:
        F1-score como float entre 0 y 1
    """
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_accuracy_at_k(retrieved_docs: List[Dict], ground_truth_links: Set[str], k: int) -> float:
    """
    Calcula Accuracy@k para sistemas de recuperación.
    
    En recuperación de información, Accuracy@k se define como la proporción de documentos 
    correctamente clasificados (relevantes/no relevantes) en los top k.
    
    Accuracy@k = (TP + TN) / (TP + TN + FP + FN)
    
    Donde para los top k documentos:
    - TP (True Positives): Documentos relevantes recuperados
    - FP (False Positives): Documentos no relevantes recuperados  
    - TN (True Negatives): Documentos no relevantes no recuperados
    - FN (False Negatives): Documentos relevantes no recuperados
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        k: Número de documentos top a considerar
        
    Returns:
        Accuracy@k como float entre 0 y 1
    """
    if k == 0 or not ground_truth_links:
        return 0.0
    
    # Obtener los top k documentos
    top_k_docs = retrieved_docs[:k]
    
    # Extraer enlaces de los documentos recuperados (normalizados)
    retrieved_links = set()
    for doc in top_k_docs:
        link = doc.get('link', '').strip()
        if link:
            normalized_link = normalize_url(link)
            if normalized_link:
                retrieved_links.add(normalized_link)
    
    # Calcular métricas de clasificación
    # TP: Documentos relevantes que fueron recuperados en top k
    true_positives = len(retrieved_links.intersection(ground_truth_links))
    
    # FP: Documentos no relevantes que fueron recuperados en top k
    false_positives = len(retrieved_links - ground_truth_links)
    
    # FN: Documentos relevantes que NO fueron recuperados en top k
    false_negatives = len(ground_truth_links - retrieved_links)
    
    # TN: Para sistemas de recuperación, TN es conceptualmente difícil de definir
    # porque el "universo" de documentos no relevantes es muy grande.
    # Usaremos una aproximación: TN = k - TP - FP (espacios no usados en top k)
    true_negatives = max(0, k - true_positives - false_positives)
    
    # Calcular accuracy
    total = true_positives + false_positives + true_negatives + false_negatives
    
    if total == 0:
        return 0.0
    
    accuracy = (true_positives + true_negatives) / total
    return accuracy


def calculate_binary_accuracy_at_k(retrieved_docs: List[Dict], ground_truth_links: Set[str], k: int) -> float:
    """
    Calcula Binary Accuracy@k - proporción de predicciones correctas en los top k.
    
    Esta es una versión simplificada donde cada posición en top k se clasifica como
    correcta (1) si el documento es relevante, incorrecta (0) si no lo es.
    
    Binary_Accuracy@k = (documentos_relevantes_en_top_k) / k
    
    Esta métrica es equivalente a Precision@k, pero presentada como "accuracy".
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        k: Número de documentos top a considerar
        
    Returns:
        Binary Accuracy@k como float entre 0 y 1
    """
    if k == 0:
        return 0.0
    
    # Obtener los top k documentos
    top_k_docs = retrieved_docs[:k]
    
    # Contar documentos relevantes en top k (usando enlaces normalizados)
    relevant_count = 0
    for doc in top_k_docs:
        link = doc.get('link', '').strip()
        if link:
            normalized_link = normalize_url(link)
            if normalized_link and normalized_link in ground_truth_links:
                relevant_count += 1
    
    # Accuracy = documentos correctos / total de documentos evaluados
    binary_accuracy = relevant_count / k
    return binary_accuracy


def calculate_ranking_accuracy(retrieved_docs: List[Dict], ground_truth_links: Set[str], k: int) -> float:
    """
    Calcula Ranking Accuracy@k - qué tan bien el sistema rankea documentos relevantes.
    
    Mide si los documentos relevantes aparecen en posiciones más altas que los no relevantes
    dentro de los top k documentos.
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        k: Número de documentos top a considerar
        
    Returns:
        Ranking Accuracy como float entre 0 y 1
    """
    if k == 0 or not ground_truth_links:
        return 0.0
    
    # Obtener los top k documentos con sus posiciones
    top_k_docs = retrieved_docs[:k]
    
    # Identificar posiciones de documentos relevantes y no relevantes
    relevant_positions = []
    non_relevant_positions = []
    
    for pos, doc in enumerate(top_k_docs, 1):
        link = doc.get('link', '').strip()
        if link:
            normalized_link = normalize_url(link)
            if normalized_link and normalized_link in ground_truth_links:
                relevant_positions.append(pos)
            else:
                non_relevant_positions.append(pos)
        else:
            non_relevant_positions.append(pos)
    
    if not relevant_positions or not non_relevant_positions:
        # Si todos son relevantes o todos son no relevantes, accuracy = 1.0
        return 1.0
    
    # Contar pares (relevante, no_relevante) donde relevante aparece antes
    correct_pairs = 0
    total_pairs = 0
    
    for rel_pos in relevant_positions:
        for non_rel_pos in non_relevant_positions:
            total_pairs += 1
            if rel_pos < non_rel_pos:  # Relevante aparece antes que no relevante
                correct_pairs += 1
    
    if total_pairs == 0:
        return 0.0
    
    ranking_accuracy = correct_pairs / total_pairs
    return ranking_accuracy


def calculate_mrr(retrieved_docs: List[Dict], ground_truth_links: Set[str]) -> float:
    """
    Calcula Mean Reciprocal Rank (MRR): Posición promedio del primer resultado relevante.
    
    MRR = 1 / rank_of_first_relevant_document
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        
    Returns:
        MRR como float entre 0 y 1
    """
    if not ground_truth_links:
        return 0.0
    
    # Buscar el primer documento relevante (usando enlaces normalizados)
    for rank, doc in enumerate(retrieved_docs, 1):
        link = doc.get('link', '').strip()
        if link:
            normalized_link = normalize_url(link)
            if normalized_link and normalized_link in ground_truth_links:
                return 1.0 / rank
    
    # Si no se encontró ningún documento relevante
    return 0.0


def calculate_retrieval_metrics(
    retrieved_docs: List[Dict], 
    ground_truth_links: Set[str], 
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Calcula todas las métricas de recuperación para diferentes valores de k.
    
    Args:
        retrieved_docs: Lista de documentos recuperados ordenados por relevancia
        ground_truth_links: Set de enlaces de referencia (ground truth)
        k_values: Lista de valores k para calcular métricas
        
    Returns:
        Diccionario con todas las métricas calculadas
    """
    metrics = {}
    
    # Calcular MRR (independiente de k)
    metrics['MRR'] = calculate_mrr(retrieved_docs, ground_truth_links)
    
    # Calcular métricas para cada valor de k
    for k in k_values:
        # Asegurar que k no exceda el número de documentos recuperados
        effective_k = min(k, len(retrieved_docs))
        
        # Calcular Recall@k
        recall_k = calculate_recall_at_k(retrieved_docs, ground_truth_links, effective_k)
        metrics[f'Recall@{k}'] = recall_k
        
        # Calcular Precision@k
        precision_k = calculate_precision_at_k(retrieved_docs, ground_truth_links, effective_k)
        metrics[f'Precision@{k}'] = precision_k
        
        # Calcular F1@k
        f1_k = calculate_f1_score(precision_k, recall_k)
        metrics[f'F1@{k}'] = f1_k
        
        # Calcular Accuracy@k (múltiples variantes)
        accuracy_k = calculate_accuracy_at_k(retrieved_docs, ground_truth_links, effective_k)
        metrics[f'Accuracy@{k}'] = accuracy_k
        
        # Calcular Binary Accuracy@k (equivalente a Precision@k pero con nombre "accuracy")
        binary_accuracy_k = calculate_binary_accuracy_at_k(retrieved_docs, ground_truth_links, effective_k)
        metrics[f'BinaryAccuracy@{k}'] = binary_accuracy_k
        
        # Calcular Ranking Accuracy@k
        ranking_accuracy_k = calculate_ranking_accuracy(retrieved_docs, ground_truth_links, effective_k)
        metrics[f'RankingAccuracy@{k}'] = ranking_accuracy_k
    
    return metrics


def calculate_before_after_reranking_metrics(
    question: str,
    docs_before_reranking: List[Dict],
    docs_after_reranking: List[Dict],
    ground_truth_answer: str,
    ms_links: List[str] = None,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas de recuperación antes y después del reranking.
    
    Args:
        question: Pregunta original
        docs_before_reranking: Documentos antes del reranking
        docs_after_reranking: Documentos después del reranking
        ground_truth_answer: Respuesta aceptada de referencia
        ms_links: Enlaces de Microsoft Learn extraídos previamente
        k_values: Lista de valores k para calcular métricas
        
    Returns:
        Diccionario con métricas 'before' y 'after' reranking
    """
    # Extraer ground truth links
    ground_truth_links = extract_ground_truth_links(ground_truth_answer, ms_links)
    
    # Calcular métricas antes del reranking
    metrics_before = calculate_retrieval_metrics(
        docs_before_reranking, 
        ground_truth_links, 
        k_values
    )
    
    # Calcular métricas después del reranking
    metrics_after = calculate_retrieval_metrics(
        docs_after_reranking, 
        ground_truth_links, 
        k_values
    )
    
    return {
        'before_reranking': metrics_before,
        'after_reranking': metrics_after,
        'ground_truth_links_count': len(ground_truth_links),
        'docs_before_count': len(docs_before_reranking),
        'docs_after_count': len(docs_after_reranking)
    }


def format_metrics_for_display(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Formatea las métricas para mostrar en una tabla legible.
    
    Args:
        metrics: Diccionario con métricas before/after
        
    Returns:
        String formateado para mostrar
    """
    before = metrics['before_reranking']
    after = metrics['after_reranking']
    
    # Crear tabla de comparación
    lines = []
    lines.append("📊 MÉTRICAS DE RECUPERACIÓN - COMPARACIÓN BEFORE/AFTER RERANKING")
    lines.append("=" * 80)
    lines.append(f"Ground Truth Links: {metrics['ground_truth_links_count']}")
    lines.append(f"Docs Before: {metrics['docs_before_count']}, Docs After: {metrics['docs_after_count']}")
    lines.append("-" * 80)
    
    # Encabezado
    lines.append(f"{'Métrica':<15} {'Before':<10} {'After':<10} {'Mejora':<10} {'% Mejora':<10}")
    lines.append("-" * 80)
    
    # MRR
    mrr_before = before['MRR']
    mrr_after = after['MRR']
    mrr_improvement = mrr_after - mrr_before
    mrr_pct = (mrr_improvement / mrr_before * 100) if mrr_before > 0 else 0
    lines.append(f"{'MRR':<15} {mrr_before:<10.4f} {mrr_after:<10.4f} {mrr_improvement:<10.4f} {mrr_pct:<10.2f}%")
    
    # Métricas por k
    for k in [1, 3, 5, 10]:
        # Recall@k
        recall_before = before.get(f'Recall@{k}', 0)
        recall_after = after.get(f'Recall@{k}', 0)
        recall_improvement = recall_after - recall_before
        recall_pct = (recall_improvement / recall_before * 100) if recall_before > 0 else 0
        lines.append(f"{'Recall@' + str(k):<15} {recall_before:<10.4f} {recall_after:<10.4f} {recall_improvement:<10.4f} {recall_pct:<10.2f}%")
        
        # Precision@k
        precision_before = before.get(f'Precision@{k}', 0)
        precision_after = after.get(f'Precision@{k}', 0)
        precision_improvement = precision_after - precision_before
        precision_pct = (precision_improvement / precision_before * 100) if precision_before > 0 else 0
        lines.append(f"{'Precision@' + str(k):<15} {precision_before:<10.4f} {precision_after:<10.4f} {precision_improvement:<10.4f} {precision_pct:<10.2f}%")
        
        # F1@k
        f1_before = before.get(f'F1@{k}', 0)
        f1_after = after.get(f'F1@{k}', 0)
        f1_improvement = f1_after - f1_before
        f1_pct = (f1_improvement / f1_before * 100) if f1_before > 0 else 0
        lines.append(f"{'F1@' + str(k):<15} {f1_before:<10.4f} {f1_after:<10.4f} {f1_improvement:<10.4f} {f1_pct:<10.2f}%")
        
        if k < 10:  # Separador entre valores de k
            lines.append("-" * 50)
    
    return "\n".join(lines)


def calculate_aggregated_metrics(
    all_metrics: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas agregadas sobre múltiples consultas.
    
    Args:
        all_metrics: Lista de métricas individuales por consulta
        
    Returns:
        Diccionario con métricas promedio, mediana y desviación estándar
    """
    if not all_metrics:
        return {}
    
    # Extraer todas las claves de métricas
    sample_metrics = all_metrics[0]['before_reranking']
    metric_keys = list(sample_metrics.keys())
    
    aggregated = {
        'before_reranking': {},
        'after_reranking': {},
        'improvement': {}
    }
    
    for metric_key in metric_keys:
        # Recopilar valores para esta métrica
        before_values = [m['before_reranking'][metric_key] for m in all_metrics]
        after_values = [m['after_reranking'][metric_key] for m in all_metrics]
        improvement_values = [after - before for before, after in zip(before_values, after_values)]
        
        # Calcular estadísticas
        aggregated['before_reranking'][metric_key] = {
            'mean': np.mean(before_values),
            'median': np.median(before_values),
            'std': np.std(before_values),
            'min': np.min(before_values),
            'max': np.max(before_values)
        }
        
        aggregated['after_reranking'][metric_key] = {
            'mean': np.mean(after_values),
            'median': np.median(after_values),
            'std': np.std(after_values),
            'min': np.min(after_values),
            'max': np.max(after_values)
        }
        
        aggregated['improvement'][metric_key] = {
            'mean': np.mean(improvement_values),
            'median': np.median(improvement_values),
            'std': np.std(improvement_values),
            'min': np.min(improvement_values),
            'max': np.max(improvement_values)
        }
    
    return aggregated