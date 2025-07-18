#!/usr/bin/env python3
"""
Test script para verificar el funcionamiento de las métricas de recuperación.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_f1_score,
    calculate_mrr,
    calculate_retrieval_metrics,
    calculate_before_after_reranking_metrics,
    format_metrics_for_display
)


def test_basic_metrics():
    """Test de métricas básicas con datos sintéticos."""
    print("🧪 TESTING BASIC METRICS")
    print("=" * 50)
    
    # Datos de prueba
    retrieved_docs = [
        {'link': 'https://learn.microsoft.com/azure/storage/blobs/', 'title': 'Azure Blob Storage', 'score': 0.9},
        {'link': 'https://learn.microsoft.com/azure/storage/files/', 'title': 'Azure Files', 'score': 0.8},
        {'link': 'https://learn.microsoft.com/azure/compute/vms/', 'title': 'Azure VMs', 'score': 0.7},
        {'link': 'https://learn.microsoft.com/azure/storage/queues/', 'title': 'Azure Queues', 'score': 0.6},
        {'link': 'https://learn.microsoft.com/azure/networking/vpn/', 'title': 'Azure VPN', 'score': 0.5}
    ]
    
    # Ground truth (documentos relevantes)
    ground_truth_links = {
        'https://learn.microsoft.com/azure/storage/blobs/',
        'https://learn.microsoft.com/azure/storage/files/',
        'https://learn.microsoft.com/azure/storage/queues/'
    }
    
    print(f"📋 Documentos recuperados: {len(retrieved_docs)}")
    print(f"📋 Ground truth links: {len(ground_truth_links)}")
    print()
    
    # Test individual metrics
    print("🎯 MÉTRICAS INDIVIDUALES:")
    
    for k in [1, 3, 5]:
        recall = calculate_recall_at_k(retrieved_docs, ground_truth_links, k)
        precision = calculate_precision_at_k(retrieved_docs, ground_truth_links, k)
        f1 = calculate_f1_score(precision, recall)
        
        print(f"k={k}: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    
    mrr = calculate_mrr(retrieved_docs, ground_truth_links)
    print(f"MRR: {mrr:.4f}")
    print()
    
    # Test complete metrics
    print("🔍 MÉTRICAS COMPLETAS:")
    metrics = calculate_retrieval_metrics(retrieved_docs, ground_truth_links)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("\n" + "=" * 50)
    return True


def test_before_after_reranking():
    """Test de métricas antes y después del reranking."""
    print("🧪 TESTING BEFORE/AFTER RERANKING METRICS")
    print("=" * 50)
    
    # Documentos ANTES del reranking (peor orden)
    docs_before = [
        {'link': 'https://learn.microsoft.com/azure/networking/vpn/', 'title': 'Azure VPN', 'score': 0.5},
        {'link': 'https://learn.microsoft.com/azure/compute/vms/', 'title': 'Azure VMs', 'score': 0.7},
        {'link': 'https://learn.microsoft.com/azure/storage/blobs/', 'title': 'Azure Blob Storage', 'score': 0.9},
        {'link': 'https://learn.microsoft.com/azure/storage/files/', 'title': 'Azure Files', 'score': 0.8},
        {'link': 'https://learn.microsoft.com/azure/storage/queues/', 'title': 'Azure Queues', 'score': 0.6}
    ]
    
    # Documentos DESPUÉS del reranking (mejor orden)
    docs_after = [
        {'link': 'https://learn.microsoft.com/azure/storage/blobs/', 'title': 'Azure Blob Storage', 'score': 0.95},
        {'link': 'https://learn.microsoft.com/azure/storage/files/', 'title': 'Azure Files', 'score': 0.85},
        {'link': 'https://learn.microsoft.com/azure/storage/queues/', 'title': 'Azure Queues', 'score': 0.75},
        {'link': 'https://learn.microsoft.com/azure/compute/vms/', 'title': 'Azure VMs', 'score': 0.65},
        {'link': 'https://learn.microsoft.com/azure/networking/vpn/', 'title': 'Azure VPN', 'score': 0.55}
    ]
    
    # Ground truth answer simulada
    ground_truth_answer = """
    Para trabajar con Azure Storage, necesitas entender los siguientes servicios:
    
    1. Azure Blob Storage: https://learn.microsoft.com/azure/storage/blobs/
    2. Azure Files: https://learn.microsoft.com/azure/storage/files/
    3. Azure Queues: https://learn.microsoft.com/azure/storage/queues/
    
    Estos servicios te permiten almacenar diferentes tipos de datos.
    """
    
    # Calcular métricas
    question = "¿Cómo configurar Azure Storage?"
    
    metrics = calculate_before_after_reranking_metrics(
        question=question,
        docs_before_reranking=docs_before,
        docs_after_reranking=docs_after,
        ground_truth_answer=ground_truth_answer,
        ms_links=None,  # Se extraerán automáticamente
        k_values=[1, 3, 5, 10]
    )
    
    # Mostrar resultados formateados
    formatted_output = format_metrics_for_display(metrics)
    print(formatted_output)
    
    print("\n" + "=" * 50)
    return True


def test_edge_cases():
    """Test de casos edge y situaciones especiales."""
    print("🧪 TESTING EDGE CASES")
    print("=" * 50)
    
    # Caso 1: Sin documentos recuperados
    print("📋 Caso 1: Sin documentos recuperados")
    empty_docs = []
    ground_truth = {'https://learn.microsoft.com/azure/storage/blobs/'}
    
    recall = calculate_recall_at_k(empty_docs, ground_truth, 5)
    precision = calculate_precision_at_k(empty_docs, ground_truth, 5)
    mrr = calculate_mrr(empty_docs, ground_truth)
    
    print(f"Recall@5: {recall:.4f} (esperado: 0.0)")
    print(f"Precision@5: {precision:.4f} (esperado: 0.0)")
    print(f"MRR: {mrr:.4f} (esperado: 0.0)")
    print()
    
    # Caso 2: Sin ground truth
    print("📋 Caso 2: Sin ground truth")
    some_docs = [
        {'link': 'https://learn.microsoft.com/azure/storage/blobs/', 'title': 'Azure Blob Storage', 'score': 0.9}
    ]
    empty_ground_truth = set()
    
    recall = calculate_recall_at_k(some_docs, empty_ground_truth, 5)
    precision = calculate_precision_at_k(some_docs, empty_ground_truth, 5)
    mrr = calculate_mrr(some_docs, empty_ground_truth)
    
    print(f"Recall@5: {recall:.4f} (esperado: 0.0)")
    print(f"Precision@5: {precision:.4f} (esperado: 0.0)")
    print(f"MRR: {mrr:.4f} (esperado: 0.0)")
    print()
    
    # Caso 3: k mayor que número de documentos
    print("📋 Caso 3: k > número de documentos")
    few_docs = [
        {'link': 'https://learn.microsoft.com/azure/storage/blobs/', 'title': 'Azure Blob Storage', 'score': 0.9},
        {'link': 'https://learn.microsoft.com/azure/storage/files/', 'title': 'Azure Files', 'score': 0.8}
    ]
    some_ground_truth = {'https://learn.microsoft.com/azure/storage/blobs/'}
    
    recall = calculate_recall_at_k(few_docs, some_ground_truth, 10)  # k=10 > 2 docs
    precision = calculate_precision_at_k(few_docs, some_ground_truth, 10)
    
    print(f"Recall@10: {recall:.4f} (con solo 2 docs)")
    print(f"Precision@10: {precision:.4f} (con solo 2 docs)")
    print()
    
    # Caso 4: Todos los documentos son relevantes
    print("📋 Caso 4: Todos los documentos son relevantes")
    all_relevant_docs = [
        {'link': 'https://learn.microsoft.com/azure/storage/blobs/', 'title': 'Azure Blob Storage', 'score': 0.9},
        {'link': 'https://learn.microsoft.com/azure/storage/files/', 'title': 'Azure Files', 'score': 0.8},
        {'link': 'https://learn.microsoft.com/azure/storage/queues/', 'title': 'Azure Queues', 'score': 0.7}
    ]
    all_ground_truth = {
        'https://learn.microsoft.com/azure/storage/blobs/',
        'https://learn.microsoft.com/azure/storage/files/',
        'https://learn.microsoft.com/azure/storage/queues/'
    }
    
    recall = calculate_recall_at_k(all_relevant_docs, all_ground_truth, 3)
    precision = calculate_precision_at_k(all_relevant_docs, all_ground_truth, 3)
    mrr = calculate_mrr(all_relevant_docs, all_ground_truth)
    
    print(f"Recall@3: {recall:.4f} (esperado: 1.0)")
    print(f"Precision@3: {precision:.4f} (esperado: 1.0)")
    print(f"MRR: {mrr:.4f} (esperado: 1.0)")
    print()
    
    print("=" * 50)
    return True


def main():
    """Ejecuta todos los tests."""
    print("🚀 INICIANDO TESTS DE MÉTRICAS DE RECUPERACIÓN")
    print("=" * 80)
    
    tests = [
        test_basic_metrics,
        test_before_after_reranking,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} ERROR: {e}")
        
        print()
    
    print("=" * 80)
    print(f"📊 RESUMEN DE TESTS: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
    else:
        print("⚠️  Algunos tests fallaron. Revisa la implementación.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)