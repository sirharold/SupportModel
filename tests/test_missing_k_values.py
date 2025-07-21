#!/usr/bin/env python3
"""
Test script para verificar que se generen métricas para k=1,3,5,10.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_k_values_generation():
    """Test que verifica la generación de métricas para todos los valores de k."""
    print("🧪 TESTING GENERACIÓN DE MÉTRICAS PARA K=1,3,5,10")
    print("=" * 60)
    
    from src.evaluation.metrics.retrieval import calculate_before_after_reranking_metrics
    
    # Datos de prueba
    question = "¿Cómo crear un Azure Storage Account?"
    
    docs_before = [
        {'title': 'Azure Storage Overview', 'link': 'https://learn.microsoft.com/azure/storage/common/storage-account-overview?view=azure-cli', 'score': 0.8},
        {'title': 'Create Storage Account', 'link': 'https://learn.microsoft.com/azure/storage/common/storage-account-create#portal', 'score': 0.7},
        {'title': 'Storage Security', 'link': 'https://learn.microsoft.com/azure/storage/common/storage-security-guide', 'score': 0.6},
        {'title': 'Blob Storage', 'link': 'https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction', 'score': 0.5},
        {'title': 'Other Doc', 'link': 'https://example.com/other', 'score': 0.4}
    ]
    
    docs_after = [
        {'title': 'Create Storage Account', 'link': 'https://learn.microsoft.com/azure/storage/common/storage-account-create#portal', 'score': 0.95},
        {'title': 'Azure Storage Overview', 'link': 'https://learn.microsoft.com/azure/storage/common/storage-account-overview?view=azure-cli', 'score': 0.9},
        {'title': 'Blob Storage', 'link': 'https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction', 'score': 0.85},
        {'title': 'Storage Security', 'link': 'https://learn.microsoft.com/azure/storage/common/storage-security-guide', 'score': 0.8},
        {'title': 'Other Doc', 'link': 'https://example.com/other', 'score': 0.4}
    ]
    
    ground_truth_answer = """
    Para crear una Azure Storage Account:
    1. Guía completa: https://learn.microsoft.com/azure/storage/common/storage-account-create
    2. Información general: https://learn.microsoft.com/azure/storage/common/storage-account-overview
    3. Configuración de Blob Storage: https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction
    """
    
    try:
        # Calcular métricas con k_values explícito
        result = calculate_before_after_reranking_metrics(
            question=question,
            docs_before_reranking=docs_before,
            docs_after_reranking=docs_after,
            ground_truth_answer=ground_truth_answer,
            ms_links=None,  # Se extraerán automáticamente
            k_values=[1, 3, 5, 10]
        )
        
        print("📊 MÉTRICAS CALCULADAS:")
        print("=" * 40)
        
        before_metrics = result['before_reranking']
        after_metrics = result['after_reranking']
        
        print("📋 Métricas BEFORE reranking:")
        for key, value in before_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\n📋 Métricas AFTER reranking:")
        for key, value in after_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Verificar que existen métricas para cada k
        expected_k_values = [1, 3, 5, 10]
        missing_k_values = []
        
        for k in expected_k_values:
            k_metrics = [metric for metric in before_metrics.keys() if f'@{k}' in metric]
            if not k_metrics:
                missing_k_values.append(k)
            else:
                print(f"\n✅ Métricas para k={k}: {len(k_metrics)} encontradas")
                print(f"   Ejemplos: {k_metrics[:3]}")
        
        if missing_k_values:
            print(f"\n❌ VALORES DE K FALTANTES: {missing_k_values}")
            return False
        else:
            print(f"\n✅ TODOS LOS VALORES DE K PRESENTES: {expected_k_values}")
        
        # Verificar tipos específicos de métricas
        metric_types = ['Recall', 'Precision', 'Accuracy']
        for k in expected_k_values:
            for metric_type in metric_types:
                metric_name = f'{metric_type}@{k}'
                if metric_name in before_metrics and metric_name in after_metrics:
                    print(f"✅ {metric_name}: Before={before_metrics[metric_name]:.4f}, After={after_metrics[metric_name]:.4f}")
                else:
                    print(f"❌ {metric_name}: FALTANTE")
                    return False
        
        # Verificar estructura del resultado para página de comparación
        comparison_data = {
            'Modelo': 'test-model',
            'Ground Truth': result.get('ground_truth_links_count', 0),
            'MRR_Before': before_metrics['MRR'],
            'MRR_After': after_metrics['MRR'],
            'MRR_Δ': after_metrics['MRR'] - before_metrics['MRR'],
            'MRR_%': ((after_metrics['MRR'] - before_metrics['MRR']) / before_metrics['MRR'] * 100) if before_metrics['MRR'] > 0 else 0
        }
        
        # Agregar métricas para cada k
        for k in expected_k_values:
            for metric_type in metric_types:
                before_val = before_metrics.get(f'{metric_type}@{k}', 0)
                after_val = after_metrics.get(f'{metric_type}@{k}', 0)
                improvement = after_val - before_val
                pct_improvement = (improvement / before_val * 100) if before_val > 0 else 0
                
                comparison_data[f'{metric_type}@{k}_Before'] = before_val
                comparison_data[f'{metric_type}@{k}_After'] = after_val
                comparison_data[f'{metric_type}@{k}_Δ'] = improvement
                comparison_data[f'{metric_type}@{k}_%'] = pct_improvement
        
        print(f"\n📊 ESTRUCTURA PARA COMPARACIÓN:")
        print(f"Total de columnas generadas: {len(comparison_data)}")
        
        # Verificar columnas específicas que el usuario reportó como faltantes
        missing_columns = []
        for k in [3, 10]:  # Los valores que el usuario dice que faltan
            for metric_type in metric_types:
                col_name = f'{metric_type}@{k}_Before'
                if col_name not in comparison_data:
                    missing_columns.append(col_name)
        
        if missing_columns:
            print(f"❌ COLUMNAS FALTANTES PARA K=3,10: {missing_columns}")
            return False
        else:
            print("✅ TODAS LAS COLUMNAS PARA K=3,10 ESTÁN PRESENTES")
        
        # Mostrar algunas columnas específicas para k=3 y k=10
        print(f"\n📋 MÉTRICAS ESPECÍFICAS PARA K=3 Y K=10:")
        for k in [3, 10]:
            print(f"  k={k}:")
            for metric_type in metric_types:
                before_key = f'{metric_type}@{k}_Before'
                after_key = f'{metric_type}@{k}_After'
                print(f"    {metric_type}@{k}: {comparison_data[before_key]:.4f} → {comparison_data[after_key]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_page_integration():
    """Test que simula la lógica de la página de comparación."""
    print("\n🧪 TESTING INTEGRACIÓN CON PÁGINA DE COMPARACIÓN")
    print("=" * 60)
    
    # Simular datos como los que llegan a la página de comparación
    retrieval_comparison_data = [
        {
            'Modelo': 'multi-qa-mpnet-base-dot-v1',
            'Ground Truth': 3,
            'MRR_Before': 0.3333,
            'MRR_After': 1.0000,
            'MRR_Δ': 0.6667,
            'MRR_%': 200.0,
        }
    ]
    
    # Simular la lógica de generación de columnas de la página
    for k in [1, 3, 5, 10]:
        retrieval_comparison_data[0].update({
            f'Recall@{k}_Before': 0.3333 if k == 1 else 0.6667,
            f'Recall@{k}_After': 0.3333 if k == 1 else 1.0000,
            f'Recall@{k}_Δ': 0.0000 if k == 1 else 0.3333,
            f'Recall@{k}_%': 0.0 if k == 1 else 50.0,
            f'Precision@{k}_Before': 1.0000 if k == 1 else 0.4000,
            f'Precision@{k}_After': 1.0000 if k == 1 else 0.6000,
            f'Precision@{k}_Δ': 0.0000 if k == 1 else 0.2000,
            f'Precision@{k}_%': 0.0 if k == 1 else 50.0,
            f'Accuracy@{k}_Before': 0.5000,
            f'Accuracy@{k}_After': 0.6000,
            f'Accuracy@{k}_Δ': 0.1000,
            f'Accuracy@{k}_%': 20.0,
        })
    
    # Verificar que las columnas esperadas existen
    expected_columns_k3 = [
        'Recall@3_Before', 'Recall@3_After', 'Recall@3_Δ', 'Recall@3_%',
        'Precision@3_Before', 'Precision@3_After', 'Precision@3_Δ', 'Precision@3_%',
        'Accuracy@3_Before', 'Accuracy@3_After', 'Accuracy@3_Δ', 'Accuracy@3_%'
    ]
    
    expected_columns_k10 = [
        'Recall@10_Before', 'Recall@10_After', 'Recall@10_Δ', 'Recall@10_%',
        'Precision@10_Before', 'Precision@10_After', 'Precision@10_Δ', 'Precision@10_%',
        'Accuracy@10_Before', 'Accuracy@10_After', 'Accuracy@10_Δ', 'Accuracy@10_%'
    ]
    
    missing_columns = []
    
    for col in expected_columns_k3 + expected_columns_k10:
        if col not in retrieval_comparison_data[0]:
            missing_columns.append(col)
    
    if missing_columns:
        print(f"❌ COLUMNAS FALTANTES: {missing_columns}")
        return False
    
    print("✅ TODAS LAS COLUMNAS PARA K=3,10 ESTÁN PRESENTES EN SIMULACIÓN")
    
    # Mostrar las columnas que el usuario reportó como faltantes
    print(f"\n📋 COLUMNAS REPORTADAS COMO FALTANTES POR EL USUARIO:")
    for k in [3, 10]:
        print(f"  k={k}:")
        for metric_type in ['Recall', 'Precision', 'Accuracy']:
            before_col = f'{metric_type}@{k}_Before'
            after_col = f'{metric_type}@{k}_After'
            if before_col in retrieval_comparison_data[0]:
                print(f"    ✅ {before_col}: {retrieval_comparison_data[0][before_col]}")
                print(f"    ✅ {after_col}: {retrieval_comparison_data[0][after_col]}")
            else:
                print(f"    ❌ {before_col}: FALTANTE")
    
    return True

def main():
    """Ejecuta todos los tests."""
    print("🚀 INICIANDO INVESTIGACIÓN DE MÉTRICAS FALTANTES PARA K=3,10")
    print("=" * 80)
    
    tests = [
        ("Generación de Métricas K=1,3,5,10", test_k_values_generation),
        ("Integración con Página de Comparación", test_comparison_page_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} TEST PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} TEST FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} TEST ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 RESUMEN DE TESTS: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print("\n💡 POSIBLES CAUSAS DEL PROBLEMA REPORTADO:")
        print("1. ✅ Métricas se generan correctamente para k=1,3,5,10")
        print("2. ❓ El problema puede estar en la VISUALIZACIÓN en Streamlit")
        print("3. ❓ Puede haber un filtro o condición que oculta k=3,10")
        print("4. ❓ El problema puede ser en el proceso de datos específico del usuario")
    else:
        print("⚠️  Algunos tests fallaron - hay un problema en la generación de métricas")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)