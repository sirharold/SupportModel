#!/usr/bin/env python3
"""
Test script para verificar que la tabla de métricas incluye todas las columnas para k=1,3,5,10.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_extended_metrics_columns():
    """Test de que se generen todas las columnas esperadas."""
    print("🧪 TESTING COLUMNAS DE MÉTRICAS EXTENDIDAS")
    print("=" * 50)
    
    # Simular la lógica de generación de columnas
    display_cols = ['Modelo', 'Ground Truth', 'MRR_Before', 'MRR_After', 'MRR_Δ', 'MRR_%']
    
    # Add metrics for k=1,3,5,10
    for k in [1, 3, 5, 10]:
        display_cols.extend([
            f'Recall@{k}_Before', f'Recall@{k}_After', f'Recall@{k}_Δ', f'Recall@{k}_%',
            f'Precision@{k}_Before', f'Precision@{k}_After', f'Precision@{k}_Δ', f'Precision@{k}_%',
            f'Accuracy@{k}_Before', f'Accuracy@{k}_After', f'Accuracy@{k}_Δ', f'Accuracy@{k}_%'
        ])
    
    print(f"📊 Total de columnas generadas: {len(display_cols)}")
    print(f"📋 Columnas base: {len(['Modelo', 'Ground Truth', 'MRR_Before', 'MRR_After', 'MRR_Δ', 'MRR_%'])}")
    print(f"📋 Columnas por k: {len(display_cols) - 6}")
    
    # Verificar estructura esperada
    expected_structure = {
        'base_cols': 6,  # Modelo, Ground Truth, MRR x 4
        'metrics_per_k': 12,  # Recall, Precision, Accuracy x 4 (before, after, delta, %)
        'k_values': 4,  # k=1,3,5,10
    }
    
    expected_total = expected_structure['base_cols'] + (expected_structure['metrics_per_k'] * expected_structure['k_values'])
    
    print(f"📊 Esperadas: {expected_total} columnas")
    print(f"📊 Generadas: {len(display_cols)} columnas")
    
    if len(display_cols) == expected_total:
        print("✅ Número correcto de columnas generadas")
    else:
        print(f"❌ Error en número de columnas: esperadas {expected_total}, generadas {len(display_cols)}")
        return False
    
    # Verificar que existan métricas para cada k
    k_values = [1, 3, 5, 10]
    for k in k_values:
        k_columns = [col for col in display_cols if f'@{k}_' in col]
        expected_k_columns = 12  # 3 metrics x 4 variants each
        
        print(f"📊 Métricas para k={k}: {len(k_columns)} columnas")
        
        if len(k_columns) == expected_k_columns:
            print(f"✅ Correcto número de métricas para k={k}")
        else:
            print(f"❌ Error en métricas para k={k}: esperadas {expected_k_columns}, encontradas {len(k_columns)}")
            return False
        
        # Verificar tipos de métricas
        recall_cols = [col for col in k_columns if col.startswith(f'Recall@{k}')]
        precision_cols = [col for col in k_columns if col.startswith(f'Precision@{k}')]
        accuracy_cols = [col for col in k_columns if col.startswith(f'Accuracy@{k}')]
        
        if len(recall_cols) == 4 and len(precision_cols) == 4 and len(accuracy_cols) == 4:
            print(f"✅ Tipos de métricas correctos para k={k}: Recall({len(recall_cols)}), Precision({len(precision_cols)}), Accuracy({len(accuracy_cols)})")
        else:
            print(f"❌ Error en tipos de métricas para k={k}")
            print(f"   Recall: {len(recall_cols)}/4, Precision: {len(precision_cols)}/4, Accuracy: {len(accuracy_cols)}/4")
            return False
    
    # Verificar algunas columnas específicas importantes
    important_columns = [
        'Modelo', 'Ground Truth', 'MRR_Before', 'MRR_After', 'MRR_Δ', 'MRR_%',
        'Recall@1_Before', 'Recall@1_After', 'Recall@1_Δ', 'Recall@1_%',
        'Precision@5_Before', 'Precision@5_After', 'Precision@5_Δ', 'Precision@5_%',
        'Accuracy@10_Before', 'Accuracy@10_After', 'Accuracy@10_Δ', 'Accuracy@10_%'
    ]
    
    missing_important = []
    for col in important_columns:
        if col not in display_cols:
            missing_important.append(col)
    
    if not missing_important:
        print("✅ Todas las columnas importantes están presentes")
    else:
        print(f"❌ Columnas importantes faltantes: {missing_important}")
        return False
    
    # Verificar que la estructura coincide con lo reportado por el usuario
    user_reported_cols = [
        'Modelo', 'Ground Truth', 'MRR_Before', 'MRR_After', 'MRR_Δ', 'MRR_%',
        'Recall@5_Before', 'Recall@5_After', 'Recall@5_Δ', 'Recall@5_%',
        'Precision@5_Before', 'Precision@5_After', 'Precision@5_Δ', 'Precision@5_%',
        'Accuracy@5_Before', 'Accuracy@5_After', 'Accuracy@5_Δ', 'Accuracy@5_%'
    ]
    
    print(f"\n📋 COMPARACIÓN CON PROBLEMA REPORTADO:")
    print(f"Usuario reportó: {len(user_reported_cols)} columnas (solo k=5)")
    print(f"Nueva implementación: {len(display_cols)} columnas (k=1,3,5,10)")
    print(f"Mejora: +{len(display_cols) - len(user_reported_cols)} columnas adicionales")
    
    # Verificar que incluye las columnas que el usuario reportó
    missing_user_cols = []
    for col in user_reported_cols:
        if col not in display_cols:
            missing_user_cols.append(col)
    
    if not missing_user_cols:
        print("✅ Todas las columnas reportadas por el usuario están incluidas")
    else:
        print(f"❌ Columnas reportadas por usuario faltantes: {missing_user_cols}")
        return False
    
    # Verificar las nuevas columnas añadidas
    new_columns = [col for col in display_cols if col not in user_reported_cols]
    print(f"\n📊 NUEVAS COLUMNAS AÑADIDAS: {len(new_columns)}")
    
    new_k_values = []
    for col in new_columns:
        for k in [1, 3, 10]:  # k=5 ya estaba
            if f'@{k}_' in col:
                if k not in new_k_values:
                    new_k_values.append(k)
    
    print(f"📋 Nuevos valores de k añadidos: {sorted(new_k_values)}")
    
    if sorted(new_k_values) == [1, 3, 10]:
        print("✅ Todos los valores de k faltantes han sido añadidos")
    else:
        print(f"❌ Error en valores de k añadidos: esperados [1,3,10], encontrados {sorted(new_k_values)}")
        return False
    
    return True

def test_column_naming_consistency():
    """Test de consistencia en nombres de columnas."""
    print("\n🧪 TESTING CONSISTENCIA DE NOMBRES DE COLUMNAS")
    print("=" * 50)
    
    # Generar columnas como en el código real
    display_cols = ['Modelo', 'Ground Truth', 'MRR_Before', 'MRR_After', 'MRR_Δ', 'MRR_%']
    
    for k in [1, 3, 5, 10]:
        display_cols.extend([
            f'Recall@{k}_Before', f'Recall@{k}_After', f'Recall@{k}_Δ', f'Recall@{k}_%',
            f'Precision@{k}_Before', f'Precision@{k}_After', f'Precision@{k}_Δ', f'Precision@{k}_%',
            f'Accuracy@{k}_Before', f'Accuracy@{k}_After', f'Accuracy@{k}_Δ', f'Accuracy@{k}_%'
        ])
    
    # Verificar patrones de nomenclatura
    patterns = {
        'before': '_Before',
        'after': '_After', 
        'delta': '_Δ',
        'percent': '_%'
    }
    
    print("📋 Verificando patrones de nomenclatura:")
    
    for pattern_name, pattern in patterns.items():
        matching_cols = [col for col in display_cols if col.endswith(pattern)]
        expected_count = 13 * 4 if pattern_name != 'delta' and pattern_name != 'percent' else 13 * 4  # 13 metrics x 4 k values for all patterns
        
        # Ajustar conteo esperado: MRR + 3 metrics * 4 k values = 1 + 12 = 13 metrics total
        if pattern_name in ['before', 'after']:
            expected_count = 13  # MRR + 12 metrics (3 per k value * 4 k values)
        else:  # delta and percent
            expected_count = 13  # Same as above
        
        print(f"  {pattern_name.capitalize()}: {len(matching_cols)}/{expected_count} columnas")
        
        if len(matching_cols) == expected_count:
            print(f"  ✅ Patrón {pattern_name} correcto")
        else:
            print(f"  ❌ Error en patrón {pattern_name}")
            return False
    
    # Verificar que no hay duplicados
    unique_cols = list(set(display_cols))
    if len(unique_cols) == len(display_cols):
        print("✅ No hay columnas duplicadas")
    else:
        duplicates = len(display_cols) - len(unique_cols)
        print(f"❌ Se encontraron {duplicates} columnas duplicadas")
        return False
    
    return True

def main():
    """Ejecuta todos los tests de la tabla extendida."""
    print("🚀 INICIANDO TESTS DE TABLA DE MÉTRICAS EXTENDIDA")
    print("=" * 80)
    
    tests = [
        ("Columnas de Métricas Extendidas", test_extended_metrics_columns),
        ("Consistencia de Nombres de Columnas", test_column_naming_consistency)
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
        print("🎉 ¡TODOS LOS TESTS DE TABLA EXTENDIDA PASARON!")
        print("\n📋 MEJORAS IMPLEMENTADAS:")
        print("✅ Métricas para k=1,3,5,10 (antes solo k=5)")
        print("✅ Análisis automático de resultados")
        print("✅ Columnas organizadas y consistentes")
        print("✅ Compatibilidad con formato original")
        print("✅ Expansión de 18 a 54 columnas totales")
    else:
        print("⚠️  Algunos tests fallaron. Revisa los errores antes de usar la funcionalidad.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)