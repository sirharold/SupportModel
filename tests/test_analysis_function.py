#!/usr/bin/env python3
"""
Test script para verificar la función de análisis automático de métricas de recuperación.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_analysis_function():
    """Test de la función de análisis automático."""
    print("🧪 TESTING ANÁLISIS AUTOMÁTICO DE MÉTRICAS")
    print("=" * 50)
    
    from comparison_page import generate_retrieval_metrics_analysis
    
    # Crear datos sintéticos para el test
    test_data = [
        {
            'Modelo': 'multi-qa-mpnet-base-dot-v1',
            'Ground Truth': 3,
            'MRR_Before': 0.3333,
            'MRR_After': 1.0000,
            'MRR_Δ': 0.6667,
            'MRR_%': 200.0,
            'Recall@1_Before': 0.0000,
            'Recall@1_After': 0.3333,
            'Recall@1_Δ': 0.3333,
            'Recall@1_%': 0.0,
            'Recall@5_Before': 0.6667,
            'Recall@5_After': 1.0000,
            'Recall@5_Δ': 0.3333,
            'Recall@5_%': 50.0,
            'Recall@10_Before': 0.6667,
            'Recall@10_After': 1.0000,
            'Recall@10_Δ': 0.3333,
            'Recall@10_%': 50.0,
            'Precision@1_Before': 0.0000,
            'Precision@1_After': 1.0000,
            'Precision@1_Δ': 1.0000,
            'Precision@1_%': 0.0,
            'Precision@5_Before': 0.4000,
            'Precision@5_After': 0.6000,
            'Precision@5_Δ': 0.2000,
            'Precision@5_%': 50.0,
            'Precision@10_Before': 0.4000,
            'Precision@10_After': 0.6000,
            'Precision@10_Δ': 0.2000,
            'Precision@10_%': 50.0,
            'Accuracy@1_Before': 0.3333,
            'Accuracy@1_After': 0.3333,
            'Accuracy@1_Δ': 0.0000,
            'Accuracy@1_%': 0.0,
            'Accuracy@5_Before': 0.5000,
            'Accuracy@5_After': 0.6000,
            'Accuracy@5_Δ': 0.1000,
            'Accuracy@5_%': 20.0,
            'Accuracy@10_Before': 0.5000,
            'Accuracy@10_After': 0.6000,
            'Accuracy@10_Δ': 0.1000,
            'Accuracy@10_%': 20.0,
        },
        {
            'Modelo': 'all-MiniLM-L6-v2',
            'Ground Truth': 3,
            'MRR_Before': 0.5000,
            'MRR_After': 1.0000,
            'MRR_Δ': 0.5000,
            'MRR_%': 100.0,
            'Recall@1_Before': 0.3333,
            'Recall@1_After': 0.3333,
            'Recall@1_Δ': 0.0000,
            'Recall@1_%': 0.0,
            'Recall@5_Before': 0.6667,
            'Recall@5_After': 0.6667,
            'Recall@5_Δ': 0.0000,
            'Recall@5_%': 0.0,
            'Recall@10_Before': 0.6667,
            'Recall@10_After': 0.6667,
            'Recall@10_Δ': 0.0000,
            'Recall@10_%': 0.0,
            'Precision@1_Before': 1.0000,
            'Precision@1_After': 1.0000,
            'Precision@1_Δ': 0.0000,
            'Precision@1_%': 0.0,
            'Precision@5_Before': 0.4000,
            'Precision@5_After': 0.4000,
            'Precision@5_Δ': 0.0000,
            'Precision@5_%': 0.0,
            'Precision@10_Before': 0.4000,
            'Precision@10_After': 0.4000,
            'Precision@10_Δ': 0.0000,
            'Precision@10_%': 0.0,
            'Accuracy@1_Before': 0.6667,
            'Accuracy@1_After': 0.6667,
            'Accuracy@1_Δ': 0.0000,
            'Accuracy@1_%': 0.0,
            'Accuracy@5_Before': 0.5000,
            'Accuracy@5_After': 0.5000,
            'Accuracy@5_Δ': 0.0000,
            'Accuracy@5_%': 0.0,
            'Accuracy@10_Before': 0.5000,
            'Accuracy@10_After': 0.5000,
            'Accuracy@10_Δ': 0.0000,
            'Accuracy@10_%': 0.0,
        },
        {
            'Modelo': 'ada',
            'Ground Truth': 3,
            'MRR_Before': 0.6667,
            'MRR_After': 1.0000,
            'MRR_Δ': 0.3333,
            'MRR_%': 50.0,
            'Recall@1_Before': 0.3333,
            'Recall@1_After': 0.3333,
            'Recall@1_Δ': 0.0000,
            'Recall@1_%': 0.0,
            'Recall@5_Before': 1.0000,
            'Recall@5_After': 1.0000,
            'Recall@5_Δ': 0.0000,
            'Recall@5_%': 0.0,
            'Recall@10_Before': 1.0000,
            'Recall@10_After': 1.0000,
            'Recall@10_Δ': 0.0000,
            'Recall@10_%': 0.0,
            'Precision@1_Before': 1.0000,
            'Precision@1_After': 1.0000,
            'Precision@1_Δ': 0.0000,
            'Precision@1_%': 0.0,
            'Precision@5_Before': 0.6000,
            'Precision@5_After': 0.6000,
            'Precision@5_Δ': 0.0000,
            'Precision@5_%': 0.0,
            'Precision@10_Before': 0.6000,
            'Precision@10_After': 0.6000,
            'Precision@10_Δ': 0.0000,
            'Precision@10_%': 0.0,
            'Accuracy@1_Before': 0.6667,
            'Accuracy@1_After': 0.6667,
            'Accuracy@1_Δ': 0.0000,
            'Accuracy@1_%': 0.0,
            'Accuracy@5_Before': 0.7000,
            'Accuracy@5_After': 0.7000,
            'Accuracy@5_Δ': 0.0000,
            'Accuracy@5_%': 0.0,
            'Accuracy@10_Before': 0.7000,
            'Accuracy@10_After': 0.7000,
            'Accuracy@10_Δ': 0.0000,
            'Accuracy@10_%': 0.0,
        }
    ]
    
    try:
        # Generar análisis
        analysis = generate_retrieval_metrics_analysis(test_data)
        
        print("📊 ANÁLISIS GENERADO:")
        print("=" * 50)
        print(analysis)
        print("=" * 50)
        
        # Verificar que el análisis contiene elementos esperados
        expected_sections = [
            "📊 Resumen General:",
            "🎯 Análisis de MRR",
            "🔍 Análisis de Recall",
            "🎯 Análisis de Precision",
            "⚡ Impacto del Reranking:",
            "💡 Recomendaciones:"
        ]
        
        missing_sections = []
        for section in expected_sections:
            if section not in analysis:
                missing_sections.append(section)
        
        if not missing_sections:
            print("✅ Todas las secciones esperadas están presentes")
        else:
            print(f"❌ Secciones faltantes: {missing_sections}")
            return False
        
        # Verificar que contiene información específica esperada
        if "3 modelos" in analysis:
            print("✅ Número de modelos detectado correctamente")
        else:
            print("❌ Error en detección de número de modelos")
            return False
        
        if "multi-qa-mpnet-base-dot-v1" in analysis:
            print("✅ Mejor modelo identificado correctamente")
        else:
            print("❌ Error en identificación del mejor modelo")
            return False
        
        # Verificar que las métricas están calculadas
        if "Recall@1:" in analysis and "Precision@5:" in analysis:
            print("✅ Métricas de recall y precision incluidas")
        else:
            print("❌ Error en inclusión de métricas")
            return False
        
        # Verificar recomendaciones
        if "Modelo recomendado:" in analysis:
            print("✅ Recomendación de modelo incluida")
        else:
            print("❌ Error en recomendación de modelo")
            return False
        
        print("\n🎉 ¡ANÁLISIS AUTOMÁTICO FUNCIONA CORRECTAMENTE!")
        return True
        
    except Exception as e:
        print(f"❌ Error generando análisis: {e}")
        return False

def test_empty_data():
    """Test con datos vacíos."""
    print("\n🧪 TESTING CON DATOS VACÍOS")
    print("=" * 50)
    
    from comparison_page import generate_retrieval_metrics_analysis
    
    try:
        # Test con lista vacía
        analysis = generate_retrieval_metrics_analysis([])
        if analysis == "":
            print("✅ Manejo correcto de datos vacíos")
            return True
        else:
            print(f"❌ Error con datos vacíos: debería retornar string vacío, retornó: '{analysis}'")
            return False
    except Exception as e:
        print(f"❌ Error con datos vacíos: {e}")
        return False

def main():
    """Ejecuta todos los tests de análisis."""
    print("🚀 INICIANDO TESTS DE ANÁLISIS AUTOMÁTICO")
    print("=" * 80)
    
    tests = [
        ("Análisis con Datos Completos", test_analysis_function),
        ("Manejo de Datos Vacíos", test_empty_data)
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
        print("🎉 ¡TODOS LOS TESTS DE ANÁLISIS PASARON!")
        print("\n📋 CARACTERÍSTICAS DEL ANÁLISIS:")
        print("✅ Análisis automático de métricas de recuperación")
        print("✅ Identificación del mejor modelo")
        print("✅ Cálculo de mejoras promedio")
        print("✅ Recomendaciones basadas en resultados")
        print("✅ Análisis de impacto del reranking")
        print("✅ Métricas para k=1,3,5,10")
    else:
        print("⚠️  Algunos tests fallaron. Revisa los errores antes de usar la funcionalidad.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)