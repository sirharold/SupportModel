#!/usr/bin/env python3
"""
Test script para verificar que la integración de métricas de recuperación 
en la página de comparación funcione correctamente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test de importaciones necesarias."""
    print("🧪 TESTING IMPORTS")
    print("=" * 50)
    
    try:
        # Test importaciones principales
        from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
        print("✅ answer_question_with_retrieval_metrics imported successfully")
        
        from utils.retrieval_metrics import format_metrics_for_display
        print("✅ format_metrics_for_display imported successfully")
        
        # Test importaciones de comparison_page
        import comparison_page
        print("✅ comparison_page imported successfully")
        
        # Test función principal
        from comparison_page import show_comparison_page
        print("✅ show_comparison_page imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_synthetic_retrieval_metrics():
    """Test con métricas sintéticas para verificar la visualización."""
    print("\n🧪 TESTING SYNTHETIC RETRIEVAL METRICS")
    print("=" * 50)
    
    try:
        # Crear datos sintéticos que simulan la estructura esperada
        synthetic_comparison_results = {
            "multi-qa-mpnet-base-dot-v1": {
                "results": [
                    {"link": "https://learn.microsoft.com/azure/storage/blobs/", "title": "Azure Blob Storage", "score": 0.95},
                    {"link": "https://learn.microsoft.com/azure/storage/files/", "title": "Azure Files", "score": 0.85},
                    {"link": "https://learn.microsoft.com/azure/compute/vms/", "title": "Azure VMs", "score": 0.75}
                ],
                "debug_info": "Test debug info",
                "summary": "Test summary",
                "time": 2.5,
                "content_metrics": {"BERT_F1": 0.85, "ROUGE1": 0.75},
                "retrieval_metrics": {
                    "before_reranking": {
                        "MRR": 0.3333,
                        "Recall@1": 0.0,
                        "Recall@5": 0.6667,
                        "Precision@1": 0.0,
                        "Precision@5": 0.4,
                        "F1@1": 0.0,
                        "F1@5": 0.5
                    },
                    "after_reranking": {
                        "MRR": 1.0,
                        "Recall@1": 0.3333,
                        "Recall@5": 1.0,
                        "Precision@1": 1.0,
                        "Precision@5": 0.6,
                        "F1@1": 0.5,
                        "F1@5": 0.75
                    },
                    "ground_truth_links_count": 3,
                    "docs_before_count": 5,
                    "docs_after_count": 5
                }
            },
            "all-MiniLM-L6-v2": {
                "results": [
                    {"link": "https://learn.microsoft.com/azure/storage/files/", "title": "Azure Files", "score": 0.88},
                    {"link": "https://learn.microsoft.com/azure/storage/blobs/", "title": "Azure Blob Storage", "score": 0.82},
                    {"link": "https://learn.microsoft.com/azure/networking/vpn/", "title": "Azure VPN", "score": 0.70}
                ],
                "debug_info": "Test debug info",
                "summary": "Test summary",
                "time": 1.8,
                "content_metrics": {"BERT_F1": 0.80, "ROUGE1": 0.70},
                "retrieval_metrics": {
                    "before_reranking": {
                        "MRR": 0.5,
                        "Recall@1": 0.3333,
                        "Recall@5": 0.6667,
                        "Precision@1": 1.0,
                        "Precision@5": 0.4,
                        "F1@1": 0.5,
                        "F1@5": 0.5
                    },
                    "after_reranking": {
                        "MRR": 1.0,
                        "Recall@1": 0.3333,
                        "Recall@5": 0.6667,
                        "Precision@1": 1.0,
                        "Precision@5": 0.4,
                        "F1@1": 0.5,
                        "F1@5": 0.5
                    },
                    "ground_truth_links_count": 3,
                    "docs_before_count": 5,
                    "docs_after_count": 5
                }
            }
        }
        
        print("✅ Synthetic comparison results created")
        
        # Test que los datos tengan la estructura correcta
        for model_key, data in synthetic_comparison_results.items():
            assert "retrieval_metrics" in data, f"Missing retrieval_metrics for {model_key}"
            assert "before_reranking" in data["retrieval_metrics"], f"Missing before_reranking for {model_key}"
            assert "after_reranking" in data["retrieval_metrics"], f"Missing after_reranking for {model_key}"
            print(f"✅ {model_key} structure verified")
        
        # Test función de formateo
        from utils.retrieval_metrics import format_metrics_for_display
        
        for model_key, data in synthetic_comparison_results.items():
            if data.get("retrieval_metrics"):
                formatted = format_metrics_for_display(data["retrieval_metrics"])
                assert len(formatted) > 0, f"Empty formatted output for {model_key}"
                print(f"✅ {model_key} metrics formatting successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing synthetic metrics: {e}")
        return False


def test_comparison_page_structure():
    """Test de estructura de la página de comparación."""
    print("\n🧪 TESTING COMPARISON PAGE STRUCTURE")
    print("=" * 50)
    
    try:
        # Test que las funciones necesarias existan
        import comparison_page
        
        # Verificar que la función principal existe
        assert hasattr(comparison_page, 'show_comparison_page'), "Missing show_comparison_page function"
        print("✅ show_comparison_page function exists")
        
        # Verificar imports en comparison_page
        import importlib
        import inspect
        
        # Obtener el código fuente para verificar imports
        source = inspect.getsource(comparison_page)
        
        required_imports = [
            'answer_question_with_retrieval_metrics',
            'format_metrics_for_display'
        ]
        
        for required_import in required_imports:
            if required_import in source:
                print(f"✅ {required_import} found in comparison_page imports")
            else:
                print(f"⚠️  {required_import} not found in imports (may cause runtime errors)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing comparison page structure: {e}")
        return False


def test_config_integration():
    """Test de integración con configuración."""
    print("\n🧪 TESTING CONFIG INTEGRATION")
    print("=" * 50)
    
    try:
        from config import EMBEDDING_MODELS, MODEL_DESCRIPTIONS
        
        print(f"✅ EMBEDDING_MODELS loaded: {list(EMBEDDING_MODELS.keys())}")
        print(f"✅ MODEL_DESCRIPTIONS loaded: {list(MODEL_DESCRIPTIONS.keys())}")
        
        # Verificar que los modelos esperados estén presentes
        expected_models = ["multi-qa-mpnet-base-dot-v1", "all-MiniLM-L6-v2", "ada"]
        
        for model in expected_models:
            if model in EMBEDDING_MODELS:
                print(f"✅ {model} found in EMBEDDING_MODELS")
            else:
                print(f"⚠️  {model} not found in EMBEDDING_MODELS")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing config integration: {e}")
        return False


def main():
    """Ejecuta todos los tests de integración."""
    print("🚀 INICIANDO TESTS DE INTEGRACIÓN - MÉTRICAS DE RECUPERACIÓN EN COMPARISON PAGE")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Synthetic Retrieval Metrics", test_synthetic_retrieval_metrics),
        ("Comparison Page Structure", test_comparison_page_structure),
        ("Config Integration", test_config_integration)
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
        print("🎉 ¡TODOS LOS TESTS DE INTEGRACIÓN PASARON!")
        print("\n📋 NEXT STEPS:")
        print("1. Ejecuta la aplicación Streamlit: streamlit run app.py")
        print("2. Ve a la página de comparación")
        print("3. Habilita 'Métricas de Recuperación' en la configuración")
        print("4. Ejecuta una comparación para ver las nuevas métricas")
    else:
        print("⚠️  Algunos tests fallaron. Revisa los errores antes de usar la funcionalidad.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)