#!/usr/bin/env python3
"""
Test script para verificar la normalización de URLs en el sistema de métricas de recuperación.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_url_normalization():
    """Test de la función de normalización de URLs."""
    print("🧪 TESTING URL NORMALIZATION")
    print("=" * 50)
    
    from utils.extract_links import normalize_url
    
    # Test cases: (input_url, expected_output)
    test_cases = [
        # Basic Microsoft Learn URLs
        (
            "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview",
            "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview"
        ),
        # URL with query parameters
        (
            "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview?view=azure-cli-latest",
            "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview"
        ),
        # URL with anchor/fragment
        (
            "https://learn.microsoft.com/en-us/azure/virtual-machines/windows/quick-create-portal#create-vm",
            "https://learn.microsoft.com/en-us/azure/virtual-machines/windows/quick-create-portal"
        ),
        # URL with both query and fragment
        (
            "https://learn.microsoft.com/azure/storage/blobs/storage-blob-overview?view=azure-cli-latest&tabs=portal#overview",
            "https://learn.microsoft.com/azure/storage/blobs/storage-blob-overview"
        ),
        # URL with multiple query parameters
        (
            "https://learn.microsoft.com/azure/virtual-machines/linux/quick-create-cli?tabs=ubuntu&pivots=azure-cli",
            "https://learn.microsoft.com/azure/virtual-machines/linux/quick-create-cli"
        ),
        # Complex URL with Azure CLI version and deep anchor
        (
            "https://learn.microsoft.com/en-us/cli/azure/vm?view=azure-cli-latest#az-vm-create",
            "https://learn.microsoft.com/en-us/cli/azure/vm"
        ),
        # URL with PowerShell module version
        (
            "https://learn.microsoft.com/en-us/powershell/module/az.compute/new-azvm?view=azps-9.0.1",
            "https://learn.microsoft.com/en-us/powershell/module/az.compute/new-azvm"
        ),
        # Edge cases
        ("", ""),
        ("  ", ""),
        ("https://example.com", "https://example.com"),
        ("https://learn.microsoft.com/", "https://learn.microsoft.com/"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (input_url, expected) in enumerate(test_cases, 1):
        try:
            result = normalize_url(input_url)
            if result == expected:
                print(f"✅ Test {i}: PASSED")
                print(f"   Input:    '{input_url}'")
                print(f"   Expected: '{expected}'")
                print(f"   Got:      '{result}'")
                passed += 1
            else:
                print(f"❌ Test {i}: FAILED")
                print(f"   Input:    '{input_url}'")
                print(f"   Expected: '{expected}'")
                print(f"   Got:      '{result}'")
                failed += 1
        except Exception as e:
            print(f"❌ Test {i}: ERROR - {e}")
            print(f"   Input: '{input_url}'")
            failed += 1
        
        print()
    
    print(f"📊 Results: {passed} PASSED, {failed} FAILED")
    return failed == 0

def test_extract_urls_with_normalization():
    """Test de extracción de URLs con normalización automática."""
    print("🧪 TESTING URL EXTRACTION WITH NORMALIZATION")
    print("=" * 50)
    
    from utils.extract_links import extract_urls_from_answer
    
    # Sample answer with various URL formats
    test_answer = """
    Para configurar Azure Blob Storage, consulta estos recursos:
    
    1. Guía principal: https://learn.microsoft.com/azure/storage/blobs/storage-blob-overview?view=azure-cli-latest
    2. CLI reference: https://learn.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest#az-storage-blob-upload
    3. Portal guide: https://learn.microsoft.com/azure/storage/blobs/storage-quickstart-blobs-portal?tabs=azure-portal#create-container
    4. REST API: https://learn.microsoft.com/rest/api/storageservices/blob-service-rest-api
    """
    
    expected_normalized = [
        "https://learn.microsoft.com/azure/storage/blobs/storage-blob-overview",
        "https://learn.microsoft.com/en-us/cli/azure/storage/blob",
        "https://learn.microsoft.com/azure/storage/blobs/storage-quickstart-blobs-portal",
        "https://learn.microsoft.com/rest/api/storageservices/blob-service-rest-api"
    ]
    
    try:
        extracted_urls = extract_urls_from_answer(test_answer)
        
        print("📋 Extracted URLs:")
        for i, url in enumerate(extracted_urls, 1):
            print(f"   {i}. {url}")
        
        print(f"\n📊 Expected {len(expected_normalized)} URLs, got {len(extracted_urls)}")
        
        # Check if all expected URLs are present
        missing_urls = []
        for expected_url in expected_normalized:
            if expected_url not in extracted_urls:
                missing_urls.append(expected_url)
        
        extra_urls = []
        for extracted_url in extracted_urls:
            if extracted_url not in expected_normalized:
                extra_urls.append(extracted_url)
        
        if not missing_urls and not extra_urls:
            print("✅ All URLs extracted and normalized correctly!")
            return True
        else:
            if missing_urls:
                print("❌ Missing URLs:")
                for url in missing_urls:
                    print(f"   - {url}")
            
            if extra_urls:
                print("⚠️  Extra URLs:")
                for url in extra_urls:
                    print(f"   + {url}")
            
            return False
    
    except Exception as e:
        print(f"❌ Error extracting URLs: {e}")
        return False

def test_retrieval_metrics_with_normalized_urls():
    """Test de métricas de recuperación con URLs normalizadas."""
    print("🧪 TESTING RETRIEVAL METRICS WITH NORMALIZED URLS")
    print("=" * 50)
    
    from utils.retrieval_metrics import calculate_recall_at_k, calculate_precision_at_k, extract_ground_truth_links
    
    # Simulated retrieved documents with URLs that have parameters/anchors
    retrieved_docs = [
        {
            'title': 'Azure Blob Storage Overview',
            'link': 'https://learn.microsoft.com/azure/storage/blobs/storage-blob-overview?view=azure-cli-latest',
            'score': 0.95
        },
        {
            'title': 'Create Storage Account',
            'link': 'https://learn.microsoft.com/azure/storage/common/storage-account-create#portal',
            'score': 0.88
        },
        {
            'title': 'Unrelated Document',
            'link': 'https://example.com/other-doc',
            'score': 0.75
        }
    ]
    
    # Ground truth with normalized URLs
    ground_truth_links = {
        'https://learn.microsoft.com/azure/storage/blobs/storage-blob-overview',
        'https://learn.microsoft.com/azure/storage/common/storage-account-create',
        'https://learn.microsoft.com/azure/storage/blobs/storage-quickstart'
    }
    
    try:
        # Test Recall@2 - should find 2 out of 3 ground truth links
        recall_2 = calculate_recall_at_k(retrieved_docs, ground_truth_links, k=2)
        expected_recall = 2/3  # 2 relevant docs found out of 3 total relevant
        
        print(f"📊 Recall@2: {recall_2:.4f} (expected: {expected_recall:.4f})")
        
        # Test Precision@2 - should find 2 relevant out of 2 retrieved
        precision_2 = calculate_precision_at_k(retrieved_docs, ground_truth_links, k=2)
        expected_precision = 2/2  # 2 relevant docs out of 2 retrieved
        
        print(f"📊 Precision@2: {precision_2:.4f} (expected: {expected_precision:.4f})")
        
        # Verify the results
        recall_correct = abs(recall_2 - expected_recall) < 0.001
        precision_correct = abs(precision_2 - expected_precision) < 0.001
        
        if recall_correct and precision_correct:
            print("✅ Retrieval metrics with URL normalization working correctly!")
            return True
        else:
            print("❌ Retrieval metrics calculation failed")
            if not recall_correct:
                print(f"   Recall error: expected {expected_recall:.4f}, got {recall_2:.4f}")
            if not precision_correct:
                print(f"   Precision error: expected {expected_precision:.4f}, got {precision_2:.4f}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing retrieval metrics: {e}")
        return False

def main():
    """Ejecuta todos los tests de normalización de URLs."""
    print("🚀 INICIANDO TESTS DE NORMALIZACIÓN DE URLS")
    print("=" * 80)
    
    tests = [
        ("URL Normalization", test_url_normalization),
        ("URL Extraction with Normalization", test_extract_urls_with_normalization),
        ("Retrieval Metrics with Normalized URLs", test_retrieval_metrics_with_normalized_urls)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
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
        print("🎉 ¡TODOS LOS TESTS DE NORMALIZACIÓN DE URLs PASARON!")
        print("\n📋 BENEFICIOS DE LA NORMALIZACIÓN:")
        print("✅ Los parámetros de consulta (?view=azure-cli-latest) se ignoran")
        print("✅ Los anclajes (#section) se ignoran")
        print("✅ Las comparaciones de enlaces son más robustas")
        print("✅ Las métricas de recuperación son más precisas")
    else:
        print("⚠️  Algunos tests fallaron. Revisa los errores antes de usar la funcionalidad.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)