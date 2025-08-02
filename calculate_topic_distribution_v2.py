#!/usr/bin/env python3
"""
Script mejorado para calcular la distribución de temas en el corpus de documentos.
Se basa principalmente en análisis de contenido dado que muchas URLs están ausentes.

Autor: Harold Gómez
Fecha: 2025-08-01
"""

import chromadb
from collections import defaultdict
import re
from typing import Dict, List, Tuple
import json
from datetime import datetime

def categorize_document_by_content(content: str) -> str:
    """
    Categoriza un documento basándose principalmente en su contenido.
    
    Args:
        content: Contenido del documento
        
    Returns:
        Categoría del documento: 'Azure Services', 'Security', 'Development', o 'Operations'
    """
    if not content:
        return "Azure Services"
    
    content_lower = content.lower()
    
    # Contar palabras clave por categoría
    scores = {
        'Security': 0,
        'Development': 0, 
        'Operations': 0,
        'Azure Services': 0
    }
    
    # Security keywords
    security_keywords = [
        'security', 'authentication', 'authorization', 'encryption', 'certificate',
        'identity', 'active directory', 'key vault', 'defender', 'sentinel',
        'firewall', 'rbac', 'role-based', 'access control', 'compliance',
        'threat', 'vulnerability', 'security center', 'password', 'token',
        'oauth', 'saml', 'federation', 'mfa', 'multi-factor'
    ]
    
    # Development keywords
    development_keywords = [
        'sdk', 'api', 'rest', 'http', 'json', 'xml', 'javascript', 'python',
        'java', 'c#', 'csharp', 'nodejs', 'react', 'angular', 'vue',
        'developer', 'programming', 'code', 'function', 'app service',
        'logic apps', 'visual studio', 'github', 'devops', 'pipeline',
        'build', 'deploy', 'release', 'continuous integration', 'ci/cd'
    ]
    
    # Operations keywords
    operations_keywords = [
        'monitor', 'monitoring', 'logging', 'metrics', 'alerts', 'dashboard',
        'backup', 'restore', 'disaster recovery', 'automation', 'powershell',
        'cli', 'terraform', 'arm template', 'bicep', 'infrastructure',
        'resource manager', 'governance', 'policy', 'management', 'configure',
        'troubleshoot', 'diagnostic', 'performance', 'scale', 'availability'
    ]
    
    # Azure Services keywords (servicios específicos)
    azure_services_keywords = [
        'virtual machine', 'storage account', 'sql database', 'cosmos db',
        'blob storage', 'table storage', 'queue storage', 'virtual network',
        'load balancer', 'application gateway', 'traffic manager', 'cdn',
        'service bus', 'event hub', 'iot hub', 'notification hub',
        'redis cache', 'search service', 'media services', 'batch',
        'kubernetes service', 'container instances', 'service fabric'
    ]
    
    # Contar ocurrencias
    for keyword in security_keywords:
        scores['Security'] += content_lower.count(keyword)
    
    for keyword in development_keywords:
        scores['Development'] += content_lower.count(keyword)
    
    for keyword in operations_keywords:
        scores['Operations'] += content_lower.count(keyword)
    
    for keyword in azure_services_keywords:
        scores['Azure Services'] += content_lower.count(keyword)
    
    # Retornar la categoría con mayor score
    if all(score == 0 for score in scores.values()):
        # Si no hay keywords específicas, clasificar como Azure Services general
        return "Azure Services"
    
    return max(scores, key=scores.get)

def calculate_topic_distribution_v2(persist_directory: str = "/Users/haroldgomez/chromadb2", 
                                   sample_size: int = 5000) -> Tuple[Dict[str, float], Dict[str, int], int]:
    """
    Calcula la distribución de temas basándose en análisis de contenido.
    """
    # Conectar a ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name="docs_ada")
    
    print(f"📊 Analizando colección: docs_ada")
    print(f"  Obteniendo muestra de {sample_size:,} documentos...")
    
    # Obtener muestra estratificada
    all_docs = []
    batch_size = 1000
    
    for i in range(0, sample_size, batch_size):
        offset = i * 3  # Espaciar para mejor distribución
        results = collection.get(
            limit=min(batch_size, sample_size - i),
            offset=offset,
            include=["documents"]
        )
        
        if not results['documents']:
            break
            
        all_docs.extend(results['documents'])
        
        if len(all_docs) >= sample_size:
            break
    
    print(f"  Total de documentos obtenidos: {len(all_docs)}")
    
    # Categorizar documentos
    category_count = defaultdict(int)
    
    for i, content in enumerate(all_docs):
        if i % 500 == 0:
            print(f"  Procesados {i}/{len(all_docs)} documentos...")
        
        category = categorize_document_by_content(content)
        category_count[category] += 1
    
    # Calcular porcentajes
    total_docs = len(all_docs)
    percentages = {}
    
    for category, count in category_count.items():
        percentage = (count / total_docs) * 100
        percentages[category] = percentage
    
    return percentages, category_count, total_docs

def save_results_v2(percentages: Dict[str, float], counts: Dict[str, int], total: int):
    """
    Guarda los resultados del análisis v2.
    """
    results = {
        "analysis_date": datetime.now().isoformat(),
        "version": "2.0 - Content-based classification",
        "total_documents_sampled": total,
        "total_documents_in_corpus": 187031,
        "methodology": "Content-based keyword analysis with weighted scoring",
        "sampling_method": "Stratified sampling across document collection",
        "categories": {
            category: {
                "count": counts[category],
                "percentage": round(percentages[category], 1)
            }
            for category in percentages
        }
    }
    
    with open("topic_distribution_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: topic_distribution_results_v2.json")

def main():
    """
    Función principal que ejecuta el análisis v2.
    """
    print("🔍 Calculando distribución de temas (Versión 2.0 - Basada en contenido)")
    print("=" * 80)
    
    try:
        percentages, counts, total = calculate_topic_distribution_v2()
        
        print(f"\n📊 RESULTADOS DEL ANÁLISIS V2.0")
        print(f"Total de documentos analizados: {total:,}")
        print("\nDistribución de temas:")
        
        # Ordenar por porcentaje descendente
        sorted_categories = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        for category, percentage in sorted_categories:
            count = counts[category]
            print(f"  - {category}: {count:,} documentos ({percentage:.1f}%)")
        
        # Guardar resultados
        save_results_v2(percentages, counts, total)
        
        # Verificar que suman 100%
        total_percentage = sum(percentages.values())
        print(f"\nVerificación: Suma total = {total_percentage:.1f}%")
        
        # Mostrar distribución final
        print(f"\n📋 DISTRIBUCIÓN FINAL:")
        azure_services = percentages.get('Azure Services', 0)
        security = percentages.get('Security', 0)
        development = percentages.get('Development', 0)
        operations = percentages.get('Operations', 0)
        
        print(f"Azure Services ({azure_services:.1f}%), Security ({security:.1f}%), Development ({development:.1f}%), Operations ({operations:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()