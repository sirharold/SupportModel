#!/usr/bin/env python3
"""
Script para hacer análisis exploratorio COMPLETO del corpus con todos los 187,031 chunks.
Actualiza las estadísticas para el capítulo 4.

Autor: Harold Gómez
Fecha: 2025-08-02
"""

import chromadb
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
import tiktoken
from collections import Counter, defaultdict
import re

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Cuenta tokens usando el tokenizer de OpenAI."""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except:
        return len(text.split()) * 1.3

def analyze_full_corpus(persist_directory: str = "/Users/haroldgomez/chromadb2") -> Dict:
    """
    Analiza TODOS los chunks del corpus (187,031).
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name="docs_ada")
    
    total_count = collection.count()
    print(f"📊 Analizando CORPUS COMPLETO")
    print(f"  Total de chunks: {total_count:,}")
    
    chunk_lengths = []
    documents_by_source = defaultdict(list)
    topic_keywords = defaultdict(int)
    
    # Keywords para clasificación temática
    topic_classifiers = {
        'Development': [
            'api', 'sdk', 'code', 'development', 'programming', 'function', 'method',
            'class', 'library', 'framework', 'endpoint', 'request', 'response',
            'json', 'rest', 'http', 'javascript', 'python', 'java', 'dotnet',
            'nodejs', 'react', 'angular', 'vue', 'github', 'git', 'repository',
            'package', 'npm', 'pip', 'maven', 'gradle', 'build', 'deploy',
            'devops', 'ci/cd', 'pipeline', 'docker', 'kubernetes', 'container'
        ],
        'Operations': [
            'monitoring', 'logs', 'metrics', 'alerts', 'dashboard', 'troubleshooting',
            'performance', 'optimization', 'scaling', 'load', 'availability',
            'reliability', 'maintenance', 'backup', 'restore', 'disaster',
            'recovery', 'automation', 'script', 'powershell', 'cli', 'command',
            'configuration', 'settings', 'parameters', 'environment', 'production',
            'staging', 'development', 'testing', 'deployment', 'rollback'
        ],
        'Security': [
            'authentication', 'authorization', 'security', 'permissions', 'roles',
            'identity', 'access', 'token', 'certificate', 'ssl', 'tls', 'https',
            'encryption', 'decrypt', 'cryptography', 'hash', 'signature',
            'compliance', 'gdpr', 'hipaa', 'sox', 'pci', 'audit', 'firewall',
            'network', 'vpn', 'private', 'public', 'key', 'secret', 'vault',
            'password', 'multi-factor', 'mfa', 'oauth', 'saml', 'active directory'
        ],
        'Azure Services': [
            'azure', 'microsoft', 'subscription', 'resource group', 'tenant',
            'storage', 'compute', 'virtual machine', 'app service', 'function app',
            'logic apps', 'service bus', 'event hub', 'cosmos db', 'sql database',
            'redis', 'search', 'cognitive', 'machine learning', 'ai', 'bot',
            'iot', 'stream analytics', 'data factory', 'synapse', 'power bi',
            'active directory', 'key vault', 'application gateway', 'load balancer'
        ]
    }
    
    # Procesar en batches de 1000
    batch_size = 1000
    total_processed = 0
    
    for offset in range(0, total_count, batch_size):
        batch_limit = min(batch_size, total_count - offset)
        
        results = collection.get(
            limit=batch_limit,
            offset=offset,
            include=["metadatas", "documents"]
        )
        
        if not results['documents']:
            break
        
        # Procesar batch
        for metadata, content in zip(results['metadatas'], results['documents']):
            # Longitud del chunk
            chunk_length = count_tokens(content)
            chunk_lengths.append(chunk_length)
            
            # Agrupar por documento original
            source = metadata.get('link', metadata.get('source', 'unknown'))
            if source != 'unknown':
                documents_by_source[source].append(chunk_length)
            
            # Clasificación temática
            content_lower = content.lower()
            chunk_topics = {}
            
            for topic, keywords in topic_classifiers.items():
                score = sum(1 for keyword in keywords if keyword in content_lower)
                chunk_topics[topic] = score
            
            # Asignar a la categoría con mayor score
            if chunk_topics and max(chunk_topics.values()) > 0:
                best_topic = max(chunk_topics, key=chunk_topics.get)
                topic_keywords[best_topic] += 1
        
        total_processed += len(results['documents'])
        
        if total_processed % 10000 == 0:
            print(f"  Procesados {total_processed:,} chunks...")
    
    print(f"  ✅ Total procesado: {total_processed:,} chunks")
    print(f"  📚 Documentos únicos encontrados: {len(documents_by_source):,}")
    
    # Calcular estadísticas de chunks
    chunk_stats = {
        'mean': np.mean(chunk_lengths),
        'std': np.std(chunk_lengths),
        'median': np.median(chunk_lengths),
        'min': np.min(chunk_lengths),
        'max': np.max(chunk_lengths),
        'q25': np.percentile(chunk_lengths, 25),
        'q75': np.percentile(chunk_lengths, 75),
        'count': len(chunk_lengths),
        'cv': np.std(chunk_lengths) / np.mean(chunk_lengths) * 100  # Coeficiente de variación
    }
    
    # Calcular estadísticas de documentos originales
    document_lengths = []
    for source, chunks in documents_by_source.items():
        total_length = sum(chunks)
        document_lengths.append(total_length)
    
    document_stats = {
        'mean': np.mean(document_lengths),
        'std': np.std(document_lengths),
        'median': np.median(document_lengths),
        'min': np.min(document_lengths),
        'max': np.max(document_lengths),
        'q25': np.percentile(document_lengths, 25),
        'q75': np.percentile(document_lengths, 75),
        'count': len(document_lengths),
        'cv': np.std(document_lengths) / np.mean(document_lengths) * 100
    }
    
    # Distribución temática
    total_classified = sum(topic_keywords.values())
    topic_distribution = {}
    for topic, count in topic_keywords.items():
        topic_distribution[topic] = {
            'count': count,
            'percentage': (count / total_classified * 100) if total_classified > 0 else 0
        }
    
    return {
        'chunk_stats': chunk_stats,
        'document_stats': document_stats,
        'topic_distribution': topic_distribution,
        'total_chunks_analyzed': total_processed,
        'unique_documents': len(documents_by_source),
        'total_classified_chunks': total_classified
    }

def save_full_analysis(stats: Dict):
    """Guarda el análisis completo."""
    results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "complete_corpus_analysis",
        "methodology": "Full corpus analysis with token counting using tiktoken (cl100k_base)",
        "corpus_info": {
            "total_chunks_analyzed": stats['total_chunks_analyzed'],
            "total_unique_documents": stats['unique_documents'],
            "coverage": "100% of corpus"
        },
        "chunk_statistics": {
            "mean_tokens": round(stats['chunk_stats']['mean'], 1),
            "std_tokens": round(stats['chunk_stats']['std'], 1),
            "median_tokens": round(stats['chunk_stats']['median'], 1),
            "min_tokens": int(stats['chunk_stats']['min']),
            "max_tokens": int(stats['chunk_stats']['max']),
            "q25_tokens": round(stats['chunk_stats']['q25'], 1),
            "q75_tokens": round(stats['chunk_stats']['q75'], 1),
            "coefficient_variation": round(stats['chunk_stats']['cv'], 1)
        },
        "document_statistics": {
            "mean_tokens": round(stats['document_stats']['mean'], 1),
            "std_tokens": round(stats['document_stats']['std'], 1),
            "median_tokens": round(stats['document_stats']['median'], 1),
            "min_tokens": int(stats['document_stats']['min']),
            "max_tokens": int(stats['document_stats']['max']),
            "q25_tokens": round(stats['document_stats']['q25'], 1),
            "q75_tokens": round(stats['document_stats']['q75'], 1),
            "coefficient_variation": round(stats['document_stats']['cv'], 1)
        },
        "topic_distribution": stats['topic_distribution'],
        "total_classified_chunks": stats['total_classified_chunks']
    }
    
    with open("Docs/Analisis/full_corpus_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Análisis completo guardado en: Docs/Analisis/full_corpus_analysis.json")

def display_results(stats: Dict):
    """Muestra los resultados del análisis."""
    print(f"\n📊 ANÁLISIS COMPLETO DEL CORPUS")
    print("=" * 70)
    
    print(f"\n📈 COBERTURA:")
    print(f"  Chunks analizados: {stats['total_chunks_analyzed']:,} (100%)")
    print(f"  Documentos únicos: {stats['unique_documents']:,}")
    
    chunk_stats = stats['chunk_stats']
    print(f"\n📄 ESTADÍSTICAS DE CHUNKS (CORPUS COMPLETO):")
    print(f"  Media: {chunk_stats['mean']:.1f} tokens")
    print(f"  Desviación estándar: {chunk_stats['std']:.1f} tokens")
    print(f"  Mediana: {chunk_stats['median']:.1f} tokens")
    print(f"  Rango: {chunk_stats['min']:.0f} - {chunk_stats['max']:,.0f} tokens")
    print(f"  Q25-Q75: {chunk_stats['q25']:.0f} - {chunk_stats['q75']:.0f} tokens")
    print(f"  Coeficiente de variación: {chunk_stats['cv']:.1f}%")
    
    doc_stats = stats['document_stats']
    print(f"\n📚 ESTADÍSTICAS DE DOCUMENTOS ORIGINALES (CORPUS COMPLETO):")
    print(f"  Media: {doc_stats['mean']:.1f} tokens")
    print(f"  Desviación estándar: {doc_stats['std']:.1f} tokens")
    print(f"  Mediana: {doc_stats['median']:.1f} tokens")
    print(f"  Rango: {doc_stats['min']:.0f} - {doc_stats['max']:,.0f} tokens")
    print(f"  Q25-Q75: {doc_stats['q25']:.0f} - {doc_stats['q75']:.0f} tokens")
    print(f"  Coeficiente de variación: {doc_stats['cv']:.1f}%")
    
    print(f"\n🏷️  DISTRIBUCIÓN TEMÁTICA (CORPUS COMPLETO):")
    print(f"  Chunks clasificados: {stats['total_classified_chunks']:,}")
    
    for topic, data in sorted(stats['topic_distribution'].items(), 
                             key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        percentage = data['percentage']
        print(f"  {topic:15}: {count:,} chunks ({percentage:.1f}%)")

def main():
    """Función principal."""
    print("🚀 ANÁLISIS EXPLORATORIO COMPLETO DEL CORPUS")
    print("=" * 70)
    print("Este análisis procesa los 187,031 chunks completos")
    
    try:
        # Analizar corpus completo
        stats = analyze_full_corpus()
        
        # Mostrar resultados
        display_results(stats)
        
        # Guardar resultados
        save_full_analysis(stats)
        
        print(f"\n✨ Análisis completo terminado exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()