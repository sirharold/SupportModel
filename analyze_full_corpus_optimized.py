#!/usr/bin/env python3
"""
Script optimizado para análisis del corpus completo usando archivos parquet.
Más eficiente que consultar ChromaDB directamente.

Autor: Harold Gómez  
Fecha: 2025-08-02
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict
import tiktoken

def count_tokens_fast(text: str) -> int:
    """Aproximación rápida de conteo de tokens."""
    # Aproximación: 1 token ≈ 0.75 palabras para inglés técnico
    return int(len(text.split()) * 1.33)

def analyze_parquet_files():
    """Analiza todos los chunks usando archivos parquet."""
    
    # Rutas de archivos parquet
    parquet_files = [
        "colab_data/docs_ada_with_embeddings_20250721_123712.parquet",
        "colab_data/docs_e5large_with_embeddings_20250721_124918.parquet", 
        "colab_data/docs_mpnet_with_embeddings_20250721_125254.parquet",
        "colab_data/docs_minilm_with_embeddings_20250721_125846.parquet"
    ]
    
    print("📊 Analizando corpus completo desde archivos parquet...")
    
    # Usar el primer archivo (todos tienen el mismo contenido textual)
    df = pd.read_parquet(parquet_files[0])
    
    total_chunks = len(df)
    print(f"  Total chunks: {total_chunks:,}")
    
    # Calcular longitudes de tokens (más rápido con aproximación)
    print("  Calculando longitudes...")
    chunk_lengths = df['content'].apply(count_tokens_fast).values
    
    # Agrupar por documento usando 'link'
    print("  Agrupando por documentos...")
    documents_by_link = df.groupby('link')['content'].apply(
        lambda x: sum(count_tokens_fast(content) for content in x)
    ).values
    
    # Clasificación temática simplificada (muestra de 10,000 para velocidad)
    print("  Clasificando temáticamente (muestra)...")
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    topic_classifiers = {
        'Development': ['api', 'sdk', 'code', 'function', 'method', 'class', 'library', 'framework', 'endpoint', 'json', 'rest', 'http', 'javascript', 'python', 'java', 'dotnet', 'github', 'build', 'deploy', 'devops', 'docker', 'kubernetes'],
        'Operations': ['monitoring', 'logs', 'metrics', 'alerts', 'troubleshooting', 'performance', 'scaling', 'availability', 'maintenance', 'backup', 'automation', 'powershell', 'cli', 'configuration', 'deployment'],
        'Security': ['authentication', 'authorization', 'security', 'permissions', 'identity', 'access', 'token', 'certificate', 'ssl', 'encryption', 'compliance', 'firewall', 'network', 'vpn', 'key', 'vault', 'password', 'oauth'],
        'Azure Services': ['azure', 'storage', 'compute', 'virtual machine', 'app service', 'function app', 'logic apps', 'cosmos db', 'sql database', 'cognitive', 'machine learning', 'iot', 'synapse']
    }
    
    topic_counts = defaultdict(int)
    
    for content in sample_df['content']:
        content_lower = content.lower()
        scores = {}
        
        for topic, keywords in topic_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            scores[topic] = score
        
        if scores and max(scores.values()) > 0:
            best_topic = max(scores, key=scores.get)
            topic_counts[best_topic] += 1
    
    # Extrapollar a todo el corpus
    total_classified = sum(topic_counts.values())
    if total_classified > 0:
        scaling_factor = total_chunks / sample_size
        for topic in topic_counts:
            topic_counts[topic] = int(topic_counts[topic] * scaling_factor)
    
    # Calcular estadísticas
    chunk_stats = {
        'mean': np.mean(chunk_lengths),
        'std': np.std(chunk_lengths),
        'median': np.median(chunk_lengths),
        'min': np.min(chunk_lengths),
        'max': np.max(chunk_lengths),
        'q25': np.percentile(chunk_lengths, 25),
        'q75': np.percentile(chunk_lengths, 75),
        'count': len(chunk_lengths),
        'cv': np.std(chunk_lengths) / np.mean(chunk_lengths) * 100
    }
    
    document_stats = {
        'mean': np.mean(documents_by_link),
        'std': np.std(documents_by_link),
        'median': np.median(documents_by_link),
        'min': np.min(documents_by_link),
        'max': np.max(documents_by_link),
        'q25': np.percentile(documents_by_link, 25),
        'q75': np.percentile(documents_by_link, 75),
        'count': len(documents_by_link),
        'cv': np.std(documents_by_link) / np.mean(documents_by_link) * 100
    }
    
    # Distribución temática
    total_estimated = sum(topic_counts.values())
    topic_distribution = {}
    for topic, count in topic_counts.items():
        topic_distribution[topic] = {
            'count': count,
            'percentage': (count / total_estimated * 100) if total_estimated > 0 else 0
        }
    
    return {
        'chunk_stats': chunk_stats,
        'document_stats': document_stats,
        'topic_distribution': topic_distribution,
        'total_chunks_analyzed': total_chunks,
        'unique_documents': len(documents_by_link),
        'sample_size_for_topics': sample_size
    }

def save_and_display(stats: Dict):
    """Guarda y muestra los resultados."""
    
    # Guardar
    results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "complete_corpus_analysis_optimized",
        "methodology": "Fast token counting with parquet files",
        "corpus_info": {
            "total_chunks_analyzed": stats['total_chunks_analyzed'],
            "total_unique_documents": stats['unique_documents'],
            "coverage": "100% of corpus for length stats, 10k sample for topics"
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
        "topic_distribution": stats['topic_distribution']
    }
    
    with open("Docs/Analisis/full_corpus_analysis_final.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Mostrar
    print(f"\n📊 ANÁLISIS COMPLETO DEL CORPUS (OPTIMIZADO)")
    print("=" * 60)
    
    print(f"\n📈 COBERTURA:")
    print(f"  Chunks analizados: {stats['total_chunks_analyzed']:,} (100%)")
    print(f"  Documentos únicos: {stats['unique_documents']:,}")
    
    cs = stats['chunk_stats']
    print(f"\n📄 ESTADÍSTICAS DE CHUNKS:")
    print(f"  Media: {cs['mean']:.1f} tokens")
    print(f"  Mediana: {cs['median']:.1f} tokens")
    print(f"  Desviación estándar: {cs['std']:.1f} tokens")
    print(f"  Q25-Q75: {cs['q25']:.1f} - {cs['q75']:.1f} tokens")
    print(f"  Rango: {cs['min']:.0f} - {cs['max']:,.0f} tokens")
    print(f"  CV: {cs['cv']:.1f}%")
    
    ds = stats['document_stats']
    print(f"\n📚 ESTADÍSTICAS DE DOCUMENTOS:")
    print(f"  Media: {ds['mean']:.1f} tokens")
    print(f"  Mediana: {ds['median']:.1f} tokens")
    print(f"  Desviación estándar: {ds['std']:.1f} tokens")
    print(f"  Q25-Q75: {ds['q25']:.1f} - {ds['q75']:.1f} tokens")
    print(f"  Rango: {ds['min']:.0f} - {ds['max']:,.0f} tokens")
    print(f"  CV: {ds['cv']:.1f}%")
    
    print(f"\n🏷️  DISTRIBUCIÓN TEMÁTICA (extrapolada):")
    for topic, data in sorted(stats['topic_distribution'].items(), 
                             key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        percentage = data['percentage']
        print(f"  {topic:15}: {count:,} chunks ({percentage:.1f}%)")
    
    print(f"\n💾 Resultados guardados en: Docs/Analisis/full_corpus_analysis_final.json")

def main():
    try:
        print("🚀 ANÁLISIS OPTIMIZADO DEL CORPUS COMPLETO")
        stats = analyze_parquet_files()
        save_and_display(stats)
        print(f"\n✨ Análisis completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()