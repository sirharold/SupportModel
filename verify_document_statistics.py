#!/usr/bin/env python3
"""
Script para verificar estadísticas de longitud de documentos y chunks del corpus.
Calcula longitudes promedio, desviaciones estándar y otros estadísticos descriptivos.

Autor: Harold Gómez
Fecha: 2025-08-01
"""

import chromadb
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
import tiktoken

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Cuenta tokens usando el tokenizer de OpenAI.
    
    Args:
        text: Texto a tokenizar
        model: Modelo de tokenizer a usar
        
    Returns:
        Número de tokens
    """
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except:
        # Fallback: aproximación usando palabras
        return len(text.split()) * 1.3  # Factor de conversión aproximado

def analyze_document_lengths(persist_directory: str = "/Users/haroldgomez/chromadb2", 
                           sample_size: int = 10000) -> Dict:
    """
    Analiza las longitudes de documentos en el corpus.
    
    Args:
        persist_directory: Directorio de ChromaDB
        sample_size: Número de documentos a analizar
        
    Returns:
        Diccionario con estadísticas de longitud
    """
    # Conectar a ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name="docs_ada")
    
    print(f"📊 Analizando longitudes de documentos")
    print(f"  Muestra: {sample_size:,} documentos")
    
    # Obtener documentos
    chunk_lengths = []
    documents_by_source = {}
    
    # Obtener datos en batches
    batch_size = 1000
    total_processed = 0
    
    for offset in range(0, sample_size, batch_size):
        batch_limit = min(batch_size, sample_size - offset)
        
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
            
            # Agrupar por documento original (link en lugar de source)
            source = metadata.get('link', metadata.get('source', 'unknown'))
            if source not in documents_by_source:
                documents_by_source[source] = []
            documents_by_source[source].append(chunk_length)
        
        total_processed += len(results['documents'])
        
        if total_processed % 2000 == 0:
            print(f"  Procesados {total_processed:,} chunks...")
    
    print(f"  Total procesado: {total_processed:,} chunks")
    print(f"  Documentos únicos encontrados: {len(documents_by_source):,}")
    
    # Calcular estadísticas de chunks
    chunk_stats = {
        'mean': np.mean(chunk_lengths),
        'std': np.std(chunk_lengths),
        'median': np.median(chunk_lengths),
        'min': np.min(chunk_lengths),
        'max': np.max(chunk_lengths),
        'q25': np.percentile(chunk_lengths, 25),
        'q75': np.percentile(chunk_lengths, 75),
        'count': len(chunk_lengths)
    }
    
    # Calcular estadísticas de documentos originales
    document_lengths = []
    for source, chunks in documents_by_source.items():
        if source != 'unknown':  # Excluir documentos sin fuente
            total_length = sum(chunks)
            document_lengths.append(total_length)
    
    if document_lengths:
        document_stats = {
            'mean': np.mean(document_lengths),
            'std': np.std(document_lengths),
            'median': np.median(document_lengths),
            'min': np.min(document_lengths),
            'max': np.max(document_lengths),
            'q25': np.percentile(document_lengths, 25),
            'q75': np.percentile(document_lengths, 75),
            'count': len(document_lengths)
        }
    else:
        document_stats = None
    
    return {
        'chunk_stats': chunk_stats,
        'document_stats': document_stats,
        'total_chunks_analyzed': total_processed,
        'unique_documents': len(documents_by_source)
    }

def save_length_analysis(stats: Dict):
    """
    Guarda el análisis de longitudes en un archivo JSON.
    """
    results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "document_length_verification",
        "methodology": "Token counting using tiktoken (cl100k_base)",
        "corpus_info": {
            "total_chunks_in_corpus": 187031,
            "total_unique_documents": 62417
        },
        "sample_analysis": {
            "chunks_analyzed": stats['total_chunks_analyzed'],
            "unique_documents_found": stats['unique_documents']
        },
        "chunk_statistics": {
            "mean_tokens": round(stats['chunk_stats']['mean'], 1),
            "std_tokens": round(stats['chunk_stats']['std'], 1),
            "median_tokens": round(stats['chunk_stats']['median'], 1),
            "min_tokens": int(stats['chunk_stats']['min']),
            "max_tokens": int(stats['chunk_stats']['max']),
            "q25_tokens": round(stats['chunk_stats']['q25'], 1),
            "q75_tokens": round(stats['chunk_stats']['q75'], 1)
        }
    }
    
    if stats['document_stats']:
        results["document_statistics"] = {
            "mean_tokens": round(stats['document_stats']['mean'], 1),
            "std_tokens": round(stats['document_stats']['std'], 1),
            "median_tokens": round(stats['document_stats']['median'], 1),
            "min_tokens": int(stats['document_stats']['min']),
            "max_tokens": int(stats['document_stats']['max']),
            "q25_tokens": round(stats['document_stats']['q25'], 1),
            "q75_tokens": round(stats['document_stats']['q75'], 1)
        }
    
    with open("document_length_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Análisis guardado en: document_length_analysis.json")

def compare_with_stated_stats(chunk_stats: Dict, document_stats: Dict):
    """
    Compara con las estadísticas declaradas en el documento.
    """
    print(f"\n📋 COMPARACIÓN CON ESTADÍSTICAS DECLARADAS")
    print("=" * 60)
    
    # Estadísticas declaradas
    stated_chunk_mean = 342.7
    stated_chunk_std = 127.3
    stated_doc_mean = 1247.2
    stated_doc_std = 891.5
    
    # Comparación de chunks
    print(f"\n📄 ESTADÍSTICAS DE CHUNKS:")
    print(f"  Declarado:  Media = {stated_chunk_mean} tokens, σ = {stated_chunk_std}")
    print(f"  Calculado:  Media = {chunk_stats['mean']:.1f} tokens, σ = {chunk_stats['std']:.1f}")
    
    chunk_mean_diff = abs(chunk_stats['mean'] - stated_chunk_mean)
    chunk_std_diff = abs(chunk_stats['std'] - stated_chunk_std)
    
    print(f"  Diferencia: Media = {chunk_mean_diff:.1f} tokens, σ = {chunk_std_diff:.1f}")
    
    if chunk_mean_diff < 50 and chunk_std_diff < 20:
        print(f"  ✅ Las estadísticas de chunks son consistentes")
    else:
        print(f"  ⚠️  Las estadísticas de chunks difieren significativamente")
    
    # Comparación de documentos
    if document_stats:
        print(f"\n📚 ESTADÍSTICAS DE DOCUMENTOS ORIGINALES:")
        print(f"  Declarado:  Media = {stated_doc_mean} tokens, σ = {stated_doc_std}")
        print(f"  Calculado:  Media = {document_stats['mean']:.1f} tokens, σ = {document_stats['std']:.1f}")
        
        doc_mean_diff = abs(document_stats['mean'] - stated_doc_mean)
        doc_std_diff = abs(document_stats['std'] - stated_doc_std)
        
        print(f"  Diferencia: Media = {doc_mean_diff:.1f} tokens, σ = {doc_std_diff:.1f}")
        
        if doc_mean_diff < 200 and doc_std_diff < 150:
            print(f"  ✅ Las estadísticas de documentos son consistentes")
        else:
            print(f"  ⚠️  Las estadísticas de documentos difieren significativamente")
    else:
        print(f"\n📚 ESTADÍSTICAS DE DOCUMENTOS ORIGINALES:")
        print(f"  ❌ No se pudieron calcular (problemas con metadatos de fuente)")

def main():
    """
    Función principal que ejecuta la verificación de estadísticas.
    """
    print("🔍 Verificando estadísticas de longitud de documentos y chunks")
    print("=" * 70)
    
    try:
        # Analizar longitudes
        stats = analyze_document_lengths()
        
        # Mostrar resultados
        chunk_stats = stats['chunk_stats']
        document_stats = stats['document_stats']
        
        print(f"\n📊 RESULTADOS DEL ANÁLISIS")
        print(f"Chunks analizados: {stats['total_chunks_analyzed']:,}")
        print(f"Documentos únicos: {stats['unique_documents']:,}")
        
        print(f"\n📄 Estadísticas de Chunks:")
        print(f"  Media: {chunk_stats['mean']:.1f} tokens")
        print(f"  Desviación estándar: {chunk_stats['std']:.1f} tokens")
        print(f"  Mediana: {chunk_stats['median']:.1f} tokens")
        print(f"  Rango: {chunk_stats['min']:.0f} - {chunk_stats['max']:,.0f} tokens")
        print(f"  Q25-Q75: {chunk_stats['q25']:.0f} - {chunk_stats['q75']:.0f} tokens")
        
        if document_stats:
            print(f"\n📚 Estadísticas de Documentos Originales:")
            print(f"  Media: {document_stats['mean']:.1f} tokens")
            print(f"  Desviación estándar: {document_stats['std']:.1f} tokens")
            print(f"  Mediana: {document_stats['median']:.1f} tokens")
            print(f"  Rango: {document_stats['min']:.0f} - {document_stats['max']:,.0f} tokens")
            print(f"  Q25-Q75: {document_stats['q25']:.0f} - {document_stats['q75']:.0f} tokens")
        
        # Comparar con estadísticas declaradas
        compare_with_stated_stats(chunk_stats, document_stats)
        
        # Guardar resultados
        save_length_analysis(stats)
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()