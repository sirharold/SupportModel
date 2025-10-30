#!/usr/bin/env python3
"""
Análisis completo de tokens REALES con cl100k_base para todos los chunks
Procesa los 187,031 chunks y genera estadísticas precisas
"""

import pandas as pd
import numpy as np
import tiktoken
from pathlib import Path
import json

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "colab_data"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_chunk_tokens():
    """Analiza tokens reales de todos los chunks"""
    print("=" * 70)
    print("ANÁLISIS DE TOKENS REALES - CORPUS COMPLETO")
    print("=" * 70)

    # Cargar datos
    print("\n📂 Cargando datos de chunks...")
    ada_path = DATA_DIR / "docs_ada_with_embeddings_20250721_123712.parquet"
    df = pd.read_parquet(ada_path)
    print(f"   ✓ Cargados {len(df):,} chunks")

    # Inicializar tokenizador
    print("\n🔧 Inicializando tokenizador cl100k_base...")
    encoding = tiktoken.get_encoding("cl100k_base")
    print("   ✓ Tokenizador listo")

    # Calcular tokens para todos los chunks
    print("\n⏳ Calculando tokens para todos los chunks...")
    print("   (Esto puede tomar varios minutos...)")

    tokens_list = []
    for i, content in enumerate(df['content'].fillna(''), 1):
        if i % 10000 == 0:
            print(f"   Procesados {i:,} / {len(df):,} chunks ({i/len(df)*100:.1f}%)")
        # Ignorar tokens especiales que puedan aparecer en el contenido
        tokens_list.append(len(encoding.encode(content, disallowed_special=())))

    tokens = np.array(tokens_list)
    print(f"   ✓ Tokenización completa: {len(tokens):,} chunks procesados")

    # Calcular estadísticas
    print("\n📊 Calculando estadísticas...")
    stats = {
        'total_chunks': len(tokens),
        'mean': float(np.mean(tokens)),
        'median': float(np.median(tokens)),
        'std': float(np.std(tokens)),
        'min': int(np.min(tokens)),
        'max': int(np.max(tokens)),
        'q1': float(np.quantile(tokens, 0.25)),
        'q3': float(np.quantile(tokens, 0.75)),
        'cv': float(np.std(tokens) / np.mean(tokens) * 100)
    }

    # Guardar estadísticas
    output_file = OUTPUT_DIR / "chunk_token_stats_real.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ Estadísticas guardadas en: {output_file}")

    # Guardar array de tokens para visualizaciones
    tokens_file = OUTPUT_DIR / "chunk_tokens_real.npy"
    np.save(tokens_file, tokens)
    print(f"   ✓ Array de tokens guardado en: {tokens_file}")

    # Mostrar estadísticas
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DE TOKENS REALES (cl100k_base)")
    print("=" * 70)
    print(f"Total de chunks:        {stats['total_chunks']:,}")
    print(f"Media:                  {stats['mean']:.1f} tokens")
    print(f"Mediana:                {stats['median']:.1f} tokens")
    print(f"Desviación estándar:    {stats['std']:.1f} tokens")
    print(f"Mínimo:                 {stats['min']} tokens")
    print(f"Máximo:                 {stats['max']:,} tokens")
    print(f"Q1 (percentil 25):      {stats['q1']:.0f} tokens")
    print(f"Q3 (percentil 75):      {stats['q3']:.0f} tokens")
    print(f"Coef. de variación:     {stats['cv']:.1f}%")
    print("=" * 70)

    return stats, tokens

def analyze_document_tokens():
    """Analiza tokens de documentos completos"""
    print("\n" + "=" * 70)
    print("ANÁLISIS DE DOCUMENTOS COMPLETOS")
    print("=" * 70)

    # Cargar datos
    print("\n📂 Cargando datos de chunks...")
    ada_path = DATA_DIR / "docs_ada_with_embeddings_20250721_123712.parquet"
    df = pd.read_parquet(ada_path)

    # Cargar tokens previamente calculados
    tokens_file = OUTPUT_DIR / "chunk_tokens_real.npy"
    if tokens_file.exists():
        print("   ✓ Usando tokens previamente calculados")
        tokens = np.load(tokens_file)
        df['tokens'] = tokens
    else:
        print("   ⚠ No se encontraron tokens calculados, calcúlalos primero")
        return None

    # Agrupar por documento
    print("\n📊 Agrupando por documento...")
    if 'document' in df.columns:
        doc_col = 'document'
    elif 'doc_id' in df.columns:
        doc_col = 'doc_id'
    elif 'document_id' in df.columns:
        doc_col = 'document_id'
    elif 'url' in df.columns:
        doc_col = 'url'
    else:
        print("   ⚠ No se encontró columna de documento ID")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        return None

    doc_tokens = df.groupby(doc_col)['tokens'].sum()
    doc_tokens_array = np.array(doc_tokens)

    print(f"   ✓ Documentos únicos: {len(doc_tokens_array):,}")

    # Calcular estadísticas
    doc_stats = {
        'total_documents': len(doc_tokens_array),
        'mean': float(np.mean(doc_tokens_array)),
        'median': float(np.median(doc_tokens_array)),
        'std': float(np.std(doc_tokens_array)),
        'min': int(np.min(doc_tokens_array)),
        'max': int(np.max(doc_tokens_array)),
        'q1': float(np.quantile(doc_tokens_array, 0.25)),
        'q3': float(np.quantile(doc_tokens_array, 0.75)),
        'cv': float(np.std(doc_tokens_array) / np.mean(doc_tokens_array) * 100)
    }

    # Guardar estadísticas
    output_file = OUTPUT_DIR / "document_token_stats_real.json"
    with open(output_file, 'w') as f:
        json.dump(doc_stats, f, indent=2)
    print(f"   ✓ Estadísticas guardadas en: {output_file}")

    # Guardar array
    docs_tokens_file = OUTPUT_DIR / "document_tokens_real.npy"
    np.save(docs_tokens_file, doc_tokens_array)
    print(f"   ✓ Array de tokens guardado en: {docs_tokens_file}")

    # Mostrar estadísticas
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DE DOCUMENTOS COMPLETOS (cl100k_base)")
    print("=" * 70)
    print(f"Total de documentos:    {doc_stats['total_documents']:,}")
    print(f"Media:                  {doc_stats['mean']:.1f} tokens")
    print(f"Mediana:                {doc_stats['median']:.1f} tokens")
    print(f"Desviación estándar:    {doc_stats['std']:.1f} tokens")
    print(f"Mínimo:                 {doc_stats['min']} tokens")
    print(f"Máximo:                 {doc_stats['max']:,} tokens")
    print(f"Q1 (percentil 25):      {doc_stats['q1']:.0f} tokens")
    print(f"Q3 (percentil 75):      {doc_stats['q3']:.0f} tokens")
    print(f"Coef. de variación:     {doc_stats['cv']:.1f}%")
    print("=" * 70)

    return doc_stats, doc_tokens_array

if __name__ == "__main__":
    # Analizar chunks
    chunk_stats, chunk_tokens = analyze_chunk_tokens()

    # Analizar documentos
    result = analyze_document_tokens()
    if result:
        doc_stats, doc_tokens = result
    else:
        print("\n⚠ Análisis de documentos omitido")

    print("\n✅ Análisis completo finalizado")
    print(f"📁 Archivos generados en: {OUTPUT_DIR}")
