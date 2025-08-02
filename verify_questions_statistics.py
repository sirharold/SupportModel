#!/usr/bin/env python3
"""
Script para verificar estadísticas de preguntas y respuestas del corpus de Q&A.
Calcula longitudes promedio, desviaciones estándar y distribución temporal.

Autor: Harold Gómez
Fecha: 2025-08-01
"""

import chromadb
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import tiktoken
import re
from collections import defaultdict

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Cuenta tokens usando el tokenizer de OpenAI.
    """
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except:
        # Fallback: aproximación usando palabras
        return len(text.split()) * 1.3

def extract_date_from_text(text: str) -> str:
    """
    Extrae fecha del texto de preguntas/respuestas.
    Busca patrones de fecha comunes.
    """
    # Patrones de fecha más comunes
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # 2023-05-15
        r'(\d{2}/\d{2}/\d{4})',  # 05/15/2023
        r'(\d{1,2}/\d{1,2}/\d{4})',  # 5/15/2023
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}',  # May 15, 2023
        r'\b(202[0-5])\b'  # Solo año 2020-2025
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Normalizar diferentes formatos
            if re.match(r'\d{4}$', date_str):  # Solo año
                return f"{date_str}-01-01"
            elif re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                return date_str
            # Agregar más normalizaciones si es necesario
    
    return None

def analyze_questions_statistics(persist_directory: str = "/Users/haroldgomez/chromadb2", 
                                sample_size: int = 5000) -> Dict:
    """
    Analiza las estadísticas de preguntas y respuestas.
    """
    # Conectar a ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Usar la colección de preguntas con links (tiene más metadatos)
    collection_name = "questions_withlinks"
    collection = client.get_collection(name=collection_name)
    
    print(f"📊 Analizando estadísticas de preguntas")
    print(f"  Colección: {collection_name}")
    print(f"  Muestra: {sample_size:,} preguntas")
    
    # Obtener preguntas
    results = collection.get(
        limit=sample_size,
        include=["metadatas", "documents"]
    )
    
    if not results['documents']:
        print("❌ No se encontraron documentos en la colección")
        return {}
    
    print(f"  Total obtenido: {len(results['documents']):,} preguntas")
    
    # Analizar cada pregunta
    question_lengths = []
    answer_lengths = []
    dates_found = []
    
    for i, (metadata, content) in enumerate(zip(results['metadatas'], results['documents'])):
        if i % 500 == 0:
            print(f"  Procesadas {i:,} preguntas...")
        
        # El content debería contener la pregunta
        question_text = content
        question_length = count_tokens(question_text)
        question_lengths.append(question_length)
        
        # Buscar respuesta en metadata
        answer_text = ""
        if metadata:
            # Intentar diferentes campos para la respuesta
            answer_fields = ['answer', 'response', 'content', 'description']
            for field in answer_fields:
                if field in metadata and metadata[field]:
                    answer_text = str(metadata[field])
                    break
        
        if answer_text:
            answer_length = count_tokens(answer_text)
            answer_lengths.append(answer_length)
        
        # Extraer fecha del contenido o metadata
        date_found = None
        if metadata and 'date' in metadata:
            date_found = metadata['date']
        elif metadata and 'timestamp' in metadata:
            date_found = metadata['timestamp']
        else:
            # Buscar fecha en el texto
            date_found = extract_date_from_text(question_text)
            if not date_found and answer_text:
                date_found = extract_date_from_text(answer_text)
        
        if date_found:
            dates_found.append(date_found)
    
    # Calcular estadísticas de preguntas
    question_stats = {
        'mean': np.mean(question_lengths),
        'std': np.std(question_lengths),
        'median': np.median(question_lengths),
        'min': np.min(question_lengths),
        'max': np.max(question_lengths),
        'q25': np.percentile(question_lengths, 25),
        'q75': np.percentile(question_lengths, 75),
        'count': len(question_lengths)
    }
    
    # Calcular estadísticas de respuestas (si existen)
    answer_stats = None
    if answer_lengths:
        answer_stats = {
            'mean': np.mean(answer_lengths),
            'std': np.std(answer_lengths),
            'median': np.median(answer_lengths),
            'min': np.min(answer_lengths),
            'max': np.max(answer_lengths),
            'q25': np.percentile(answer_lengths, 25),
            'q75': np.percentile(answer_lengths, 75),
            'count': len(answer_lengths)
        }
    
    # Análisis temporal
    temporal_analysis = analyze_temporal_distribution(dates_found)
    
    return {
        'question_stats': question_stats,
        'answer_stats': answer_stats,
        'temporal_analysis': temporal_analysis,
        'total_questions_analyzed': len(results['documents']),
        'questions_with_answers': len(answer_lengths),
        'questions_with_dates': len(dates_found)
    }

def analyze_temporal_distribution(dates: List[str]) -> Dict:
    """
    Analiza la distribución temporal de las preguntas.
    """
    if not dates:
        return {'error': 'No dates found'}
    
    # Contar por año
    year_counts = defaultdict(int)
    valid_dates = 0
    
    for date_str in dates:
        try:
            # Extraer año
            if isinstance(date_str, str):
                year_match = re.search(r'(202[0-5])', date_str)
                if year_match:
                    year = int(year_match.group(1))
                    year_counts[year] += 1
                    valid_dates += 1
        except:
            continue
    
    if not year_counts:
        return {'error': 'No valid years found'}
    
    # Calcular porcentajes
    total = sum(year_counts.values())
    year_percentages = {year: (count / total) * 100 
                       for year, count in year_counts.items()}
    
    # Calcular concentración en 2023-2024
    concentration_2023_2024 = (year_percentages.get(2023, 0) + 
                              year_percentages.get(2024, 0))
    
    return {
        'year_counts': dict(year_counts),
        'year_percentages': year_percentages,
        'concentration_2023_2024': concentration_2023_2024,
        'date_range': f"{min(year_counts.keys())}-{max(year_counts.keys())}",
        'total_with_dates': valid_dates,
        'total_dates_processed': len(dates)
    }

def examine_questions_structure(persist_directory: str = "/Users/haroldgomez/chromadb2"):
    """
    Examina la estructura de la colección de preguntas.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    print(f"📊 EXAMINANDO ESTRUCTURA DE PREGUNTAS")
    print("=" * 60)
    
    # Probar diferentes colecciones
    collections_to_check = ["questions_withlinks", "questions_ada", "questions_mpnet"]
    
    for collection_name in collections_to_check:
        try:
            collection = client.get_collection(name=collection_name)
            results = collection.get(limit=3, include=["metadatas", "documents"])
            
            print(f"\n📄 Colección: {collection_name}")
            print(f"  Documentos obtenidos: {len(results['documents'])}")
            
            for i, (metadata, content) in enumerate(zip(results['metadatas'], results['documents'])):
                print(f"\n  Ejemplo {i+1}:")
                print(f"    Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
                print(f"    Content preview: {content[:100]}...")
                if metadata:
                    for key, value in list(metadata.items())[:3]:
                        print(f"    {key}: {str(value)[:100]}...")
                
        except Exception as e:
            print(f"\n❌ Error con colección {collection_name}: {e}")

def save_questions_analysis(stats: Dict):
    """
    Guarda el análisis de preguntas en un archivo JSON.
    """
    results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "questions_and_answers_verification",
        "methodology": "Token counting using tiktoken (cl100k_base)",
        "sample_analysis": {
            "questions_analyzed": stats['total_questions_analyzed'],
            "questions_with_answers": stats['questions_with_answers'],
            "questions_with_dates": stats['questions_with_dates']
        },
        "question_statistics": {
            "mean_tokens": round(stats['question_stats']['mean'], 1),
            "std_tokens": round(stats['question_stats']['std'], 1),
            "median_tokens": round(stats['question_stats']['median'], 1),
            "min_tokens": int(stats['question_stats']['min']),
            "max_tokens": int(stats['question_stats']['max']),
            "q25_tokens": round(stats['question_stats']['q25'], 1),
            "q75_tokens": round(stats['question_stats']['q75'], 1)
        }
    }
    
    if stats['answer_stats']:
        results["answer_statistics"] = {
            "mean_tokens": round(stats['answer_stats']['mean'], 1),
            "std_tokens": round(stats['answer_stats']['std'], 1),
            "median_tokens": round(stats['answer_stats']['median'], 1),
            "min_tokens": int(stats['answer_stats']['min']),
            "max_tokens": int(stats['answer_stats']['max']),
            "q25_tokens": round(stats['answer_stats']['q25'], 1),
            "q75_tokens": round(stats['answer_stats']['q75'], 1)
        }
    
    if 'temporal_analysis' in stats:
        results["temporal_analysis"] = stats['temporal_analysis']
    
    with open("questions_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Análisis guardado en: questions_analysis.json")

def compare_with_stated_stats(question_stats: Dict, answer_stats: Dict, temporal: Dict):
    """
    Compara con las estadísticas declaradas en el documento.
    """
    print(f"\n📋 COMPARACIÓN CON ESTADÍSTICAS DECLARADAS")
    print("=" * 60)
    
    # Estadísticas declaradas
    stated_question_mean = 127.3
    stated_question_std = 76.2
    stated_answer_mean = 289.7
    stated_answer_std = 194.3
    stated_temporal_concentration = 67.8
    
    # Comparación de preguntas
    print(f"\n❓ ESTADÍSTICAS DE PREGUNTAS:")
    print(f"  Declarado:  Media = {stated_question_mean} tokens, σ = {stated_question_std}")
    print(f"  Calculado:  Media = {question_stats['mean']:.1f} tokens, σ = {question_stats['std']:.1f}")
    
    q_mean_diff = abs(question_stats['mean'] - stated_question_mean)
    q_std_diff = abs(question_stats['std'] - stated_question_std)
    
    print(f"  Diferencia: Media = {q_mean_diff:.1f} tokens, σ = {q_std_diff:.1f}")
    
    if q_mean_diff < 20 and q_std_diff < 15:
        print(f"  ✅ Las estadísticas de preguntas son consistentes")
    else:
        print(f"  ⚠️  Las estadísticas de preguntas difieren")
    
    # Comparación de respuestas
    if answer_stats:
        print(f"\n💬 ESTADÍSTICAS DE RESPUESTAS:")
        print(f"  Declarado:  Media = {stated_answer_mean} tokens, σ = {stated_answer_std}")
        print(f"  Calculado:  Media = {answer_stats['mean']:.1f} tokens, σ = {answer_stats['std']:.1f}")
        
        a_mean_diff = abs(answer_stats['mean'] - stated_answer_mean)
        a_std_diff = abs(answer_stats['std'] - stated_answer_std)
        
        print(f"  Diferencia: Media = {a_mean_diff:.1f} tokens, σ = {a_std_diff:.1f}")
        
        if a_mean_diff < 50 and a_std_diff < 30:
            print(f"  ✅ Las estadísticas de respuestas son consistentes")
        else:
            print(f"  ⚠️  Las estadísticas de respuestas difieren")
    else:
        print(f"\n💬 ESTADÍSTICAS DE RESPUESTAS:")
        print(f"  ❌ No se encontraron respuestas en los datos")
    
    # Comparación temporal
    if temporal and 'concentration_2023_2024' in temporal:
        print(f"\n📅 DISTRIBUCIÓN TEMPORAL:")
        print(f"  Declarado:  Concentración 2023-2024 = {stated_temporal_concentration}%")
        print(f"  Calculado:  Concentración 2023-2024 = {temporal['concentration_2023_2024']:.1f}%")
        
        temporal_diff = abs(temporal['concentration_2023_2024'] - stated_temporal_concentration)
        print(f"  Diferencia: {temporal_diff:.1f}%")
        
        if temporal_diff < 10:
            print(f"  ✅ La distribución temporal es consistente")
        else:
            print(f"  ⚠️  La distribución temporal difiere")
    else:
        print(f"\n📅 DISTRIBUCIÓN TEMPORAL:")
        print(f"  ❌ No se pudieron extraer fechas suficientes")

def main():
    """
    Función principal que ejecuta la verificación de estadísticas de preguntas.
    """
    print("🔍 Verificando estadísticas de preguntas y respuestas")
    print("=" * 70)
    
    try:
        # Primero examinar estructura
        examine_questions_structure()
        
        # Analizar estadísticas
        stats = analyze_questions_statistics()
        
        if not stats:
            print("❌ No se pudieron obtener estadísticas")
            return
        
        # Mostrar resultados
        question_stats = stats['question_stats']
        answer_stats = stats['answer_stats']
        temporal = stats['temporal_analysis']
        
        print(f"\n📊 RESULTADOS DEL ANÁLISIS")
        print(f"Preguntas analizadas: {stats['total_questions_analyzed']:,}")
        print(f"Preguntas con respuestas: {stats['questions_with_answers']:,}")
        print(f"Preguntas con fechas: {stats['questions_with_dates']:,}")
        
        print(f"\n❓ Estadísticas de Preguntas:")
        print(f"  Media: {question_stats['mean']:.1f} tokens")
        print(f"  Desviación estándar: {question_stats['std']:.1f} tokens")
        print(f"  Mediana: {question_stats['median']:.1f} tokens")
        print(f"  Rango: {question_stats['min']:.0f} - {question_stats['max']:,.0f} tokens")
        
        if answer_stats:
            print(f"\n💬 Estadísticas de Respuestas:")
            print(f"  Media: {answer_stats['mean']:.1f} tokens")
            print(f"  Desviación estándar: {answer_stats['std']:.1f} tokens")
            print(f"  Mediana: {answer_stats['median']:.1f} tokens")
            print(f"  Rango: {answer_stats['min']:.0f} - {answer_stats['max']:,.0f} tokens")
        
        if temporal and 'year_percentages' in temporal:
            print(f"\n📅 Distribución Temporal:")
            for year, percentage in sorted(temporal['year_percentages'].items()):
                print(f"  {year}: {percentage:.1f}%")
            print(f"  Concentración 2023-2024: {temporal['concentration_2023_2024']:.1f}%")
        
        # Comparar con estadísticas declaradas
        compare_with_stated_stats(question_stats, answer_stats, temporal)
        
        # Guardar resultados
        save_questions_analysis(stats)
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()