#!/usr/bin/env python3
"""
Análisis Completo de Preguntas para Capítulo 4
Genera todas las estadísticas necesarias con tokenización cl100k_base

Autor: Claude Code
Fecha: 2025-11-02
"""

import json
import tiktoken
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_questions_complete():
    """Analiza todas las preguntas y genera estadísticas completas"""

    print("🔍 Cargando datos de preguntas...")

    # Cargar preguntas (formato JSONL - múltiples objetos JSON por línea)
    questions_file = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/ScrappingMozilla/Logs al 20250602/questions_data.json")

    questions_data = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_questions = len(questions_data)
    print(f"✅ Cargadas {total_questions:,} preguntas")

    # Inicializar tokenizador
    print("🔧 Inicializando tokenizador cl100k_base...")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Calcular longitudes en tokens
    print("📊 Calculando longitudes en tokens...")
    token_lengths = []

    for i, question in enumerate(questions_data, 1):
        if i % 1000 == 0:
            print(f"   Procesadas {i:,}/{total_questions:,} preguntas...")

        # Obtener texto de la pregunta
        question_text = question.get('question_content', '')

        if question_text:
            # Tokenizar
            tokens = tokenizer.encode(question_text)
            token_lengths.append(len(tokens))

    print(f"✅ Procesadas {len(token_lengths):,} preguntas con texto válido")

    # Calcular estadísticas
    print("📈 Calculando estadísticas...")

    token_lengths_array = np.array(token_lengths)

    stats = {
        "analysis_date": datetime.now().isoformat(),
        "version": "3.0 - Complete analysis with cl100k_base tokenization",
        "methodology": "Full analysis of all questions using tiktoken cl100k_base",
        "total_questions": total_questions,
        "questions_with_text": len(token_lengths),
        "question_length_statistics": {
            "mean_tokens": float(np.mean(token_lengths_array)),
            "median_tokens": float(np.median(token_lengths_array)),
            "std_tokens": float(np.std(token_lengths_array)),
            "min_tokens": int(np.min(token_lengths_array)),
            "max_tokens": int(np.max(token_lengths_array)),
            "q25_tokens": float(np.percentile(token_lengths_array, 25)),
            "q75_tokens": float(np.percentile(token_lengths_array, 75)),
            "coefficient_variation": float((np.std(token_lengths_array) / np.mean(token_lengths_array)) * 100)
        },
        "distribution_details": {
            "iqr": float(np.percentile(token_lengths_array, 75) - np.percentile(token_lengths_array, 25)),
            "range": int(np.max(token_lengths_array) - np.min(token_lengths_array))
        }
    }

    # Guardar resultados
    output_file = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Analisis/questions_complete_statistics.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Análisis completo guardado en: {output_file}")

    # Imprimir resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DE ESTADÍSTICAS DE PREGUNTAS")
    print("="*70)
    print(f"\nTotal de preguntas: {stats['total_questions']:,}")
    print(f"Preguntas con texto: {stats['questions_with_text']:,}")

    print("\n📏 LONGITUD EN TOKENS (cl100k_base):")
    q_stats = stats['question_length_statistics']
    print(f"  Media:               {q_stats['mean_tokens']:.1f} tokens")
    print(f"  Mediana:             {q_stats['median_tokens']:.1f} tokens")
    print(f"  Desviación estándar: {q_stats['std_tokens']:.1f} tokens")
    print(f"  Mínimo:              {q_stats['min_tokens']} tokens")
    print(f"  Máximo:              {q_stats['max_tokens']:,} tokens")
    print(f"  Q1 (25%):            {q_stats['q25_tokens']:.1f} tokens")
    print(f"  Q3 (75%):            {q_stats['q75_tokens']:.1f} tokens")
    print(f"  Coef. Variación:     {q_stats['coefficient_variation']:.1f}%")

    dist = stats['distribution_details']
    print(f"\n📊 DISTRIBUCIÓN:")
    print(f"  Rango intercuartílico: {dist['iqr']:.1f} tokens")
    print(f"  Rango total:           {dist['range']:,} tokens")

    print("\n" + "="*70)

    return stats

if __name__ == "__main__":
    try:
        stats = analyze_questions_complete()
        print("\n✅ Análisis completado exitosamente")
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
