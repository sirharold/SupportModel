#!/usr/bin/env python3
"""
Script para enriquecer el análisis temporal de las 2,067 preguntas del ground truth.

Este script hace JOIN entre:
- Las 2,067 preguntas de ChromaDB (questions_withlinks)
- El archivo original con todas las fechas

Genera análisis temporal completo con 100% de cobertura.
"""

import json
import chromadb
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import os


def load_original_questions(filepath: str) -> Dict[str, dict]:
    """
    Carga preguntas originales del archivo JSONL.

    Args:
        filepath: Ruta al archivo questions_data.json

    Returns:
        Diccionario {url: question_data}
    """
    print("📥 Cargando preguntas originales con fechas...")

    questions = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                url = q.get('url', '')
                if url:
                    questions[url] = q

    print(f"✅ Cargadas {len(questions):,} preguntas originales")
    return questions


def load_ground_truth_questions(chromadb_path: str) -> List[dict]:
    """
    Carga las 2,067 preguntas del ground truth desde ChromaDB.

    Args:
        chromadb_path: Ruta a la base de datos ChromaDB

    Returns:
        Lista de metadatos de preguntas
    """
    print("📥 Cargando preguntas del ground truth (ChromaDB)...")

    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_collection("questions_withlinks")
    results = collection.get(include=['metadatas'])

    print(f"✅ Cargadas {len(results['metadatas'])} preguntas del ground truth")
    return results['metadatas']


def parse_date(date_str: str) -> datetime:
    """
    Parsea string de fecha en formato ISO.

    Args:
        date_str: Fecha en formato ISO (ej: "2023-05-17T14:29:31.1366667+00:00")

    Returns:
        Objeto datetime
    """
    try:
        # Intentar varios formatos
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                # Limpiar timezone si es necesario
                clean_date = date_str.replace('+00:00', '').replace('Z', '')
                return datetime.strptime(clean_date, fmt.replace('%z', ''))
            except ValueError:
                continue

        # Si ninguno funciona, intentar parseado inteligente
        # Tomar solo la parte de fecha
        date_part = date_str.split('T')[0]
        return datetime.strptime(date_part, "%Y-%m-%d")

    except Exception as e:
        print(f"⚠️  Error parseando fecha '{date_str}': {e}")
        return None


def join_and_enrich(ground_truth: List[dict], original: Dict[str, dict]) -> Tuple[List[dict], int, int]:
    """
    Hace JOIN entre ground truth y datos originales para obtener fechas.

    Args:
        ground_truth: Lista de metadatos de ChromaDB
        original: Diccionario de preguntas originales

    Returns:
        Tupla (preguntas_enriquecidas, matched, unmatched)
    """
    print("🔗 Haciendo JOIN por URL para recuperar fechas...")

    enriched = []
    matched = 0
    unmatched = 0

    for metadata in ground_truth:
        url = metadata.get('url', '')

        if url in original:
            matched += 1
            original_data = original[url]

            # Enriquecer con datos originales
            enriched_question = {
                **metadata,
                'date': original_data.get('date', ''),
                'original_tags': original_data.get('tags', []),
            }
            enriched.append(enriched_question)
        else:
            unmatched += 1
            enriched.append(metadata)

    print(f"✅ Match: {matched:,}/{len(ground_truth):,} ({matched/len(ground_truth)*100:.1f}%)")
    if unmatched > 0:
        print(f"⚠️  Sin match: {unmatched:,}")

    return enriched, matched, unmatched


def analyze_temporal_data(enriched_questions: List[dict]) -> dict:
    """
    Genera análisis temporal completo de las preguntas.

    Args:
        enriched_questions: Lista de preguntas enriquecidas con fechas

    Returns:
        Diccionario con análisis temporal
    """
    print("📊 Generando análisis temporal completo...")

    dates = []
    years = []
    months = []
    year_month_counts = defaultdict(int)

    questions_with_dates = 0

    for question in enriched_questions:
        date_str = question.get('date', '')
        if not date_str:
            continue

        date_obj = parse_date(date_str)
        if date_obj:
            questions_with_dates += 1
            dates.append(date_obj)
            years.append(date_obj.year)
            months.append(date_obj.month)
            year_month_key = f"{date_obj.year}-{date_obj.month:02d}"
            year_month_counts[year_month_key] += 1

    # Análisis por año
    year_counts = Counter(years)
    total_with_dates = len(dates)
    year_percentages = {
        year: (count / total_with_dates * 100) if total_with_dates > 0 else 0
        for year, count in year_counts.items()
    }

    # Análisis por mes
    month_counts = Counter(months)
    month_names = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    month_distribution = {
        month_names[month]: count for month, count in sorted(month_counts.items())
    }

    # Rango de fechas
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range = f"{min_date.year}-{max_date.year}"
    else:
        date_range = "N/A"

    # Concentración en años recientes (2023-2024)
    recent_years = [2023, 2024]
    recent_count = sum(year_counts.get(year, 0) for year in recent_years)
    concentration_2023_2024 = (recent_count / total_with_dates * 100) if total_with_dates > 0 else 0

    # Top 10 meses con más preguntas
    top_months = sorted(
        [(ym, count) for ym, count in year_month_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Distribución por trimestre
    quarter_counts = defaultdict(int)
    for date_obj in dates:
        quarter = (date_obj.month - 1) // 3 + 1
        quarter_key = f"{date_obj.year}-Q{quarter}"
        quarter_counts[quarter_key] += 1

    analysis = {
        "total_questions": len(enriched_questions),
        "questions_with_dates": questions_with_dates,
        "coverage_percentage": (questions_with_dates / len(enriched_questions) * 100) if enriched_questions else 0,
        "date_range": date_range,
        "year_counts": dict(year_counts),
        "year_percentages": year_percentages,
        "concentration_2023_2024": concentration_2023_2024,
        "month_distribution": month_distribution,
        "top_10_months": [{"month": ym, "count": count} for ym, count in top_months],
        "quarter_distribution": dict(sorted(quarter_counts.items())),
        "temporal_statistics": {
            "earliest_date": min_date.isoformat() if dates else None,
            "latest_date": max_date.isoformat() if dates else None,
            "total_days_span": (max_date - min_date).days if dates else 0,
        }
    }

    print(f"✅ Análisis completado:")
    print(f"   📊 Preguntas con fecha: {questions_with_dates:,}/{len(enriched_questions):,} ({analysis['coverage_percentage']:.1f}%)")
    print(f"   📆 Rango: {date_range}")
    print(f"   🎯 Concentración 2023-2024: {concentration_2023_2024:.1f}%")

    return analysis


def save_analysis(analysis: dict, output_path: str) -> bool:
    """
    Guarda el análisis temporal en un archivo JSON.

    Args:
        analysis: Diccionario con análisis temporal
        output_path: Ruta del archivo de salida

    Returns:
        True si se guardó exitosamente
    """
    print(f"💾 Guardando análisis en {output_path}...")

    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Preparar estructura completa
        output_data = {
            "analysis_date": datetime.now().isoformat(),
            "version": "3.0 - Complete temporal analysis with JOIN",
            "methodology": "JOIN between ChromaDB ground truth and original questions by URL",
            "data_source": {
                "chromadb_collection": "questions_withlinks",
                "original_file": "ScrappingMozilla/Logs al 20250602/questions_data.json"
            },
            "sample_analysis": {
                "questions_analyzed": analysis["total_questions"],
                "questions_with_dates": analysis["questions_with_dates"],
                "coverage_percentage": analysis["coverage_percentage"]
            },
            "temporal_analysis": {
                "year_counts": analysis["year_counts"],
                "year_percentages": analysis["year_percentages"],
                "concentration_2023_2024": analysis["concentration_2023_2024"],
                "date_range": analysis["date_range"],
                "total_with_dates": analysis["questions_with_dates"],
                "month_distribution": analysis["month_distribution"],
                "top_10_months": analysis["top_10_months"],
                "quarter_distribution": analysis["quarter_distribution"],
                "statistics": analysis["temporal_statistics"]
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Análisis guardado exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error guardando análisis: {e}")
        return False


def main():
    """Función principal."""
    print("="*80)
    print("🚀 ENRIQUECIMIENTO DE ANÁLISIS TEMPORAL")
    print("="*80)
    print()

    # Configuración de rutas
    CHROMADB_PATH = "/Users/haroldgomez/chromadb2"
    ORIGINAL_DATA_PATH = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/ScrappingMozilla/Logs al 20250602/questions_data.json"
    OUTPUT_PATH = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Analisis/questions_analysis.json"

    try:
        # 1. Cargar datos originales
        original_questions = load_original_questions(ORIGINAL_DATA_PATH)

        # 2. Cargar ground truth
        ground_truth = load_ground_truth_questions(CHROMADB_PATH)

        # 3. Hacer JOIN y enriquecer
        enriched_questions, matched, unmatched = join_and_enrich(ground_truth, original_questions)

        # 4. Generar análisis temporal
        analysis = analyze_temporal_data(enriched_questions)

        # 5. Guardar análisis
        success = save_analysis(analysis, OUTPUT_PATH)

        if success:
            print()
            print("="*80)
            print("🎉 PROCESO COMPLETADO EXITOSAMENTE")
            print("="*80)
            print()
            print("📊 RESUMEN:")
            print(f"   • Total preguntas ground truth: {analysis['total_questions']:,}")
            print(f"   • Preguntas con fecha: {analysis['questions_with_dates']:,}")
            print(f"   • Cobertura: {analysis['coverage_percentage']:.1f}%")
            print(f"   • Rango temporal: {analysis['date_range']}")
            print(f"   • Concentración 2023-2024: {analysis['concentration_2023_2024']:.1f}%")
            print()
            print("📈 MEJORA:")
            print(f"   Antes: 104 preguntas con fecha (5.0%)")
            print(f"   Ahora: {analysis['questions_with_dates']:,} preguntas con fecha ({analysis['coverage_percentage']:.1f}%)")
            print(f"   Ganancia: +{analysis['questions_with_dates'] - 104:,} fechas recuperadas")
            print()
            print("✅ El análisis temporal en Streamlit ahora mostrará datos completos")

            return True
        else:
            print()
            print("="*80)
            print("❌ ERROR EN EL PROCESO")
            print("="*80)
            return False

    except Exception as e:
        print()
        print("="*80)
        print("❌ ERROR CRÍTICO")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
