#!/usr/bin/env python3
"""
Script de verificación para el filtro de año/semestre.
Prueba que el filtrado funciona correctamente con datos reales.
"""

import json
from datetime import datetime
from collections import Counter

# Cargar preguntas originales con fechas
original_questions_path = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/ScrappingMozilla/Logs al 20250602/questions_data.json"

print("="*80)
print("🔍 VERIFICACIÓN DE FILTRO DE AÑO/SEMESTRE")
print("="*80)
print()

# Cargar todas las preguntas
all_questions = []
with open(original_questions_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                q = json.loads(line)
                all_questions.append(q)
            except json.JSONDecodeError:
                continue

print(f"✅ Cargadas {len(all_questions):,} preguntas del archivo original")
print()

# Analizar distribución por año y semestre
year_counts = Counter()
year_semester_counts = Counter()

for q in all_questions:
    date_str = q.get('date', '')
    if not date_str:
        continue

    try:
        # Parse fecha
        date_str_clean = date_str.replace('Z', '+00:00')
        if '+' in date_str_clean:
            date_str_clean = date_str_clean.split('+')[0]

        date_obj = datetime.fromisoformat(date_str_clean)
        year = date_obj.year
        month = date_obj.month

        # Contar por año
        year_counts[year] += 1

        # Contar por año.semestre
        if month <= 6:
            year_semester_counts[f"{year}.1"] += 1
        else:
            year_semester_counts[f"{year}.2"] += 1

    except:
        continue

print("📊 DISTRIBUCIÓN POR AÑO:")
print("-" * 80)
for year in sorted(year_counts.keys(), reverse=True):
    count = year_counts[year]
    percentage = (count / len(all_questions) * 100)
    bar = "█" * int(percentage / 2)
    print(f"   {year}: {count:,} preguntas ({percentage:.1f}%) {bar}")
print()

print("📊 DISTRIBUCIÓN POR AÑO.SEMESTRE:")
print("-" * 80)
for period in sorted(year_semester_counts.keys(), reverse=True):
    count = year_semester_counts[period]
    percentage = (count / len(all_questions) * 100)
    print(f"   {period}: {count:,} preguntas ({percentage:.1f}%)")
print()

# Simular filtros
print("🔬 SIMULACIÓN DE FILTROS:")
print("-" * 80)

filters = {
    "2024": lambda y, m: y == 2024,
    "2023.1": lambda y, m: y == 2023 and m <= 6,
    "2023.2": lambda y, m: y == 2023 and m > 6,
    "2022": lambda y, m: y == 2022,
    "2020": lambda y, m: y == 2020
}

for filter_name, filter_func in filters.items():
    filtered_count = 0

    for q in all_questions:
        date_str = q.get('date', '')
        if not date_str:
            continue

        try:
            date_str_clean = date_str.replace('Z', '+00:00')
            if '+' in date_str_clean:
                date_str_clean = date_str_clean.split('+')[0]

            date_obj = datetime.fromisoformat(date_str_clean)
            year = date_obj.year
            month = date_obj.month

            if filter_func(year, month):
                filtered_count += 1
        except:
            continue

    print(f"   {filter_name}: {filtered_count:,} preguntas")

print()
print("="*80)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*80)
print()
print("💡 Estos números deben coincidir con los mostrados en el configurador.")
print("   Si coinciden, el filtro está funcionando correctamente.")
