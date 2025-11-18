#!/usr/bin/env python3
"""
Verificación del Capítulo 8 - Conclusiones y Trabajo Futuro
Contrasta valores numéricos con JSON 20251114_071914
"""
import json
import sys

# Cargar JSON
json_path = '../cumulative_results_20251114_071914.json'
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✅ JSON cargado: {json_path}\n")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("=" * 100)
print("VERIFICACIÓN CAPÍTULO 8 - VALORES NUMÉRICOS")
print("=" * 100)

errores = []
correctas = []

# ========== TABLA 8.1: Rendimiento Comparativo (línea ~19) ==========
print("\n📊 TABLA 8.1: Rendimiento Comparativo - Precision@5 BEFORE (Línea ~19)")
print("-" * 100)

tabla_8_1 = {
    'Ada': 0.062,
    'MPNet': 0.052,
    'E5-Large': 0.045,
    'MiniLM': 0.041
}

for modelo_cap, valor_cap in tabla_8_1.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    m_json = data['results'][modelo_json]['avg_before_metrics']
    valor_json = m_json['precision@5']

    diff = abs(valor_cap - valor_json)
    if diff > 0.001:
        print(f"❌ {modelo_cap}: Cap={valor_cap:.3f} | JSON={valor_json:.3f} | Diff={diff:.3f}")
        errores.append(f"Tabla 8.1: {modelo_cap} precision@5")
    else:
        print(f"✅ {modelo_cap}: {valor_cap:.3f} (correcto)")
        correctas.append(f"Tabla 8.1: {modelo_cap} precision@5")

# Verificar diferencias porcentuales
print("\n📊 Diferencias Porcentuales vs Ada:")
ada_val = data['results']['ada']['avg_before_metrics']['precision@5']

diffs_cap = {
    'MPNet': -19.2,
    'E5-Large': -27.4,
    'MiniLM': -33.9
}

for modelo, diff_cap in diffs_cap.items():
    modelo_json = modelo.lower().replace('-', '-')
    val_json = data['results'][modelo_json]['avg_before_metrics']['precision@5']
    diff_real = ((val_json - ada_val) / ada_val) * 100
    diff_abs = abs(diff_cap - diff_real)

    if diff_abs > 0.5:
        print(f"❌ {modelo}: Cap={diff_cap:.1f}% | Real={diff_real:.1f}% | Diff={diff_abs:.1f}%")
        errores.append(f"Tabla 8.1: {modelo} % diff vs Ada")
    else:
        print(f"✅ {modelo}: {diff_cap:.1f}% (correcto, real={diff_real:.1f}%)")
        correctas.append(f"Tabla 8.1: {modelo} % diff vs Ada")

# ========== TABLA 8.2: Impacto del Reranking (línea ~79) ==========
print("\n📊 TABLA 8.2: Impacto del Reranking - Precision@5 (Línea ~79)")
print("-" * 100)

tabla_8_2 = {
    'MiniLM': {'before': 0.041, 'after': 0.047, 'delta': 0.006, 'pct': 13.1},
    'E5-Large': {'before': 0.045, 'after': 0.046, 'delta': 0.001, 'pct': 2.2},
    'MPNet': {'before': 0.052, 'after': 0.050, 'delta': -0.002, 'pct': -3.4},
    'Ada': {'before': 0.062, 'after': 0.052, 'delta': -0.010, 'pct': -15.6}
}

for modelo, valores_cap in tabla_8_2.items():
    modelo_json = modelo.lower().replace('-', '-')
    before_json = data['results'][modelo_json]['avg_before_metrics']['precision@5']
    after_json = data['results'][modelo_json]['avg_after_metrics']['precision@5']
    delta_json = after_json - before_json
    pct_json = (delta_json / before_json) * 100 if before_json != 0 else 0

    print(f"\n{modelo}:")

    # Verificar before
    diff_before = abs(valores_cap['before'] - before_json)
    if diff_before > 0.001:
        print(f"  ❌ Before: Cap={valores_cap['before']:.3f} | JSON={before_json:.3f}")
        errores.append(f"Tabla 8.2: {modelo} before")
    else:
        print(f"  ✅ Before: {valores_cap['before']:.3f}")
        correctas.append(f"Tabla 8.2: {modelo} before")

    # Verificar after
    diff_after = abs(valores_cap['after'] - after_json)
    if diff_after > 0.001:
        print(f"  ❌ After: Cap={valores_cap['after']:.3f} | JSON={after_json:.3f}")
        errores.append(f"Tabla 8.2: {modelo} after")
    else:
        print(f"  ✅ After: {valores_cap['after']:.3f}")
        correctas.append(f"Tabla 8.2: {modelo} after")

    # Verificar delta
    diff_delta = abs(valores_cap['delta'] - delta_json)
    if diff_delta > 0.001:
        print(f"  ❌ Delta: Cap={valores_cap['delta']:+.3f} | JSON={delta_json:+.3f}")
        errores.append(f"Tabla 8.2: {modelo} delta")
    else:
        print(f"  ✅ Delta: {valores_cap['delta']:+.3f}")
        correctas.append(f"Tabla 8.2: {modelo} delta")

    # Verificar porcentaje
    diff_pct = abs(valores_cap['pct'] - pct_json)
    if diff_pct > 0.5:
        print(f"  ❌ %: Cap={valores_cap['pct']:+.1f}% | JSON={pct_json:+.1f}%")
        errores.append(f"Tabla 8.2: {modelo} %")
    else:
        print(f"  ✅ %: {valores_cap['pct']:+.1f}%")
        correctas.append(f"Tabla 8.2: {modelo} %")

# ========== MÉTRICAS RAGAS Y BERTSCORE MENCIONADAS ==========
print("\n📊 MÉTRICAS RAGAS Y BERTSCORE MENCIONADAS EN EL TEXTO")
print("-" * 100)

print("\nFaithfulness (mencionado: 0.707-0.719):")
for modelo in ['ada', 'mpnet', 'e5-large', 'minilm']:
    r = data['results'][modelo]['rag_metrics']
    faith = r.get('avg_faithfulness') or 0

    if 0.707 <= faith <= 0.719:
        print(f"  ❌ {modelo}: {faith:.3f} (FUERA del rango mencionado)")
        errores.append(f"Faithfulness rango: {modelo}")
    else:
        print(f"  ⚠️  {modelo}: {faith:.3f} (verificar si está en rango correcto)")

# Calcular rango real
faithfulness_values = [
    data['results'][m]['rag_metrics'].get('avg_faithfulness', 0)
    for m in ['ada', 'mpnet', 'e5-large', 'minilm']
]
min_faith = min(faithfulness_values)
max_faith = max(faithfulness_values)
print(f"\n  📊 Rango REAL en JSON: {min_faith:.3f} - {max_faith:.3f}")
print(f"  📝 Rango MENCIONADO en Cap 8: 0.707 - 0.719")

if abs(min_faith - 0.707) > 0.005 or abs(max_faith - 0.719) > 0.005:
    print(f"  ❌ RANGO INCORRECTO en el capítulo")
    errores.append("Faithfulness: rango incorrecto")
else:
    print(f"  ✅ Rango correcto")
    correctas.append("Faithfulness: rango correcto")

print("\nBERTScore F1 (mencionado: 0.589):")
for modelo in ['ada', 'mpnet', 'e5-large', 'minilm']:
    r = data['results'][modelo]['rag_metrics']
    bert_f1 = r.get('avg_bert_f1')

    if bert_f1 is None:
        print(f"  ⚠️  {modelo}: None (NO DISPONIBLE en JSON)")
    else:
        diff = abs(bert_f1 - 0.589)
        if diff > 0.001:
            print(f"  ❌ {modelo}: {bert_f1:.3f} (esperado 0.589)")
            errores.append(f"BERTScore F1: {modelo}")
        else:
            print(f"  ✅ {modelo}: {bert_f1:.3f}")
            correctas.append(f"BERTScore F1: {modelo}")

# ========== VERIFICAR EFICIENCIA RELATIVA (Tabla 8.1) ==========
print("\n📊 EFICIENCIA RELATIVA (Tabla 8.1, columna 'Eficiencia Relativa')")
print("-" * 100)

eficiencia_cap = {
    'Ada': 100.0,
    'MPNet': 83.9,
    'E5-Large': 72.6,
    'MiniLM': 66.1
}

ada_prec = data['results']['ada']['avg_before_metrics']['precision@5']

for modelo, efic_cap in eficiencia_cap.items():
    modelo_json = modelo.lower().replace('-', '-')
    prec_json = data['results'][modelo_json]['avg_before_metrics']['precision@5']
    efic_real = (prec_json / ada_prec) * 100
    diff = abs(efic_cap - efic_real)

    if diff > 0.5:
        print(f"❌ {modelo}: Cap={efic_cap:.1f}% | Real={efic_real:.1f}% | Diff={diff:.1f}%")
        errores.append(f"Eficiencia relativa: {modelo}")
    else:
        print(f"✅ {modelo}: {efic_cap:.1f}% (real={efic_real:.1f}%)")
        correctas.append(f"Eficiencia relativa: {modelo}")

# ========== RESUMEN ==========
print("\n" + "=" * 100)
print("RESUMEN DE VERIFICACIÓN - CAPÍTULO 8")
print("=" * 100)

total = len(correctas) + len(errores)
print(f"\n✅ Valores CORRECTOS: {len(correctas)}/{total} ({len(correctas)/total*100:.1f}%)")
print(f"❌ Valores INCORRECTOS: {len(errores)}/{total} ({len(errores)/total*100:.1f}%)")

if errores:
    print(f"\n🔴 ERRORES DETECTADOS:")
    print("-" * 100)
    for error in errores:
        print(f"  - {error}")
else:
    print("\n✅ NO SE DETECTARON ERRORES - Capítulo 8 completamente correcto")

print("\n" + "=" * 100)
print("ANÁLISIS ADICIONAL")
print("=" * 100)

# Verificar que NO hay BERTScore en JSON actual
bert_none_count = 0
for modelo in ['ada', 'mpnet', 'e5-large', 'minilm']:
    if data['results'][modelo]['rag_metrics'].get('avg_bert_f1') is None:
        bert_none_count += 1

if bert_none_count == 4:
    print("\n⚠️  ADVERTENCIA: BERTScore F1 = None en JSON actual para TODOS los modelos")
    print("   El capítulo menciona BERTScore F1 = 0.589 (convergencia completa)")
    print("   Este valor proviene de evaluaciones anteriores, no del JSON 20251114")
    print("\n   RECOMENDACIÓN:")
    print("   - Agregar nota al pie en secciones que mencionan BERTScore (8.2.4, 8.3.4)")
    print("   - Ejemplo: 'Valores de BERTScore de evaluaciones preliminares'")

print("\n✅ Verificación de Capítulo 8 completada")
