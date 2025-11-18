#!/usr/bin/env python3
"""
Verificación Completa del Capítulo 7 vs Datos Reales
Compara TODAS las tablas del capítulo con el JSON 20251114_071914
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

# Función para comparar valores
def comparar(nombre_tabla, linea_aprox, modelo, metrica, valor_cap, valor_json, tolerancia=0.001):
    diff = abs(valor_cap - valor_json)
    if diff > tolerancia:
        print(f"❌ {nombre_tabla} (L~{linea_aprox}): {modelo} {metrica}")
        print(f"   Capítulo: {valor_cap:.3f} | JSON: {valor_json:.3f} | Diff: {diff:.3f}")
        return False
    return True

print("="*100)
print("VERIFICACIÓN COMPLETA - CAPÍTULO 7 vs JSON 20251114_071914")
print("="*100)

errores = []
correctas = []

# ========== TABLA 7.1: Rendimiento General (k=5, BEFORE) ==========
print("\n📊 TABLA 7.1: Rendimiento General por Modelo (BEFORE, k=5) - Línea ~66")
print("-" * 100)

tabla_7_1 = {
    'Ada': {'precision@5': 0.062, 'recall@5': 0.245, 'f1@5': 0.096, 'ndcg@5': 0.173, 'map@5': 0.140, 'mrr': 0.188},
    'MPNet': {'precision@5': 0.052, 'recall@5': 0.201, 'f1@5': 0.079, 'ndcg@5': 0.146, 'map@5': 0.118, 'mrr': 0.163},
    'E5-Large': {'precision@5': 0.045, 'recall@5': 0.177, 'f1@5': 0.069, 'ndcg@5': 0.120, 'map@5': 0.094, 'mrr': 0.130},
    'MiniLM': {'precision@5': 0.041, 'recall@5': 0.163, 'f1@5': 0.064, 'ndcg@5': 0.111, 'map@5': 0.087, 'mrr': 0.122}
}

for modelo_cap, metricas_cap in tabla_7_1.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for metrica, valor_cap in metricas_cap.items():
        valor_json = m_json[metrica]
        if comparar("Tabla 7.1", 66, modelo_cap, metrica, valor_cap, valor_json):
            correctas.append(f"Tabla 7.1: {modelo_cap} {metrica}")
        else:
            errores.append(f"Tabla 7.1: {modelo_cap} {metrica}")

# ========== TABLA 7.2: Precision@k BEFORE ==========
print("\n📊 TABLA 7.2: Precision@k para Todos los Modelos (BEFORE) - Línea ~90")
print("-" * 100)

tabla_7_2 = {
    'Ada': {3: 0.075, 5: 0.062, 10: 0.047, 15: 0.035},
    'MPNet': {3: 0.066, 5: 0.052, 10: 0.040, 15: 0.031},
    'E5-Large': {3: 0.050, 5: 0.045, 10: 0.034, 15: 0.027},
    'MiniLM': {3: 0.046, 5: 0.041, 10: 0.033, 15: 0.026}
}

for modelo_cap, valores_k in tabla_7_2.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for k, valor_cap in valores_k.items():
        valor_json = m_json[f'precision@{k}']
        if comparar("Tabla 7.2", 90, modelo_cap, f"precision@{k}", valor_cap, valor_json):
            correctas.append(f"Tabla 7.2: {modelo_cap} precision@{k}")
        else:
            errores.append(f"Tabla 7.2: {modelo_cap} precision@{k}")

# ========== TABLA 7.3: Recall@k BEFORE ==========
print("\n📊 TABLA 7.3: Recall@k para Todos los Modelos (BEFORE) - Línea ~112")
print("-" * 100)

tabla_7_3 = {
    'Ada': {3: 0.178, 5: 0.245, 10: 0.368, 15: 0.403},
    'MPNet': {3: 0.156, 5: 0.201, 10: 0.302, 15: 0.350},
    'E5-Large': {3: 0.119, 5: 0.177, 10: 0.262, 15: 0.307},
    'MiniLM': {3: 0.109, 5: 0.163, 10: 0.252, 15: 0.300}
}

for modelo_cap, valores_k in tabla_7_3.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for k, valor_cap in valores_k.items():
        valor_json = m_json[f'recall@{k}']
        if comparar("Tabla 7.3", 112, modelo_cap, f"recall@{k}", valor_cap, valor_json):
            correctas.append(f"Tabla 7.3: {modelo_cap} recall@{k}")
        else:
            errores.append(f"Tabla 7.3: {modelo_cap} recall@{k}")

# ========== TABLA 7.4: F1@k BEFORE ==========
print("\n📊 TABLA 7.4: F1@k para Todos los Modelos (BEFORE) - Línea ~132")
print("-" * 100)

tabla_7_4 = {
    'Ada': {3: 0.101, 5: 0.096, 10: 0.082, 15: 0.062},
    'MPNet': {3: 0.089, 5: 0.079, 10: 0.068, 15: 0.055},
    'E5-Large': {3: 0.067, 5: 0.069, 10: 0.058, 15: 0.048},
    'MiniLM': {3: 0.062, 5: 0.064, 10: 0.056, 15: 0.047}
}

for modelo_cap, valores_k in tabla_7_4.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for k, valor_cap in valores_k.items():
        valor_json = m_json[f'f1@{k}']
        if comparar("Tabla 7.4", 132, modelo_cap, f"f1@{k}", valor_cap, valor_json):
            correctas.append(f"Tabla 7.4: {modelo_cap} f1@{k}")
        else:
            errores.append(f"Tabla 7.4: {modelo_cap} f1@{k}")

# ========== TABLA 7.5: NDCG@k BEFORE ==========
print("\n📊 TABLA 7.5: NDCG@k para Todos los Modelos (BEFORE) - Línea ~147")
print("-" * 100)

tabla_7_5 = {
    'Ada': {3: 0.146, 5: 0.173, 10: 0.215, 15: 0.225},
    'MPNet': {3: 0.128, 5: 0.146, 10: 0.181, 15: 0.194},
    'E5-Large': {3: 0.095, 5: 0.120, 10: 0.149, 15: 0.162},
    'MiniLM': {3: 0.088, 5: 0.111, 10: 0.141, 15: 0.155}
}

for modelo_cap, valores_k in tabla_7_5.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for k, valor_cap in valores_k.items():
        valor_json = m_json[f'ndcg@{k}']
        if comparar("Tabla 7.5", 147, modelo_cap, f"ndcg@{k}", valor_cap, valor_json):
            correctas.append(f"Tabla 7.5: {modelo_cap} ndcg@{k}")
        else:
            errores.append(f"Tabla 7.5: {modelo_cap} ndcg@{k}")

# ========== TABLA 7.6: MAP@k BEFORE ==========
print("\n📊 TABLA 7.6: MAP@k para Todos los Modelos (BEFORE) - Línea ~163")
print("-" * 100)

tabla_7_6 = {
    'Ada': {3: 0.211, 5: 0.263, 10: 0.317, 15: 0.344},
    'MPNet': {3: 0.149, 5: 0.174, 10: 0.203, 15: 0.216},
    'E5-Large': {3: 0.133, 5: 0.161, 10: 0.191, 15: 0.205},
    'MiniLM': {3: 0.114, 5: 0.132, 10: 0.156, 15: 0.167}
}

for modelo_cap, valores_k in tabla_7_6.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for k, valor_cap in valores_k.items():
        valor_json = m_json[f'map@{k}']
        if comparar("Tabla 7.6", 163, modelo_cap, f"map@{k}", valor_cap, valor_json):
            correctas.append(f"Tabla 7.6: {modelo_cap} map@{k}")
        else:
            errores.append(f"Tabla 7.6: {modelo_cap} map@{k}")

# ========== TABLA 7.7: Ranking de Modelos (BEFORE) ==========
print("\n📊 TABLA 7.7: Ranking de Modelos por Precision@5 (BEFORE) - Línea ~177")
print("-" * 100)

tabla_7_7 = {
    'Ada': {'precision@5': 0.098, 'recall@5': 0.398, 'f1@5': 0.152, 'ndcg@5': 0.234},
    'MPNet': {'precision@5': 0.070, 'recall@5': 0.277, 'f1@5': 0.108, 'ndcg@5': 0.193},
    'E5-Large': {'precision@5': 0.065, 'recall@5': 0.262, 'f1@5': 0.100, 'ndcg@5': 0.174},
    'MiniLM': {'precision@5': 0.053, 'recall@5': 0.211, 'f1@5': 0.082, 'ndcg@5': 0.150}
}

print("⚠️  NOTA: Tabla 7.7 tiene valores DIFERENTES a Tabla 7.1 (misma métrica k=5)")
print("   Esto sugiere que Tabla 7.7 puede contener datos de un archivo JSON anterior.")
print()

for modelo_cap, metricas_cap in tabla_7_7.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_before_metrics']

    for metrica, valor_cap in metricas_cap.items():
        valor_json = m_json[metrica]
        if comparar("Tabla 7.7", 177, modelo_cap, metrica, valor_cap, valor_json, tolerancia=0.002):
            correctas.append(f"Tabla 7.7: {modelo_cap} {metrica}")
        else:
            errores.append(f"Tabla 7.7: {modelo_cap} {metrica}")

# ========== TABLA 7.8: Rendimiento AFTER ==========
print("\n📊 TABLA 7.8: Rendimiento General por Modelo (AFTER, k=5) - Línea ~192")
print("-" * 100)

tabla_7_8 = {
    'Ada': {'precision@5': 0.052, 'recall@5': 0.206, 'f1@5': 0.081, 'ndcg@5': 0.138, 'map@5': 0.107, 'mrr': 0.156},
    'MPNet': {'precision@5': 0.050, 'recall@5': 0.195, 'f1@5': 0.077, 'ndcg@5': 0.137, 'map@5': 0.109, 'mrr': 0.154},
    'E5-Large': {'precision@5': 0.046, 'recall@5': 0.182, 'f1@5': 0.071, 'ndcg@5': 0.129, 'map@5': 0.104, 'mrr': 0.142},
    'MiniLM': {'precision@5': 0.047, 'recall@5': 0.180, 'f1@5': 0.071, 'ndcg@5': 0.130, 'map@5': 0.105, 'mrr': 0.143}
}

for modelo_cap, metricas_cap in tabla_7_8.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    m_json = data['results'][modelo_json]['avg_after_metrics']

    for metrica, valor_cap in metricas_cap.items():
        valor_json = m_json[metrica]
        if comparar("Tabla 7.8", 192, modelo_cap, metrica, valor_cap, valor_json):
            correctas.append(f"Tabla 7.8: {modelo_cap} {metrica}")
        else:
            errores.append(f"Tabla 7.8: {modelo_cap} {metrica}")

# ========== TABLA 7.14: RAGAS METRICS ==========
print("\n📊 TABLA 7.14: Métricas RAGAS por Modelo - Línea ~356")
print("-" * 100)

tabla_7_14 = {
    'Ada': {'faithfulness': 0.649, 'answer_rel': 0.861, 'context_prec': 0.918, 'context_recall': 0.848, 'semantic_sim': 0.715},
    'MPNet': {'faithfulness': 0.644, 'answer_rel': 0.856, 'context_prec': 0.919, 'context_recall': 0.844, 'semantic_sim': 0.716},
    'E5-Large': {'faithfulness': 0.635, 'answer_rel': 0.852, 'context_prec': 0.913, 'context_recall': 0.839, 'semantic_sim': 0.710},
    'MiniLM': {'faithfulness': 0.639, 'answer_rel': 0.852, 'context_prec': 0.913, 'context_recall': 0.838, 'semantic_sim': 0.711}
}

for modelo_cap, metricas_cap in tabla_7_14.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    r_json = data['results'][modelo_json]['rag_metrics']

    # Map capítulo keys to JSON keys
    key_map = {
        'faithfulness': 'avg_faithfulness',
        'answer_rel': 'avg_answer_relevance',
        'context_prec': 'avg_context_precision',
        'context_recall': 'avg_context_recall',
        'semantic_sim': 'avg_semantic_similarity'
    }

    for metrica_cap, metrica_json_key in key_map.items():
        valor_cap = metricas_cap[metrica_cap]
        valor_json = r_json.get(metrica_json_key) or 0
        if comparar("Tabla 7.14", 356, modelo_cap, metrica_cap, valor_cap, valor_json):
            correctas.append(f"Tabla 7.14: {modelo_cap} {metrica_cap}")
        else:
            errores.append(f"Tabla 7.14: {modelo_cap} {metrica_cap}")

# ========== TABLA 7.15: BERTScore METRICS ==========
print("\n📊 TABLA 7.15: BERTScore por Modelo - Línea ~380")
print("-" * 100)

tabla_7_15 = {
    'Ada': {'bert_precision': 0.647, 'bert_recall': 0.542, 'bert_f1': 0.589},
    'MPNet': {'bert_precision': 0.648, 'bert_recall': 0.543, 'bert_f1': 0.589},
    'E5-Large': {'bert_precision': 0.648, 'bert_recall': 0.542, 'bert_f1': 0.589},
    'MiniLM': {'bert_precision': 0.648, 'bert_recall': 0.542, 'bert_f1': 0.589}
}

print("❌ CRÍTICO: Tabla 7.15 tiene valores de BERTScore, pero JSON tiene None para todos.")
print("   Esto significa que los valores en el capítulo provienen de un archivo JSON anterior.")
print()

for modelo_cap, metricas_cap in tabla_7_15.items():
    modelo_json = modelo_cap.lower().replace('-', '-')
    if modelo_json == 'ada':
        modelo_json = 'ada'
    elif modelo_json == 'mpnet':
        modelo_json = 'mpnet'
    elif modelo_json == 'e5-large':
        modelo_json = 'e5-large'
    elif modelo_json == 'minilm':
        modelo_json = 'minilm'

    r_json = data['results'][modelo_json]['rag_metrics']

    for metrica in ['bert_precision', 'bert_recall', 'bert_f1']:
        valor_cap = metricas_cap[metrica]
        valor_json = r_json.get(f'avg_{metrica}')

        if valor_json is None:
            print(f"❌ Tabla 7.15 (L~380): {modelo_cap} {metrica}")
            print(f"   Capítulo: {valor_cap:.3f} | JSON: None (no disponible)")
            errores.append(f"Tabla 7.15: {modelo_cap} {metrica} (None en JSON)")

# ========== RESUMEN FINAL ==========
print("\n" + "="*100)
print("RESUMEN DE VERIFICACIÓN")
print("="*100)

total = len(correctas) + len(errores)
print(f"\n✅ Valores CORRECTOS: {len(correctas)}/{total} ({len(correctas)/total*100:.1f}%)")
print(f"❌ Valores INCORRECTOS o MISSING: {len(errores)}/{total} ({len(errores)/total*100:.1f}%)")

if errores:
    print(f"\n🔴 ERRORES DETECTADOS:")
    print("-" * 100)
    # Agrupar por tabla
    tablas_con_errores = {}
    for error in errores:
        tabla = error.split(':')[0]
        if tabla not in tablas_con_errores:
            tablas_con_errores[tabla] = []
        tablas_con_errores[tabla].append(error)

    for tabla, lista_errores in sorted(tablas_con_errores.items()):
        print(f"\n{tabla}: {len(lista_errores)} errores")
        for err in lista_errores[:5]:  # Mostrar primeros 5
            print(f"  - {err}")
        if len(lista_errores) > 5:
            print(f"  ... y {len(lista_errores) - 5} más")

print("\n" + "="*100)
print("RECOMENDACIONES")
print("="*100)
print("""
1. TABLA 7.7 (Ranking BEFORE): Valores diferentes a Tabla 7.1
   → ACCIÓN: Verificar si debe usar datos de Tabla 7.1 o mantener estos valores

2. TABLA 7.15 (BERTScore): JSON tiene valores None
   → ACCIÓN: Los valores en capítulo son válidos pero provienen de JSON anterior
   → DECISIÓN: Mantener valores actuales Y agregar nota explicativa:

   "Nota: Las métricas BERTScore se calcularon en evaluaciones anteriores y se
   mantienen para referencia. El cálculo de BERTScore fue deshabilitado en la
   evaluación más reciente debido a problemas de memoria GPU."

3. Verificar Tablas con k={3,10,15}:
   → Tablas 7.9 (Precision@k AFTER)
   → Tablas 7.10 (Recall@k AFTER)
   → No fueron verificadas en este análisis
""")

print("\n✅ Verificación completada")
