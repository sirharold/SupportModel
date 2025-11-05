"""
Script para generar tablas comparativas del Capítulo 7
Genera tablas en formato Markdown y CSV para inclusión en la tesis
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Ruta al archivo de resultados
RESULTS_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json"
OUTPUT_DIR = Path(__file__).parent / "tables"

# Valores de k para tablas (según instrucciones)
K_VALUES_TABLES = [3, 5, 10, 15]

# Colores por modelo (para referencia)
MODEL_COLORS = {
    'ada': '#1f77b4',
    'mpnet': '#ff7f0e',
    'minilm': '#2ca02c',
    'e5large': '#d62728'
}

# Nombres de modelos para display
MODEL_NAMES = {
    'ada': 'Ada (OpenAI)',
    'mpnet': 'MPNet',
    'minilm': 'MiniLM',
    'e5large': 'E5-Large'
}


def load_results() -> Dict:
    """Carga el archivo de resultados JSON"""
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Verificar que son datos reales
    if not data.get('evaluation_info', {}).get('data_verification', {}).get('is_real_data', False):
        raise ValueError("⚠️ ADVERTENCIA: Los datos NO están marcados como reales")

    print("✅ Datos verificados como REALES (no simulados)")
    return data


def extract_metrics_by_k(metrics: Dict, k_values: List[int]) -> Dict:
    """Extrae métricas para valores específicos de k"""
    result = {}

    # Métricas que dependen de k
    metric_families = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr']

    for family in metric_families:
        for k in k_values:
            key = f"{family}@{k}"
            if key in metrics:
                result[key] = metrics[key]

    # MRR es especial - no tiene @k específico, pero tiene variantes
    if 'mrr' in metrics:
        result['mrr'] = metrics['mrr']

    return result


def generate_comparison_table() -> pd.DataFrame:
    """Genera tabla comparativa de todos los modelos"""
    data = load_results()
    results = data['results']

    rows = []

    for model_key in ['ada', 'mpnet', 'minilm', 'e5large']:
        if model_key not in results:
            continue

        model_data = results[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)

        # Métricas principales (k=5 como referencia)
        before = model_data['avg_before_metrics']
        after = model_data.get('avg_after_metrics', {})

        row = {
            'Modelo': model_name,
            'Dimensiones': model_data.get('embedding_dimensions', 'N/A'),
            'Preguntas': model_data['num_questions_evaluated'],
        }

        # Métricas ANTES del reranking (k=5)
        for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']:
            if metric in before:
                row[f'{metric.upper()} (Antes)'] = f"{before[metric]:.3f}"

        # Métricas DESPUÉS del reranking (k=5)
        if after:
            for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']:
                if metric in after:
                    row[f'{metric.upper()} (Después)'] = f"{after[metric]:.3f}"

                    # Calcular delta
                    if metric in before:
                        delta = after[metric] - before[metric]
                        pct = (delta / before[metric] * 100) if before[metric] > 0 else 0
                        row[f'Δ {metric.upper()}'] = f"{delta:+.3f} ({pct:+.1f}%)"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Guardar en CSV y Markdown
    csv_path = OUTPUT_DIR / "tabla_comparativa_modelos.csv"
    md_path = OUTPUT_DIR / "tabla_comparativa_modelos.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"✅ Tabla comparativa generada:")
    print(f"   - CSV: {csv_path}")
    print(f"   - MD:  {md_path}")

    return df


def generate_metric_by_k_table(metric_family: str) -> pd.DataFrame:
    """Genera tabla de una métrica específica por valores de k"""
    data = load_results()
    results = data['results']

    rows = []

    for model_key in ['ada', 'mpnet', 'minilm', 'e5large']:
        if model_key not in results:
            continue

        model_data = results[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)

        before = model_data['avg_before_metrics']
        after = model_data.get('avg_after_metrics', {})

        # Fila ANTES del reranking
        row_before = {'Modelo': model_name, 'Etapa': 'Antes CrossEncoder'}
        for k in K_VALUES_TABLES:
            key = f"{metric_family}@{k}"
            if key in before:
                row_before[f'k={k}'] = f"{before[key]:.3f}"
        rows.append(row_before)

        # Fila DESPUÉS del reranking
        if after:
            row_after = {'Modelo': model_name, 'Etapa': 'Después CrossEncoder'}
            for k in K_VALUES_TABLES:
                key = f"{metric_family}@{k}"
                if key in after:
                    row_after[f'k={k}'] = f"{after[key]:.3f}"
            rows.append(row_after)

            # Fila de DELTA
            row_delta = {'Modelo': model_name, 'Etapa': 'Δ (cambio)'}
            for k in K_VALUES_TABLES:
                key = f"{metric_family}@{k}"
                if key in before and key in after:
                    delta = after[key] - before[key]
                    pct = (delta / before[key] * 100) if before[key] > 0 else 0
                    row_delta[f'k={k}'] = f"{delta:+.3f} ({pct:+.1f}%)"
            rows.append(row_delta)

        # Línea en blanco entre modelos
        rows.append({'Modelo': '', 'Etapa': ''})

    df = pd.DataFrame(rows)

    # Guardar
    csv_path = OUTPUT_DIR / f"tabla_{metric_family}_por_k.csv"
    md_path = OUTPUT_DIR / f"tabla_{metric_family}_por_k.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"✅ Tabla {metric_family.upper()} por k generada:")
    print(f"   - CSV: {csv_path}")
    print(f"   - MD:  {md_path}")

    return df


def generate_ranking_table() -> pd.DataFrame:
    """Genera tabla de ranking de modelos por métrica"""
    data = load_results()
    results = data['results']

    # Métricas para ranking (k=5)
    metrics_to_rank = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']

    rows = []

    for metric in metrics_to_rank:
        # Recolectar valores de todos los modelos
        model_values = []

        for model_key in ['ada', 'mpnet', 'minilm', 'e5large']:
            if model_key not in results:
                continue

            model_data = results[model_key]
            model_name = MODEL_NAMES.get(model_key, model_key)

            # Antes del reranking
            before = model_data['avg_before_metrics']
            if metric in before:
                model_values.append({
                    'model': model_name,
                    'value': before[metric],
                    'stage': 'before'
                })

            # Después del reranking
            after = model_data.get('avg_after_metrics', {})
            if metric in after:
                model_values.append({
                    'model': model_name,
                    'value': after[metric],
                    'stage': 'after'
                })

        # Ordenar por valor (mayor a menor)
        model_values.sort(key=lambda x: x['value'], reverse=True)

        # Crear filas para ANTES
        before_values = [v for v in model_values if v['stage'] == 'before']
        for i, item in enumerate(before_values, 1):
            rows.append({
                'Métrica': metric.upper(),
                'Etapa': 'Antes',
                'Ranking': i,
                'Modelo': item['model'],
                'Valor': f"{item['value']:.3f}"
            })

        # Crear filas para DESPUÉS
        after_values = [v for v in model_values if v['stage'] == 'after']
        for i, item in enumerate(after_values, 1):
            rows.append({
                'Métrica': metric.upper(),
                'Etapa': 'Después',
                'Ranking': i,
                'Modelo': item['model'],
                'Valor': f"{item['value']:.3f}"
            })

        # Línea en blanco
        rows.append({'Métrica': '', 'Etapa': '', 'Ranking': '', 'Modelo': '', 'Valor': ''})

    df = pd.DataFrame(rows)

    # Guardar
    csv_path = OUTPUT_DIR / "tabla_ranking_modelos.csv"
    md_path = OUTPUT_DIR / "tabla_ranking_modelos.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"✅ Tabla de ranking generada:")
    print(f"   - CSV: {csv_path}")
    print(f"   - MD:  {md_path}")

    return df


def generate_rag_metrics_table() -> pd.DataFrame:
    """Genera tabla de métricas RAG (si existen)"""
    data = load_results()
    results = data['results']

    rows = []

    # Métricas RAG a buscar
    rag_metrics = [
        'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy',
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1'
    ]

    for model_key in ['ada', 'mpnet', 'minilm', 'e5large']:
        if model_key not in results:
            continue

        model_data = results[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)

        # Buscar métricas RAG en avg_after_metrics o avg_before_metrics
        after = model_data.get('avg_after_metrics', {})
        before = model_data.get('avg_before_metrics', {})

        row = {'Modelo': model_name}
        has_rag_metrics = False

        for metric in rag_metrics:
            if metric in after:
                row[metric.replace('_', ' ').title()] = f"{after[metric]:.3f}"
                has_rag_metrics = True
            elif metric in before:
                row[metric.replace('_', ' ').title()] = f"{before[metric]:.3f}"
                has_rag_metrics = True

        if has_rag_metrics:
            rows.append(row)

    if not rows:
        print("⚠️  No se encontraron métricas RAG en los resultados")
        return None

    df = pd.DataFrame(rows)

    # Guardar
    csv_path = OUTPUT_DIR / "tabla_metricas_rag.csv"
    md_path = OUTPUT_DIR / "tabla_metricas_rag.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"✅ Tabla de métricas RAG generada:")
    print(f"   - CSV: {csv_path}")
    print(f"   - MD:  {md_path}")

    return df


def main():
    """Función principal para generar todas las tablas"""
    print("=" * 60)
    print("GENERACIÓN DE TABLAS - CAPÍTULO 7")
    print("=" * 60)
    print()

    # Crear directorio de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tabla comparativa general
    print("\n[1/6] Generando tabla comparativa de modelos...")
    generate_comparison_table()

    # 2-5. Tablas por familia de métrica
    metric_families = ['precision', 'recall', 'f1', 'ndcg', 'map']
    for i, metric_family in enumerate(metric_families, 2):
        print(f"\n[{i}/6] Generando tabla de {metric_family.upper()} por k...")
        generate_metric_by_k_table(metric_family)

    # 6. Tabla de ranking
    print(f"\n[6/6] Generando tabla de ranking de modelos...")
    generate_ranking_table()

    # Bonus: Métricas RAG
    print(f"\n[Bonus] Generando tabla de métricas RAG...")
    generate_rag_metrics_table()

    print("\n" + "=" * 60)
    print("✅ TODAS LAS TABLAS GENERADAS EXITOSAMENTE")
    print("=" * 60)
    print(f"\nTablas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
