"""
Script para generar gráficos del Capítulo 7
Genera gráficos en alta resolución (300 DPI) para inclusión en la tesis
Usa TODOS los valores de k disponibles (1-15)
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Configuración de matplotlib para alta calidad
matplotlib.use('Agg')  # Backend sin GUI
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Ruta al archivo de resultados
RESULTS_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251114_071914.json"
OUTPUT_DIR = Path(__file__).parent / "charts"

# Valores de k para gráficos (TODOS los disponibles)
K_VALUES = list(range(1, 16))  # 1-15

# Colores por modelo
MODEL_COLORS = {
    'ada': '#1f77b4',
    'mpnet': '#ff7f0e',
    'minilm': '#2ca02c',
    'e5-large': '#d62728'
}

# Nombres de modelos para display
MODEL_NAMES = {
    'ada': 'Ada (OpenAI)',
    'mpnet': 'MPNet',
    'minilm': 'MiniLM',
    'e5-large': 'E5-Large'
}

# Estilo de seaborn
sns.set_style("whitegrid")


def load_results() -> Dict:
    """Carga el archivo de resultados JSON"""
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Verificar que son datos reales
    if not data.get('evaluation_info', {}).get('data_verification', {}).get('is_real_data', False):
        raise ValueError("⚠️ ADVERTENCIA: Los datos NO están marcados como reales")

    print("✅ Datos verificados como REALES (no simulados)")
    return data


def extract_metric_values_by_k(metrics: Dict, metric_family: str, k_values: List[int]) -> List[float]:
    """Extrae valores de una métrica para todos los valores de k"""
    values = []
    for k in k_values:
        key = f"{metric_family}@{k}"
        if key in metrics:
            values.append(metrics[key])
        else:
            values.append(np.nan)
    return values


def plot_metric_by_k_all_models(metric_family: str, stage: str = 'before'):
    """Genera gráfico de una métrica por k para todos los modelos"""
    data = load_results()
    results = data['results']

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key in ['ada', 'mpnet', 'minilm', 'e5-large']:
        if model_key not in results:
            continue

        model_data = results[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)

        # Seleccionar antes o después
        if stage == 'before':
            metrics = model_data['avg_before_metrics']
        else:
            metrics = model_data.get('avg_after_metrics', {})
            if not metrics:
                continue

        # Extraer valores
        values = extract_metric_values_by_k(metrics, metric_family, K_VALUES)

        # Plotear
        ax.plot(K_VALUES, values,
                marker='o',
                label=model_name,
                color=MODEL_COLORS[model_key],
                linewidth=2,
                markersize=4)

    ax.set_xlabel('k (número de documentos recuperados)', fontweight='bold')
    ax.set_ylabel(f'{metric_family.upper()}@k', fontweight='bold')

    stage_label = 'Antes de CrossEncoder' if stage == 'before' else 'Después de CrossEncoder'
    ax.set_title(f'{metric_family.upper()} por k - {stage_label}', fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(K_VALUES)

    plt.tight_layout()

    # Guardar
    filename = f"{metric_family}_por_k_{stage}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_metric_comparison_before_after(metric_family: str, model_key: str):
    """Genera gráfico comparando antes/después para un modelo específico"""
    data = load_results()
    results = data['results']

    if model_key not in results:
        print(f"⚠️  Modelo {model_key} no encontrado")
        return None

    model_data = results[model_key]
    model_name = MODEL_NAMES.get(model_key, model_key)

    before_metrics = model_data['avg_before_metrics']
    after_metrics = model_data.get('avg_after_metrics', {})

    if not after_metrics:
        print(f"⚠️  No hay métricas después del reranking para {model_key}")
        return None

    # Extraer valores
    before_values = extract_metric_values_by_k(before_metrics, metric_family, K_VALUES)
    after_values = extract_metric_values_by_k(after_metrics, metric_family, K_VALUES)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(K_VALUES, before_values,
            marker='o',
            label='Antes de CrossEncoder',
            color=MODEL_COLORS[model_key],
            linewidth=2,
            markersize=5,
            linestyle='-')

    ax.plot(K_VALUES, after_values,
            marker='s',
            label='Después de CrossEncoder',
            color=MODEL_COLORS[model_key],
            linewidth=2,
            markersize=5,
            linestyle='--',
            alpha=0.7)

    ax.set_xlabel('k (número de documentos recuperados)', fontweight='bold')
    ax.set_ylabel(f'{metric_family.upper()}@k', fontweight='bold')
    ax.set_title(f'{metric_family.upper()} por k - {model_name}\nComparación Antes vs Después', fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(K_VALUES)

    plt.tight_layout()

    # Guardar
    filename = f"{metric_family}_comparison_{model_key}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_all_metrics_single_model(model_key: str, stage: str = 'before'):
    """Genera gráfico con múltiples métricas para un solo modelo"""
    data = load_results()
    results = data['results']

    if model_key not in results:
        print(f"⚠️  Modelo {model_key} no encontrado")
        return None

    model_data = results[model_key]
    model_name = MODEL_NAMES.get(model_key, model_key)

    # Seleccionar métricas
    if stage == 'before':
        metrics = model_data['avg_before_metrics']
    else:
        metrics = model_data.get('avg_after_metrics', {})
        if not metrics:
            return None

    # Familias de métricas a graficar
    metric_families = ['precision', 'recall', 'f1', 'ndcg', 'map']

    fig, ax = plt.subplots(figsize=(12, 7))

    for metric_family in metric_families:
        values = extract_metric_values_by_k(metrics, metric_family, K_VALUES)
        ax.plot(K_VALUES, values,
                marker='o',
                label=metric_family.upper(),
                linewidth=2,
                markersize=4)

    ax.set_xlabel('k (número de documentos recuperados)', fontweight='bold')
    ax.set_ylabel('Valor de la métrica', fontweight='bold')

    stage_label = 'Antes de CrossEncoder' if stage == 'before' else 'Después de CrossEncoder'
    ax.set_title(f'Todas las Métricas por k - {model_name}\n{stage_label}', fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(K_VALUES)

    plt.tight_layout()

    # Guardar
    filename = f"all_metrics_{model_key}_{stage}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_delta_heatmap():
    """Genera heatmap de cambios (deltas) para todas las métricas y modelos"""
    data = load_results()
    results = data['results']

    # Preparar datos
    metric_families = ['precision', 'recall', 'f1', 'ndcg', 'map']
    k_values_subset = [3, 5, 10, 15]  # Usar subset para heatmap
    models = ['ada', 'mpnet', 'minilm', 'e5-large']

    # Crear matriz de deltas
    delta_data = []

    for model_key in models:
        if model_key not in results:
            continue

        model_data = results[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)

        before = model_data['avg_before_metrics']
        after = model_data.get('avg_after_metrics', {})

        if not after:
            continue

        for metric_family in metric_families:
            for k in k_values_subset:
                key = f"{metric_family}@{k}"
                if key in before and key in after:
                    delta_pct = ((after[key] - before[key]) / before[key] * 100) if before[key] > 0 else 0
                    delta_data.append({
                        'Modelo': model_name,
                        'Métrica': f'{metric_family.upper()}@{k}',
                        'Delta %': delta_pct
                    })

    if not delta_data:
        print("⚠️  No hay suficientes datos para generar heatmap")
        return None

    # Crear DataFrame y pivot
    df = pd.DataFrame(delta_data)
    pivot = df.pivot(index='Modelo', columns='Métrica', values='Delta %')

    # Crear heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(pivot,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 'Cambio Porcentual (%)'},
                linewidths=0.5,
                ax=ax)

    ax.set_title('Cambio Porcentual en Métricas: Antes vs Después de CrossEncoder', fontweight='bold', pad=20)
    ax.set_xlabel('Métrica', fontweight='bold')
    ax.set_ylabel('Modelo', fontweight='bold')

    plt.tight_layout()

    # Guardar
    filename = "delta_heatmap.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_model_ranking_bar():
    """Genera gráfico de barras con ranking de modelos para métricas clave"""
    data = load_results()
    results = data['results']

    # Métricas clave (k=5)
    metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Recolectar valores
        model_names = []
        before_values = []
        after_values = []

        for model_key in ['ada', 'mpnet', 'minilm', 'e5-large']:
            if model_key not in results:
                continue

            model_data = results[model_key]
            model_name = MODEL_NAMES.get(model_key, model_key)

            before = model_data['avg_before_metrics']
            after = model_data.get('avg_after_metrics', {})

            if metric in before:
                model_names.append(model_name)
                before_values.append(before[metric])

                if metric in after:
                    after_values.append(after[metric])
                else:
                    after_values.append(0)

        # Crear barras
        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, before_values, width, label='Antes', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_values, width, label='Después', alpha=0.8)

        # Colorear barras según modelo
        for i, model_name in enumerate(model_names):
            for model_key, name in MODEL_NAMES.items():
                if name == model_name:
                    bars1[i].set_color(MODEL_COLORS[model_key])
                    bars2[i].set_color(MODEL_COLORS[model_key])
                    bars2[i].set_alpha(0.6)

        ax.set_ylabel('Valor', fontweight='bold')
        ax.set_title(metric.upper().replace('@', ' @ '), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Eliminar subplot extra
    axes[-1].remove()

    plt.suptitle('Comparación de Modelos - Métricas Clave\nAntes vs Después de CrossEncoder',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()

    # Guardar
    filename = "model_ranking_bars.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_combined_before_after_by_metric(metric_family: str):
    """Genera gráfico combinado mostrando antes/después para todos los modelos en una sola métrica"""
    data = load_results()
    results = data['results']

    fig, ax = plt.subplots(figsize=(12, 7))

    for model_key in ['ada', 'mpnet', 'minilm', 'e5-large']:
        if model_key not in results:
            continue

        model_data = results[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)
        color = MODEL_COLORS[model_key]

        before_metrics = model_data['avg_before_metrics']
        after_metrics = model_data.get('avg_after_metrics', {})

        # Extraer valores
        before_values = extract_metric_values_by_k(before_metrics, metric_family, K_VALUES)
        after_values = extract_metric_values_by_k(after_metrics, metric_family, K_VALUES)

        # Plotear BEFORE (línea sólida)
        ax.plot(K_VALUES, before_values,
                marker='o',
                label=f'{model_name} (Antes)',
                color=color,
                linewidth=2.5,
                markersize=5,
                linestyle='-')

        # Plotear AFTER (línea punteada)
        if after_values and any(v for v in after_values if not np.isnan(v)):
            ax.plot(K_VALUES, after_values,
                    marker='s',
                    label=f'{model_name} (Después)',
                    color=color,
                    linewidth=2,
                    markersize=4,
                    linestyle='--',
                    alpha=0.7)

    ax.set_xlabel('k (número de documentos recuperados)', fontweight='bold')
    ax.set_ylabel(f'{metric_family.upper()}@k', fontweight='bold')
    ax.set_title(f'{metric_family.upper()} por k - Comparación Antes vs Después de CrossEncoder\n(Todos los Modelos)',
                 fontweight='bold', fontsize=12)

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(K_VALUES)

    plt.tight_layout()

    # Guardar
    filename = f"{metric_family}_combined_before_after.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_ragas_metrics():
    """Genera gráfico de barras con métricas RAGAS para todos los modelos"""
    data = load_results()
    results = data['results']

    # Métricas RAGAS (con prefijo avg_ como están en el JSON)
    ragas_metrics = [
        'avg_faithfulness',
        'avg_answer_relevance',
        'avg_answer_correctness',
        'avg_context_precision',
        'avg_context_recall',
        'avg_semantic_similarity'
    ]

    ragas_labels = [
        'Faithfulness',
        'Answer Relevance',
        'Answer Correctness',
        'Context Precision',
        'Context Recall',
        'Semantic Similarity'
    ]

    # Recolectar valores
    model_data_dict = {}
    for model_key in ['ada', 'mpnet', 'e5-large', 'minilm']:
        if model_key not in results:
            continue

        model_name = MODEL_NAMES.get(model_key, model_key)
        rag_data = results[model_key].get('rag_metrics', {})

        values = []
        for metric in ragas_metrics:
            val = rag_data.get(metric, np.nan)
            values.append(val if val is not None else np.nan)

        model_data_dict[model_name] = values

    if not model_data_dict:
        print("⚠️  No hay datos RAGAS disponibles")
        return None

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(ragas_labels))
    width = 0.2

    for i, (model_name, values) in enumerate(model_data_dict.items()):
        # Encontrar color del modelo
        color = None
        for model_key, name in MODEL_NAMES.items():
            if name == model_name:
                color = MODEL_COLORS[model_key]
                break

        offset = (i - len(model_data_dict)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.8)

    ax.set_xlabel('Métrica RAGAS', fontweight='bold')
    ax.set_ylabel('Valor', fontweight='bold')
    ax.set_title('Métricas RAGAS por Modelo', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ragas_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    # Guardar
    filename = "ragas_metrics_comparison.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def plot_bertscore_metrics():
    """Genera gráfico de barras con métricas BERTScore para todos los modelos"""
    data = load_results()
    results = data['results']

    # Métricas BERTScore (con prefijo avg_ como están en el JSON)
    bertscore_metrics = ['avg_bert_precision', 'avg_bert_recall', 'avg_bert_f1']
    bertscore_labels = ['BERTScore Precision', 'BERTScore Recall', 'BERTScore F1']

    # Recolectar valores
    model_data_dict = {}
    for model_key in ['ada', 'mpnet', 'e5-large', 'minilm']:
        if model_key not in results:
            continue

        model_name = MODEL_NAMES.get(model_key, model_key)
        rag_data = results[model_key].get('rag_metrics', {})

        values = []
        for metric in bertscore_metrics:
            val = rag_data.get(metric, np.nan)
            # Si avg_bert_f1 es null, calcularlo como F1 = 2 * (P * R) / (P + R)
            if metric == 'avg_bert_f1' and (val is None or np.isnan(val)):
                precision = rag_data.get('avg_bert_precision', np.nan)
                recall = rag_data.get('avg_bert_recall', np.nan)
                if precision and recall and not np.isnan(precision) and not np.isnan(recall):
                    val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
            values.append(val if val is not None else np.nan)

        model_data_dict[model_name] = values

    if not model_data_dict:
        print("⚠️  No hay datos BERTScore disponibles")
        return None

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bertscore_labels))
    width = 0.2

    for i, (model_name, values) in enumerate(model_data_dict.items()):
        # Encontrar color del modelo
        color = None
        for model_key, name in MODEL_NAMES.items():
            if name == model_name:
                color = MODEL_COLORS[model_key]
                break

        offset = (i - len(model_data_dict)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.8)

    ax.set_xlabel('Métrica BERTScore', fontweight='bold')
    ax.set_ylabel('Valor', fontweight='bold')
    ax.set_title('Métricas BERTScore por Modelo', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bertscore_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    # Guardar
    filename = "bertscore_metrics_comparison.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico generado: {filename}")
    return filepath


def main():
    """Función principal para generar todos los gráficos"""
    print("=" * 60)
    print("GENERACIÓN DE GRÁFICOS - CAPÍTULO 7")
    print("=" * 60)
    print()

    # Crear directorio de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_families = ['precision', 'recall', 'f1', 'ndcg', 'map']
    models = ['ada', 'mpnet', 'minilm', 'e5-large']

    total_charts = 0

    # 1. Gráficos de métricas por k (todos los modelos)
    print("\n[1] Generando gráficos de métricas por k (todos los modelos)...")
    for metric in metric_families:
        plot_metric_by_k_all_models(metric, 'before')
        plot_metric_by_k_all_models(metric, 'after')
        total_charts += 2

    # 2. Gráficos de comparación antes/después por modelo
    print("\n[2] Generando gráficos de comparación antes/después...")
    for model in models:
        for metric in metric_families:
            plot_metric_comparison_before_after(metric, model)
            total_charts += 1

    # 3. Gráficos de todas las métricas para cada modelo
    print("\n[3] Generando gráficos de todas las métricas por modelo...")
    for model in models:
        plot_all_metrics_single_model(model, 'before')
        plot_all_metrics_single_model(model, 'after')
        total_charts += 2

    # 4. Heatmap de deltas
    print("\n[4] Generando heatmap de cambios...")
    plot_delta_heatmap()
    total_charts += 1

    # 5. Gráfico de barras de ranking
    print("\n[5] Generando gráfico de ranking de modelos...")
    plot_model_ranking_bar()
    total_charts += 1

    # 6. Gráficos combinados before/after por métrica
    print("\n[6] Generando gráficos combinados before/after por métrica...")
    for metric in metric_families:
        plot_combined_before_after_by_metric(metric)
        total_charts += 1

    # 7. Gráficos de métricas RAGAS (NUEVO)
    print("\n[7] Generando gráfico de métricas RAGAS...")
    plot_ragas_metrics()
    total_charts += 1

    # 8. Gráficos de métricas BERTScore (NUEVO)
    print("\n[8] Generando gráfico de métricas BERTScore...")
    plot_bertscore_metrics()
    total_charts += 1

    print("\n" + "=" * 60)
    print(f"✅ {total_charts} GRÁFICOS GENERADOS EXITOSAMENTE")
    print("=" * 60)
    print(f"\nGráficos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
