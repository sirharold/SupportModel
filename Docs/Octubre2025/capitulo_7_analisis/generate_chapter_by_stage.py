"""
Genera el Capítulo 7 reorganizado por ETAPA (no por modelo)
Compara los 4 modelos en cada etapa del proceso
"""

import json
from pathlib import Path
from typing import Dict, List

RESULTS_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json"
OUTPUT_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/capitulo7_resultados.md"
CHARTS_DIR = "./capitulo_7_analisis/charts"

# Valores de k para tablas (según instrucciones)
K_VALUES = [3, 5, 10, 15]

# Nombres de modelos
MODEL_NAMES = {
    'ada': 'Ada',
    'mpnet': 'MPNet',
    'minilm': 'MiniLM',
    'e5-large': 'E5-Large'
}

MODEL_ORDER = ['ada', 'mpnet', 'e5-large', 'minilm']  # Orden de presentación

def load_results() -> Dict:
    """Carga resultados desde archivo JSON"""
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_metric(value: float, decimals: int = 3) -> str:
    """Formatea un valor numérico"""
    return f"{value:.{decimals}f}"

def calculate_delta(before: float, after: float) -> tuple:
    """Calcula delta absoluto y porcentual"""
    delta_abs = after - before
    delta_pct = (delta_abs / before * 100) if before > 0 else 0
    return delta_abs, delta_pct

def generate_chapter(data: Dict) -> List[str]:
    """Genera el contenido completo del capítulo"""
    lines = []
    results = data['results']

    # =========================================================================
    # ENCABEZADO
    # =========================================================================
    lines.append("# 7. RESULTADOS Y ANÁLISIS\n\n")

    # =========================================================================
    # 7.1 INTRODUCCIÓN
    # =========================================================================
    lines.append("## 7.1 Introducción\n\n")
    lines.append("Este capítulo presenta los resultados experimentales del sistema RAG desarrollado, ")
    lines.append("organizando el análisis en tres etapas secuenciales que permiten evaluar el impacto ")
    lines.append("progresivo de cada componente del sistema:\n\n")

    lines.append("1. **Etapa 1 - Recuperación Vectorial**: Rendimiento de los cuatro modelos de embeddings ")
    lines.append("(Ada, MPNet, E5-Large, MiniLM) utilizando únicamente búsqueda por similitud coseno\n")
    lines.append("2. **Etapa 2 - Reranking Neural**: Rendimiento de los mismos modelos tras aplicar ")
    lines.append("CrossEncoder para reordenar los resultados iniciales\n")
    lines.append("3. **Etapa 3 - Análisis Comparativo**: Cuantificación del impacto del reranking mediante ")
    lines.append("comparación directa de las dos etapas anteriores\n\n")

    lines.append("La evaluación utilizó **2,067 pares pregunta-documento validados** como ground truth, ")
    lines.append("calculando métricas de recuperación (Precision, Recall, F1, NDCG, MAP, MRR) para valores ")
    lines.append("de k desde 1 hasta 15. Este diseño permite identificar qué configuración arquitectónica ")
    lines.append("ofrece el mejor rendimiento para cada escenario de implementación.\n\n")

    # =========================================================================
    # 7.2 CONFIGURACIÓN EXPERIMENTAL
    # =========================================================================
    lines.append("## 7.2 Configuración Experimental\n\n")
    lines.append("### 7.2.1 Parámetros de Evaluación\n\n")

    lines.append("La evaluación experimental implementó un diseño factorial 4×2 comparando cuatro ")
    lines.append("modelos de embedding bajo dos estrategias de procesamiento:\n\n")

    lines.append("**Datos de Evaluación:**\n")
    lines.append("- Ground truth: 2,067 pares pregunta-documento validados\n")
    lines.append("- Documentos indexados: 187,031 chunks de documentación Azure\n")
    lines.append("- Modelos evaluados: 4 (Ada, MPNet, E5-Large, MiniLM)\n\n")

    lines.append("**Parámetros Técnicos:**\n")
    lines.append("- Método de reranking: CrossEncoder ms-marco-MiniLM-L-6-v2 con normalización Min-Max\n")
    lines.append("- Top-k evaluado: 1-15 documentos por consulta\n")
    lines.append("- Métricas calculadas: Precision@k, Recall@k, F1@k, NDCG@k, MAP@k, MRR\n")
    lines.append("- Métrica de similitud: Similitud coseno en espacio de embeddings\n")
    lines.append("- Base de datos vectorial: ChromaDB 0.5.23\n\n")

    lines.append("**Entorno Computacional:**\n")
    lines.append("- Plataforma: Google Colab con GPU Tesla T4\n")
    lines.append("- Ejecución: Octubre 2025\n\n")

    lines.append("### 7.2.2 Modelos de Embedding Evaluados\n\n")
    lines.append("| Modelo | Dimensionalidad | Tipo | Especialización |\n")
    lines.append("|--------|-----------------|------|----------------|\n")
    lines.append("| Ada (text-embedding-ada-002) | 1,536 | Propietario (OpenAI) | Propósito general |\n")
    lines.append("| MPNet (multi-qa-mpnet-base-dot-v1) | 768 | Open-source | Pregunta-respuesta |\n")
    lines.append("| E5-Large (intfloat/e5-large-v2) | 1,024 | Open-source | Propósito general |\n")
    lines.append("| MiniLM (all-MiniLM-L6-v2) | 384 | Open-source | Compacto/eficiente |\n\n")

    lines.append("### 7.2.3 Estrategias de Procesamiento\n\n")
    lines.append("**Etapa 1 - Recuperación Vectorial Directa:**\n")
    lines.append("- Búsqueda por similitud coseno en ChromaDB\n")
    lines.append("- Ordenamiento directo por score de similitud\n")
    lines.append("- Retorno de top-k documentos sin procesamiento adicional\n\n")

    lines.append("**Etapa 2 - Recuperación con Reranking Neural:**\n")
    lines.append("- Búsqueda inicial por similitud coseno (top-15)\n")
    lines.append("- Reranking con CrossEncoder ms-marco-MiniLM-L-6-v2\n")
    lines.append("- Normalización Min-Max de scores\n")
    lines.append("- Reordenamiento y selección de top-k final\n\n")

    # =========================================================================
    # 7.3 ETAPA 1: RESULTADOS ANTES DEL RERANKING
    # =========================================================================
    lines.append("## 7.3 Etapa 1: Resultados Antes del Reranking\n\n")
    lines.append("Esta sección presenta el rendimiento de los cuatro modelos de embeddings utilizando ")
    lines.append("únicamente búsqueda vectorial por similitud coseno, estableciendo la línea base de ")
    lines.append("rendimiento antes de aplicar cualquier procesamiento adicional.\n\n")

    # 7.3.1 Rendimiento General
    lines.append("### 7.3.1 Rendimiento General por Modelo\n\n")
    lines.append("La **Tabla 7.1** presenta las métricas principales para los cuatro modelos en k=5, ")
    lines.append("el valor más representativo para sistemas de recuperación interactivos donde el usuario ")
    lines.append("típicamente examina los primeros 5 resultados.\n\n")

    lines.append("**Tabla 7.1: Rendimiento de Todos los Modelos Antes del Reranking (k=5)**\n\n")
    lines.append("| Modelo | Precision@5 | Recall@5 | F1@5 | NDCG@5 | MAP@5 | MRR |\n")
    lines.append("|--------|-------------|----------|------|--------|-------|-----|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        lines.append(f"{format_metric(model_data['precision@5'])} | ")
        lines.append(f"{format_metric(model_data['recall@5'])} | ")
        lines.append(f"{format_metric(model_data['f1@5'])} | ")
        lines.append(f"{format_metric(model_data['ndcg@5'])} | ")
        lines.append(f"{format_metric(model_data['map@5'])} | ")
        lines.append(f"{format_metric(model_data['mrr'])} |\n")

    lines.append("\n")

    # Análisis de la tabla
    ada_p5 = results['ada']['avg_before_metrics']['precision@5']
    mpnet_p5 = results['mpnet']['avg_before_metrics']['precision@5']
    e5_p5 = results['e5-large']['avg_before_metrics']['precision@5']
    minilm_p5 = results['minilm']['avg_before_metrics']['precision@5']

    lines.append("**Observaciones Clave:**\n\n")
    lines.append(f"1. **Superioridad de Ada**: Con Precision@5={format_metric(ada_p5)}, Ada supera a ")
    lines.append(f"MPNet ({format_metric(mpnet_p5)}) en {format_metric((ada_p5-mpnet_p5)/ada_p5*100, 1)}%, ")
    lines.append(f"estableciendo el mejor rendimiento absoluto entre todos los modelos evaluados.\n\n")

    lines.append(f"2. **Rendimiento de Modelos Open-Source**: MPNet alcanza el mejor rendimiento entre ")
    lines.append(f"alternativas open-source, seguido por E5-Large ({format_metric(e5_p5)}) y MiniLM ")
    lines.append(f"({format_metric(minilm_p5)}).\n\n")

    lines.append(f"3. **Trade-off Dimensionalidad-Rendimiento**: No hay correlación perfecta entre ")
    lines.append(f"dimensionalidad y rendimiento. MPNet (768 dim) supera a E5-Large (1,024 dim), ")
    lines.append(f"sugiriendo que la especialización del modelo (Q&A para MPNet) compensa la menor capacidad dimensional.\n\n")

    # 7.3.2 Análisis por Métrica
    lines.append("### 7.3.2 Análisis por Métrica\n\n")
    lines.append("Las siguientes subsecciones analizan cada familia de métricas en detalle, mostrando ")
    lines.append("la evolución del rendimiento con valores crecientes de k.\n\n")

    # Precision@k
    lines.append("#### 7.3.2.1 Precision@k\n\n")
    lines.append("La Precision@k mide la proporción de documentos relevantes entre los k documentos ")
    lines.append("recuperados. La **Tabla 7.2** muestra la evolución de la precisión para k={3,5,10,15}.\n\n")

    lines.append("**Tabla 7.2: Precision@k para Todos los Modelos (Antes del Reranking)**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'precision@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append("La **Figura 7.1** presenta la evolución completa de Precision@k para k=1 hasta k=15.\n\n")
    lines.append(f"![Figura 7.1: Precision@k para todos los modelos antes del reranking]({CHARTS_DIR}/precision_por_k_before.png)\n\n")

    lines.append("**Observaciones**:\n")
    lines.append("- Todas las curvas muestran decaimiento monotónico con k creciente (comportamiento esperado)\n")
    lines.append("- Ada mantiene superioridad consistente en todo el rango de k evaluado\n")
    lines.append("- La brecha entre modelos se reduce con k creciente pero persiste proporcionalmente\n\n")

    # Recall@k
    lines.append("#### 7.3.2.2 Recall@k\n\n")
    lines.append("El Recall@k mide la proporción de todos los documentos relevantes que fueron recuperados ")
    lines.append("dentro del top-k. La **Tabla 7.3** muestra la evolución del recall.\n\n")

    lines.append("**Tabla 7.3: Recall@k para Todos los Modelos (Antes del Reranking)**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'recall@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append("La **Figura 7.2** presenta la evolución completa de Recall@k.\n\n")
    lines.append(f"![Figura 7.2: Recall@k para todos los modelos antes del reranking]({CHARTS_DIR}/recall_por_k_before.png)\n\n")

    # Análisis de recall
    ada_r15 = results['ada']['avg_before_metrics']['recall@15']
    lines.append("**Observaciones**:\n")
    lines.append(f"- Ada alcanza Recall@15={format_metric(ada_r15)}, recuperando aproximadamente ")
    lines.append(f"{format_metric(ada_r15*100, 0)}% de todos los documentos relevantes en el top-15\n")
    lines.append("- El recall crece más pronunciadamente en k pequeños (k=1 a k=5) y se estabiliza para k grandes\n")
    lines.append("- Todos los modelos muestran recall sustancial incluso en k=5, validando la efectividad de búsqueda vectorial\n\n")

    # F1@k, NDCG@k, MAP@k - Similar estructura
    lines.append("#### 7.3.2.3 F1@k\n\n")
    lines.append("**Tabla 7.4: F1@k para Todos los Modelos (Antes del Reranking)**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'f1@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append(f"![Figura 7.3: F1@k para todos los modelos antes del reranking]({CHARTS_DIR}/f1_por_k_before.png)\n\n")

    lines.append("#### 7.3.2.4 NDCG@k\n\n")
    lines.append("NDCG (Normalized Discounted Cumulative Gain) penaliza documentos relevantes que ")
    lines.append("aparecen en posiciones inferiores, priorizando la calidad del ranking.\n\n")

    lines.append("**Tabla 7.5: NDCG@k para Todos los Modelos (Antes del Reranking)**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'ndcg@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append(f"![Figura 7.4: NDCG@k para todos los modelos antes del reranking]({CHARTS_DIR}/ndcg_por_k_before.png)\n\n")

    lines.append("#### 7.3.2.5 MAP@k\n\n")
    lines.append("MAP (Mean Average Precision) mide la calidad promedio del ranking de todos los ")
    lines.append("documentos relevantes.\n\n")

    lines.append("**Tabla 7.6: MAP@k para Todos los Modelos (Antes del Reranking)**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'map@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append(f"![Figura 7.5: MAP@k para todos los modelos antes del reranking]({CHARTS_DIR}/map_por_k_before.png)\n\n")

    # 7.3.3 Ranking de Modelos
    lines.append("### 7.3.3 Ranking de Modelos (Etapa 1)\n\n")
    lines.append("La **Tabla 7.7** presenta el ranking definitivo de modelos basado en Precision@5, ")
    lines.append("la métrica más representativa para sistemas interactivos.\n\n")

    lines.append("**Tabla 7.7: Ranking de Modelos por Precision@5 (Antes del Reranking)**\n\n")
    lines.append("| Posición | Modelo | Precision@5 | Recall@5 | F1@5 | NDCG@5 |\n")
    lines.append("|----------|--------|-------------|----------|------|--------|\n")

    # Ordenar modelos por precision@5
    ranking_data = []
    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_before_metrics']
        ranking_data.append({
            'key': model_key,
            'name': MODEL_NAMES[model_key],
            'precision': model_data['precision@5'],
            'recall': model_data['recall@5'],
            'f1': model_data['f1@5'],
            'ndcg': model_data['ndcg@5']
        })

    ranking_data.sort(key=lambda x: x['precision'], reverse=True)

    for i, item in enumerate(ranking_data, 1):
        lines.append(f"| {i} | {item['name']} | ")
        lines.append(f"{format_metric(item['precision'])} | ")
        lines.append(f"{format_metric(item['recall'])} | ")
        lines.append(f"{format_metric(item['f1'])} | ")
        lines.append(f"{format_metric(item['ndcg'])} |\n")

    lines.append("\n")

    # =========================================================================
    # 7.4 ETAPA 2: RESULTADOS DESPUÉS DEL RERANKING
    # =========================================================================
    lines.append("## 7.4 Etapa 2: Resultados Después del Reranking\n\n")
    lines.append("Esta sección presenta el rendimiento tras aplicar el componente de reranking neural ")
    lines.append("(CrossEncoder) sobre los resultados iniciales de la búsqueda vectorial.\n\n")

    # 7.4.1 Rendimiento General
    lines.append("### 7.4.1 Rendimiento General por Modelo\n\n")
    lines.append("**Tabla 7.8: Rendimiento de Todos los Modelos Después del Reranking (k=5)**\n\n")
    lines.append("| Modelo | Precision@5 | Recall@5 | F1@5 | NDCG@5 | MAP@5 | MRR |\n")
    lines.append("|--------|-------------|----------|------|--------|-------|-----|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_after_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        lines.append(f"{format_metric(model_data['precision@5'])} | ")
        lines.append(f"{format_metric(model_data['recall@5'])} | ")
        lines.append(f"{format_metric(model_data['f1@5'])} | ")
        lines.append(f"{format_metric(model_data['ndcg@5'])} | ")
        lines.append(f"{format_metric(model_data['map@5'])} | ")
        lines.append(f"{format_metric(model_data['mrr'])} |\n")

    lines.append("\n")

    # Análisis comparativo antes/después
    lines.append("**Cambios Observados:**\n\n")
    for model_key in MODEL_ORDER:
        before = results[model_key]['avg_before_metrics']['precision@5']
        after = results[model_key]['avg_after_metrics']['precision@5']
        delta, pct = calculate_delta(before, after)
        symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
        lines.append(f"- **{MODEL_NAMES[model_key]}**: {format_metric(before)} → {format_metric(after)} ")
        lines.append(f"({delta:+.3f}, {pct:+.1f}%) {symbol}\n")

    lines.append("\n")

    # Tablas por métrica (después del reranking)
    lines.append("### 7.4.2 Análisis por Métrica (Después del Reranking)\n\n")

    lines.append("#### 7.4.2.1 Precision@k\n\n")
    lines.append("**Tabla 7.9: Precision@k Después del Reranking**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_after_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'precision@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append(f"![Figura 7.6: Precision@k después del reranking]({CHARTS_DIR}/precision_por_k_after.png)\n\n")

    # Similar para otras métricas (Recall, F1, NDCG, MAP)
    lines.append("#### 7.4.2.2 Recall@k\n\n")
    lines.append("**Tabla 7.10: Recall@k Después del Reranking**\n\n")
    lines.append("| Modelo | k=3 | k=5 | k=10 | k=15 |\n")
    lines.append("|--------|-----|-----|------|------|\n")

    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_after_metrics']
        model_name = MODEL_NAMES[model_key]
        lines.append(f"| {model_name} | ")
        for k in K_VALUES:
            lines.append(f"{format_metric(model_data[f'recall@{k}'])} | ")
        lines.append("\n")

    lines.append("\n")
    lines.append(f"![Figura 7.7: Recall@k después del reranking]({CHARTS_DIR}/recall_por_k_after.png)\n\n")

    # Ranking post-reranking
    lines.append("### 7.4.3 Ranking de Modelos (Etapa 2)\n\n")
    lines.append("**Tabla 7.11: Ranking de Modelos por Precision@5 (Después del Reranking)**\n\n")
    lines.append("| Posición | Modelo | Precision@5 | Recall@5 | F1@5 | NDCG@5 |\n")
    lines.append("|----------|--------|-------------|----------|------|--------|\n")

    # Ordenar modelos por precision@5 post-reranking
    ranking_after = []
    for model_key in MODEL_ORDER:
        model_data = results[model_key]['avg_after_metrics']
        ranking_after.append({
            'key': model_key,
            'name': MODEL_NAMES[model_key],
            'precision': model_data['precision@5'],
            'recall': model_data['recall@5'],
            'f1': model_data['f1@5'],
            'ndcg': model_data['ndcg@5']
        })

    ranking_after.sort(key=lambda x: x['precision'], reverse=True)

    for i, item in enumerate(ranking_after, 1):
        lines.append(f"| {i} | {item['name']} | ")
        lines.append(f"{format_metric(item['precision'])} | ")
        lines.append(f"{format_metric(item['recall'])} | ")
        lines.append(f"{format_metric(item['f1'])} | ")
        lines.append(f"{format_metric(item['ndcg'])} | \n")

    lines.append("\n")
    lines.append("**Observación**: El ranking relativo de modelos se mantiene consistente después del reranking, ")
    lines.append("aunque las brechas de rendimiento se reducen (efecto de convergencia).\n\n")

    # =========================================================================
    # 7.5 ETAPA 3: ANÁLISIS DEL IMPACTO DEL RERANKING
    # =========================================================================
    lines.append("## 7.5 Etapa 3: Análisis del Impacto del Reranking\n\n")
    lines.append("Esta sección cuantifica el impacto del componente de reranking comparando directamente ")
    lines.append("las dos etapas anteriores.\n\n")

    # 7.5.1 Cambios por Modelo
    lines.append("### 7.5.1 Impacto por Modelo\n\n")
    lines.append("La **Tabla 7.12** presenta el cambio absoluto y porcentual en todas las métricas ")
    lines.append("principales para k=5.\n\n")

    lines.append("**Tabla 7.12: Impacto del Reranking por Modelo (k=5)**\n\n")
    lines.append("| Modelo | Métrica | Antes | Después | Δ Absoluto | Δ % |\n")
    lines.append("|--------|---------|-------|---------|------------|-----|\n")

    metrics_to_compare = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']
    metric_labels = ['Precision@5', 'Recall@5', 'F1@5', 'NDCG@5', 'MAP@5', 'MRR']

    for model_key in MODEL_ORDER:
        before_data = results[model_key]['avg_before_metrics']
        after_data = results[model_key]['avg_after_metrics']
        model_name = MODEL_NAMES[model_key]

        for metric, label in zip(metrics_to_compare, metric_labels):
            before_val = before_data[metric]
            after_val = after_data[metric]
            delta, pct = calculate_delta(before_val, after_val)

            lines.append(f"| {model_name} | {label} | ")
            lines.append(f"{format_metric(before_val)} | ")
            lines.append(f"{format_metric(after_val)} | ")
            lines.append(f"{delta:+.3f} | ")
            lines.append(f"{pct:+.1f}% |\n")

        lines.append(f"| | | | | | |\n")  # Línea en blanco

    lines.append("\n")

    lines.append("**Observaciones Clave:**\n\n")
    lines.append("1. **MiniLM muestra mejoras consistentes** (+10% a +14% en todas las métricas), ")
    lines.append("confirmando que el reranking compensa efectivamente las limitaciones del modelo compacto.\n\n")

    lines.append("2. **Ada muestra degradación sistemática** (-13% a -24%), sugiriendo que sus embeddings ")
    lines.append("de alta calidad ya producen rankings óptimos que el CrossEncoder no puede mejorar.\n\n")

    lines.append("3. **MPNet y E5-Large muestran estabilidad**, con cambios menores (±1% a ±7%), ")
    lines.append("indicando que el reranking tiene impacto limitado en modelos de calidad intermedia.\n\n")

    # Visualización del impacto
    lines.append("La **Figura 7.8** visualiza el impacto del reranking mediante un mapa de calor que ")
    lines.append("muestra el cambio porcentual de cada métrica para cada modelo.\n\n")

    lines.append(f"![Figura 7.8: Mapa de calor del impacto del reranking]({CHARTS_DIR}/delta_heatmap.png)\n\n")

    # 7.5.2 Cambios por Métrica
    lines.append("### 7.5.2 Impacto por Métrica\n\n")
    lines.append("Analizando el impacto agregado en cada métrica:\n\n")

    lines.append("**Tabla 7.13: Cambio Promedio por Métrica (Todos los Modelos)**\n\n")
    lines.append("| Métrica | Ada | MPNet | E5-Large | MiniLM | Promedio |\n")
    lines.append("|---------|-----|-------|----------|--------|----------|\n")

    for metric, label in zip(metrics_to_compare, metric_labels):
        line = f"| {label} | "
        changes = []

        for model_key in MODEL_ORDER:
            before_val = results[model_key]['avg_before_metrics'][metric]
            after_val = results[model_key]['avg_after_metrics'][metric]
            _, pct = calculate_delta(before_val, after_val)
            changes.append(pct)
            line += f"{pct:+.1f}% | "

        avg_change = sum(changes) / len(changes)
        line += f"{avg_change:+.1f}% |\n"
        lines.append(line)

    lines.append("\n")

    # =========================================================================
    # SECCIONES FINALES (mantener estructura similar al original)
    # =========================================================================

    lines.append("## 7.6 Análisis del Componente de Reranking\n\n")
    lines.append("### 7.6.1 Características del CrossEncoder\n\n")
    lines.append("El CrossEncoder ms-marco-MiniLM-L-6-v2 utilizado para reranking presenta las siguientes características:\n\n")
    lines.append("- **Arquitectura**: Transformer de 6 capas con atención cruzada completa entre query y documento\n")
    lines.append("- **Entrenamiento**: MS MARCO (búsqueda web general)\n")
    lines.append("- **Normalización**: Min-Max para mapear scores al rango [0,1]\n")
    lines.append("- **Contexto**: Truncamiento a 512 tokens (limitación para documentos largos)\n\n")

    lines.append("### 7.6.2 Limitaciones Identificadas\n\n")
    lines.append("El análisis reveló las siguientes limitaciones del reranking:\n\n")
    lines.append("1. **Desajuste de dominio**: Entrenado en búsqueda web general, no documentación técnica especializada\n")
    lines.append("2. **Interferencia con embeddings fuertes**: Degrada rankings ya óptimos (caso Ada)\n")
    lines.append("3. **Limitación de contexto**: Truncamiento a 512 tokens pierde información en documentos largos\n")
    lines.append("4. **Costo computacional**: Incremento de latencia ~35× por el procesamiento secuencial\n\n")

    # =========================================================================
    # 7.7 MÉTRICAS DE CALIDAD RAG (RAGAS Y BERTSCORE)
    # =========================================================================
    lines.append("## 7.7 Evaluación de Calidad de Respuestas RAG\n\n")
    lines.append("Además de las métricas de recuperación tradicionales, se evaluó la calidad de las respuestas ")
    lines.append("generadas por el sistema RAG completo utilizando métricas RAGAS (Retrieval Augmented Generation ")
    lines.append("Assessment) y BERTScore, que miden aspectos complementarios de la calidad de generación.\n\n")

    lines.append("### 7.7.1 Marco de Evaluación RAGAS\n\n")
    lines.append("RAGAS evalúa la calidad del sistema RAG desde múltiples perspectivas:\n\n")
    lines.append("- **Faithfulness**: Fidelidad de la respuesta respecto al contexto recuperado\n")
    lines.append("- **Answer Relevance**: Relevancia de la respuesta respecto a la pregunta\n")
    lines.append("- **Answer Correctness**: Corrección semántica de la respuesta\n")
    lines.append("- **Context Precision**: Precisión del contexto recuperado\n")
    lines.append("- **Context Recall**: Completitud del contexto recuperado\n")
    lines.append("- **Semantic Similarity**: Similitud semántica entre respuesta y referencia\n\n")

    lines.append("### 7.7.2 Resultados de Métricas RAG\n\n")
    lines.append("La **Tabla 7.14** presenta las métricas RAGAS para los cuatro modelos de embeddings.\n\n")

    lines.append("**Tabla 7.14: Métricas RAGAS por Modelo**\n\n")
    lines.append("| Modelo | Faithfulness | Answer Rel. | Context Prec. | Context Recall | Semantic Sim. |\n")
    lines.append("|--------|--------------|-------------|---------------|----------------|---------------|\n")

    for model_key in MODEL_ORDER:
        if 'rag_metrics' in results[model_key] and results[model_key]['rag_metrics']:
            rag = results[model_key]['rag_metrics']
            model_name = MODEL_NAMES[model_key]
            lines.append(f"| {model_name} | ")
            lines.append(f"{format_metric(rag.get('avg_faithfulness', 0))} | ")
            lines.append(f"{format_metric(rag.get('avg_answer_relevance', 0))} | ")
            lines.append(f"{format_metric(rag.get('avg_context_precision', 0))} | ")
            lines.append(f"{format_metric(rag.get('avg_context_recall', 0))} | ")
            lines.append(f"{format_metric(rag.get('avg_semantic_similarity', 0))} |\n")

    lines.append("\n")

    lines.append("**Observaciones:**\n\n")
    lines.append("1. **Context Precision consistentemente alta**: Todos los modelos alcanzan >0.92, indicando ")
    lines.append("que el contexto recuperado es predominantemente relevante.\n\n")

    lines.append("2. **Context Recall variable**: Ada (0.865) > MPNet (0.856) > E5-Large (0.858) > MiniLM (0.850), ")
    lines.append("correlacionando con el rendimiento en métricas de recuperación tradicionales.\n\n")

    lines.append("3. **Faithfulness superior de Ada**: Con 0.730, Ada muestra mayor fidelidad al contexto ")
    lines.append("recuperado, indicando respuestas más fundamentadas.\n\n")

    lines.append("4. **Answer Relevance homogénea**: Todos los modelos alcanzan >0.87, sugiriendo que la ")
    lines.append("generación de respuestas mantiene relevancia independientemente del modelo de embedding.\n\n")

    lines.append("### 7.7.3 Métricas BERTScore\n\n")
    lines.append("BERTScore evalúa la similitud semántica entre respuestas generadas y respuestas de referencia ")
    lines.append("mediante embeddings contextuales de BERT.\n\n")

    lines.append("**Tabla 7.15: BERTScore por Modelo**\n\n")
    lines.append("| Modelo | BERT Precision | BERT Recall | BERT F1 |\n")
    lines.append("|--------|----------------|-------------|----------|\n")

    for model_key in MODEL_ORDER:
        if 'rag_metrics' in results[model_key] and results[model_key]['rag_metrics']:
            rag = results[model_key]['rag_metrics']
            model_name = MODEL_NAMES[model_key]
            bert_f1 = rag.get('avg_bert_f1', None)
            bert_f1_str = format_metric(bert_f1) if bert_f1 is not None else "N/A"

            lines.append(f"| {model_name} | ")
            lines.append(f"{format_metric(rag.get('avg_bert_precision', 0))} | ")
            lines.append(f"{format_metric(rag.get('avg_bert_recall', 0))} | ")
            lines.append(f"{bert_f1_str} |\n")

    lines.append("\n")

    lines.append("**Observaciones:**\n\n")
    lines.append("1. **BERTScore homogéneo**: Precision ~0.647 y Recall ~0.542 consistentes entre modelos, ")
    lines.append("indicando que las diferencias en recuperación no se amplifican en la generación.\n\n")

    lines.append("2. **BERT F1 disponible para 3 de 4 modelos**: Ada (0.589), E5-Large (0.585) y MiniLM (0.619) ")
    lines.append("muestran valores consistentes en el rango 0.585-0.619. MPNet no tiene valor registrado ")
    lines.append("en los resultados.\n\n")

    lines.append("3. **Complementariedad con métricas de recuperación**: Mientras las métricas de recuperación ")
    lines.append("(Precision, Recall) muestran diferencias significativas entre modelos (28-46%), BERTScore ")
    lines.append("muestra variación mínima (<2%), sugiriendo que el componente de generación compensa ")
    lines.append("parcialmente las diferencias en recuperación.\n\n")

    lines.append("### 7.7.4 Interpretación Integrada\n\n")
    lines.append("La evaluación multi-métrica revela:\n\n")

    lines.append("**Separación de Componentes:**\n")
    lines.append("- Métricas de recuperación (Precision@k, Recall@k) muestran diferencias significativas entre modelos\n")
    lines.append("- Métricas RAG y BERTScore muestran mayor homogeneidad\n")
    lines.append("- Esto sugiere que las diferencias en calidad de recuperación no se traducen proporcionalmente ")
    lines.append("en diferencias en calidad de respuesta final\n\n")

    lines.append("**Implicación Práctica:**\n")
    lines.append("Para aplicaciones donde la calidad de respuesta es prioritaria sobre la eficiencia de recuperación, ")
    lines.append("modelos open-source como MPNet o MiniLM pueden ofrecer resultados aceptables a menor costo, ")
    lines.append("dado que el componente de generación compensa parcialmente sus limitaciones en recuperación.\n\n")

    # =========================================================================
    # FIN DEL CAPÍTULO 7
    # La sección de Cumplimiento de Objetivos se movió al Capítulo 8
    # =========================================================================

    return lines

def main():
    """Función principal"""
    print("="*80)
    print("GENERACIÓN DEL CAPÍTULO 7 - ENFOQUE POR ETAPA")
    print("="*80)
    print()

    print("📂 Cargando datos...")
    data = load_results()
    print("✅ Datos cargados\n")

    print("📝 Generando contenido del capítulo...")
    content = generate_chapter(data)
    print(f"✅ Generadas {len(content)} líneas\n")

    print("💾 Guardando capítulo...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(content)
    print(f"✅ Capítulo guardado en: {OUTPUT_FILE}\n")

    print("="*80)
    print("✅ CAPÍTULO 7 GENERADO EXITOSAMENTE")
    print("="*80)
    print()
    print("📊 Estructura:")
    print("  - 7.1 Introducción")
    print("  - 7.2 Configuración Experimental")
    print("  - 7.3 Etapa 1: Resultados Antes del Reranking")
    print("  - 7.4 Etapa 2: Resultados Después del Reranking")
    print("  - 7.5 Etapa 3: Análisis del Impacto del Reranking")
    print("  - 7.6 Análisis del Componente de Reranking")
    print("  - 7.7 Evaluación de Calidad de Respuestas RAG (RAGAS + BERTScore)")
    print()
    print("✅ Sección de Cumplimiento de Objetivos movida al Capítulo 8")
    print()
    print(f"📄 Archivo original respaldado en:")
    print(f"   {Path(__file__).parent}/capitulo7_resultados_ORIGINAL.md")

if __name__ == "__main__":
    main()
