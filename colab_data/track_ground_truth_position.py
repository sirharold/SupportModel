"""
Ground Truth Position Tracking
================================

Funciones para trackear la posición del documento correcto (ground truth)
antes y después del CrossEncoder reranking.

Uso:
1. Copiar estas funciones a tu notebook de Colab
2. Agregar después de la función normalize_url()
3. Usar en el loop principal de evaluación

Author: Claude Code
Date: 2025-10-31
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from urllib.parse import urlparse, urlunparse


def track_ground_truth_position(retrieved_docs: List[Dict], ground_truth_links: List[str]) -> Optional[int]:
    """
    Encuentra la posición del primer documento de ground truth en la lista recuperada.

    Parameters
    ----------
    retrieved_docs : list[dict]
        Lista de documentos recuperados, cada uno con al menos {'link': str}
    ground_truth_links : list[str]
        Lista de links del ground truth para esta pregunta

    Returns
    -------
    int or None
        Posición (1-indexed) del primer doc de ground truth encontrado.
        None si no se encuentra ninguno en la lista.

    Examples
    --------
    >>> docs = [
    ...     {'link': 'https://example.com/a', 'score': 0.9},
    ...     {'link': 'https://example.com/b', 'score': 0.8},
    ...     {'link': 'https://example.com/c', 'score': 0.7}
    ... ]
    >>> gt = ['https://example.com/b']
    >>> track_ground_truth_position(docs, gt)
    2
    """
    if not ground_truth_links or not retrieved_docs:
        return None

    # Normalizar ground truth links
    normalized_gt = {normalize_url(link) for link in ground_truth_links if link}

    # Buscar primera coincidencia
    for position, doc in enumerate(retrieved_docs, start=1):
        doc_link = normalize_url(doc.get('link', ''))
        if doc_link in normalized_gt:
            return position

    return None  # No encontrado en la lista


def calculate_position_stats(positions: List[int]) -> Dict[str, float]:
    """
    Calcula estadísticas sobre las posiciones del ground truth.

    Parameters
    ----------
    positions : list[int]
        Lista de posiciones donde se encontró el ground truth

    Returns
    -------
    dict
        Diccionario con estadísticas: mean, median, min, max, std
    """
    if not positions:
        return {
            'mean': None,
            'median': None,
            'min': None,
            'max': None,
            'std': None,
            'count': 0
        }

    return {
        'mean': float(np.mean(positions)),
        'median': float(np.median(positions)),
        'min': int(np.min(positions)),
        'max': int(np.max(positions)),
        'std': float(np.std(positions)),
        'count': len(positions)
    }


def analyze_crossencoder_impact(positions_before: List[int],
                                positions_after: List[int]) -> Dict[str, any]:
    """
    Analiza el impacto del CrossEncoder comparando posiciones antes y después.

    Parameters
    ----------
    positions_before : list[int]
        Posiciones antes del CrossEncoder
    positions_after : list[int]
        Posiciones después del CrossEncoder

    Returns
    -------
    dict
        Análisis detallado del impacto del CrossEncoder
    """
    if len(positions_before) != len(positions_after):
        raise ValueError("Las listas de posiciones deben tener la misma longitud")

    improved = 0
    worsened = 0
    unchanged = 0
    total_improvement = 0
    total_degradation = 0

    for pos_before, pos_after in zip(positions_before, positions_after):
        diff = pos_before - pos_after  # Positivo = mejora (subió en ranking)

        if diff > 0:
            improved += 1
            total_improvement += diff
        elif diff < 0:
            worsened += 1
            total_degradation += abs(diff)
        else:
            unchanged += 1

    total = len(positions_before)

    return {
        'total_questions': total,
        'improved': improved,
        'improved_pct': (improved / total * 100) if total > 0 else 0,
        'worsened': worsened,
        'worsened_pct': (worsened / total * 100) if total > 0 else 0,
        'unchanged': unchanged,
        'unchanged_pct': (unchanged / total * 100) if total > 0 else 0,
        'avg_improvement': (total_improvement / improved) if improved > 0 else 0,
        'avg_degradation': (total_degradation / worsened) if worsened > 0 else 0,
        'net_improvement': total_improvement - total_degradation,
        'stats_before': calculate_position_stats(positions_before),
        'stats_after': calculate_position_stats(positions_after)
    }


def print_position_analysis(analysis: Dict[str, any], model_name: str = ""):
    """
    Imprime un reporte legible del análisis de posiciones.

    Parameters
    ----------
    analysis : dict
        Resultado de analyze_crossencoder_impact()
    model_name : str, optional
        Nombre del modelo para el reporte
    """
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISIS DE POSICIÓN DEL GROUND TRUTH - {model_name.upper()}")
    print(f"{'='*70}")

    print(f"\n🎯 POSICIONES ANTES DEL CROSSENCODER:")
    stats_before = analysis['stats_before']
    if stats_before['count'] > 0:
        print(f"   Media:   {stats_before['mean']:.2f}")
        print(f"   Mediana: {stats_before['median']:.1f}")
        print(f"   Min/Max: {stats_before['min']} / {stats_before['max']}")
        print(f"   Std Dev: {stats_before['std']:.2f}")
        print(f"   Casos:   {stats_before['count']}")
    else:
        print("   ❌ No hay datos")

    print(f"\n🔄 POSICIONES DESPUÉS DEL CROSSENCODER:")
    stats_after = analysis['stats_after']
    if stats_after['count'] > 0:
        print(f"   Media:   {stats_after['mean']:.2f}")
        print(f"   Mediana: {stats_after['median']:.1f}")
        print(f"   Min/Max: {stats_after['min']} / {stats_after['max']}")
        print(f"   Std Dev: {stats_after['std']:.2f}")
        print(f"   Casos:   {stats_after['count']}")
    else:
        print("   ❌ No hay datos")

    print(f"\n📈 IMPACTO DEL CROSSENCODER:")
    print(f"   ✅ Mejoró:    {analysis['improved']:4d} casos ({analysis['improved_pct']:5.1f}%)")
    if analysis['improved'] > 0:
        print(f"      → Promedio de mejora: {analysis['avg_improvement']:.2f} posiciones")

    print(f"   ❌ Empeoró:   {analysis['worsened']:4d} casos ({analysis['worsened_pct']:5.1f}%)")
    if analysis['worsened'] > 0:
        print(f"      → Promedio de degradación: {analysis['avg_degradation']:.2f} posiciones")

    print(f"   ➡️  Sin cambio: {analysis['unchanged']:4d} casos ({analysis['unchanged_pct']:5.1f}%)")

    print(f"\n🎯 MEJORA NETA: {analysis['net_improvement']:+.2f} posiciones")

    # Interpretación
    print(f"\n💡 INTERPRETACIÓN:")
    if analysis['net_improvement'] > 0:
        print(f"   ✅ El CrossEncoder está AYUDANDO (mejora neta positiva)")
    elif analysis['net_improvement'] < 0:
        print(f"   ❌ El CrossEncoder está PERJUDICANDO (mejora neta negativa)")
    else:
        print(f"   ➡️  El CrossEncoder tiene impacto NEUTRO")

    if stats_before['count'] > 0 and stats_after['count'] > 0:
        delta_mean = stats_after['mean'] - stats_before['mean']
        if delta_mean < -1:
            print(f"   ✅ Posición promedio MEJORÓ en {abs(delta_mean):.2f} posiciones")
        elif delta_mean > 1:
            print(f"   ❌ Posición promedio EMPEORÓ en {delta_mean:.2f} posiciones")
        else:
            print(f"   ➡️  Posición promedio casi SIN CAMBIO ({delta_mean:+.2f})")

    print(f"{'='*70}\n")


# ============================================================================
# EJEMPLO DE INTEGRACIÓN EN EL CÓDIGO PRINCIPAL
# ============================================================================

def integration_example():
    """
    Ejemplo de cómo integrar estas funciones en tu código de evaluación.

    INSTRUCCIONES:
    --------------
    1. Copiar las funciones anteriores a tu notebook
    2. En tu loop principal de evaluación, agregar el tracking:

    # Inicializar trackers ANTES del loop
# The code snippet you provided is initializing two dictionaries `positions_before_ce` and
# `positions_after_ce` with keys representing different model names ('ada', 'mpnet', 'minilm',
# 'e5-large') and empty lists as their corresponding values. These dictionaries are intended to track
# the positions of the ground truth documents before and after applying the CrossEncoder reranking
# process for each model during evaluation.
    positions_before_ce = {
        'ada': [],
        'mpnet': [],
        'minilm': [],
        'e5-large': []
    }

    positions_after_ce = {
        'ada': [],
        'mpnet': [],
        'minilm': [],
        'e5-large': []
    }

    # DENTRO del loop de evaluación de cada pregunta:
    for model_name in ['ada', 'mpnet', 'minilm', 'e5-large']:

        # 1. Después de la búsqueda semántica (ANTES del CrossEncoder)
        docs_before = semantic_search(question, model_name, top_k=15)
        pos_before = track_ground_truth_position(docs_before, question['ground_truth_links'])
        if pos_before:
            positions_before_ce[model_name].append(pos_before)

        # 2. Después del CrossEncoder reranking
        docs_after = apply_crossencoder(docs_before)
        pos_after = track_ground_truth_position(docs_after, question['ground_truth_links'])
        if pos_after:
            positions_after_ce[model_name].append(pos_after)

        # ... continuar con el cálculo de métricas normal ...

    # DESPUÉS del loop, analizar resultados
    for model_name in ['ada', 'mpnet', 'minilm', 'e5-large']:
        if positions_before_ce[model_name] and positions_after_ce[model_name]:
            analysis = analyze_crossencoder_impact(
                positions_before_ce[model_name],
                positions_after_ce[model_name]
            )
            print_position_analysis(analysis, model_name)
    """
    pass


# ============================================================================
# VERSIÓN SIMPLIFICADA PARA DEBUGGING RÁPIDO
# ============================================================================

def quick_debug_position(docs_before, docs_after, gt_links):
    """
    Función simplificada para debugging rápido en una sola pregunta.

    Úsala cuando estés debuggeando una pregunta específica.
    """
    pos_before = track_ground_truth_position(docs_before, gt_links)
    pos_after = track_ground_truth_position(docs_after, gt_links)

    print(f"\n🔍 Posición del documento correcto:")
    print(f"   ANTES del CrossEncoder:  {pos_before if pos_before else 'No encontrado'}")
    print(f"   DESPUÉS del CrossEncoder: {pos_after if pos_after else 'No encontrado'}")

    if pos_before and pos_after:
        diff = pos_before - pos_after
        if diff > 0:
            print(f"   ✅ MEJORÓ {diff} posiciones")
        elif diff < 0:
            print(f"   ❌ EMPEORÓ {abs(diff)} posiciones")
        else:
            print(f"   ➡️  SIN CAMBIO")

    # Mostrar top-3 docs antes y después
    print(f"\n📄 Top-3 ANTES:")
    for i, doc in enumerate(docs_before[:3], start=1):
        score = doc.get('cosine_similarity', doc.get('score', 'N/A'))
        print(f"   {i}. {doc.get('title', 'Sin título')[:50]}... (score: {score:.3f})")

    print(f"\n📄 Top-3 DESPUÉS:")
    for i, doc in enumerate(docs_after[:3], start=1):
        score = doc.get('crossencoder_score', 'N/A')
        print(f"   {i}. {doc.get('title', 'Sin título')[:50]}... (CE: {score:.3f})")


if __name__ == "__main__":
    print("✅ Funciones de tracking cargadas correctamente")
    print("\nFunciones disponibles:")
    print("  - track_ground_truth_position()")
    print("  - calculate_position_stats()")
    print("  - analyze_crossencoder_impact()")
    print("  - print_position_analysis()")
    print("  - quick_debug_position()")
    print("\nVer integration_example() para ejemplo de uso")
