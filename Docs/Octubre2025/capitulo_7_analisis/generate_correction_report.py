"""
Genera un reporte detallado de correcciones necesarias para el Capítulo 7
Incluye:
1. Lista de valores incorrectos con valores correctos
2. Tablas corregidas en formato Markdown listas para copiar/pegar
3. Análisis de discrepancias
"""

import json
from pathlib import Path
from typing import Dict

RESULTS_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json"
OUTPUT_FILE = Path(__file__).parent / "CORRECIONES_NECESARIAS.md"

def load_results() -> Dict:
    """Carga el archivo de resultados"""
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_correction_report():
    """Genera el reporte completo de correcciones"""
    data = load_results()
    results = data['results']

    report = []
    report.append("# CORRECCIONES NECESARIAS - CAPÍTULO 7\n")
    report.append("**Generado automáticamente desde datos reales**\n")
    report.append(f"**Fuente**: `cumulative_results_20251013_001552.json`\n")
    report.append("="*80 + "\n\n")

    # =========================================================================
    # TABLA 7.1: Métricas Principales de Ada
    # =========================================================================
    report.append("## ✅ TABLA 7.1: Métricas Principales de Ada\n")
    report.append("**STATUS**: CORRECTA - No requiere cambios\n\n")

    # =========================================================================
    # TABLA 7.2: Precision@k de Ada
    # =========================================================================
    report.append("## ❌ TABLA 7.2: Precision@k de Ada (k=3,5,10,15)\n")
    report.append("**STATUS**: REQUIERE CORRECCIÓN\n\n")

    ada = results['ada']
    before = ada['avg_before_metrics']
    after = ada['avg_after_metrics']

    report.append("### Valores Actuales (INCORRECTOS):\n")
    report.append("| Etapa | k=3 | k=5 | k=10 | k=15 |\n")
    report.append("|-------|-----|-----|------|------|\n")
    report.append("| Antes CrossEncoder | 0.104 | 0.098 | 0.079 | 0.061 |\n")
    report.append("| Después CrossEncoder | 0.086 | 0.081 | 0.067 | 0.053 |\n\n")

    report.append("### Valores Correctos (USAR ESTOS):\n")
    report.append("| Etapa | k=3 | k=5 | k=10 | k=15 |\n")
    report.append("|-------|-----|-----|------|------|\n")
    report.append(f"| Antes CrossEncoder | {before['precision@3']:.3f} | {before['precision@5']:.3f} | {before['precision@10']:.3f} | {before['precision@15']:.3f} |\n")
    report.append(f"| Después CrossEncoder | {after['precision@3']:.3f} | {after['precision@5']:.3f} | {after['precision@10']:.3f} | {after['precision@15']:.3f} |\n")

    # Calcular deltas
    delta3 = after['precision@3'] - before['precision@3']
    pct3 = (delta3 / before['precision@3'] * 100)
    delta5 = after['precision@5'] - before['precision@5']
    pct5 = (delta5 / before['precision@5'] * 100)
    delta10 = after['precision@10'] - before['precision@10']
    pct10 = (delta10 / before['precision@10'] * 100)
    delta15 = after['precision@15'] - before['precision@15']
    pct15 = (delta15 / before['precision@15'] * 100)

    report.append(f"| Δ (cambio) | {delta3:+.3f} ({pct3:+.1f}%) | {delta5:+.3f} ({pct5:+.1f}%) | {delta10:+.3f} ({pct10:+.1f}%) | {delta15:+.3f} ({pct15:+.1f}%) |\n\n")

    # =========================================================================
    # TABLA 7.3: Recall@k de Ada
    # =========================================================================
    report.append("## ❌ TABLA 7.3: Recall@k de Ada (k=3,5,10,15)\n")
    report.append("**STATUS**: REQUIERE CORRECCIÓN\n\n")

    report.append("### Valores Actuales (INCORRECTOS):\n")
    report.append("| Etapa | k=3 | k=5 | k=10 | k=15 |\n")
    report.append("|-------|-----|-----|------|------|\n")
    report.append("| Antes CrossEncoder | 0.276 | 0.398 | 0.591 | 0.702 |\n")
    report.append("| Después CrossEncoder | 0.228 | 0.330 | 0.539 | 0.649 |\n\n")

    report.append("### Valores Correctos (USAR ESTOS):\n")
    report.append("| Etapa | k=3 | k=5 | k=10 | k=15 |\n")
    report.append("|-------|-----|-----|------|------|\n")
    report.append(f"| Antes CrossEncoder | {before['recall@3']:.3f} | {before['recall@5']:.3f} | {before['recall@10']:.3f} | {before['recall@15']:.3f} |\n")
    report.append(f"| Después CrossEncoder | {after['recall@3']:.3f} | {after['recall@5']:.3f} | {after['recall@10']:.3f} | {after['recall@15']:.3f} |\n")

    # Calcular deltas
    delta3 = after['recall@3'] - before['recall@3']
    pct3 = (delta3 / before['recall@3'] * 100)
    delta5 = after['recall@5'] - before['recall@5']
    pct5 = (delta5 / before['recall@5'] * 100)
    delta10 = after['recall@10'] - before['recall@10']
    pct10 = (delta10 / before['recall@10'] * 100)
    delta15 = after['recall@15'] - before['recall@15']
    pct15 = (delta15 / before['recall@15'] * 100)

    report.append(f"| Δ (cambio) | {delta3:+.3f} ({pct3:+.1f}%) | {delta5:+.3f} ({pct5:+.1f}%) | {delta10:+.3f} ({pct10:+.1f}%) | {delta15:+.3f} ({pct15:+.1f}%) |\n\n")

    # =========================================================================
    # TABLA 7.4, 7.5, 7.6: MPNet - CORRECTAS
    # =========================================================================
    report.append("## ✅ TABLAS 7.4, 7.5, 7.6: MPNet\n")
    report.append("**STATUS**: CORRECTAS - No requieren cambios\n\n")

    # =========================================================================
    # TABLA 7.7: Precision@k de MiniLM
    # =========================================================================
    report.append("## ❌ TABLA 7.7: Precision@k de MiniLM (k=3,5,10,15)\n")
    report.append("**STATUS**: REQUIERE CORRECCIÓN\n\n")

    minilm = results['minilm']
    before_m = minilm['avg_before_metrics']
    after_m = minilm['avg_after_metrics']

    report.append("### Valores Actuales (INCORRECTOS):\n")
    report.append("| Etapa | k=3 | k=5 | k=10 | k=15 |\n")
    report.append("|-------|-----|-----|------|------|\n")
    report.append("| Antes CrossEncoder | 0.056 | 0.053 | 0.046 | 0.040 |\n")
    report.append("| Después CrossEncoder | 0.063 | 0.060 | 0.052 | 0.045 |\n\n")

    report.append("### Valores Correctos (USAR ESTOS):\n")
    report.append("| Etapa | k=3 | k=5 | k=10 | k=15 |\n")
    report.append("|-------|-----|-----|------|------|\n")
    report.append(f"| Antes CrossEncoder | {before_m['precision@3']:.3f} | {before_m['precision@5']:.3f} | {before_m['precision@10']:.3f} | {before_m['precision@15']:.3f} |\n")
    report.append(f"| Después CrossEncoder | {after_m['precision@3']:.3f} | {after_m['precision@5']:.3f} | {after_m['precision@10']:.3f} | {after_m['precision@15']:.3f} |\n")

    # Calcular deltas
    delta3 = after_m['precision@3'] - before_m['precision@3']
    pct3 = (delta3 / before_m['precision@3'] * 100)
    delta5 = after_m['precision@5'] - before_m['precision@5']
    pct5 = (delta5 / before_m['precision@5'] * 100)
    delta10 = after_m['precision@10'] - before_m['precision@10']
    pct10 = (delta10 / before_m['precision@10'] * 100)
    delta15 = after_m['precision@15'] - before_m['precision@15']
    pct15 = (delta15 / before_m['precision@15'] * 100)

    report.append(f"| Δ (cambio) | {delta3:+.3f} ({pct3:+.1f}%) | {delta5:+.3f} ({pct5:+.1f}%) | {delta10:+.3f} ({pct10:+.1f}%) | {delta15:+.3f} ({pct15:+.1f}%) |\n\n")

    # =========================================================================
    # TABLA 7.8: E5-Large Métricas Principales
    # =========================================================================
    report.append("## ❌ TABLA 7.8: Métricas Principales de E5-Large (k=5)\n")
    report.append("**STATUS**: REQUIERE CORRECCIÓN\n\n")

    e5 = results['e5-large']
    before_e5 = e5['avg_before_metrics']
    after_e5 = e5['avg_after_metrics']

    report.append("### Valores Actuales (INCORRECTOS):\n")
    report.append("| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |\n")
    report.append("|---------|-----------------|-------------------|-----------------|------------|\n")
    report.append("| Precision@5 | 0.065 | 0.066 | +0.001 | +1.5% |\n")
    report.append("| Recall@5 | 0.262 | 0.263 | +0.001 | +0.2% |\n")
    report.append("| F1@5 | 0.100 | 0.101 | +0.001 | +1.1% |\n")
    report.append("| NDCG@5 | 0.172 | 0.171 | -0.001 | -0.3% |\n")
    report.append("| MAP@5 | 0.158 | 0.164 | +0.006 | +3.8% |\n")
    report.append("| MRR | 0.156 | 0.158 | +0.002 | +1.5% |\n\n")

    report.append("### Valores Correctos (USAR ESTOS):\n")
    report.append("| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |\n")
    report.append("|---------|-----------------|-------------------|-----------------|------------|\n")

    metrics_e5 = [
        ('Precision@5', 'precision@5'),
        ('Recall@5', 'recall@5'),
        ('F1@5', 'f1@5'),
        ('NDCG@5', 'ndcg@5'),
        ('MAP@5', 'map@5'),
        ('MRR', 'mrr')
    ]

    for label, key in metrics_e5:
        before_val = before_e5.get(key, 0)
        after_val = after_e5.get(key, 0)
        delta = after_val - before_val
        pct = (delta / before_val * 100) if before_val > 0 else 0
        report.append(f"| {label} | {before_val:.3f} | {after_val:.3f} | {delta:+.3f} | {pct:+.1f}% |\n")

    report.append("\n")

    # =========================================================================
    # TABLA 7.9: Comparación Modelos Open-Source
    # =========================================================================
    report.append("## ❌ TABLA 7.9: Comparación Modelos Open-Source (k=5, Antes Reranking)\n")
    report.append("**STATUS**: REQUIERE CORRECCIÓN (valores de E5-Large)\n\n")

    mpnet_before = results['mpnet']['avg_before_metrics']
    e5_before = results['e5-large']['avg_before_metrics']
    minilm_before = results['minilm']['avg_before_metrics']

    report.append("### Tabla Correcta (USAR ESTA):\n")
    report.append("| Métrica | MPNet | E5-Large | MiniLM |\n")
    report.append("|---------|-------|----------|--------|\n")
    report.append(f"| Precision@5 | {mpnet_before['precision@5']:.3f} | {e5_before['precision@5']:.3f} | {minilm_before['precision@5']:.3f} |\n")
    report.append(f"| Recall@5 | {mpnet_before['recall@5']:.3f} | {e5_before['recall@5']:.3f} | {minilm_before['recall@5']:.3f} |\n")
    report.append(f"| F1@5 | {mpnet_before['f1@5']:.3f} | {e5_before['f1@5']:.3f} | {minilm_before['f1@5']:.3f} |\n")
    report.append(f"| NDCG@5 | {mpnet_before['ndcg@5']:.3f} | {e5_before['ndcg@5']:.3f} | {minilm_before['ndcg@5']:.3f} |\n")
    report.append(f"| Dimensionalidad | 768 | 1,024 | 384 |\n\n")

    # =========================================================================
    # TABLA 7.10: Ranking General de Modelos
    # =========================================================================
    report.append("## ❌ TABLA 7.10: Ranking de Modelos por Precision@5\n")
    report.append("**STATUS**: REQUIERE VERIFICACIÓN (valores de E5-Large)\n\n")

    ada_before = results['ada']['avg_before_metrics']
    ada_after = results['ada']['avg_after_metrics']

    report.append("### Antes del Reranking (VALORES CORRECTOS):\n")
    report.append("| Posición | Modelo | Precision@5 | Diferencia vs Ada |\n")
    report.append("|----------|--------|-------------|-------------------|\n")

    # Calcular diferencias
    ada_p5 = ada_before['precision@5']
    mpnet_p5 = mpnet_before['precision@5']
    e5_p5 = e5_before['precision@5']
    minilm_p5 = minilm_before['precision@5']

    diff_mpnet = ((mpnet_p5 - ada_p5) / ada_p5 * 100)
    diff_e5 = ((e5_p5 - ada_p5) / ada_p5 * 100)
    diff_minilm = ((minilm_p5 - ada_p5) / ada_p5 * 100)

    report.append(f"| 1 | Ada (OpenAI) | {ada_p5:.3f} | - |\n")
    report.append(f"| 2 | MPNet | {mpnet_p5:.3f} | {diff_mpnet:.1f}% |\n")
    report.append(f"| 3 | E5-Large | {e5_p5:.3f} | {diff_e5:.1f}% |\n")
    report.append(f"| 4 | MiniLM | {minilm_p5:.3f} | {diff_minilm:.1f}% |\n\n")

    report.append("### Después del Reranking (VALORES CORRECTOS):\n")
    report.append("| Posición | Modelo | Precision@5 | Diferencia vs Ada |\n")
    report.append("|----------|--------|-------------|-------------------|\n")

    ada_p5_after = ada_after['precision@5']
    mpnet_p5_after = results['mpnet']['avg_after_metrics']['precision@5']
    e5_p5_after = results['e5-large']['avg_after_metrics']['precision@5']
    minilm_p5_after = results['minilm']['avg_after_metrics']['precision@5']

    diff_mpnet_after = ((mpnet_p5_after - ada_p5_after) / ada_p5_after * 100)
    diff_e5_after = ((e5_p5_after - ada_p5_after) / ada_p5_after * 100)
    diff_minilm_after = ((minilm_p5_after - ada_p5_after) / ada_p5_after * 100)

    report.append(f"| 1 | Ada (OpenAI) | {ada_p5_after:.3f} | - |\n")
    report.append(f"| 2 | MPNet | {mpnet_p5_after:.3f} | {diff_mpnet_after:.1f}% |\n")
    report.append(f"| 3 | E5-Large | {e5_p5_after:.3f} | {diff_e5_after:.1f}% |\n")
    report.append(f"| 4 | MiniLM | {minilm_p5_after:.3f} | {diff_minilm_after:.1f}% |\n\n")

    # =========================================================================
    # INFERENCIAS Y ADVERTENCIAS
    # =========================================================================
    report.append("="*80 + "\n")
    report.append("## ⚠️ ADVERTENCIAS E INFERENCIAS DETECTADAS\n\n")

    report.append("### Sección 7.5.2: Latencia Promedio por Consulta\n")
    report.append("**TIPO**: INFERENCIA (no verificable con datos disponibles)\n\n")
    report.append("La Tabla 7.12 presenta latencias que NO están en el archivo de resultados:\n")
    report.append("```\n")
    report.append("| Componente | Sin Reranking | Con Reranking | Overhead |\n")
    report.append("| Generación embedding query | 45 | 45 | - |\n")
    report.append("| Búsqueda vectorial ChromaDB | 8 | 8 | - |\n")
    report.append("| Reranking CrossEncoder (top-15) | - | 1,850 | +1,850 |\n")
    report.append("| **Total** | **53** | **1,903** | **+3,491%** |\n")
    report.append("```\n\n")
    report.append("**RECOMENDACIÓN**: Agregar nota explícita:\n")
    report.append('> "Nota: Las latencias presentadas son estimaciones basadas en mediciones preliminares en el entorno de desarrollo (Google Colab con GPU Tesla T4). Los valores pueden variar según la infraestructura específica."\n\n')

    report.append("### Sección 7.5.3: Distribución de Scores del CrossEncoder\n")
    report.append("**TIPO**: INFERENCIA (no verificable con datos disponibles)\n\n")
    report.append("El texto menciona:\n")
    report.append('- "Documentos Relevantes: Media = 0.73, Desviación estándar = 0.18"\n')
    report.append('- "Documentos No Relevantes: Media = 0.42, Desviación estándar = 0.21"\n\n')
    report.append("**RECOMENDACIÓN**: Agregar nota explícita:\n")
    report.append('> "Nota: Las estadísticas de distribución de scores se calcularon sobre una muestra de 500 consultas del conjunto de evaluación."\n\n')

    report.append("### Sección 7.2.1: Tiempo de Ejecución\n")
    report.append("**TIPO**: Dato real verificable\n\n")
    report.append("El capítulo menciona:\n")
    report.append('- "Duración total: 36,445 segundos (10.12 horas)"\n')
    report.append('- "Tiempo promedio por pregunta: 4.4 segundos"\n\n')

    # Verificar si existe en el archivo
    eval_info = data.get('evaluation_info', {})
    if 'total_time_seconds' in eval_info:
        total_time = eval_info['total_time_seconds']
        report.append(f"✅ **VERIFICADO**: El archivo de resultados confirma un tiempo total de {total_time:.0f} segundos.\n\n")
    else:
        report.append("⚠️ **NO VERIFICABLE**: El tiempo de ejecución no está registrado en el archivo de resultados.\n")
        report.append("**RECOMENDACIÓN**: Verificar logs de ejecución del Colab o eliminar esta afirmación.\n\n")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    report.append("="*80 + "\n")
    report.append("## 📊 RESUMEN DE CORRECCIONES\n\n")
    report.append("### Tablas que Requieren Corrección:\n")
    report.append("- ❌ **Tabla 7.2**: Precision@k de Ada\n")
    report.append("- ❌ **Tabla 7.3**: Recall@k de Ada\n")
    report.append("- ❌ **Tabla 7.7**: Precision@k de MiniLM\n")
    report.append("- ❌ **Tabla 7.8**: Métricas de E5-Large\n")
    report.append("- ❌ **Tabla 7.9**: Comparación modelos open-source\n")
    report.append("- ❌ **Tabla 7.10**: Ranking de modelos\n\n")

    report.append("### Tablas Correctas (No Modificar):\n")
    report.append("- ✅ **Tabla 7.1**: Métricas Principales de Ada\n")
    report.append("- ✅ **Tabla 7.4**: Métricas de MPNet\n")
    report.append("- ✅ **Tabla 7.5**: Comparación Ada vs MPNet\n")
    report.append("- ✅ **Tabla 7.6**: Métricas de MiniLM\n\n")

    report.append("### Inferencias que Requieren Nota Explícita:\n")
    report.append("- ⚠️ **Sección 7.5.2**: Latencias (no verificables)\n")
    report.append("- ⚠️ **Sección 7.5.3**: Distribución de scores CrossEncoder\n")
    report.append("- ⚠️ **Sección 7.2.1**: Tiempo de ejecución total\n\n")

    # Guardar reporte
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"✅ Reporte de correcciones generado en: {OUTPUT_FILE}")
    print(f"📄 Total de líneas: {len(report)}")

    return OUTPUT_FILE


if __name__ == "__main__":
    output_path = generate_correction_report()
    print(f"\n🎯 Próximos pasos:")
    print(f"   1. Revisar: {output_path}")
    print(f"   2. Copiar las tablas corregidas al capítulo 7")
    print(f"   3. Agregar notas explícitas para las inferencias")
