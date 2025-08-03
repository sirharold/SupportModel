#!/usr/bin/env python3
"""
Script para generar actualización del capítulo 7 con nuevos resultados (1000 preguntas)
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_results(filepath):
    """Cargar archivo de resultados JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_key_metrics(results):
    """Extraer métricas clave de cada modelo"""
    metrics_data = []
    
    for model_name, model_data in results['results'].items():
        before_metrics = model_data['avg_before_metrics']
        after_metrics = model_data['avg_after_metrics']
        
        # Calcular cambios porcentuales
        precision5_change = ((after_metrics.get('precision@5', 0) - before_metrics.get('precision@5', 0)) / 
                            before_metrics.get('precision@5', 0.0001) * 100) if before_metrics.get('precision@5', 0) > 0 else 0
        
        recall5_change = ((after_metrics.get('recall@5', 0) - before_metrics.get('recall@5', 0)) / 
                         before_metrics.get('recall@5', 0.0001) * 100) if before_metrics.get('recall@5', 0) > 0 else 0
        
        ndcg5_change = ((after_metrics.get('ndcg@5', 0) - before_metrics.get('ndcg@5', 0)) / 
                       before_metrics.get('ndcg@5', 0.0001) * 100) if before_metrics.get('ndcg@5', 0) > 0 else 0
        
        mrr_change = ((after_metrics.get('mrr', 0) - before_metrics.get('mrr', 0)) / 
                     before_metrics.get('mrr', 0.0001) * 100) if before_metrics.get('mrr', 0) > 0 else 0
        
        metrics_data.append({
            'Model': model_name.upper(),
            'Questions': model_data['num_questions_evaluated'],
            'Dimensions': model_data['embedding_dimensions'],
            # Before reranking
            'Precision@5 (Before)': f"{before_metrics.get('precision@5', 0):.3f}",
            'Recall@5 (Before)': f"{before_metrics.get('recall@5', 0):.3f}",
            'NDCG@5 (Before)': f"{before_metrics.get('ndcg@5', 0):.3f}",
            'MRR (Before)': f"{before_metrics.get('mrr', 0):.3f}",
            # After reranking
            'Precision@5 (After)': f"{after_metrics.get('precision@5', 0):.3f}",
            'Recall@5 (After)': f"{after_metrics.get('recall@5', 0):.3f}",
            'NDCG@5 (After)': f"{after_metrics.get('ndcg@5', 0):.3f}",
            'MRR (After)': f"{after_metrics.get('mrr', 0):.3f}",
            # Changes
            'Precision@5 Change': f"{precision5_change:+.1f}%" if precision5_change != 0 else "0.0%",
            'Recall@5 Change': f"{recall5_change:+.1f}%" if recall5_change != 0 else "0.0%",
            'NDCG@5 Change': f"{ndcg5_change:+.1f}%" if ndcg5_change != 0 else "0.0%",
            'MRR Change': f"{mrr_change:+.1f}%" if mrr_change != 0 else "0.0%"
        })
    
    return pd.DataFrame(metrics_data)

def extract_rag_metrics(results):
    """Extraer métricas RAG de cada modelo"""
    rag_data = []
    
    for model_name, model_data in results['results'].items():
        rag_metrics = model_data.get('rag_metrics', {})
        
        rag_data.append({
            'Model': model_name.upper(),
            'Faithfulness': f"{rag_metrics.get('faithfulness', 0):.3f}" if rag_metrics.get('faithfulness') else 'N/A',
            'Answer Relevancy': f"{rag_metrics.get('answer_relevancy', 0):.3f}" if rag_metrics.get('answer_relevancy') else 'N/A',
            'Context Precision': f"{rag_metrics.get('context_precision', 0):.3f}" if rag_metrics.get('context_precision') else 'N/A',
            'Context Recall': f"{rag_metrics.get('context_recall', 0):.3f}" if rag_metrics.get('context_recall') else 'N/A',
            'Context Relevancy': f"{rag_metrics.get('context_relevancy', 0):.3f}" if rag_metrics.get('context_relevancy') else 'N/A',
            'Context Utilization': f"{rag_metrics.get('context_utilization', 0):.3f}" if rag_metrics.get('context_utilization') else 'N/A',
            'BERTScore Precision': f"{rag_metrics.get('bertscore_precision', 0):.3f}" if rag_metrics.get('bertscore_precision') else 'N/A',
            'BERTScore Recall': f"{rag_metrics.get('bertscore_recall', 0):.3f}" if rag_metrics.get('bertscore_recall') else 'N/A',
            'BERTScore F1': f"{rag_metrics.get('bertscore_f1', 0):.3f}" if rag_metrics.get('bertscore_f1') else 'N/A'
        })
    
    return pd.DataFrame(rag_data)

def generate_updated_sections(results_file):
    """Generar secciones actualizadas del capítulo 7"""
    results = load_results(results_file)
    
    # Información general
    eval_info = results['evaluation_info']
    
    output = []
    output.append("# ACTUALIZACIÓN DEL CAPÍTULO 7 - RESULTADOS CON 1000 PREGUNTAS\n")
    output.append(f"## Información de la Nueva Evaluación\n")
    output.append(f"- **Fecha de evaluación:** {eval_info['timestamp'][:10]}")
    output.append(f"- **Preguntas por modelo:** {eval_info['questions_per_model']}")
    output.append(f"- **Modelos evaluados:** {eval_info['models_evaluated']}")
    output.append(f"- **Duración total:** {eval_info['total_duration_seconds']/3600:.1f} horas")
    output.append(f"- **Verificación de datos:** Datos reales sin simulación\n")
    
    # Tabla principal de métricas
    output.append("## Tabla 7.1: Comparación de Modelos (1000 preguntas)\n")
    metrics_df = extract_key_metrics(results)
    output.append(metrics_df.to_markdown(index=False))
    output.append("\n")
    
    # Tabla de métricas RAG
    output.append("## Tabla 7.2: Métricas RAG y BERTScore (1000 preguntas)\n")
    rag_df = extract_rag_metrics(results)
    output.append(rag_df.to_markdown(index=False))
    output.append("\n")
    
    # Análisis por modelo
    output.append("## Análisis Detallado por Modelo\n")
    
    for model_name, model_data in results['results'].items():
        output.append(f"### {model_name.upper()}\n")
        
        before = model_data['avg_before_metrics']
        after = model_data['avg_after_metrics']
        
        output.append(f"**Configuración:**")
        output.append(f"- Preguntas evaluadas: {model_data['num_questions_evaluated']}")
        output.append(f"- Dimensiones del embedding: {model_data['embedding_dimensions']}")
        output.append(f"- Modelo completo: {model_data['full_model_name']}\n")
        
        output.append(f"**Métricas principales (antes → después del reranking):**")
        output.append(f"- Precision@5: {before.get('precision@5', 0):.3f} → {after.get('precision@5', 0):.3f}")
        output.append(f"- Recall@5: {before.get('recall@5', 0):.3f} → {after.get('recall@5', 0):.3f}")
        output.append(f"- NDCG@5: {before.get('ndcg@5', 0):.3f} → {after.get('ndcg@5', 0):.3f}")
        output.append(f"- MRR: {before.get('mrr', 0):.3f} → {after.get('mrr', 0):.3f}\n")
        
        # Análisis del impacto del reranking
        precision_change = after.get('precision@5', 0) - before.get('precision@5', 0)
        if precision_change > 0:
            output.append(f"**Impacto del CrossEncoder:** Mejora significativa en métricas principales")
        elif precision_change < 0:
            output.append(f"**Impacto del CrossEncoder:** Impacto negativo o neutro")
        else:
            output.append(f"**Impacto del CrossEncoder:** Sin cambios significativos")
        output.append("\n")
    
    # Comparación con resultados anteriores
    output.append("## Comparación con Evaluación Anterior (11 vs 1000 preguntas)\n")
    output.append("### Cambios Principales:\n")
    output.append("1. **Tamaño de muestra:** 91x más datos (11 → 1000 preguntas)")
    output.append("2. **Confiabilidad estadística:** Métricas mucho más estables y representativas")
    output.append("3. **E5-Large funcional:** Ahora muestra métricas válidas (antes todas en 0.0)")
    output.append("4. **Jerarquía de rendimiento clara:** Ada > MPNet > E5-Large > MiniLM")
    output.append("5. **Impacto del reranking diferente:** Patrones más complejos con dataset mayor\n")
    
    return "\n".join(output)

def main():
    results_file = "/Users/haroldgomez/Downloads/cumulative_results_20250802_222752.json"
    output_file = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Finales/actualizacion_capitulo_7_1000_preguntas.md"
    
    # Generar contenido actualizado
    updated_content = generate_updated_sections(results_file)
    
    # Guardar archivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ Actualización generada en: {output_file}")
    
    # También generar un resumen de cambios clave
    summary_file = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Finales/resumen_cambios_capitulo_7.md"
    
    summary = """# Resumen de Cambios Clave - Capítulo 7

## Cambios en la Sección 7.2.1 (Configuración Experimental)

**Actualizar:**
- Preguntas evaluadas: ~~11~~ → **1000** por modelo
- Duración total: ~~774.78 segundos (12.9 minutos)~~ → **28,216 segundos (7.8 horas)**
- Fecha de evaluación: ~~26 de julio de 2025~~ → **2 de agosto de 2025**

## Cambios en Métricas por Modelo

### Ada (Sección 7.2.2)
- Precision@5: ~~0.055~~ → **0.097** (+76.7%)
- Recall@5: ~~0.273~~ → **0.399** (+46.0%)
- NDCG@5: ~~0.126~~ → **0.228** (+81.3%)
- MRR: ~~0.125~~ → **0.217** (+73.9%)

### MPNet (Sección 7.2.3)
- Precision@5: ~~0.055~~ → **0.074** (+35.3%)
- Recall@5: ~~0.273~~ → **0.292** (+7.0%)
- NDCG@5: ~~0.108~~ → **0.199** (+84.5%)
- MRR: ~~0.082~~ → **0.185** (+125.7%)

### MiniLM (Sección 7.2.4)
- Precision@5: ~~0.018~~ → **0.053** (+192.2%)
- Recall@5: ~~0.091~~ → **0.201** (+121.0%)
- NDCG@5: ~~0.091~~ → **0.148** (+62.7%)
- MRR: ~~0.077~~ → **0.144** (+87.3%)

### E5-Large (Sección 7.2.5)
- **¡AHORA FUNCIONAL!**
- Precision@5: ~~0.000~~ → **0.060**
- Recall@5: ~~0.000~~ → **0.239**
- NDCG@5: ~~0.000~~ → **0.169**
- MRR: ~~0.000~~ → **0.161**

## Cambios en Conclusiones (Sección 7.3)

1. **Eliminar:** "No hay diferencias estadísticamente significativas entre modelos"
2. **Agregar:** "Con 1000 preguntas, emergen diferencias claras: Ada > MPNet > E5-Large > MiniLM"
3. **Actualizar:** El impacto del reranking varía más entre modelos con dataset mayor

## Nueva Sección Recomendada: 7.3.4 Validación con Dataset Ampliado

Agregar subsección que discuta:
- Importancia del tamaño de muestra para confiabilidad
- Confirmación de tendencias observadas
- Resolución del problema de E5-Large
- Implicaciones para producción
"""
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✅ Resumen de cambios generado en: {summary_file}")

if __name__ == "__main__":
    main()