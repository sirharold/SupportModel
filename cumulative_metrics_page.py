"""
Página de Métricas Acumulativas - Evalúa múltiples preguntas y calcula promedios
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import re
import gc
import psutil
import os
from typing import List, Dict, Any
from utils.clients import initialize_clients
from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from config import EMBEDDING_MODELS, GENERATIVE_MODELS, WEAVIATE_CLASS_CONFIG
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

# Helper functions to interpret metric quality
def grade_metric(value: float) -> str:
    """Return a qualitative label for a metric value."""
    if value >= 0.7:
        return "🟢 Muy bueno"
    elif value >= 0.4:
        return "🟡 Bueno"
    else:
        return "🔴 Malo"


def color_metric(val: float) -> str:
    """Return background color for dataframe styling."""
    if pd.isna(val):
        return ""
    if val >= 0.7:
        return "background-color: #c6f5c6"  # light green
    elif val >= 0.4:
        return "background-color: #fff4b3"  # light yellow
    else:
        return "background-color: #f7c6c6"  # light red


def style_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply coloring to before/after metric columns."""
    return df.style.applymap(color_metric, subset=["Before", "After"])

def extract_ms_links(accepted_answer: str) -> List[str]:
    """
    Extrae links de Microsoft Learn de la respuesta aceptada.
    
    Args:
        accepted_answer: Texto de la respuesta aceptada
        
    Returns:
        Lista de links de Microsoft Learn encontrados
    """
    # Patrón para encontrar links de Microsoft Learn
    pattern = r'https://learn\.microsoft\.com[\w/\-\?=&%\.]+'
    links = re.findall(pattern, accepted_answer)
    return list(set(links))  # Eliminar duplicados

def filter_questions_with_links(questions_and_answers: List[Dict]) -> List[Dict]:
    """
    Filtra preguntas que tienen links en la respuesta aceptada.
    
    Args:
        questions_and_answers: Lista de preguntas y respuestas
        
    Returns:
        Lista filtrada de preguntas con links en respuesta aceptada
    """
    filtered_questions = []
    
    for qa in questions_and_answers:
        accepted_answer = qa.get('accepted_answer', '')
        
        # Extraer links de Microsoft Learn
        ms_links = extract_ms_links(accepted_answer)
        
        # Solo incluir si hay links
        if ms_links and len(ms_links) > 0:
            # Añadir los links extraídos al diccionario
            qa_copy = qa.copy()
            qa_copy['ms_links'] = ms_links
            qa_copy['question'] = qa.get('question_content', qa.get('title', ''))
            filtered_questions.append(qa_copy)
    
    return filtered_questions

def get_memory_usage() -> float:
    """Obtiene el uso actual de memoria en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Fuerza la limpieza de memoria."""
    gc.collect()

def get_color_palette():
    """Retorna una paleta de colores personalizada."""
    return [
        '#1f77b4',  # azul
        '#ff7f0e',  # naranja
        '#2ca02c',  # verde
        '#d62728',  # rojo
        '#9467bd',  # púrpura
        '#8c564b',  # marrón
        '#e377c2',  # rosa
        '#7f7f7f',  # gris
        '#bcbd22',  # oliva
        '#17becf'   # cian
    ]

def add_metric_definitions_page(pdf, ax):
    """Agrega página con definiciones de métricas."""
    ax.axis('off')
    
    # Título
    ax.text(0.5, 0.95, 'DEFINICIONES DE MÉTRICAS', 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Definiciones
    definitions_text = """
MÉTRICAS DE RECUPERACIÓN:

• Precision@k: Proporción de documentos relevantes entre los k documentos recuperados.
  Fórmula: |documentos relevantes en top-k| / k
  Interpretación: Qué tan preciso es el sistema al recuperar documentos.

• Recall@k: Proporción de documentos relevantes recuperados del total de relevantes.
  Fórmula: |documentos relevantes en top-k| / |total documentos relevantes|
  Interpretación: Qué tan completa es la recuperación del sistema.

• F1@k: Media armónica entre Precision@k y Recall@k.
  Fórmula: 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)
  Interpretación: Balance entre precisión y cobertura.

• Accuracy@k: Proporción de consultas donde al menos 1 documento relevante está en top-k.
  Fórmula: |consultas con ≥1 documento relevante en top-k| / |total consultas|
  Interpretación: Tasa de éxito del sistema.

• MRR (Mean Reciprocal Rank): Promedio del recíproco de la posición del primer documento relevante.
  Fórmula: (1/N) × Σ(1/posición_primer_relevante)
  Interpretación: Qué tan rápido encuentra el sistema el primer documento relevante.

• nDCG@k (Normalized Discounted Cumulative Gain): Ganancia acumulativa con descuento normalizada.
  Fórmula: DCG@k / IDCG@k, donde DCG considera la posición de documentos relevantes.
  Interpretación: Calidad del ranking considerando el orden de relevancia.

METODOLOGÍA DE EVALUACIÓN:

• Ground Truth: Enlaces de Microsoft Learn extraídos de respuestas aceptadas.
• Proceso: Para cada pregunta se recuperan documentos y se comparan con enlaces esperados.
• Reranking LLM: Reordenamiento de documentos usando GPT-4 para mejorar relevancia.
• Métricas "Antes": Basadas en similaridad de embeddings únicamente.
• Métricas "Después": Basadas en reordenamiento inteligente con LLM.

INTERPRETACIÓN DE VALORES:

• 0.7 - 1.0: 🟢 Muy Bueno - Rendimiento excelente
• 0.4 - 0.7: 🟡 Bueno - Rendimiento aceptable  
• 0.0 - 0.4: 🔴 Malo - Necesita mejoras
"""
    
    ax.text(0.05, 0.88, definitions_text, fontsize=9, transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

def generate_pdf_report(results: Dict[str, Any], model_name: str, use_llm_reranker: bool, 
                       evaluation_time: float, generative_model_name: str, num_questions: int, 
                       top_k: int) -> bytes:
    """
    Genera un reporte PDF completo con métricas, gráficos y tablas.
    Incluye toda la información presentada en la página web.
    
    Args:
        results: Resultados de la evaluación
        model_name: Nombre del modelo evaluado
        use_llm_reranker: Si se usó reranking LLM
        evaluation_time: Tiempo de evaluación
        generative_model_name: Modelo generativo usado
        num_questions: Número de preguntas evaluadas
        top_k: Valor de top-k usado
        
    Returns:
        bytes: Contenido del PDF generado
    """
    # Configurar matplotlib para mejor apariencia
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Crear buffer para el PDF
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Página 1: Portada y resumen ejecutivo
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Título principal
        ax.text(0.5, 0.9, 'REPORTE DE MÉTRICAS ACUMULATIVAS', 
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Información general
        info_text = f"""
INFORMACIÓN GENERAL:
• Modelo de Embedding: {model_name}
• Modelo Generativo: {generative_model_name}
• Preguntas Evaluadas: {num_questions}
• Top-K Documentos: {top_k}
• Reranking LLM: {'Habilitado' if use_llm_reranker else 'Deshabilitado'}
• Tiempo de Evaluación: {evaluation_time:.1f} segundos
• Fecha de Generación: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        ax.text(0.1, 0.75, info_text, fontsize=12, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Resumen de métricas principales
        avg_before = results['avg_before_metrics']
        avg_after = results['avg_after_metrics']
        
        main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
        
        summary_text = "RESUMEN DE MÉTRICAS PRINCIPALES:\n\n"
        for metric in main_metrics:
            if metric in avg_before:
                before_val = avg_before[metric]
                after_val = avg_after.get(metric, before_val)
                delta = after_val - before_val if use_llm_reranker else 0
                
                quality = grade_metric(after_val)
                summary_text += f"• {metric}: {after_val:.3f} ({quality})"
                if use_llm_reranker:
                    summary_text += f" [Δ: {delta:+.3f}]"
                summary_text += "\n"
        
        ax.text(0.1, 0.45, summary_text, fontsize=11, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        # Estadísticas de memoria si están disponibles
        if 'memory_stats' in results:
            memory_stats = results['memory_stats']
            memory_text = f"""
ESTADÍSTICAS DE MEMORIA:
• Memoria Inicial: {memory_stats.get('initial_memory', 0):.1f} MB
• Memoria Final: {memory_stats.get('final_memory', 0):.1f} MB  
• Incremento: {memory_stats.get('memory_increase', 0):.1f} MB
"""
            ax.text(0.1, 0.15, memory_text, fontsize=10, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 2: Definiciones de métricas
        fig, ax = plt.subplots(figsize=(8.5, 11))
        add_metric_definitions_page(pdf, ax)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 3: Gráfico de barras comparativo principal
        if use_llm_reranker and avg_before and avg_after:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = [m for m in main_metrics if m in avg_before]
            before_values = [avg_before[m] for m in metrics]
            after_values = [avg_after.get(m, avg_before[m]) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            colors = get_color_palette()
            
            bars1 = ax.bar(x - width/2, before_values, width, label='Antes del Reranking', 
                          color=colors[0], alpha=0.7)
            bars2 = ax.bar(x + width/2, after_values, width, label='Después del Reranking', 
                          color=colors[1], alpha=0.7)
            
            ax.set_xlabel('Métricas')
            ax.set_ylabel('Valor')
            ax.set_title(f'Comparación de Métricas Principales - {model_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Agregar valores en las barras
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Página 4: Gráficos por valores de K (k=1,3,5,10) - Todos en una página
        k_values = [1, 3, 5, 10]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, k in enumerate(k_values):
            ax = axes[idx]
            k_metrics = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}']
            
            before_vals = [avg_before.get(m, 0) for m in k_metrics]
            after_vals = [avg_after.get(m, 0) for m in k_metrics]
            
            x = np.arange(len(k_metrics))
            width = 0.35
            colors = get_color_palette()
            
            bars1 = ax.bar(x - width/2, before_vals, width, label='Antes', color=colors[0], alpha=0.7)
            if use_llm_reranker:
                bars2 = ax.bar(x + width/2, after_vals, width, label='Después', color=colors[1], alpha=0.7)
            
            ax.set_title(f'Métricas para K={k}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Métrica')
            ax.set_ylabel('Valor')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(f'@{k}', '') for m in k_metrics])
            if idx == 0:  # Solo mostrar leyenda en el primer gráfico
                ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Agregar valores en las barras
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            if use_llm_reranker:
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'Análisis Detallado por Valores de K - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 5: Tabla detallada de métricas
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Preparar datos de la tabla
        table_data = []
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            for metric_type in ['Precision', 'Recall', 'F1', 'Accuracy']:
                metric_name = f'{metric_type}@{k}'
                if metric_name in avg_before:
                    before_val = avg_before[metric_name]
                    after_val = avg_after.get(metric_name, before_val)
                    delta = after_val - before_val if use_llm_reranker else 0
                    
                    table_data.append([
                        metric_name,
                        f'{before_val:.3f}',
                        f'{after_val:.3f}',
                        f'{delta:+.3f}',
                        grade_metric(after_val)
                    ])
        
        # Agregar MRR y nDCG
        for metric in ['MRR@5', 'nDCG@5']:
            if metric in avg_before:
                before_val = avg_before[metric]
                after_val = avg_after.get(metric, before_val)
                delta = after_val - before_val if use_llm_reranker else 0
                
                table_data.append([
                    metric,
                    f'{before_val:.3f}',
                    f'{after_val:.3f}',
                    f'{delta:+.3f}',
                    grade_metric(after_val)
                ])
        
        # Crear tabla
        headers = ['Métrica', 'Antes', 'Después', 'Δ', 'Calidad']
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Colorear header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorear filas alternadas
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Tabla Detallada de Métricas', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 6: Evolución por pregunta (muestra de primeras 20 preguntas)
        if results['all_questions_data']:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Tomar muestra de primeras 20 preguntas
            sample_data = results['all_questions_data'][:20]
            question_nums = [q['question_num'] for q in sample_data]
            
            colors = get_color_palette()
            
            # Gráfico 1: Precision@5
            before_precision = [q['before_precision_5'] for q in sample_data]
            after_precision = [q['after_precision_5'] for q in sample_data]
            
            ax1.plot(question_nums, before_precision, 'o-', label='Antes del Reranking', 
                    color=colors[0], markersize=4)
            if use_llm_reranker:
                ax1.plot(question_nums, after_precision, 'o-', label='Después del Reranking', 
                        color=colors[1], markersize=4)
            
            ax1.set_xlabel('Número de Pregunta')
            ax1.set_ylabel('Precision@5')
            ax1.set_title('Evolución de Precision@5 por Pregunta (Primeras 20)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: F1@5
            before_f1 = [q['before_f1_5'] for q in sample_data]
            after_f1 = [q['after_f1_5'] for q in sample_data]
            
            ax2.plot(question_nums, before_f1, 'o-', label='Antes del Reranking', 
                    color=colors[2], markersize=4)
            if use_llm_reranker:
                ax2.plot(question_nums, after_f1, 'o-', label='Después del Reranking', 
                        color=colors[3], markersize=4)
            
            ax2.set_xlabel('Número de Pregunta')
            ax2.set_ylabel('F1@5')
            ax2.set_title('Evolución de F1@5 por Pregunta (Primeras 20)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Página 5: Distribución de métricas
        if results['all_questions_data']:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Extraer datos para histogramas
            all_data = results['all_questions_data']
            before_precision = [q['before_precision_5'] for q in all_data]
            after_precision = [q['after_precision_5'] for q in all_data]
            before_recall = [q['before_recall_5'] for q in all_data]
            after_recall = [q['after_recall_5'] for q in all_data]
            
            colors = get_color_palette()
            
            # Histograma 1: Precision@5
            ax1.hist(before_precision, bins=20, alpha=0.7, label='Antes', color=colors[0])
            if use_llm_reranker:
                ax1.hist(after_precision, bins=20, alpha=0.7, label='Después', color=colors[1])
            ax1.set_xlabel('Precision@5')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('Distribución de Precision@5')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histograma 2: Recall@5
            ax2.hist(before_recall, bins=20, alpha=0.7, label='Antes', color=colors[2])
            if use_llm_reranker:
                ax2.hist(after_recall, bins=20, alpha=0.7, label='Después', color=colors[3])
            ax2.set_xlabel('Recall@5')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('Distribución de Recall@5')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Scatter plot: Precision vs Recall (antes)
            ax3.scatter(before_precision, before_recall, alpha=0.6, color=colors[0], s=20)
            ax3.set_xlabel('Precision@5')
            ax3.set_ylabel('Recall@5')
            ax3.set_title('Precision vs Recall (Antes)')
            ax3.grid(True, alpha=0.3)
            
            # Scatter plot: Precision vs Recall (después)
            if use_llm_reranker:
                ax4.scatter(after_precision, after_recall, alpha=0.6, color=colors[1], s=20)
                ax4.set_xlabel('Precision@5')
                ax4.set_ylabel('Recall@5')
                ax4.set_title('Precision vs Recall (Después)')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Reranking LLM\nDeshabilitado', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_xticks([])
                ax4.set_yticks([])
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Página 7: Tabla de evolución por pregunta (completa)
        if results['all_questions_data']:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Preparar datos de la tabla de evolución
            questions_data = results['all_questions_data'][:30]  # Mostrar primeras 30 preguntas
            
            table_data = []
            for q_data in questions_data:
                row = [
                    str(q_data['question_num']),
                    str(q_data['ground_truth_links']),
                    str(q_data['docs_retrieved']),
                    f"{q_data['before_precision_5']:.3f}",
                    f"{q_data['after_precision_5']:.3f}",
                    f"{q_data['before_recall_5']:.3f}",
                    f"{q_data['after_recall_5']:.3f}",
                    f"{q_data['before_f1_5']:.3f}",
                    f"{q_data['after_f1_5']:.3f}"
                ]
                table_data.append(row)
            
            headers = ['Q#', 'GT Links', 'Docs', 'P@5 (A)', 'P@5 (D)', 'R@5 (A)', 'R@5 (D)', 'F1@5 (A)', 'F1@5 (D)']
            table = ax.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            colWidths=[0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
            
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.2, 1.2)
            
            # Colorear header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear filas alternadas
            for i in range(1, len(table_data) + 1):
                if i % 2 == 0:
                    for j in range(len(headers)):
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            ax.set_title(f'Evolución por Pregunta (Primeras {len(questions_data)} preguntas)', 
                        fontsize=14, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Página 8: Análisis estadístico y resumen
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Calcular estadísticas adicionales
        before_metrics = results['individual_before_metrics']
        after_metrics = results['individual_after_metrics']
        
        # Estadísticas para MRR
        mrr_before = [m.get('MRR@5', 0) for m in before_metrics]
        mrr_after = [m.get('MRR@5', 0) for m in after_metrics]
        
        # Estadísticas para Precision@5
        p5_before = [m.get('Precision@5', 0) for m in before_metrics]
        p5_after = [m.get('Precision@5', 0) for m in after_metrics]
        
        # Estadísticas para F1@5
        f1_before = [m.get('F1@5', 0) for m in before_metrics]
        f1_after = [m.get('F1@5', 0) for m in after_metrics]
        
        stats_text = f"""
ANÁLISIS ESTADÍSTICO DETALLADO

MRR@5 (Mean Reciprocal Rank):
• Antes del Reranking: μ={np.mean(mrr_before):.3f}, σ={np.std(mrr_before):.3f}, min={np.min(mrr_before):.3f}, max={np.max(mrr_before):.3f}
• Después del Reranking: μ={np.mean(mrr_after):.3f}, σ={np.std(mrr_after):.3f}, min={np.min(mrr_after):.3f}, max={np.max(mrr_after):.3f}
• Mejora promedio: {np.mean(mrr_after) - np.mean(mrr_before):+.3f}

Precision@5:
• Antes del Reranking: μ={np.mean(p5_before):.3f}, σ={np.std(p5_before):.3f}, min={np.min(p5_before):.3f}, max={np.max(p5_before):.3f}
• Después del Reranking: μ={np.mean(p5_after):.3f}, σ={np.std(p5_after):.3f}, min={np.min(p5_after):.3f}, max={np.max(p5_after):.3f}
• Mejora promedio: {np.mean(p5_after) - np.mean(p5_before):+.3f}

F1@5:
• Antes del Reranking: μ={np.mean(f1_before):.3f}, σ={np.std(f1_before):.3f}, min={np.min(f1_before):.3f}, max={np.max(f1_before):.3f}
• Después del Reranking: μ={np.mean(f1_after):.3f}, σ={np.std(f1_after):.3f}, min={np.min(f1_after):.3f}, max={np.max(f1_after):.3f}
• Mejora promedio: {np.mean(f1_after) - np.mean(f1_before):+.3f}

INTERPRETACIÓN DE RESULTADOS:
• El reranking LLM {'mejoró' if np.mean(mrr_after) > np.mean(mrr_before) else 'no mejoró'} significativamente el MRR
• La precisión {'aumentó' if np.mean(p5_after) > np.mean(p5_before) else 'no aumentó'} con el reranking
• El F1-score {'se benefició' if np.mean(f1_after) > np.mean(f1_before) else 'no se benefició'} del reranking LLM

CONCLUSIONES:
• Modelo evaluado: {model_name}
• Número total de preguntas: {num_questions}
• Tiempo de evaluación: {evaluation_time:.1f} segundos
• Reranking LLM: {'Efectivo' if use_llm_reranker and np.mean(mrr_after) > np.mean(mrr_before) else 'No aplicado o poco efectivo'}
"""
        
        ax.text(0.05, 0.95, stats_text, fontsize=10, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        ax.set_title('Análisis Estadístico y Conclusiones', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 9: Recomendaciones y metodología
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        methodology_text = """
METODOLOGÍA DE EVALUACIÓN

1. PREPARACIÓN DE DATOS:
   • Se utilizaron preguntas de Stack Overflow sobre Azure
   • Se filtraron preguntas con respuestas que contienen enlaces de Microsoft Learn
   • Se extrajo el ground truth de enlaces relevantes de las respuestas aceptadas

2. PROCESO DE EVALUACIÓN:
   • Para cada pregunta se ejecutó el pipeline de recuperación
   • Se recuperaron documentos usando embeddings de similaridad
   • Se aplicó reranking LLM (GPT-4) para mejorar el orden de relevancia
   • Se calcularon métricas antes y después del reranking

3. MÉTRICAS CALCULADAS:
   • Precision@k: Proporción de documentos relevantes en top-k
   • Recall@k: Proporción de documentos relevantes recuperados
   • F1@k: Balance entre precisión y recall
   • MRR: Posición promedio del primer documento relevante
   • nDCG@k: Ganancia acumulativa con descuento normalizada

4. RECOMENDACIONES:
   • Para aplicaciones que requieren alta precisión: usar k=1 o k=3
   • Para aplicaciones que requieren alta cobertura: usar k=10
   • El reranking LLM es especialmente útil cuando se requiere orden de relevancia
   • Monitorear el balance entre precisión y recall según el caso de uso

5. LIMITACIONES:
   • El ground truth está limitado a enlaces de Microsoft Learn
   • La evaluación depende de la calidad de las respuestas de Stack Overflow
   • Los resultados pueden variar según el dominio específico
"""
        
        ax.text(0.05, 0.95, methodology_text, fontsize=10, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Metodología y Recomendaciones', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    pdf_buffer.seek(0)
    return pdf_buffer.read()

def generate_multi_model_pdf_report(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool,
                                  evaluation_time: float, generative_model_name: str, 
                                  num_questions: int, top_k: int) -> bytes:
    """
    Genera un reporte PDF comparativo completo para múltiples modelos.
    Incluye todos los gráficos y tablas mostrados en la interfaz web.
    
    Args:
        results: Resultados de múltiples modelos
        use_llm_reranker: Si se usó reranking LLM
        evaluation_time: Tiempo total de evaluación
        generative_model_name: Modelo generativo usado
        num_questions: Número de preguntas evaluadas
        top_k: Valor de top-k usado
        
    Returns:
        bytes: Contenido del PDF generado
    """
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    pdf_buffer = BytesIO()
    
    # Preparar datos para múltiples visualizaciones
    main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
    all_k_metrics = []
    for k in [1, 3, 5, 10]:
        all_k_metrics.extend([f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}'])
    all_k_metrics.extend(['MRR@5', 'nDCG@5'])
    
    models_data = {}
    comparison_data = []
    improvement_data = []
    
    for model_name, res in results.items():
        before_metrics = res['avg_before_metrics']
        after_metrics = res['avg_after_metrics']
        
        models_data[model_name] = {
            'before': before_metrics,
            'after': after_metrics,
            'num_questions': res['num_questions_evaluated']
        }
        
        # Datos para gráfico de barras comparativo
        metrics = after_metrics if use_llm_reranker else before_metrics
        for m in main_metrics:
            comparison_data.append({
                'Modelo': model_name, 
                'Métrica': m, 
                'Valor': metrics.get(m, 0),
                'Tipo': 'Después del Reranking' if use_llm_reranker else 'Antes del Reranking'
            })
        
        # Datos para gráfico de mejora
        if use_llm_reranker:
            for m in main_metrics:
                before_val = before_metrics.get(m, 0)
                after_val = after_metrics.get(m, 0)
                improvement = ((after_val - before_val) / before_val * 100) if before_val > 0 else 0
                improvement_data.append({
                    'Modelo': model_name,
                    'Métrica': m,
                    'Mejora (%)': improvement,
                    'Delta': after_val - before_val
                })
    
    with PdfPages(pdf_buffer) as pdf:
        # Página 1: Portada
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'REPORTE COMPARATIVO DE MODELOS', 
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        models_list = list(results.keys())
        info_text = f"""
INFORMACIÓN GENERAL:
• Modelos Evaluados: {', '.join(models_list)}
• Modelo Generativo: {generative_model_name}
• Preguntas Evaluadas: {num_questions}
• Top-K Documentos: {top_k}
• Reranking LLM: {'Habilitado' if use_llm_reranker else 'Deshabilitado'}
• Tiempo Total: {evaluation_time:.1f} segundos
• Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        ax.text(0.1, 0.7, info_text, fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Resumen de mejor modelo
        main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
        
        # Calcular puntajes
        model_scores = {}
        weights = {'Precision@5': 0.25, 'Recall@5': 0.25, 'F1@5': 0.25, 'MRR@5': 0.15, 'nDCG@5': 0.10}
        
        for model_name, res in results.items():
            metrics = res['avg_after_metrics'] if use_llm_reranker else res['avg_before_metrics']
            score = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
            model_scores[model_name] = score
        
        best_model = max(model_scores, key=model_scores.get)
        
        summary_text = f"""
RESUMEN EJECUTIVO:
• Mejor Modelo: {best_model}
• Puntaje: {model_scores[best_model]*100:.1f}%

RANKING DE MODELOS:
"""
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(sorted_models):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            summary_text += f"{medal} {model}: {score*100:.1f}%\n"
        
        ax.text(0.1, 0.35, summary_text, fontsize=11, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 2: Comparación General - Gráfico de barras agrupado
        fig, ax = plt.subplots(figsize=(14, 8))
        
        models = list(results.keys())
        metrics = main_metrics
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)  # Ajustar ancho basado en número de modelos
        colors = get_color_palette()
        
        for i, model in enumerate(models):
            model_metrics = results[model]['avg_after_metrics'] if use_llm_reranker else results[model]['avg_before_metrics']
            values = [model_metrics.get(m, 0) for m in metrics]
            
            bars = ax.bar(x + i*width, values, width, label=model, 
                         color=colors[i % len(colors)], alpha=0.8)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Agregar líneas de referencia
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Muy Bueno (≥0.7)')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Bueno (≥0.4)')
        
        ax.set_xlabel('Métricas')
        ax.set_ylabel('Valor')
        ax.set_title(f'Comparación de Métricas {"Después del Reranking" if use_llm_reranker else "Antes del Reranking"}')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 3: Gráfico Radar - Vista Multidimensional
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Ángulos para el radar
        angles = [n / float(len(main_metrics)) * 2 * np.pi for n in range(len(main_metrics))]
        angles += angles[:1]  # Cerrar el círculo
        
        for i, (model_name, data) in enumerate(models_data.items()):
            metrics = data['after'] if use_llm_reranker else data['before']
            values = [metrics.get(m, 0) for m in main_metrics]
            values += values[:1]  # Cerrar el radar
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                   color=colors[i % len(colors)], alpha=0.8)
            ax.fill(angles, values, alpha=0.3, color=colors[i % len(colors)])
        
        # Configurar labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(main_metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Comparación Multidimensional de Modelos', y=1.08, fontsize=16, fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 4: Mapa de Calor - Rendimiento Detallado
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Preparar datos para heatmap con todas las métricas disponibles
        all_metrics = set()
        for data in models_data.values():
            metrics = data['after'] if use_llm_reranker else data['before']
            all_metrics.update(metrics.keys())
        
        relevant_metrics = [m for m in all_k_metrics if m in all_metrics]
        relevant_metrics.sort()
        
        heatmap_data = []
        for model_name, data in models_data.items():
            metrics = data['after'] if use_llm_reranker else data['before']
            model_values = [metrics.get(m, 0) for m in relevant_metrics]
            heatmap_data.append(model_values)
        
        # Crear heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Configurar etiquetas
        ax.set_xticks(range(len(relevant_metrics)))
        ax.set_xticklabels(relevant_metrics, rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(list(models_data.keys()))
        
        # Agregar valores en el heatmap
        for i in range(len(models)):
            for j in range(len(relevant_metrics)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Valor de Métrica', rotation=270, labelpad=20)
        
        ax.set_title('Mapa de Calor - Todas las Métricas', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 5: Mejoras con Reranking LLM (si aplica)
        if use_llm_reranker and improvement_data:
            # Gráfico de mejoras porcentuales
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Preparar datos para el gráfico
            improvement_by_metric = {}
            delta_by_metric = {}
            
            for item in improvement_data:
                metric = item['Métrica']
                if metric not in improvement_by_metric:
                    improvement_by_metric[metric] = []
                    delta_by_metric[metric] = []
                improvement_by_metric[metric].append((item['Modelo'], item['Mejora (%)']))
                delta_by_metric[metric].append((item['Modelo'], item['Delta']))
            
            # Gráfico de mejoras porcentuales
            x_pos = np.arange(len(main_metrics))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                improvements = [next((imp for mod, imp in improvement_by_metric[metric] if mod == model), 0) 
                              for metric in main_metrics]
                
                bars = ax1.bar(x_pos + i*width, improvements, width, label=model,
                              color=colors[i % len(colors)], alpha=0.8)
                
                # Agregar valores en las barras
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                           f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
            
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_xlabel('Métrica')
            ax1.set_ylabel('Mejora (%)')
            ax1.set_title('Mejora Porcentual por Modelo y Métrica')
            ax1.set_xticks(x_pos + width * (len(models) - 1) / 2)
            ax1.set_xticklabels(main_metrics, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Gráfico de deltas absolutas
            for i, model in enumerate(models):
                deltas = [next((delta for mod, delta in delta_by_metric[metric] if mod == model), 0) 
                         for metric in main_metrics]
                
                bars = ax2.bar(x_pos + i*width, deltas, width, label=model,
                              color=colors[i % len(colors)], alpha=0.8)
                
                # Agregar valores en las barras
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.005),
                           f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Métrica')
            ax2.set_ylabel('Delta (Después - Antes)')
            ax2.set_title('Mejora Absoluta por Modelo y Métrica')
            ax2.set_xticks(x_pos + width * (len(models) - 1) / 2)
            ax2.set_xticklabels(main_metrics, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Página 6: Comparación Detallada por Valores de K (todos en una página)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        k_values = [1, 3, 5, 10]
        axes = [ax1, ax2, ax3, ax4]
        
        for i, k in enumerate(k_values):
            ax = axes[i]
            k_metrics = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}']
            available_k_metrics = [m for m in k_metrics if any(m in res['avg_after_metrics'] for res in results.values())]
            
            if available_k_metrics:
                x_pos = np.arange(len(available_k_metrics))
                width = 0.8 / len(models)
                
                for j, model in enumerate(models):
                    model_metrics = results[model]['avg_after_metrics'] if use_llm_reranker else results[model]['avg_before_metrics']
                    values = [model_metrics.get(m, 0) for m in available_k_metrics]
                    
                    bars = ax.bar(x_pos + j*width, values, width, label=model,
                                 color=colors[j % len(colors)], alpha=0.8)
                    
                    # Agregar valores en las barras
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xlabel('Métrica')
                ax.set_ylabel('Valor')
                ax.set_title(f'Métricas para K={k}', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
                ax.set_xticklabels([m.replace(f'@{k}', '') for m in available_k_metrics])
                if i == 0:  # Solo mostrar leyenda en el primer gráfico
                    ax.legend()
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No hay datos\ndisponibles para K={k}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('Comparación Detallada por Valores de K', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 7: Tabla Comparativa Completa
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')
        
        # Preparar datos de la tabla con todas las métricas principales
        table_data = []
        for metric in main_metrics:
            row = [metric]
            for model in models:
                model_metrics = results[model]['avg_after_metrics'] if use_llm_reranker else results[model]['avg_before_metrics']
                value = model_metrics.get(metric, 0)
                quality = grade_metric(value)
                row.append(f'{value:.3f}\n{quality}')
            table_data.append(row)
        
        headers = ['Métrica'] + models
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.2] + [0.8/len(models)] * len(models))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)
        
        # Colorear header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorear filas alternadas
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Tabla Comparativa Completa con Interpretaciones', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 8: Resultados Detallados por Modelo Individual
        for model_name, res in results.items():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            before_metrics = res['avg_before_metrics']
            after_metrics = res['avg_after_metrics']
            
            # Gráfico 1: Comparación antes/después para este modelo
            k_values = [1, 3, 5, 10]
            metrics_types = ['Precision', 'Recall', 'F1', 'Accuracy']
            
            for i, metric_type in enumerate(metrics_types):
                ax = [ax1, ax2, ax3, ax4][i]
                
                before_vals = [before_metrics.get(f'{metric_type}@{k}', 0) for k in k_values]
                after_vals = [after_metrics.get(f'{metric_type}@{k}', 0) for k in k_values]
                
                x = np.arange(len(k_values))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, before_vals, width, label='Antes', alpha=0.7, color=colors[0])
                if use_llm_reranker:
                    bars2 = ax.bar(x + width/2, after_vals, width, label='Después', alpha=0.7, color=colors[1])
                
                # Agregar valores en las barras
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                if use_llm_reranker:
                    for bar in bars2:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'{metric_type}@k - {model_name}')
                ax.set_xlabel('K')
                ax.set_ylabel(f'{metric_type}')
                ax.set_xticks(x)
                ax.set_xticklabels(k_values)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
            plt.suptitle(f'Análisis Detallado - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Página 9: Resumen de Rankings
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        ranking_text = "RANKING DETALLADO POR MÉTRICA\n\n"
        
        # Calcular puntajes
        model_scores = {}
        weights = {'Precision@5': 0.25, 'Recall@5': 0.25, 'F1@5': 0.25, 'MRR@5': 0.15, 'nDCG@5': 0.10}
        
        for model_name, res in results.items():
            metrics = res['avg_after_metrics'] if use_llm_reranker else res['avg_before_metrics']
            score = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
            model_scores[model_name] = score
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranking_text += "RANKING GENERAL:\n"
        for i, (model, score) in enumerate(sorted_models):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            ranking_text += f"{medal} {model}: {score*100:.1f}% (Score ponderado)\n"
        
        ranking_text += "\nRANKING POR MÉTRICA INDIVIDUAL:\n\n"
        
        for metric in main_metrics:
            metric_values = [(name, res['avg_after_metrics' if use_llm_reranker else 'avg_before_metrics'].get(metric, 0)) 
                           for name, res in results.items()]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            
            ranking_text += f"{metric}:\n"
            for i, (name, val) in enumerate(metric_values):
                ranking_text += f"  {i+1}. {name}: {val:.3f}\n"
            ranking_text += "\n"
        
        # Estadísticas de mejora por modelo
        if use_llm_reranker:
            ranking_text += "ESTADÍSTICAS DE MEJORA CON LLM RERANKING:\n\n"
            for model_name in models:
                model_improvements = [item for item in improvement_data if item['Modelo'] == model_name]
                if model_improvements:
                    avg_improvement = np.mean([item['Mejora (%)'] for item in model_improvements])
                    positive_improvements = sum(1 for item in model_improvements if item['Mejora (%)'] > 0)
                    ranking_text += f"{model_name}:\n"
                    ranking_text += f"  • Mejora promedio: {avg_improvement:+.1f}%\n"
                    ranking_text += f"  • Métricas mejoradas: {positive_improvements}/{len(model_improvements)}\n\n"
        
        ax.text(0.05, 0.95, ranking_text, fontsize=10, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        ax.set_title('Ranking Detallado y Estadísticas', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 10: Definiciones de métricas y conclusiones
        fig, ax = plt.subplots(figsize=(8.5, 11))
        add_metric_definitions_page(pdf, ax)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Página 11: Conclusiones y recomendaciones finales
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        best_model = max(model_scores, key=model_scores.get)
        worst_model = min(model_scores, key=model_scores.get)
        
        conclusions_text = f"""
CONCLUSIONES Y RECOMENDACIONES FINALES

RESUMEN EJECUTIVO:
• Mejor Modelo Overall: {best_model} (Score: {model_scores[best_model]*100:.1f}%)
• Modelo con Mayor Margen de Mejora: {worst_model} (Score: {model_scores[worst_model]*100:.1f}%)
• Número de Modelos Evaluados: {len(models)}
• Preguntas Evaluadas por Modelo: {num_questions}
• Reranking LLM: {'Habilitado' if use_llm_reranker else 'Deshabilitado'}

HALLAZGOS PRINCIPALES:
• Diferencia entre mejor y peor modelo: {(model_scores[best_model] - model_scores[worst_model])*100:.1f} puntos porcentuales
• Todos los modelos {'se beneficiaron' if use_llm_reranker and all(any(item['Mejora (%)'] > 0 for item in improvement_data if item['Modelo'] == model) for model in models) else 'mostraron rendimiento variable'} {'del reranking LLM' if use_llm_reranker else 'en la evaluación'}

RECOMENDACIONES GENERALES:
1. Para aplicaciones de alta precisión: Utilizar {best_model}
2. Para aplicaciones balanceadas: Considerar top 3 modelos del ranking
3. Para optimización de recursos: Evaluar trade-off entre rendimiento y eficiencia
4. Para casos específicos: Revisar métricas individuales según caso de uso

PRÓXIMOS PASOS:
• Evaluar rendimiento en datos específicos del dominio
• Considerar fine-tuning de modelos prometedores
• Implementar A/B testing con los top 2 modelos
• Monitorear rendimiento en producción

LIMITACIONES DEL ESTUDIO:
• Dataset limitado a preguntas de Stack Overflow sobre Azure
• Ground truth basado en enlaces de Microsoft Learn
• Evaluación realizada en un momento específico
• Posible sesgo hacia ciertos tipos de consultas
"""
        
        ax.text(0.05, 0.95, conclusions_text, fontsize=10, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        ax.set_title('Conclusiones y Recomendaciones Finales', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    pdf_buffer.seek(0)
    return pdf_buffer.read()

def validate_data_integrity(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida la integridad de los datos antes de la descarga.
    
    Args:
        results: Diccionario con resultados de evaluación
        
    Returns:
        Diccionario con resultado de validación
    """
    validation_report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Validar estructura básica
        required_keys = ['num_questions_evaluated', 'avg_before_metrics', 'avg_after_metrics', 'all_questions_data']
        for key in required_keys:
            if key not in results:
                validation_report['errors'].append(f"Falta clave requerida: {key}")
                validation_report['is_valid'] = False
        
        # Validar datos de preguntas
        if 'all_questions_data' in results and results['all_questions_data']:
            questions_data = results['all_questions_data']
            validation_report['stats']['num_questions'] = len(questions_data)
            
            # Verificar que todas las preguntas tengan datos completos
            incomplete_questions = []
            for i, q_data in enumerate(questions_data):
                required_fields = ['question_num', 'ground_truth_links', 'docs_retrieved']
                for field in required_fields:
                    if field not in q_data:
                        incomplete_questions.append(f"Pregunta {i+1}: falta {field}")
            
            if incomplete_questions:
                validation_report['warnings'].extend(incomplete_questions)
            
            # Verificar métricas numéricas
            numeric_fields = ['before_precision_5', 'after_precision_5', 'before_recall_5', 'after_recall_5']
            for q_data in questions_data:
                for field in numeric_fields:
                    if field in q_data and not isinstance(q_data[field], (int, float)):
                        validation_report['warnings'].append(f"Pregunta {q_data.get('question_num', 'N/A')}: {field} no es numérico")
        
        # Validar métricas promedio
        for metrics_type in ['avg_before_metrics', 'avg_after_metrics']:
            if metrics_type in results:
                metrics = results[metrics_type]
                validation_report['stats'][metrics_type] = len(metrics)
                
                # Verificar que las métricas sean numéricas
                for metric_name, value in metrics.items():
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        validation_report['warnings'].append(f"{metrics_type}.{metric_name}: valor inválido ({value})")
        
        # Validar consistencia entre métricas individuales y promedio
        if 'individual_before_metrics' in results and 'avg_before_metrics' in results:
            individual_count = len(results['individual_before_metrics'])
            evaluated_count = results.get('num_questions_evaluated', 0)
            
            if individual_count != evaluated_count:
                validation_report['warnings'].append(
                    f"Inconsistencia: {individual_count} métricas individuales vs {evaluated_count} preguntas evaluadas"
                )
        
    except Exception as e:
        validation_report['errors'].append(f"Error durante validación: {str(e)}")
        validation_report['is_valid'] = False
    
    return validation_report
    
def create_question_data_record(question_num: int, question: str, ms_links: List[str], 
                               docs_count: int, before_metrics: Dict, after_metrics: Dict) -> Dict:
    """
    Crea un registro optimizado de datos de pregunta para reducir uso de memoria.
    
    Args:
        question_num: Número de pregunta
        question: Texto de la pregunta
        ms_links: Links de Microsoft Learn
        docs_count: Número de documentos recuperados
        before_metrics: Métricas antes del reranking
        after_metrics: Métricas después del reranking
        
    Returns:
        Diccionario con datos esenciales de la pregunta
    """
    return {
        'question_num': question_num,
        'question': question[:100] + '...' if len(question) > 100 else question,
        'ground_truth_links': len(ms_links),
        'docs_retrieved': docs_count,
        'before_precision_5': before_metrics.get('Precision@5', 0),
        'after_precision_5': after_metrics.get('Precision@5', 0),
        'before_recall_5': before_metrics.get('Recall@5', 0),
        'after_recall_5': after_metrics.get('Recall@5', 0),
        'before_f1_5': before_metrics.get('F1@5', 0),
        'after_f1_5': after_metrics.get('F1@5', 0)
    }

def calculate_average_metrics(all_metrics: List[Dict]) -> Dict[str, float]:
    """
    Calcula métricas promedio de una lista de métricas de forma eficiente.
    
    Args:
        all_metrics: Lista de diccionarios con métricas
        
    Returns:
        Diccionario con métricas promedio
    """
    if not all_metrics:
        return {}
    
    # Usar defaultdict para evitar verificaciones repetidas
    from collections import defaultdict
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    
    # Sumar todas las métricas de forma eficiente
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_sums[key] += value
                metric_counts[key] += 1
    
    # Calcular promedios usando comprehension
    average_metrics = {
        key: metric_sums[key] / metric_counts[key] 
        for key in metric_sums 
        if metric_counts[key] > 0
    }
    
    return average_metrics


def run_cumulative_metrics_for_models(
    questions_and_answers: List[Dict],
    num_questions: int,
    model_names: List[str],
    generative_model_name: str,
    top_k: int = 10,
    use_llm_reranker: bool = True,
    batch_size: int = 50
) -> Dict[str, Dict[str, Any]]:
    """Run cumulative evaluation for several embedding models."""
    results = {}
    total_models = len(model_names)

    for i, model in enumerate(model_names):
        with st.spinner(f"Evaluando {model} ({i+1}/{total_models})..."):
            model_result = run_cumulative_metrics_evaluation(
                questions_and_answers=questions_and_answers,
                num_questions=num_questions,
                model_name=model,
                generative_model_name=generative_model_name,
                top_k=top_k,
                use_llm_reranker=use_llm_reranker,
                batch_size=batch_size
            )
            results[model] = model_result
            
            # Limpiar memoria entre modelos
            gc.collect()

    return results

def run_cumulative_metrics_evaluation(
    questions_and_answers: List[Dict],
    num_questions: int,
    model_name: str,
    generative_model_name: str,
    top_k: int = 10,
    use_llm_reranker: bool = True,
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Ejecuta evaluación de métricas acumulativas para múltiples preguntas.
    
    Args:
        questions_and_answers: Lista de preguntas y respuestas
        num_questions: Número de preguntas a evaluar
        model_name: Nombre del modelo de embedding
        generative_model_name: Nombre del modelo generativo
        top_k: Número de documentos top-k
        use_llm_reranker: Si usar LLM reranking
        
    Returns:
        Diccionario con resultados de evaluación
    """
    # Filtrar preguntas con links
    filtered_questions = filter_questions_with_links(questions_and_answers)
    
    if len(filtered_questions) < num_questions:
        st.warning(f"⚠️ Solo hay {len(filtered_questions)} preguntas con links disponibles. Se usarán todas.")
        num_questions = len(filtered_questions)
    
    # Seleccionar preguntas aleatoriamente
    if len(filtered_questions) > num_questions:
        selected_indices = np.random.choice(len(filtered_questions), num_questions, replace=False)
        selected_questions = [filtered_questions[i] for i in selected_indices]
    else:
        selected_questions = filtered_questions
    
    # Inicializar clientes
    weaviate_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(
        model_name, generative_model_name
    )
    
    # Listas para almacenar métricas
    before_reranking_metrics = []
    after_reranking_metrics = []
    all_questions_data = []
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Procesar preguntas en lotes
    num_batches = (num_questions + batch_size - 1) // batch_size
    initial_memory = get_memory_usage()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_questions)
        batch_questions = selected_questions[start_idx:end_idx]
        
        current_memory = get_memory_usage()
        status_text.text(f"Procesando lote {batch_idx + 1}/{num_batches} ({end_idx - start_idx} preguntas) - Memoria: {current_memory:.1f} MB")
        
        # Procesar cada pregunta en el lote
        for i, qa in enumerate(batch_questions):
            global_idx = start_idx + i
            question = qa['question']
            ground_truth_answer = qa.get('accepted_answer', '')
            ms_links = qa.get('ms_links', [])
            
            try:
                # Ejecutar pipeline con métricas
                result = answer_question_with_retrieval_metrics(
                    question=question,
                    weaviate_wrapper=weaviate_wrapper,
                    embedding_client=embedding_client,
                    openai_client=openai_client,
                    gemini_client=gemini_client,
                    local_tinyllama_client=local_tinyllama_client,
                    local_mistral_client=local_mistral_client,
                    openrouter_client=openrouter_client,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    generate_answer=False,  # Sin RAG como se solicitó
                    calculate_metrics=True,
                    ground_truth_answer=ground_truth_answer,
                    ms_links=ms_links,
                    generative_model_name=generative_model_name
                )
                
                if len(result) >= 3:
                    docs, debug_info, retrieval_metrics = result
                    
                    # Extraer métricas antes y después del reranking
                    before_metrics = retrieval_metrics.get('before_reranking', {})
                    after_metrics = retrieval_metrics.get('after_reranking', {})
                    
                    before_reranking_metrics.append(before_metrics)
                    after_reranking_metrics.append(after_metrics)
                    
                    # Almacenar datos de la pregunta para referencia usando función optimizada
                    question_data = create_question_data_record(
                        question_num=global_idx + 1,
                        question=question,
                        ms_links=ms_links,
                        docs_count=len(docs),
                        before_metrics=before_metrics,
                        after_metrics=after_metrics
                    )
                    all_questions_data.append(question_data)
                    
            except Exception as e:
                st.error(f"Error evaluando pregunta {global_idx + 1}: {e}")
                continue
            
            progress_bar.progress((global_idx + 1) / num_questions)
        
        # Limpiar memoria después de cada lote
        cleanup_memory()
        
        # Pausa breve para permitir que el sistema respire
        time.sleep(0.1)
        
        # Mostrar información de memoria cada 5 lotes
        if (batch_idx + 1) % 5 == 0:
            memory_used = get_memory_usage()
            memory_increase = memory_used - initial_memory
            if memory_increase > 0:
                st.info(f"📊 Memoria utilizada: {memory_used:.1f} MB (+{memory_increase:.1f} MB desde inicio)")
    
    # Limpieza final
    cleanup_memory()
    final_memory = get_memory_usage()
    total_memory_increase = final_memory - initial_memory
    
    # Calcular métricas promedio
    avg_before_metrics = calculate_average_metrics(before_reranking_metrics)
    avg_after_metrics = calculate_average_metrics(after_reranking_metrics)
    
    # Limpiar interfaz
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar estadísticas de memoria
    if total_memory_increase > 50:  # Solo mostrar si el incremento es significativo
        st.info(f"💾 Evaluación completada. Incremento total de memoria: {total_memory_increase:.1f} MB")
    
    return {
        'num_questions_evaluated': len(before_reranking_metrics),
        'avg_before_metrics': avg_before_metrics,
        'avg_after_metrics': avg_after_metrics,
        'all_questions_data': all_questions_data,
        'individual_before_metrics': before_reranking_metrics,
        'individual_after_metrics': after_reranking_metrics,
        'memory_stats': {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_increase': total_memory_increase
        }
    }

def display_cumulative_metrics(results: Dict[str, Any], model_name: str, use_llm_reranker: bool):
    """
    Muestra los resultados de métricas acumulativas en la interfaz.
    
    Args:
        results: Resultados de la evaluación
        model_name: Nombre del modelo usado
        use_llm_reranker: Si se usó LLM reranking
    """
    num_questions = results['num_questions_evaluated']
    avg_before = results['avg_before_metrics']
    avg_after = results['avg_after_metrics']
    
    st.success(f"✅ Evaluación completada para {num_questions} preguntas")
    
    # Métricas principales en columnas
    st.subheader("📊 Métricas Promedio")
    
    # Métricas principales para diferentes valores de k
    k_values = [1, 3, 5, 10]
    main_metrics = []
    for k in k_values:
        main_metrics.extend([f'Precision@{k}', f'Recall@{k}', f'F1@{k}'])
    main_metrics.extend(['MRR@5', 'nDCG@5'])
    
    col1, col2 = st.columns(2)
    
    # Organizar métricas por k en tabs
    tab1, tab2, tab3, tab4 = st.tabs([f"📊 Top-{k}" for k in k_values])
    
    for i, k in enumerate(k_values):
        with [tab1, tab2, tab3, tab4][i]:
            k_metrics = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}']
            if k == 5:  # Agregar MRR y nDCG solo para k=5
                k_metrics.extend(['MRR@5', 'nDCG@5'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔍 Antes del Reranking**")
                for metric in k_metrics:
                    if metric in avg_before:
                        st.metric(
                            label=metric,
                            value=f"{avg_before[metric]:.3f}",
                            help=f"Promedio de {metric} antes del reranking LLM"
                        )
            
            with col2:
                if use_llm_reranker:
                    st.markdown("**🤖 Después del Reranking LLM**")
                    for metric in k_metrics:
                        if metric in avg_after:
                            # Calcular delta
                            delta = avg_after[metric] - avg_before.get(metric, 0)
                            st.metric(
                                label=metric,
                                value=f"{avg_after[metric]:.3f}",
                                delta=f"{delta:+.3f}",
                                help=f"Promedio de {metric} después del reranking LLM"
                            )
                else:
                    st.info("ℹ️ Reranking LLM deshabilitado")
    
    # Gráfico de comparación
    if use_llm_reranker and avg_before and avg_after:
        st.subheader("📈 Comparación Visual")
        
        # Preparar datos para el gráfico
        metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
        before_values = [avg_before.get(m, 0) for m in metrics_to_plot]
        after_values = [avg_after.get(m, 0) for m in metrics_to_plot]
        
        # Crear gráfico de barras comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Antes del Reranking',
            x=metrics_to_plot,
            y=before_values,
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Después del Reranking',
            x=metrics_to_plot,
            y=after_values,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title=f'Comparación de Métricas Promedio ({num_questions} preguntas)',
            xaxis_title='Métricas',
            yaxis_title='Valor Promedio',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"individual_comparison_{model_name}")

    # Tabla resumen con interpretación
    summary_metrics = ['MRR']
    for k in [1, 3, 5, 10]:
        summary_metrics.extend([
            f'Recall@{k}', f'Precision@{k}', f'F1@{k}', f'Accuracy@{k}'
        ])

    rows = []
    for metric in summary_metrics:
        before_val = avg_before.get(metric, np.nan)
        after_val = avg_after.get(metric, np.nan) if use_llm_reranker else before_val
        delta_val = after_val - before_val if use_llm_reranker else 0
        rows.append({
            'Métrica': metric,
            'Before': before_val,
            'After': after_val,
            'Δ': delta_val,
            'Interpretación': grade_metric(after_val)
        })

    metrics_df = pd.DataFrame(rows)

    st.markdown("### 📋 Resumen de Métricas")
    st.dataframe(
        style_metrics_df(metrics_df).format({'Before': '{:.3f}', 'After': '{:.3f}', 'Δ': '{:+.3f}'}),
        use_container_width=True
    )

    # Métricas adicionales en expander
    with st.expander("📋 Métricas Detalladas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Antes del Reranking**")
            before_df = pd.DataFrame([avg_before]).T
            before_df.columns = ['Valor Promedio']
            before_df.index.name = 'Métrica'
            st.dataframe(before_df.style.format({'Valor Promedio': '{:.4f}'}))
        
        with col2:
            if use_llm_reranker and avg_after:
                st.markdown("**Después del Reranking**")
                after_df = pd.DataFrame([avg_after]).T
                after_df.columns = ['Valor Promedio']
                after_df.index.name = 'Métrica'
                st.dataframe(after_df.style.format({'Valor Promedio': '{:.4f}'}))
    
    # Tabla de evolución por pregunta
    with st.expander("📊 Evolución por Pregunta"):
        questions_df = pd.DataFrame(results['all_questions_data'])
        if not questions_df.empty:
            # Renombrar columnas para mejor legibilidad
            column_names = {
                'question_num': 'Pregunta #',
                'ground_truth_links': 'Links GT',
                'docs_retrieved': 'Docs Recuperados',
                'before_precision_5': 'Precision@5 (Antes)',
                'after_precision_5': 'Precision@5 (Después)',
                'before_f1_5': 'F1@5 (Antes)',
                'after_f1_5': 'F1@5 (Después)',
                'before_recall_5': 'Recall@5 (Antes)',
                'after_recall_5': 'Recall@5 (Después)'
            }
            
            # Mostrar tabla con métricas por pregunta
            if use_llm_reranker:
                display_columns = ['question_num', 'ground_truth_links', 'docs_retrieved', 
                                 'before_precision_5', 'after_precision_5', 
                                 'before_f1_5', 'after_f1_5']
            else:
                display_columns = ['question_num', 'ground_truth_links', 'docs_retrieved', 
                                 'before_precision_5', 'before_f1_5']
            
            questions_df_display = questions_df[display_columns].rename(columns=column_names)
            
            st.dataframe(questions_df_display.style.format({
                'Precision@5 (Antes)': '{:.3f}',
                'Precision@5 (Después)': '{:.3f}',
                'F1@5 (Antes)': '{:.3f}',
                'F1@5 (Después)': '{:.3f}'
            }), use_container_width=True)


def display_models_comparison(results: Dict[str, Dict[str, Any]], use_llm_reranker: bool) -> None:
    """Display comprehensive side-by-side comparison of multiple models."""
    st.subheader("📈 Comparación Completa Entre Modelos")

    # Métricas principales para visualización (enfoque en k=5 para simplicidad en comparación)
    main_metrics = ['Precision@5', 'Recall@5', 'F1@5', 'MRR@5', 'nDCG@5']
    
    # Métricas completas para análisis detallado
    all_k_metrics = []
    for k in [1, 3, 5, 10]:
        all_k_metrics.extend([f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}'])
    all_k_metrics.extend(['MRR@5', 'nDCG@5'])
    
    # Preparar datos para múltiples visualizaciones
    models_data = {}
    comparison_data = []
    improvement_data = []
    
    for model_name, res in results.items():
        before_metrics = res['avg_before_metrics']
        after_metrics = res['avg_after_metrics']
        
        models_data[model_name] = {
            'before': before_metrics,
            'after': after_metrics,
            'num_questions': res['num_questions_evaluated']
        }
        
        # Datos para gráfico de barras comparativo
        metrics = after_metrics if use_llm_reranker else before_metrics
        for m in main_metrics:
            comparison_data.append({
                'Modelo': model_name, 
                'Métrica': m, 
                'Valor': metrics.get(m, 0),
                'Tipo': 'Después del Reranking' if use_llm_reranker else 'Antes del Reranking'
            })
        
        # Datos para gráfico de mejora
        if use_llm_reranker:
            for m in main_metrics:
                before_val = before_metrics.get(m, 0)
                after_val = after_metrics.get(m, 0)
                improvement = ((after_val - before_val) / before_val * 100) if before_val > 0 else 0
                improvement_data.append({
                    'Modelo': model_name,
                    'Métrica': m,
                    'Mejora (%)': improvement,
                    'Delta': after_val - before_val
                })

    # Tab layout para múltiples visualizaciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Comparación General", "🎯 Gráfico Radar", "🔥 Mapa de Calor", "📈 Mejoras", "🔢 Métricas por K"])
    
    with tab1:
        st.markdown("#### Comparación de Métricas Principales")
        
        # Gráfico de barras mejorado
        df_comparison = pd.DataFrame(comparison_data)
        fig_bar = px.bar(
            df_comparison, 
            x='Métrica', 
            y='Valor', 
            color='Modelo',
            barmode='group',
            title=f"Comparación de Métricas {'Después del Reranking' if use_llm_reranker else 'Antes del Reranking'}",
            height=500
        )
        
        fig_bar.update_layout(
            xaxis_title="Métrica",
            yaxis_title="Valor",
            legend_title="Modelo de Embedding",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Agregar líneas de referencia
        fig_bar.add_hline(y=0.7, line_dash="dash", line_color="green", 
                         annotation_text="Muy Bueno (≥0.7)")
        fig_bar.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                         annotation_text="Bueno (≥0.4)")
        
        st.plotly_chart(fig_bar, use_container_width=True, key="comparison_bar_chart")
        
        # Tabla resumen
        st.markdown("#### Resumen Numérico")
        summary_table = []
        for model_name, data in models_data.items():
            metrics = data['after'] if use_llm_reranker else data['before']
            summary_table.append({
                'Modelo': model_name,
                'Precision@5': f"{metrics.get('Precision@5', 0):.3f}",
                'Recall@5': f"{metrics.get('Recall@5', 0):.3f}",
                'F1@5': f"{metrics.get('F1@5', 0):.3f}",
                'MRR@5': f"{metrics.get('MRR@5', 0):.3f}",
                'nDCG@5': f"{metrics.get('nDCG@5', 0):.3f}",
                'Preguntas': data['num_questions']
            })
        
        summary_df = pd.DataFrame(summary_table)
        st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### Gráfico Radar - Vista Multidimensional")
        
        # Crear gráfico radar
        fig_radar = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (model_name, data) in enumerate(models_data.items()):
            metrics = data['after'] if use_llm_reranker else data['before']
            
            values = [metrics.get(m, 0) for m in main_metrics]
            values.append(values[0])  # Cerrar el radar
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=main_metrics + [main_metrics[0]],
                fill='toself',
                name=model_name,
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                )
            ),
            showlegend=True,
            title="Comparación Multidimensional de Modelos",
            height=600
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
        
        # Interpretación del radar
        st.markdown("##### 🎯 Interpretación:")
        st.markdown("""
        - **Área más grande**: Mejor rendimiento general
        - **Forma regular**: Rendimiento equilibrado entre métricas
        - **Picos específicos**: Fortalezas particulares del modelo
        """)
    
    with tab3:
        st.markdown("#### Mapa de Calor - Rendimiento Detallado")
        
        # Preparar datos para heatmap
        all_metrics = set()
        for data in models_data.values():
            metrics = data['after'] if use_llm_reranker else data['before']
            all_metrics.update(metrics.keys())
        
        # Usar métricas detalladas para el heatmap
        relevant_metrics = [m for m in all_k_metrics if m in all_metrics]
        relevant_metrics.sort()
        
        heatmap_data = []
        for model_name, data in models_data.items():
            metrics = data['after'] if use_llm_reranker else data['before']
            model_values = [metrics.get(m, 0) for m in relevant_metrics]
            heatmap_data.append(model_values)
        
        # Crear heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=relevant_metrics,
            y=list(models_data.keys()),
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Valor de Métrica"),
            hoverongaps=False,
            text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title="Mapa de Calor - Todas las Métricas",
            xaxis_title="Métrica",
            yaxis_title="Modelo",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart")
        
        # Ranking de modelos
        st.markdown("##### 🏆 Ranking por Métrica:")
        for metric in main_metrics:
            metric_values = [(name, data['after' if use_llm_reranker else 'before'].get(metric, 0)) 
                           for name, data in models_data.items()]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            
            ranking_text = " > ".join([f"**{name}** ({val:.3f})" for name, val in metric_values])
            st.markdown(f"**{metric}:** {ranking_text}")
    
    with tab4:
        if use_llm_reranker and improvement_data:
            st.markdown("#### Mejoras con Reranking LLM")
            
            # Gráfico de mejoras porcentuales
            df_improvement = pd.DataFrame(improvement_data)
            
            fig_improvement = px.bar(
                df_improvement,
                x='Métrica',
                y='Mejora (%)',
                color='Modelo',
                barmode='group',
                title="Mejora Porcentual por Modelo y Métrica",
                height=500
            )
            
            fig_improvement.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig_improvement.update_layout(
                xaxis_title="Métrica",
                yaxis_title="Mejora (%)",
                legend_title="Modelo"
            )
            
            st.plotly_chart(fig_improvement, use_container_width=True, key="improvement_chart")
            
            # Gráfico de deltas absolutas
            fig_delta = px.bar(
                df_improvement,
                x='Métrica',
                y='Delta',
                color='Modelo',
                barmode='group',
                title="Mejora Absoluta por Modelo y Métrica",
                height=400
            )
            
            fig_delta.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig_delta.update_layout(
                xaxis_title="Métrica",
                yaxis_title="Delta (Después - Antes)",
                legend_title="Modelo"
            )
            
            st.plotly_chart(fig_delta, use_container_width=True, key="delta_chart")
            
            # Estadísticas de mejora
            st.markdown("##### 📊 Estadísticas de Mejora:")
            for model_name in models_data.keys():
                model_improvements = [item for item in improvement_data if item['Modelo'] == model_name]
                avg_improvement = np.mean([item['Mejora (%)'] for item in model_improvements])
                positive_improvements = sum(1 for item in model_improvements if item['Mejora (%)'] > 0)
                
                st.markdown(f"**{model_name}:** {avg_improvement:+.1f}% mejora promedio, {positive_improvements}/{len(model_improvements)} métricas mejoradas")
        else:
            st.info("ℹ️ Las mejoras solo se muestran cuando el reranking LLM está habilitado")
    
    with tab5:
        st.markdown("#### Comparación Detallada por Valores de K")
        
        # Crear subtabs para cada valor de k
        k_tab1, k_tab2, k_tab3, k_tab4 = st.tabs([f"📊 K={k}" for k in [1, 3, 5, 10]])
        
        for i, k in enumerate([1, 3, 5, 10]):
            with [k_tab1, k_tab2, k_tab3, k_tab4][i]:
                k_metrics = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'Accuracy@{k}']
                
                # Datos para gráfico de barras por k
                k_data = []
                for model_name, res in results.items():
                    metrics = res['avg_after_metrics'] if use_llm_reranker else res['avg_before_metrics']
                    for metric in k_metrics:
                        if metric in metrics:
                            k_data.append({
                                'Modelo': model_name,
                                'Métrica': metric,
                                'Valor': metrics[metric]
                            })
                
                if k_data:
                    df_k = pd.DataFrame(k_data)
                    fig_k = px.bar(
                        df_k,
                        x='Métrica',
                        y='Valor',
                        color='Modelo',
                        barmode='group',
                        title=f"Comparación de Métricas para K={k}",
                        height=400
                    )
                    
                    fig_k.update_layout(
                        xaxis_title="Métrica",
                        yaxis_title="Valor",
                        legend_title="Modelo"
                    )
                    
                    st.plotly_chart(fig_k, use_container_width=True, key=f"k_comparison_{k}")
                    
                    # Tabla resumen para este k
                    st.markdown(f"##### Tabla Resumen para K={k}")
                    k_summary = []
                    for model_name, res in results.items():
                        metrics = res['avg_after_metrics'] if use_llm_reranker else res['avg_before_metrics']
                        row = {'Modelo': model_name}
                        for metric in k_metrics:
                            row[metric] = f"{metrics.get(metric, 0):.3f}"
                        k_summary.append(row)
                    
                    k_summary_df = pd.DataFrame(k_summary)
                    st.dataframe(k_summary_df, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para K={k}")
    
    # Resumen general al final
    st.markdown("---")
    st.markdown("### 🏆 Resumen General de Rendimiento")
    
    # Calcular puntaje general por modelo
    model_scores = {}
    for model_name, data in models_data.items():
        metrics = data['after'] if use_llm_reranker else data['before']
        
        # Calcular puntaje ponderado (puedes ajustar los pesos según importancia)
        weights = {
            'Precision@5': 0.25,
            'Recall@5': 0.25,
            'F1@5': 0.25,
            'MRR@5': 0.15,
            'nDCG@5': 0.10
        }
        
        score = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
        model_scores[model_name] = {
            'score': score,
            'num_questions': data['num_questions']
        }
    
    # Ordenar modelos por puntaje
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Mostrar ranking
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🥇 Ranking Global")
        for i, (model_name, data) in enumerate(sorted_models):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            score_pct = data['score'] * 100
            st.markdown(f"{medal} **{model_name}**: {score_pct:.1f}% puntaje general")
    
    with col2:
        st.markdown("#### 📈 Métricas del Ganador")
        if sorted_models:
            winner_name = sorted_models[0][0]
            winner_metrics = models_data[winner_name]['after' if use_llm_reranker else 'before']
            
            for metric in main_metrics:
                value = winner_metrics.get(metric, 0)
                quality = grade_metric(value)
                st.metric(
                    label=metric,
                    value=f"{value:.3f}",
                    help=f"Calidad: {quality}"
                )
    
    # Gráfico de puntaje general
    st.markdown("#### 📊 Puntaje General Comparativo")
    score_data = []
    for model_name, data in model_scores.items():
        score_data.append({
            'Modelo': model_name,
            'Puntaje General': data['score'],
            'Puntaje (%)': data['score'] * 100
        })
    
    score_df = pd.DataFrame(score_data)
    fig_score = px.bar(
        score_df,
        x='Modelo',
        y='Puntaje (%)',
        title="Puntaje General por Modelo (Ponderado)",
        color='Puntaje (%)',
        color_continuous_scale='RdYlGn',
        height=400
    )
    
    fig_score.update_layout(
        xaxis_title="Modelo de Embedding",
        yaxis_title="Puntaje General (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig_score, use_container_width=True, key="score_chart")
    
    # Recomendaciones
    st.markdown("#### 💡 Recomendaciones")
    
    if sorted_models:
        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]
        
        best_metrics = models_data[best_model]['after' if use_llm_reranker else 'before']
        worst_metrics = models_data[worst_model]['after' if use_llm_reranker else 'before']
        
        st.success(f"🏆 **Mejor modelo general**: {best_model}")
        st.info(f"📊 **Uso recomendado**: {best_model} para aplicaciones que requieren balance entre precisión y recall")
        
        # Encontrar fortalezas específicas
        for metric in main_metrics:
            metric_ranking = [(name, data['after' if use_llm_reranker else 'before'].get(metric, 0)) 
                            for name, data in models_data.items()]
            metric_ranking.sort(key=lambda x: x[1], reverse=True)
            
            if metric_ranking[0][0] != best_model:
                st.info(f"🎯 **Especialista en {metric}**: {metric_ranking[0][0]} ({metric_ranking[0][1]:.3f})")
    
    # Mostrar estadísticas adicionales
    with st.expander("📋 Estadísticas Detalladas"):
        st.markdown("##### Variabilidad entre modelos:")
        for metric in main_metrics:
            values = [data['after' if use_llm_reranker else 'before'].get(metric, 0) 
                     for data in models_data.values()]
            std_dev = np.std(values)
            mean_val = np.mean(values)
            cv = (std_dev / mean_val) * 100 if mean_val > 0 else 0
            
            st.markdown(f"**{metric}**: μ={mean_val:.3f}, σ={std_dev:.3f}, CV={cv:.1f}%")

def load_questions_from_json(file_path: str) -> List[Dict]:
    """
    Carga preguntas desde archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Lista de preguntas y respuestas
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo '{file_path}'")
        return []
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al leer el archivo JSON: {e}")
        return []

def display_download_section(cached_results):
    """Display download section for cached results without causing interface resets."""
    results = cached_results['results']
    evaluation_time = cached_results['evaluation_time']
    evaluate_all_models = cached_results['evaluate_all_models']
    params = cached_results['params']
    
    st.subheader("📥 Descargar Resultados Cached")
    
    if evaluate_all_models:
        st.markdown("**Comparación Multi-Modelo**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                # Multi-model CSV comparison
                comparison_data = []
                for model_name_iter, res in results.items():
                    metrics_before = res['avg_before_metrics']
                    metrics_after = res['avg_after_metrics']
                    
                    all_metrics = set(metrics_before.keys()) | set(metrics_after.keys())
                    
                    for metric in all_metrics:
                        before_val = metrics_before.get(metric, 0)
                        after_val = metrics_after.get(metric, 0)
                        delta = after_val - before_val
                        improvement = (delta / before_val * 100) if before_val > 0 else 0
                        
                        comparison_data.append({
                            'Model': model_name_iter,
                            'Metric': metric,
                            'Before_Reranking': before_val,
                            'After_Reranking': after_val,
                            'Delta': delta,
                            'Improvement_Percent': improvement
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"models_comparison_{params['num_questions']}q_{timestamp}.csv"
                csv = comparison_df.to_csv(index=False, encoding='utf-8')
                
                st.download_button(
                    label="📈 CSV Multi-Modelo",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    help="Comparación entre modelos",
                    key=f"multi_csv_{timestamp}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col2:
            try:
                # Multi-model JSON report
                multi_model_report = {
                    'evaluation_info': {
                        'models_evaluated': list(results.keys()),
                        'generative_model': params['generative_model_name'],
                        'num_questions': params['num_questions'],
                        'evaluation_time': evaluation_time,
                        'use_llm_reranker': params['use_llm_reranker'],
                        'top_k': params['top_k'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    'results_by_model': results
                }
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"multi_model_report_{params['num_questions']}q_{timestamp}.json"
                json_data = json.dumps(multi_model_report, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📋 JSON Multi-Modelo",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    help="Reporte completo JSON",
                    key=f"multi_json_{timestamp}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col3:
            try:
                # Multi-model PDF report
                pdf_data = generate_multi_model_pdf_report(
                    results=results,
                    use_llm_reranker=params['use_llm_reranker'],
                    evaluation_time=evaluation_time,
                    generative_model_name=params['generative_model_name'],
                    num_questions=params['num_questions'],
                    top_k=params['top_k']
                )
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"multi_model_comparison_{params['num_questions']}q_{timestamp}.pdf"
                
                st.download_button(
                    label="📄 PDF Multi-Modelo",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    help="Reporte PDF comparativo",
                    key=f"multi_pdf_{timestamp}"
                )
            except Exception as e:
                st.warning(f"PDF no disponible: {e}")
    
    else:
        st.markdown("**Modelo Individual**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                # Detailed CSV
                results_df = pd.DataFrame(results['all_questions_data'])
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detailed_{params['model_name']}_{params['num_questions']}q_{timestamp}.csv"
                csv = results_df.to_csv(index=False, encoding='utf-8')
                
                st.download_button(
                    label="📄 CSV Detallado",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    help="Resultados por pregunta",
                    key=f"single_detailed_{timestamp}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col2:
            try:
                # Average metrics CSV
                all_metrics = set()
                all_metrics.update(results['avg_before_metrics'].keys())
                all_metrics.update(results['avg_after_metrics'].keys())
                
                avg_metrics_data = {
                    'Metric': [], 'Before_Reranking': [], 'After_Reranking': [], 
                    'Delta': [], 'Improvement_Percent': []
                }

                for metric in sorted(all_metrics):
                    before_val = results['avg_before_metrics'].get(metric, 0)
                    after_val = results['avg_after_metrics'].get(metric, 0)
                    delta = after_val - before_val
                    improvement = (delta / before_val * 100) if before_val > 0 else 0
                    
                    avg_metrics_data['Metric'].append(metric)
                    avg_metrics_data['Before_Reranking'].append(before_val)
                    avg_metrics_data['After_Reranking'].append(after_val)
                    avg_metrics_data['Delta'].append(delta)
                    avg_metrics_data['Improvement_Percent'].append(improvement)

                avg_df = pd.DataFrame(avg_metrics_data)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"avg_{params['model_name']}_{params['num_questions']}q_{timestamp}.csv"
                csv = avg_df.to_csv(index=False, encoding='utf-8')
                
                st.download_button(
                    label="📈 CSV Promedio",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    help="Métricas promedio",
                    key=f"single_avg_{timestamp}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col3:
            try:
                # JSON report
                report_data = {
                    'evaluation_info': {
                        'model_name': params['model_name'],
                        'generative_model': params['generative_model_name'],
                        'num_questions': params['num_questions'],
                        'evaluation_time': evaluation_time,
                        'use_llm_reranker': params['use_llm_reranker'],
                        'top_k': params['top_k'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    'memory_stats': results.get('memory_stats', {}),
                    'avg_before_metrics': results['avg_before_metrics'],
                    'avg_after_metrics': results['avg_after_metrics'],
                    'questions_data': results['all_questions_data']
                }
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"report_{params['model_name']}_{params['num_questions']}q_{timestamp}.json"
                json_data = json.dumps(report_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📋 Reporte JSON",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    help="Reporte completo JSON",
                    key=f"single_json_{timestamp}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col4:
            try:
                # PDF report
                pdf_data = generate_pdf_report(
                    results=results,
                    model_name=params['model_name'],
                    use_llm_reranker=params['use_llm_reranker'],
                    evaluation_time=evaluation_time,
                    generative_model_name=params['generative_model_name'],
                    num_questions=params['num_questions'],
                    top_k=params['top_k']
                )
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"report_{params['model_name']}_{params['num_questions']}q_{timestamp}.pdf"
                
                st.download_button(
                    label="📄 Reporte PDF",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    help="Reporte PDF completo",
                    key=f"single_pdf_{timestamp}"
                )
            except Exception as e:
                st.warning(f"PDF no disponible: {e}")


def show_cumulative_metrics_page():
    """Muestra la página de métricas acumulativas."""
    st.title("📈 Métricas Acumulativas")
    st.markdown("""
    Esta página evalúa múltiples preguntas y calcula **métricas promedio** para obtener una visión general 
    del rendimiento del sistema. Utiliza el dataset completo de **3035 preguntas** que contienen enlaces 
    de Microsoft Learn en sus respuestas aceptadas.
    """)
    
    # Initialize session state to persist results and prevent resets
    if 'cumulative_results' not in st.session_state:
        st.session_state.cumulative_results = None
    if 'evaluation_params' not in st.session_state:
        st.session_state.evaluation_params = None
    
    # Cargar dataset completo (train + val)
    val_questions = load_questions_from_json('data/val_set.json')
    train_questions = load_questions_from_json('data/train_set.json')
    
    # Combinar ambos datasets
    questions_and_answers = []
    if val_questions:
        questions_and_answers.extend(val_questions)
    if train_questions:
        questions_and_answers.extend(train_questions)
    
    if not questions_and_answers:
        st.error("❌ No se pudieron cargar los datasets")
        st.stop()
    
    # Filtrar preguntas con links
    filtered_questions = filter_questions_with_links(questions_and_answers)
    
    # Mostrar estadísticas del dataset
    st.info(f"📊 **Dataset completo**: {len(questions_and_answers)} preguntas total (Validación: {len(val_questions)}, Entrenamiento: {len(train_questions)})")
    st.info(f"🔗 **Preguntas con enlaces MS Learn**: {len(filtered_questions)} preguntas disponibles para evaluación")
    
    if len(filtered_questions) == 0:
        st.error("❌ No hay preguntas con enlaces de Microsoft Learn para evaluar")
        st.stop()
    
    # Configuración
    st.subheader("⚙️ Configuración de Evaluación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Número de preguntas
        num_questions = st.slider(
            "Número de preguntas a evaluar",
            min_value=min(5, len(filtered_questions)),
            max_value=len(filtered_questions),
            value=min(100, len(filtered_questions)),
            help="Cantidad de preguntas para evaluar (seleccionadas aleatoriamente)"
        )
        
        # Botones de valores sugeridos
        st.markdown("**Valores sugeridos:**")
        col_btns = st.columns(5)
        with col_btns[0]:
            if st.button("50"):
                num_questions = 50
        with col_btns[1]:
            if st.button("100"):
                num_questions = 100
        with col_btns[2]:
            if st.button("500"):
                num_questions = 500
        with col_btns[3]:
            if st.button("1000"):
                num_questions = 1000
        with col_btns[4]:
            if st.button("Todas"):
                num_questions = len(filtered_questions)

        # Modelo de embedding
        model_name = st.selectbox(
            "Modelo de Embedding",
            options=list(EMBEDDING_MODELS.keys()),
            index=0,
            help="Modelo para generar embeddings de documentos"
        )
        evaluate_all_models = st.checkbox(
            "Evaluar los 3 modelos",
            value=False,
            help="Ejecuta la evaluación para MiniLM, ada y Mpnet en una sola corrida"
        )
    
    with col2:
        # Configuración de retrieval
        top_k = st.slider(
            "Top-K documentos",
            min_value=5,
            max_value=20,
            value=10,
            help="Número de documentos a recuperar"
        )
        
        # LLM Reranking
        use_llm_reranker = st.checkbox(
            "Usar LLM Reranking",
            value=True,
            help="Usar GPT-4 para reordenar documentos (necesario para métricas antes/después)"
        )
    
    # Modelo generativo (solo para reranking)
    generative_model_name = st.selectbox(
        "Modelo Generativo (para reranking)",
        options=list(GENERATIVE_MODELS.keys()),
        index=0,
        help="Modelo usado para el reranking LLM"
    )
    
    # Configuración de memoria y rendimiento
    with st.expander("⚙️ Configuración Avanzada", expanded=False):
        batch_size = st.slider(
            "Tamaño del lote",
            min_value=10,
            max_value=100,
            value=50,
            help="Número de preguntas a procesar por lote (menor = menos memoria, mayor = más rápido)"
        )
    
    # Mostrar estimación de tiempo
    if num_questions > 100:
        estimated_time = num_questions * 3  # ~3 segundos por pregunta con reranking
        if evaluate_all_models:
            estimated_time *= 3  # 3 modelos
        
        if estimated_time > 300:  # Más de 5 minutos
            st.warning(f"⏱️ **Tiempo estimado**: {estimated_time//60} minutos aprox. para {num_questions} preguntas")
            st.info("💡 **Recomendación**: Usa lotes grandes (50-100) para optimizar memoria y velocidad")
    
    # Display previous results if available
    if st.session_state.cumulative_results is not None:
        st.info("📊 **Resultados previos disponibles**. Puedes descargar los reportes sin necesidad de re-ejecutar la evaluación.")
        
        # Add button to clear previous results
        if st.button("🗑️ Limpiar Resultados Previos"):
            st.session_state.cumulative_results = None
            st.rerun()
        
        st.markdown("---")
        
        # Display download section for previous results
        display_download_section(st.session_state.cumulative_results)
        
        st.markdown("---")
    
    # Botón para ejecutar evaluación
    if st.button("🚀 Ejecutar Evaluación", type="primary"):
        if len(filtered_questions) < num_questions:
            st.error(f"❌ Solo hay {len(filtered_questions)} preguntas con links disponibles")
            st.stop()
        
        with st.spinner(f"🔍 Evaluando {num_questions} preguntas en lotes de {batch_size}..."):
            start_time = time.time()

            if evaluate_all_models:
                model_list = list(EMBEDDING_MODELS.keys())
                results = run_cumulative_metrics_for_models(
                    questions_and_answers=filtered_questions,
                    num_questions=num_questions,
                    model_names=model_list,
                    generative_model_name=generative_model_name,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    batch_size=batch_size
                )
            else:
                results = run_cumulative_metrics_evaluation(
                    questions_and_answers=filtered_questions,
                    num_questions=num_questions,
                    model_name=model_name,
                    generative_model_name=generative_model_name,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    batch_size=batch_size
                )

            evaluation_time = time.time() - start_time
            
            # Store results in session state to prevent loss on downloads
            st.session_state.cumulative_results = {
                'results': results,
                'evaluation_time': evaluation_time,
                'evaluate_all_models': evaluate_all_models,
                'params': {
                    'model_name': model_name,
                    'generative_model_name': generative_model_name,
                    'num_questions': num_questions,
                    'top_k': top_k,
                    'use_llm_reranker': use_llm_reranker
                }
            }

            if evaluate_all_models:
                # Mostrar métricas individuales en acordeones
                st.subheader("📊 Métricas Individuales por Modelo")
                for m_name, res in results.items():
                    with st.expander(f"📈 {m_name} - Resultados Detallados", expanded=False):
                        display_cumulative_metrics(res, m_name, use_llm_reranker)
                
                # Mostrar comparación completa
                st.markdown("---")
                display_models_comparison(results, use_llm_reranker)
                
                # Agregar descarga para múltiples modelos
                st.markdown("---")
                st.subheader("📥 Descargar Comparación de Modelos")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    try:
                        # Crear comparación de métricas entre modelos
                        comparison_data = []
                        for model_name_iter, res in results.items():
                            metrics_before = res['avg_before_metrics']
                            metrics_after = res['avg_after_metrics']
                            
                            # Obtener todas las métricas únicas
                            all_metrics = set(metrics_before.keys()) | set(metrics_after.keys())
                            
                            for metric in all_metrics:
                                before_val = metrics_before.get(metric, 0)
                                after_val = metrics_after.get(metric, 0)
                                delta = after_val - before_val
                                improvement = (delta / before_val * 100) if before_val > 0 else 0
                                
                                comparison_data.append({
                                    'Model': model_name_iter,
                                    'Metric': metric,
                                    'Before_Reranking': before_val,
                                    'After_Reranking': after_val,
                                    'Delta': delta,
                                    'Improvement_Percent': improvement
                                })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"models_comparison_{num_questions}q_{timestamp}.csv"
                        
                        csv = comparison_df.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📈 Descargar Comparación CSV",
                            data=csv,
                            file_name=filename,
                            mime="text/csv",
                            help="Descarga comparación de métricas entre todos los modelos",
                            key=f"live_multi_csv_{timestamp}"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error preparando comparación: {e}")
                
                with col2:
                    try:
                        # Crear reporte completo multi-modelo
                        multi_model_report = {
                            'evaluation_info': {
                                'models_evaluated': list(results.keys()),
                                'generative_model': generative_model_name,
                                'num_questions': num_questions,
                                'evaluation_time': evaluation_time,
                                'use_llm_reranker': use_llm_reranker,
                                'top_k': top_k,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            },
                            'results_by_model': results
                        }
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"multi_model_report_{num_questions}q_{timestamp}.json"
                        
                        import json
                        json_data = json.dumps(multi_model_report, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📋 Descargar Reporte JSON",
                            data=json_data,
                            file_name=filename,
                            mime="application/json",
                            help="Descarga reporte completo de todos los modelos",
                            key=f"live_multi_json_{timestamp}"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error preparando reporte multi-modelo: {e}")
                
                with col3:
                    try:
                        # Generar reporte PDF multi-modelo
                        pdf_data = generate_multi_model_pdf_report(
                            results=results,
                            use_llm_reranker=use_llm_reranker,
                            evaluation_time=evaluation_time,
                            generative_model_name=generative_model_name,
                            num_questions=num_questions,
                            top_k=top_k
                        )
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"multi_model_comparison_{num_questions}q_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📄 Descargar Reporte PDF",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf",
                            help="Descarga reporte comparativo en formato PDF con gráficos",
                            key=f"live_multi_pdf_{timestamp}"
                        )
                        
                    except ImportError as e:
                        st.warning(f"⚠️ Reporte PDF no disponible: Faltan dependencias de matplotlib")
                        st.info("💡 El reporte PDF requiere matplotlib. Usa los reportes CSV y JSON como alternativa.")
                    except Exception as e:
                        st.error(f"❌ Error generando reporte PDF multi-modelo: {e}")
                        st.info("💡 Usa los reportes CSV y JSON como alternativa.")
            else:
                display_cumulative_metrics(results, model_name, use_llm_reranker)
            
            if not evaluate_all_models:
                # Estadísticas adicionales
                st.markdown("---")
                st.subheader("📊 Estadísticas de Evaluación")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "📝 Preguntas Evaluadas",
                        f"{results['num_questions_evaluated']}",
                        help="Número total de preguntas procesadas"
                    )

                with col2:
                    if results['all_questions_data']:
                        avg_gt_links = np.mean([q['ground_truth_links'] for q in results['all_questions_data']])
                        st.metric(
                            "🔗 Links GT Promedio",
                            f"{avg_gt_links:.1f}",
                            help="Número promedio de links de ground truth por pregunta"
                        )

                with col3:
                    st.metric(
                        "⏱️ Tiempo Total",
                        f"{evaluation_time:.1f}s",
                        help="Tiempo total de evaluación"
                    )

                # Opción para descargar resultados
                st.markdown("---")
                st.subheader("📥 Descargar Resultados")
                
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    try:
                        # Validar integridad de datos
                        validation = validate_data_integrity(results)
                        
                        if validation['is_valid']:
                            results_df = pd.DataFrame(results['all_questions_data'])
                            
                            # Agregar timestamp y metadatos
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"cumulative_metrics_detailed_{model_name}_{num_questions}q_{timestamp}.csv"
                            
                            csv = results_df.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="📄 Descargar CSV Detallado",
                                data=csv,
                                file_name=filename,
                                mime="text/csv",
                                help="Descarga resultados detallados por pregunta",
                                key=f"live_single_detailed_{timestamp}"
                            )
                            
                            # Mostrar advertencias si existen
                            if validation['warnings']:
                                with st.expander("⚠️ Advertencias de validación"):
                                    for warning in validation['warnings'][:5]:
                                        st.warning(f"• {warning}")
                                    
                            # Mostrar estadísticas de validación
                            if validation['stats']:
                                with st.expander("📊 Estadísticas de Validación"):
                                    st.json(validation['stats'])
                        else:
                            st.error("❌ Error de integridad de datos - no se puede descargar")
                            for error in validation['errors']:
                                st.error(f"• {error}")
                                
                    except Exception as e:
                        st.error(f"❌ Error preparando descarga detallada: {e}")

                with col2:
                    try:
                        # Incluir todas las métricas disponibles, no solo las principales
                        all_metrics = set()
                        all_metrics.update(results['avg_before_metrics'].keys())
                        all_metrics.update(results['avg_after_metrics'].keys())
                        
                        avg_metrics_data = {
                            'Metric': [],
                            'Before_Reranking': [],
                            'After_Reranking': [],
                            'Delta': [],
                            'Improvement_Percent': []
                        }

                        for metric in sorted(all_metrics):
                            before_val = results['avg_before_metrics'].get(metric, 0)
                            after_val = results['avg_after_metrics'].get(metric, 0)
                            delta = after_val - before_val
                            improvement = (delta / before_val * 100) if before_val > 0 else 0
                            
                            avg_metrics_data['Metric'].append(metric)
                            avg_metrics_data['Before_Reranking'].append(before_val)
                            avg_metrics_data['After_Reranking'].append(after_val)
                            avg_metrics_data['Delta'].append(delta)
                            avg_metrics_data['Improvement_Percent'].append(improvement)

                        avg_df = pd.DataFrame(avg_metrics_data)
                        
                        # Agregar timestamp y metadatos
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"cumulative_metrics_avg_{model_name}_{num_questions}q_{timestamp}.csv"
                        
                        csv = avg_df.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📈 Descargar CSV Promedio",
                            data=csv,
                            file_name=filename,
                            mime="text/csv",
                            help="Descarga métricas promedio con comparación antes/después",
                            key=f"live_single_avg_{timestamp}"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error preparando descarga promedio: {e}")
                
                with col3:
                    try:
                        # Crear un reporte completo con metadatos
                        report_data = {
                            'evaluation_info': {
                                'model_name': model_name,
                                'generative_model': generative_model_name,
                                'num_questions': num_questions,
                                'evaluation_time': evaluation_time,
                                'use_llm_reranker': use_llm_reranker,
                                'top_k': top_k,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            },
                            'memory_stats': results.get('memory_stats', {}),
                            'avg_before_metrics': results['avg_before_metrics'],
                            'avg_after_metrics': results['avg_after_metrics'],
                            'questions_data': results['all_questions_data']
                        }
                        
                        # Convertir a JSON para descarga
                        import json
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"cumulative_metrics_report_{model_name}_{num_questions}q_{timestamp}.json"
                        
                        json_data = json.dumps(report_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📋 Descargar Reporte JSON",
                            data=json_data,
                            file_name=filename,
                            mime="application/json",
                            help="Descarga reporte completo en formato JSON",
                            key=f"live_single_json_{timestamp}"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error preparando reporte completo: {e}")
                
                with col4:
                    try:
                        # Generar reporte PDF
                        pdf_data = generate_pdf_report(
                            results=results,
                            model_name=model_name,
                            use_llm_reranker=use_llm_reranker,
                            evaluation_time=evaluation_time,
                            generative_model_name=generative_model_name,
                            num_questions=num_questions,
                            top_k=top_k
                        )
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"cumulative_metrics_report_{model_name}_{num_questions}q_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📄 Descargar Reporte PDF",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf",
                            help="Descarga reporte completo en formato PDF con gráficos",
                            key=f"live_single_pdf_{timestamp}"
                        )
                        
                    except ImportError as e:
                        st.warning(f"⚠️ Reporte PDF no disponible: Faltan dependencias de matplotlib")
                        st.info("💡 El reporte PDF requiere matplotlib. Usa los reportes CSV y JSON como alternativa.")
                    except Exception as e:
                        st.error(f"❌ Error generando reporte PDF: {e}")
                        st.info("💡 Usa los reportes CSV y JSON como alternativa.")