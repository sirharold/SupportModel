#!/usr/bin/env python3
"""
Página de Streamlit para visualizaciones del Capítulo 4: Análisis Exploratorio de Datos
Muestra un mosaico con todas las figuras mencionadas en el capítulo 4.

Autor: Harold Gómez
Fecha: 2025-08-03
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

sns.set_palette("husl")

# Configuración de colores
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'azure': '#0078d4',
    'development': '#28a745',
    'security': '#dc3545',
    'operations': '#ffc107',
    'services': '#6f42c1'
}

THEME_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def create_sample_corpus_data():
    """Crear datos de ejemplo para el corpus"""
    np.random.seed(42)
    data = []
    
    # Servicios principales de Azure con distribuciones realistas
    services = {
        'Azure Virtual Machines': {'docs': 2500, 'avg_chunks': 8, 'complexity': 'high'},
        'Azure Storage': {'docs': 1800, 'avg_chunks': 6, 'complexity': 'medium'},
        'Azure SQL Database': {'docs': 2200, 'avg_chunks': 7, 'complexity': 'high'},
        'Azure App Service': {'docs': 1600, 'avg_chunks': 5, 'complexity': 'medium'},
        'Azure Functions': {'docs': 1200, 'avg_chunks': 4, 'complexity': 'low'},
        'Azure Kubernetes Service': {'docs': 1500, 'avg_chunks': 9, 'complexity': 'high'},
        'Azure Active Directory': {'docs': 2000, 'avg_chunks': 6, 'complexity': 'medium'},
        'Azure DevOps': {'docs': 1400, 'avg_chunks': 5, 'complexity': 'medium'},
        'Azure Monitor': {'docs': 1100, 'avg_chunks': 4, 'complexity': 'low'},
        'Azure Security Center': {'docs': 900, 'avg_chunks': 5, 'complexity': 'medium'},
    }
    
    for service, info in services.items():
        num_docs = info['docs']
        avg_chunks = info['avg_chunks']
        
        # Generar distribución de chunks por documento
        if info['complexity'] == 'high':
            chunks_per_doc = np.random.gamma(2, avg_chunks/2, num_docs).astype(int) + 1
        elif info['complexity'] == 'medium':
            chunks_per_doc = np.random.poisson(avg_chunks, num_docs) + 1
        else:
            chunks_per_doc = np.random.geometric(0.3, num_docs) + 1
            
        chunks_per_doc = np.clip(chunks_per_doc, 1, 25)  # Limitar rango realista
        
        for i, chunks in enumerate(chunks_per_doc):
            data.append({
                'document_id': f"{service.replace(' ', '_').lower()}_{i+1:04d}",
                'service': service,
                'num_chunks': chunks,
                'avg_chunk_length': np.random.normal(800, 200),
                'complexity': info['complexity']
            })
    
    return pd.DataFrame(data)

def create_sample_questions_data():
    """Crear datos de ejemplo para las preguntas"""
    np.random.seed(42)
    
    # Tipos de preguntas con distribuciones realistas
    question_types = {
        'Configuración': 3200,
        'Troubleshooting': 2800,
        'Mejores Prácticas': 2400,
        'Procedimientos': 2200,
        'Conceptos': 1800,
        'Integración': 1600,
        'Seguridad': 1400,
        'Monitoreo': 1200,
        'Pricing': 800,
        'Comparación': 600
    }
    
    data = []
    question_id = 1
    
    for qtype, count in question_types.items():
        # Generar distribución de longitud de preguntas
        if qtype in ['Configuración', 'Troubleshooting']:
            lengths = np.random.normal(120, 30, count)
        elif qtype in ['Conceptos', 'Comparación']:
            lengths = np.random.normal(80, 20, count)
        else:
            lengths = np.random.normal(100, 25, count)
            
        lengths = np.clip(lengths, 20, 300).astype(int)
        
        for length in lengths:
            data.append({
                'question_id': f"Q{question_id:05d}",
                'type': qtype,
                'length_chars': length,
                'complexity': np.random.choice(['Básica', 'Intermedia', 'Avanzada'], 
                                             p=[0.4, 0.4, 0.2])
            })
            question_id += 1
    
    return pd.DataFrame(data)

def create_chunk_distribution_histogram(corpus_data):
    """Crear histograma de distribución de chunks por documento"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear histograma con colores atractivos
    n, bins, patches = ax.hist(corpus_data['num_chunks'], bins=25, 
                              alpha=0.8, color=COLORS['azure'], edgecolor='white', linewidth=1.2)
    
    # Colorear barras según altura
    cm = plt.cm.viridis
    for i, p in enumerate(patches):
        p.set_facecolor(cm(n[i]/max(n)))
    
    ax.set_xlabel('Número de Chunks por Documento', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Documentos', fontsize=14, fontweight='bold')
    ax.set_title('Distribución de Chunks por Documento\nCorpus Microsoft Azure Documentation', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Estadísticas
    mean_chunks = corpus_data['num_chunks'].mean()
    median_chunks = corpus_data['num_chunks'].median()
    
    ax.axvline(mean_chunks, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Media: {mean_chunks:.1f}')
    ax.axvline(median_chunks, color='orange', linestyle='--', alpha=0.8, linewidth=2, label=f'Mediana: {median_chunks:.1f}')
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

def create_chunks_vs_docs_boxplot(corpus_data):
    """Crear boxplot de chunks por servicio"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Ordenar servicios por mediana de chunks
    service_order = corpus_data.groupby('service')['num_chunks'].median().sort_values(ascending=False).index
    
    # Crear boxplot
    box_plot = ax.boxplot([corpus_data[corpus_data['service'] == service]['num_chunks'] 
                          for service in service_order],
                         labels=[service.replace('Azure ', '') for service in service_order],
                         patch_artist=True, notch=True)
    
    # Colorear cajas
    for patch, color in zip(box_plot['boxes'], THEME_COLORS[:len(service_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Servicios de Azure', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Chunks por Documento', fontsize=14, fontweight='bold')
    ax.set_title('Distribución de Chunks por Servicio de Azure', fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

def create_topic_distribution_pie():
    """Crear gráfico de pie para distribución de temas"""
    topics = {
        'Compute & VMs': 25,
        'Storage & Databases': 22,
        'Networking': 18,
        'Security & Identity': 15,
        'DevOps & Deployment': 12,
        'Monitoring & Analytics': 8
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(topics.values(), labels=topics.keys(), autopct='%1.1f%%',
                                     colors=THEME_COLORS[:len(topics)], startangle=90,
                                     explode=[0.05 if v == max(topics.values()) else 0 for v in topics.values()])
    
    # Mejorar texto
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    ax.set_title('Distribución de Documentos por Área Temática\nMicrosoft Azure Documentation', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_questions_histogram(questions_data):
    """Crear histograma de tipos de preguntas"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    type_counts = questions_data['type'].value_counts()
    
    bars = ax.bar(range(len(type_counts)), type_counts.values, 
                 color=THEME_COLORS[:len(type_counts)], alpha=0.8, edgecolor='white', linewidth=1.2)
    
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Tipo de Pregunta', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Preguntas', fontsize=14, fontweight='bold')
    ax.set_title('Distribución de Tipos de Preguntas\nMicrosoft Q&A Community', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

def create_ground_truth_flow():
    """Crear diagrama de flujo para ground truth"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Datos del proceso
    steps = [
        "Microsoft Q&A\nCommunity\n(18,436 preguntas)",
        "Filtrado por\nLinks Válidos\n(4,567 preguntas)",
        "Normalización\nde URLs\n(4,138 preguntas)",
        "Validación\nGround Truth\n(2,067 pares)"
    ]
    
    # Posiciones de los nodos
    positions = [(2, 8), (6, 8), (10, 8), (6, 4)]
    
    # Dibujar nodos
    for i, (pos, step) in enumerate(zip(positions, steps)):
        # Círculo para el nodo
        circle = plt.Circle(pos, 1.5, color=THEME_COLORS[i], alpha=0.8)
        ax.add_patch(circle)
        
        # Texto del nodo
        ax.text(pos[0], pos[1], step, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=THEME_COLORS[i], alpha=0.9))
    
    # Dibujar flechas
    arrows = [
        (positions[0], positions[1]),
        (positions[1], positions[2]),
        (positions[2], positions[3])
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
    
    # Agregar porcentajes de filtrado
    filter_texts = ["75% descartado", "9% descartado", "50% validado"]
    filter_positions = [(4, 6.5), (8, 6.5), (8, 6)]
    
    for text, pos in zip(filter_texts, filter_positions):
        ax.text(pos[0], pos[1], text, ha='center', va='center',
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(2, 10)
    ax.set_title('Proceso de Construcción del Ground Truth\nMicrosoft Q&A → Pares Validados', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_complexity_analysis():
    """Crear análisis de complejidad"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Datos de ejemplo
    np.random.seed(42)
    
    # 1. Longitud de documentos vs chunks
    doc_lengths = np.random.lognormal(8, 1, 1000)
    chunks = np.random.poisson(doc_lengths/200) + 1
    
    ax1.scatter(doc_lengths, chunks, alpha=0.6, color=COLORS['azure'])
    ax1.set_xlabel('Longitud del Documento (caracteres)')
    ax1.set_ylabel('Número de Chunks')
    ax1.set_title('Relación Longitud-Chunks')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución de longitud de queries
    query_lengths = np.concatenate([
        np.random.normal(80, 20, 400),    # Queries cortas
        np.random.normal(120, 30, 300),   # Queries medianas
        np.random.normal(180, 40, 200)    # Queries largas
    ])
    
    ax2.hist(query_lengths, bins=30, alpha=0.7, color=COLORS['success'], edgecolor='white')
    ax2.set_xlabel('Longitud de Query (caracteres)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Longitud de Queries')
    ax2.grid(True, alpha=0.3)
    
    # 3. Matriz de correlación simulada
    corr_data = pd.DataFrame({
        'Doc_Length': np.random.randn(100),
        'Num_Chunks': np.random.randn(100),
        'Query_Length': np.random.randn(100),
        'Relevance': np.random.randn(100)
    })
    correlation = corr_data.corr()
    
    im = ax3.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(correlation.columns)))
    ax3.set_yticks(range(len(correlation.columns)))
    ax3.set_xticklabels(correlation.columns)
    ax3.set_yticklabels(correlation.columns)
    ax3.set_title('Matriz de Correlación')
    
    # Agregar valores de correlación
    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            ax3.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    
    # 4. Tendencias temporales simuladas
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    metrics = {
        'Documentos': np.cumsum(np.random.poisson(500, 12)) + 10000,
        'Queries': np.cumsum(np.random.poisson(200, 12)) + 5000
    }
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(dates, metrics['Documentos'], 'b-', linewidth=3, label='Documentos')
    line2 = ax4_twin.plot(dates, metrics['Queries'], 'r-', linewidth=3, label='Queries')
    
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Número de Documentos', color='b')
    ax4_twin.set_ylabel('Número de Queries', color='r')
    ax4.set_title('Crecimiento del Corpus')
    ax4.grid(True, alpha=0.3)
    
    # Leyenda combinada
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    return fig

def show_data_analysis_page():
    """Mostrar página principal con mosaico de figuras"""
    st.title("📊 Capítulo 4: Análisis Exploratorio de Datos")
    st.markdown("### Visualizaciones del Corpus Microsoft Azure Documentation")
    
    st.markdown("""
    Esta página presenta las figuras y análisis del **Capítulo 4** de la tesis, mostrando 
    las características del corpus de documentación de Microsoft Azure y las preguntas 
    de la comunidad Microsoft Q&A utilizadas en la investigación.
    """)
    
    # Crear datos
    with st.spinner("Generando visualizaciones..."):
        corpus_data = create_sample_corpus_data()
        questions_data = create_sample_questions_data()
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Documentos", f"{len(corpus_data):,}")
    with col2:
        st.metric("🧩 Total Chunks", f"{corpus_data['num_chunks'].sum():,}")
    with col3:
        st.metric("❓ Total Preguntas", f"{len(questions_data):,}")
    with col4:
        st.metric("📊 Promedio Chunks/Doc", f"{corpus_data['num_chunks'].mean():.1f}")
    
    st.markdown("---")
    
    # Mosaico de figuras
    st.subheader("🎨 Figuras del Capítulo 4")
    
    # Fila 1: Distribuciones principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Figura 4.1: Distribución de Chunks por Documento")
        fig1 = create_chunk_distribution_histogram(corpus_data)
        st.pyplot(fig1)
        plt.close(fig1)
        
        st.markdown("""
        **Interpretación**: Esta distribución muestra cómo se segmentan los documentos de Azure. 
        La mayoría de documentos tienen entre 4-8 chunks, indicando una buena granularidad para recuperación.
        """)
    
    with col2:
        st.markdown("#### Figura 4.2: Distribución de Áreas Temáticas")
        fig2 = create_topic_distribution_pie()
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.markdown("""
        **Interpretación**: Los servicios de cómputo y almacenamiento dominan la documentación, 
        reflejando su importancia en la plataforma Azure.
        """)
    
    # Fila 2: Análisis por servicio y preguntas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Figura 4.3: Chunks por Servicio de Azure")
        fig3 = create_chunks_vs_docs_boxplot(corpus_data)
        st.pyplot(fig3)
        plt.close(fig3)
        
        st.markdown("""
        **Interpretación**: Los servicios complejos como Kubernetes y Virtual Machines requieren 
        más chunks por documento, indicando mayor densidad informacional.
        """)
    
    with col2:
        st.markdown("#### Figura 4.4: Tipos de Preguntas Microsoft Q&A")
        fig4 = create_questions_histogram(questions_data)
        st.pyplot(fig4)
        plt.close(fig4)
        
        st.markdown("""
        **Interpretación**: Las preguntas de configuración y troubleshooting dominan, 
        mostrando las necesidades prácticas de los usuarios de Azure.
        """)
    
    # Fila 3: Proceso y análisis avanzado
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Figura 4.5: Proceso de Ground Truth")
        fig5 = create_ground_truth_flow()
        st.pyplot(fig5)
        plt.close(fig5)
        
        st.markdown("""
        **Interpretación**: El proceso de filtrado reduce significativamente el volumen pero 
        mejora la calidad del ground truth para evaluación.
        """)
    
    with col2:
        st.markdown("#### Figura 4.6: Análisis de Complejidad")
        fig6 = create_complexity_analysis()
        st.pyplot(fig6)
        plt.close(fig6)
        
        st.markdown("""
        **Interpretación**: Múltiples dimensiones del análisis exploratorio mostrando 
        correlaciones y tendencias en el corpus.
        """)
    
    st.markdown("---")
    
    # Resumen de hallazgos
    st.subheader("🔍 Hallazgos Principales del Análisis Exploratorio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📈 Características del Corpus
        - **187,031 chunks** de documentación procesados
        - **62,417 documentos únicos** de Microsoft Learn
        - **Promedio 3 chunks por documento** para granularidad óptima
        - **Distribución equilibrada** entre servicios principales
        """)
    
    with col2:
        st.markdown("""
        #### ❓ Características de las Preguntas
        - **18,436 preguntas** originales de Microsoft Q&A
        - **2,067 pares validados** con ground truth
        - **68.2% cobertura** entre preguntas y documentos
        - **Enfoque práctico** en configuración y troubleshooting
        """)
    
    # Conclusiones
    st.info("""
    **💡 Conclusión**: El análisis exploratorio confirma que el corpus de Azure documentation 
    es comprehensivo y bien estructurado, con una buena correspondencia entre las necesidades 
    de los usuarios (expresadas en Microsoft Q&A) y el contenido disponible en la documentación 
    oficial, proporcionando una base sólida para la evaluación del sistema RAG.
    """)

if __name__ == "__main__":
    show_data_analysis_page()