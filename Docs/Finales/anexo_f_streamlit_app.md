# F. STREAMLIT APP

## F.1 Introducción

Este anexo documenta la aplicación web interactiva desarrollada con Streamlit para la exploración y visualización de los resultados experimentales del sistema RAG. La aplicación proporciona una interfaz intuitiva para analizar el rendimiento de los diferentes modelos de embedding, visualizar métricas comparativas, y explorar casos específicos de recuperación de documentos.

## F.2 Arquitectura de la Aplicación

### F.2.1 Estructura del Proyecto Streamlit

```
src/apps/                         # Aplicaciones Streamlit del proyecto
├── main_qa_app.py                # Aplicación principal de Q&A
├── cumulative_metrics_results_matplotlib.py  # Visualización de resultados
├── comparison_page.py            # Comparación de modelos
├── cumulative_comparison.py      # Comparación acumulativa
├── batch_queries_page.py         # Procesamiento de consultas en lote
├── data_analysis_page.py         # Análisis de datos experimentales
└── question_answer_comparison.py # Comparación de respuestas

src/ui/                           # Interfaces de usuario compartidas
├── display.py                    # Funciones de visualización
├── enhanced_metrics_display.py   # Visualización de métricas avanzadas
├── metrics_display.py           # Visualización básica de métricas
└── pdf_generator.py             # Generación de reportes PDF

.streamlit/                       # Configuración de Streamlit (directorio raíz)
└── config.toml                   # Configuración global de Streamlit
```

## F.3 Funcionalidades Principales

### F.3.1 Página Principal (Dashboard)

#### F.3.1.1 Resumen Ejecutivo

La página principal presenta un dashboard con las métricas clave del sistema:

```python
# Métricas principales mostradas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📊 Modelos Evaluados", 
        value="4",
        help="Ada, MPNet, MiniLM, E5-Large"
    )

with col2:
    st.metric(
        label="📋 Preguntas Evaluadas", 
        value="11",
        help="Por modelo, total 44 evaluaciones"
    )

with col3:
    st.metric(
        label="📚 Documentos Indexados", 
        value="187,031",
        help="Chunks de documentación Azure"
    )

with col4:
    st.metric(
        label="⏱️ Tiempo Total", 
        value="12.9 min",
        help="774.78 segundos de evaluación"
    )
```

#### F.3.1.2 Selector de Archivos de Resultados

La aplicación permite cargar diferentes archivos de resultados experimentales:

```python
# Selector de archivos de resultados
results_files = [
    "cumulative_results_1753578255.json",
    "cumulative_results_20250731_140825.json"
]

selected_file = st.selectbox(
    "📁 Seleccionar archivo de resultados:",
    results_files,
    help="Selecciona el archivo de resultados experimentales a analizar"
)
```

### F.3.2 Comparación de Modelos

#### F.3.2.1 Tabla Comparativa Interactiva

```python
def create_comparison_table():
    """Crea tabla comparativa de modelos con métricas clave"""
    
    comparison_data = {
        'Modelo': ['Ada', 'MPNet', 'MiniLM', 'E5-Large'],
        'Dimensiones': [1536, 768, 384, 1024],
        'Precision@5': [0.055, 0.055, 0.036, 0.000],
        'Recall@5': [0.273, 0.273, 0.182, 0.000],
        'NDCG@5': [0.162, 0.189, 0.103, 0.000],
        'BERTScore F1': [0.732, 0.739, 0.729, 0.739],
        'Faithfulness': [0.482, 0.518, 0.509, 0.591]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Aplicar formato condicional
    st.dataframe(
        df.style.background_gradient(subset=['Precision@5', 'Recall@5']),
        use_container_width=True
    )
```

#### F.3.2.2 Gráfico Radar Comparativo

```python
def create_radar_chart():
    """Crea gráfico radar para comparación multi-dimensional"""
    
    fig = go.Figure()
    
    metrics = ['Precision@5', 'Recall@5', 'NDCG@5', 'BERTScore F1', 'Faithfulness']
    
    for model_name, values in model_data.items():
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=model_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="📊 Comparación Multi-Dimensional de Modelos"
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

### F.3.3 Análisis de Métricas

#### F.3.3.1 Visualización de Impacto del Reranking

```python
def plot_reranking_impact():
    """Visualiza el impacto del reranking por modelo"""
    
    models = ['Ada', 'MPNet', 'MiniLM', 'E5-Large']
    pre_reranking = [0.126, 0.108, 0.091, 0.000]
    post_reranking = [0.162, 0.189, 0.103, 0.000]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, pre_reranking, width, 
                   label='Pre-Reranking', alpha=0.8)
    bars2 = ax.bar(x + width/2, post_reranking, width,
                   label='Post-Reranking', alpha=0.8)
    
    # Añadir etiquetas de mejora porcentual
    for i, (pre, post) in enumerate(zip(pre_reranking, post_reranking)):
        if pre > 0:
            improvement = ((post - pre) / pre) * 100
            ax.text(i, post + 0.01, f'+{improvement:.1f}%', 
                   ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Modelos')
    ax.set_ylabel('NDCG@5')
    ax.set_title('🎯 Impacto del CrossEncoder Reranking')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    st.pyplot(fig)
```

#### F.3.3.2 Análisis Estadístico (Wilcoxon)

```python
def display_statistical_analysis():
    """Muestra resultados de tests estadísticos"""
    
    st.subheader("📊 Análisis de Significancia Estadística")
    
    # Cargar resultados de Wilcoxon
    wilcoxon_data = pd.read_csv('wilcoxon_test_results.csv')
    
    # Filtrar por métrica seleccionada
    metric = st.selectbox(
        "Seleccionar métrica:",
        ['precision@5', 'recall@5', 'f1@5', 'ndcg@5']
    )
    
    filtered_data = wilcoxon_data[wilcoxon_data['metric'] == metric]
    
    # Crear heatmap de p-valores
    pivot_table = filtered_data.pivot_table(
        values='p_value', 
        index='model1', 
        columns='model2'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', 
                center=0.05, ax=ax)
    ax.set_title(f'P-valores Test de Wilcoxon - {metric.upper()}')
    
    st.pyplot(fig)
    
    # Interpretación
    significant_pairs = filtered_data[filtered_data['significant'] == True]
    
    if len(significant_pairs) == 0:
        st.warning("⚠️ No se encontraron diferencias estadísticamente significativas (p > 0.05)")
    else:
        st.success(f"✅ {len(significant_pairs)} comparaciones estadísticamente significativas")
```

### F.3.4 Explorador de Consultas

#### F.3.4.1 Búsqueda Interactiva

```python
def create_query_explorer():
    """Interfaz para explorar consultas específicas"""
    
    st.subheader("🔍 Explorador de Consultas")
    
    # Selector de consulta
    query_options = load_query_list()
    selected_query = st.selectbox(
        "Seleccionar consulta:",
        query_options,
        help="Elige una consulta para ver resultados detallados"
    )
    
    # Selector de modelo
    model_options = ['Ada', 'MPNet', 'MiniLM', 'E5-Large']
    selected_model = st.selectbox(
        "Seleccionar modelo:",
        model_options
    )
    
    # Mostrar resultados
    if st.button("🔍 Buscar"):
        results = get_query_results(selected_query, selected_model)
        display_query_results(results)

def display_query_results(results):
    """Muestra resultados detallados de una consulta"""
    
    st.write(f"**Consulta:** {results['query']}")
    st.write(f"**Modelo:** {results['model']}")
    
    # Métricas de la consulta
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precision@5", f"{results['precision_5']:.3f}")
    with col2:
        st.metric("NDCG@5", f"{results['ndcg_5']:.3f}")
    with col3:
        st.metric("MRR", f"{results['mrr']:.3f}")
    
    # Top 10 documentos recuperados
    st.subheader("📋 Top 10 Documentos Recuperados")
    
    for i, doc in enumerate(results['top_documents'][:10], 1):
        with st.expander(f"#{i} - Score: {doc['score']:.3f}"):
            st.write(f"**Título:** {doc['title']}")
            st.write(f"**URL:** {doc['url']}")
            st.write(f"**Snippet:** {doc['content'][:200]}...")
            
            # Indicador de relevancia
            if doc['is_relevant']:
                st.success("✅ Documento relevante según ground truth")
            else:
                st.info("ℹ️ Documento no marcado como relevante")
```

### F.3.5 Visualizaciones Avanzadas

#### F.3.5.1 Distribución de Scores de Similitud

```python
def plot_similarity_distribution():
    """Visualiza distribución de scores de similitud por modelo"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('📊 Distribución de Scores de Similitud Coseno')
    
    models = ['Ada', 'MPNet', 'MiniLM', 'E5-Large']
    
    for i, model in enumerate(models):
        ax = axes[i//2, i%2]
        
        # Obtener scores del modelo
        scores = get_similarity_scores(model)
        
        if len(scores) > 0:
            ax.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{model}')
            ax.set_xlabel('Cosine Similarity Score')
            ax.set_ylabel('Frequency')
            
            # Añadir línea vertical para el promedio
            mean_score = np.mean(scores)
            ax.axvline(mean_score, color='red', linestyle='--', 
                      label=f'Media: {mean_score:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model} - Sin datos')
    
    plt.tight_layout()
    st.pyplot(fig)
```

#### F.3.5.2 Análisis de Correlación entre Métricas

```python
def plot_metrics_correlation():
    """Visualiza correlaciones entre diferentes métricas"""
    
    # Crear matriz de correlación
    metrics_data = prepare_correlation_data()
    correlation_matrix = metrics_data.corr()
    
    # Crear heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',
                ax=ax)
    
    ax.set_title('🔗 Matriz de Correlación entre Métricas')
    
    st.pyplot(fig)
    
    # Interpretación de correlaciones importantes
    st.subheader("🔍 Interpretación de Correlaciones")
    
    high_correlations = find_high_correlations(correlation_matrix)
    
    for correlation in high_correlations:
        if correlation['value'] > 0.7:
            st.success(f"✅ **{correlation['metric1']}** y **{correlation['metric2']}**: "
                      f"Correlación alta ({correlation['value']:.3f})")
        elif correlation['value'] < -0.7:
            st.warning(f"⚠️ **{correlation['metric1']}** y **{correlation['metric2']}**: "
                      f"Correlación negativa fuerte ({correlation['value']:.3f})")
```

### F.3.6 Exportación de Reportes

#### F.3.6.1 Generación de Reportes PDF

```python
def generate_pdf_report():
    """Genera reporte PDF con todos los resultados"""
    
    st.subheader("📄 Generar Reporte PDF")
    
    # Opciones de reporte
    include_sections = st.multiselect(
        "Seleccionar secciones a incluir:",
        [
            "Resumen Ejecutivo",
            "Comparación de Modelos", 
            "Análisis de Métricas",
            "Casos de Ejemplo",
            "Análisis Estadístico",
            "Recomendaciones"
        ],
        default=["Resumen Ejecutivo", "Comparación de Modelos"]
    )
    
    if st.button("📄 Generar Reporte"):
        with st.spinner("Generando reporte PDF..."):
            pdf_buffer = create_pdf_report(include_sections)
            
            st.download_button(
                label="📥 Descargar Reporte PDF",
                data=pdf_buffer,
                file_name=f"reporte_sistema_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
```

#### F.3.6.2 Exportación de Datos

```python
def export_data_section():
    """Sección para exportar datos experimentales"""
    
    st.subheader("💾 Exportar Datos")
    
    export_format = st.radio(
        "Formato de exportación:",
        ["CSV", "JSON", "Excel"]
    )
    
    data_type = st.selectbox(
        "Tipo de datos:",
        [
            "Métricas por modelo",
            "Resultados por consulta",
            "Análisis estadístico",
            "Datos completos"
        ]
    )
    
    if st.button("💾 Exportar"):
        data = prepare_export_data(data_type)
        
        if export_format == "CSV":
            csv_buffer = data.to_csv(index=False)
            st.download_button(
                "📥 Descargar CSV",
                csv_buffer,
                f"datos_{data_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        elif export_format == "JSON":
            json_buffer = data.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Descargar JSON", 
                json_buffer,
                f"datos_{data_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
            )
```

## F.4 Configuración y Despliegue

### F.4.1 Configuración de Streamlit

**Archivo `.streamlit/config.toml`:**

```toml
[global]
dataFrameSerialization = "legacy"

[server]
port = 8501
address = "localhost"
maxUploadSize = 200
enableCORS = true
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#0078d4"              # Azure blue
backgroundColor = "#ffffff"           # White background
secondaryBackgroundColor = "#f5f5f5"  # Light gray
textColor = "#000000"                 # Black text
font = "sans serif"

[logger]
level = "info"
```

### F.4.2 Variables de Ambiente

```bash
# .env para Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_THEME_PRIMARY_COLOR=#0078d4

# Paths de datos
RESULTS_DATA_PATH=./data/
CHROMADB_PATH=/Users/haroldgomez/chromadb2

# APIs (si se requieren)
OPENAI_API_KEY=your_api_key_here
```

### F.4.3 Comandos de Ejecución

```bash
# Desarrollo local - Aplicación principal Q&A
streamlit run src/apps/main_qa_app.py

# Aplicación de resultados experimentales
streamlit run src/apps/cumulative_metrics_results_matplotlib.py

# Aplicación de comparación de modelos
streamlit run src/apps/comparison_page.py

# Con configuración específica
streamlit run src/apps/main_qa_app.py --server.port 8502

# Modo debug
streamlit run src/apps/main_qa_app.py --logger.level debug
```

## F.5 Funcionalidades de Usuario

### F.5.1 Navegación Intuitiva

- **Sidebar navigation:** Navegación entre páginas mediante sidebar
- **Breadcrumbs:** Indicadores de ubicación actual
- **Search functionality:** Búsqueda rápida de consultas y documentos

### F.5.2 Interactividad

- **Filtros dinámicos:** Filtrar resultados por modelo, métrica, rango de fechas
- **Zoom en gráficos:** Gráficos interactivos con Plotly
- **Tooltips informativos:** Ayuda contextual en métricas y visualizaciones

### F.5.3 Personalización

- **Temas:** Soporte para modo claro/oscuro
- **Exportación:** Múltiples formatos de exportación
- **Configuración:** Preferencias de usuario persistentes

## F.6 Casos de Uso de la Aplicación

### F.6.1 Investigación Académica

- **Análisis exploratorio:** Identificar patrones en los datos
- **Validación de hipótesis:** Verificar hallazgos experimentales
- **Generación de figuras:** Crear visualizaciones para publicaciones

### F.6.2 Desarrollo de Sistema

- **Debugging:** Identificar problemas en modelos específicos
- **Optimización:** Comparar configuraciones y parámetros
- **Monitoreo:** Tracking de performance a lo largo del tiempo

### F.6.3 Presentaciones y Demos

- **Demos interactivas:** Mostrar capacidades del sistema en tiempo real
- **Presentaciones:** Generar visualizaciones para audiencias técnicas
- **Reportes ejecutivos:** Crear resúmenes para stakeholders no técnicos

## F.7 Métricas de Performance de la Aplicación

### F.7.1 Tiempo de Carga

| Componente | Tiempo Promedio | Optimización |
|------------|-----------------|--------------|
| **Carga inicial** | 2.3 segundos | Caching de datos |
| **Cambio de página** | 0.8 segundos | Session state |
| **Generación de gráficos** | 1.5 segundos | Plotly optimizado |
| **Exportación PDF** | 4.2 segundos | Procesamiento asíncrono |

### F.7.2 Uso de Recursos

- **Memoria RAM:** ~150MB (datos cargados)
- **CPU:** Picos del 20% durante generación de gráficos
- **Storage:** ~50MB cache de visualizaciones
- **Network:** Mínimo (datos locales)

## F.8 Mantenimiento y Actualizaciones

### F.8.1 Actualizaciones de Datos

```python
def update_data_sources():
    """Actualiza fuentes de datos experimentales"""
    
    # Detectar nuevos archivos de resultados
    new_files = scan_for_new_results()
    
    # Validar formato y completitud
    validated_files = validate_results_files(new_files)
    
    # Actualizar cache de la aplicación
    update_app_cache(validated_files)
    
    # Notificar a usuarios activos
    st.rerun()
```

### F.8.2 Monitoreo de Performance

```python
def monitor_app_performance():
    """Monitorea performance de la aplicación"""
    
    metrics = {
        'load_time': measure_load_time(),
        'memory_usage': get_memory_usage(),
        'active_users': count_active_sessions(),
        'error_rate': calculate_error_rate()
    }
    
    # Log métricas
    logger.info(f"App metrics: {metrics}")
    
    # Alertas si performance degrada
    if metrics['load_time'] > 5.0:
        send_performance_alert(metrics)
```

## F.9 Conclusión

La aplicación Streamlit proporciona una interfaz comprehensiva para explorar y analizar los resultados experimentales del sistema RAG. Su arquitectura modular permite fácil extensión y mantenimiento, mientras que sus capacidades de visualización facilitan el entendimiento de patrones complejos en los datos experimentales.

### F.9.1 Beneficios Principales

1. **Accesibilidad:** Interfaz web intuitiva sin necesidad de conocimientos técnicos
2. **Interactividad:** Exploración dinámica de resultados experimentales
3. **Reproducibilidad:** Visualizaciones consistentes basadas en datos verificables
4. **Extensibilidad:** Arquitectura modular para agregar nuevas funcionalidades

### F.9.2 Uso Recomendado

- **Análisis exploratorio** de resultados experimentales
- **Validación** de hallazgos de investigación
- **Generación de reportes** para audiencias diversas
- **Desarrollo iterativo** del sistema RAG

---

**Acceso:** Las aplicaciones están disponibles ejecutando:
- `streamlit run src/apps/main_qa_app.py` (aplicación principal)
- `streamlit run src/apps/cumulative_metrics_results_matplotlib.py` (resultados experimentales)

Después de seguir las instrucciones de configuración del Anexo C.