# 🔬 Diagrama Detallado: Página de Métricas Acumulativas
## Flujo Técnico Completo con Implementación

```mermaid
flowchart TD
    %% Definición de estilos
    classDef userInput fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef models fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    classDef decision fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#000
    classDef evaluation fill:#f0f4c3,stroke:#827717,stroke-width:2px,color:#000
    classDef utils fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#000
    
    %% ENTRADA DEL USUARIO
    USER[👤 Usuario accede a página<br/>show_cumulative_metrics_page()]:::userInput
    
    %% CONFIGURACIÓN UI
    CONFIG_UI[⚙️ Configuración UI<br/>• num_questions: 1-2000<br/>• selected_models: multiselect<br/>• generative_model_name<br/>• top_k: 1-50<br/>• use_llm_reranker: checkbox<br/>• batch_size: 10-100]:::userInput
    
    %% VALIDACIÓN
    VALIDATION[✅ Validación<br/>• Modelos seleccionados > 0<br/>• Parámetros válidos<br/>• Estimación memoria]:::processing
    
    %% CLICK EJECUTAR
    EXECUTE_CLICK[🚀 Click 'Ejecutar Evaluación'<br/>run_evaluation()]:::userInput
    
    %% DECISIÓN TIPO EVALUACIÓN
    EVAL_TYPE{🎯 evaluate_all_models<br/>¿Múltiples modelos?}:::decision
    
    %% INICIALIZACIÓN
    INIT_CLIENTS[🔧 initialize_clients()<br/>• WeaviateWrapper<br/>• EmbeddingClient<br/>• OpenAI, Gemini<br/>• Local models]:::processing
    
    %% EXTRACCIÓN DE DATOS
    FETCH_QUESTIONS[🔍 fetch_random_questions_from_weaviate()<br/>• Determinar clase por modelo<br/>• Sample size inteligente<br/>• Filtrar por MS Learn links<br/>• Random sampling]:::processing
    
    %% DETALLE EXTRACCIÓN
    WEAVIATE_QUERY[🌐 Weaviate Query<br/>questions_collection.query.fetch_objects()<br/>• limit: sample_size<br/>• return_properties: title, content, answer<br/>• Conversión a diccionarios]:::storage
    
    %% FILTRADO
    FILTER_LINKS[🔗 Filtrar enlaces MS Learn<br/>• Regex: https://learn.microsoft.com<br/>• extract_ms_links()<br/>• Solo preguntas con enlaces<br/>• Agregar ms_links al dict]:::processing
    
    %% EVALUACIÓN INDIVIDUAL
    SINGLE_MODEL[🔧 Evaluación Individual<br/>run_cumulative_metrics_evaluation()]:::evaluation
    
    %% EVALUACIÓN MÚLTIPLE
    MULTI_MODEL[🔄 Evaluación Múltiple<br/>run_cumulative_metrics_for_models()<br/>Bucle secuencial]:::evaluation
    
    %% PROCESAMIENTO POR LOTES
    BATCH_LOOP[📦 Procesamiento por Lotes<br/>• Calcular num_batches<br/>• Barra de progreso<br/>• batch_questions = selected[start:end]<br/>• Memoria inicial]:::processing
    
    %% BUCLE PREGUNTA INDIVIDUAL
    QUESTION_LOOP[🔄 Bucle Individual<br/>Para cada pregunta en lote<br/>• Extraer question, answer, links<br/>• Actualizar progreso]:::processing
    
    %% PIPELINE RAG COMPLETO
    RAG_PIPELINE[🤖 answer_question_with_retrieval_metrics()<br/>• question: str<br/>• ground_truth_answer: str<br/>• ms_links: List[str]<br/>• calculate_metrics: True]:::processing
    
    %% REFINAMIENTO
    REFINE_QUERY[📝 refine_and_prepare_query()<br/>• Análisis de intención<br/>• Expansión contextual<br/>• Preparación para embedding]:::processing
    
    %% EMBEDDING
    EMBEDDING[🔤 Generación Embedding<br/>• embedding_client.encode()<br/>• Modelo específico<br/>• Vectorización query]:::models
    
    %% BÚSQUEDA VECTORIAL
    VECTOR_SEARCH[🔍 Búsqueda Vectorial<br/>• Weaviate hybrid search<br/>• Múltiples colecciones<br/>• top_k documentos]:::storage
    
    %% RERANKING
    RERANKING[📊 Reranking LLM<br/>• CrossEncoder scoring<br/>• Reordenamiento contextual<br/>• Filtrado por relevancia]:::processing
    
    %% CÁLCULO MÉTRICAS
    METRICS_CALC[📈 Cálculo Métricas RAG<br/>• before_reranking_metrics<br/>• after_reranking_metrics<br/>• rag_stats]:::evaluation
    
    %% MÉTRICAS DETALLADAS
    DETAILED_METRICS[📊 Métricas Detalladas<br/>• Precision@k (k=1,3,5,10)<br/>• Recall@k (k=1,3,5,10)<br/>• F1@k (k=1,3,5,10)<br/>• MRR, nDCG<br/>• BinaryAccuracy@k<br/>• RankingAccuracy@k]:::evaluation
    
    %% ESTADÍSTICAS RAG
    RAG_STATS[📊 Estadísticas RAG<br/>• ground_truth_links_count<br/>• docs_before_count<br/>• docs_after_count<br/>• memory_usage]:::evaluation
    
    %% ALMACENAMIENTO TEMPORAL
    TEMP_STORAGE[💾 Almacenamiento Temporal<br/>• before_reranking_metrics[]<br/>• after_reranking_metrics[]<br/>• rag_stats_list[]<br/>• all_questions_data[]]:::storage
    
    %% CLEANUP LOTE
    BATCH_CLEANUP[🧹 Cleanup Lote<br/>• cleanup_memory()<br/>• gc.collect()<br/>• Pausa 0.1s<br/>• Monitoreo memoria]:::processing
    
    %% AGREGACIÓN FINAL
    FINAL_AGGREGATION[📊 Agregación Final<br/>• calculate_average_metrics()<br/>• Promedios por métrica<br/>• Estadísticas finales<br/>• Validación integridad]:::processing
    
    %% CACHÉ DE RESULTADOS
    CACHE_RESULTS[💾 Caché de Resultados<br/>• st.session_state[cache_key]<br/>• Timestamp único<br/>• Parámetros evaluación<br/>• Resultados completos]:::storage
    
    %% VISUALIZACIÓN
    DISPLAY_RESULTS[📈 Visualización<br/>• display_cumulative_metrics()<br/>• display_models_comparison()<br/>• Métricas por modelo<br/>• Gráficos interactivos]:::output
    
    %% EXPORTACIÓN
    EXPORT_OPTIONS[📥 Opciones Exportación<br/>• display_download_section()<br/>• PDF: generate_cumulative_pdf_report()<br/>• CSV: DataFrame.to_csv()<br/>• JSON: export completo]:::output
    
    %% GESTIÓN CACHÉ
    CACHE_MANAGEMENT[🔧 Gestión Caché<br/>• Múltiples resultados<br/>• Selección dropdown<br/>• Limpieza manual<br/>• Estadísticas memoria]:::processing
    
    %% FLUJO PRINCIPAL
    USER --> CONFIG_UI
    CONFIG_UI --> VALIDATION
    VALIDATION --> EXECUTE_CLICK
    EXECUTE_CLICK --> EVAL_TYPE
    
    EVAL_TYPE -->|Individual| SINGLE_MODEL
    EVAL_TYPE -->|Múltiple| MULTI_MODEL
    
    SINGLE_MODEL --> INIT_CLIENTS
    MULTI_MODEL --> INIT_CLIENTS
    
    INIT_CLIENTS --> FETCH_QUESTIONS
    FETCH_QUESTIONS --> WEAVIATE_QUERY
    WEAVIATE_QUERY --> FILTER_LINKS
    
    FILTER_LINKS --> BATCH_LOOP
    BATCH_LOOP --> QUESTION_LOOP
    QUESTION_LOOP --> RAG_PIPELINE
    
    RAG_PIPELINE --> REFINE_QUERY
    REFINE_QUERY --> EMBEDDING
    EMBEDDING --> VECTOR_SEARCH
    VECTOR_SEARCH --> RERANKING
    
    RERANKING --> METRICS_CALC
    METRICS_CALC --> DETAILED_METRICS
    DETAILED_METRICS --> RAG_STATS
    RAG_STATS --> TEMP_STORAGE
    
    TEMP_STORAGE --> BATCH_CLEANUP
    BATCH_CLEANUP --> QUESTION_LOOP
    QUESTION_LOOP --> FINAL_AGGREGATION
    
    FINAL_AGGREGATION --> CACHE_RESULTS
    CACHE_RESULTS --> DISPLAY_RESULTS
    DISPLAY_RESULTS --> EXPORT_OPTIONS
    
    EXPORT_OPTIONS --> CACHE_MANAGEMENT
    
    %% UTILIDADES REFACTORIZADAS
    UTILS_DATA[🔧 utils/data_processing.py<br/>• fetch_random_questions_from_weaviate()<br/>• extract_ms_links()<br/>• filter_questions_with_links()]:::utils
    
    UTILS_EVAL[🔧 utils/cumulative_evaluation.py<br/>• run_cumulative_metrics_evaluation()<br/>• run_cumulative_metrics_for_models()]:::utils
    
    UTILS_DISPLAY[🔧 utils/metrics_display.py<br/>• display_cumulative_metrics()<br/>• display_models_comparison()]:::utils
    
    UTILS_FILE[🔧 utils/file_utils.py<br/>• display_download_section()<br/>• CSV/PDF/JSON export]:::utils
    
    UTILS_MEMORY[🔧 utils/memory_utils.py<br/>• get_memory_usage()<br/>• cleanup_memory()]:::utils
    
    UTILS_METRICS[🔧 utils/metrics.py<br/>• calculate_average_metrics()<br/>• validate_data_integrity()]:::utils
    
    %% CONEXIONES UTILIDADES
    FETCH_QUESTIONS --> UTILS_DATA
    SINGLE_MODEL --> UTILS_EVAL
    MULTI_MODEL --> UTILS_EVAL
    DISPLAY_RESULTS --> UTILS_DISPLAY
    EXPORT_OPTIONS --> UTILS_FILE
    BATCH_CLEANUP --> UTILS_MEMORY
    FINAL_AGGREGATION --> UTILS_METRICS
    
    %% COMENTARIOS TÉCNICOS
    FETCH_QUESTIONS -.- TECH1["🔍 WEAVIATE API v4:<br/>collections.get(class).query.fetch_objects()<br/>Reemplaza GraphQL query.raw()"]
    
    BATCH_LOOP -.- TECH2["📦 GESTIÓN MEMORIA:<br/>Procesamiento por lotes con límites<br/>Cleanup automático entre lotes"]
    
    METRICS_CALC -.- TECH3["📈 MÉTRICAS AVANZADAS:<br/>Incluye BinaryAccuracy@k y RankingAccuracy@k<br/>Estadísticas RAG especializadas"]
    
    CACHE_RESULTS -.- TECH4["💾 CACHÉ INTELIGENTE:<br/>Múltiples resultados simultáneos<br/>Gestión automática de memoria"]
    
    %% BENEFICIOS TÉCNICOS
    TECH_BENEFITS[🏆 BENEFICIOS TÉCNICOS<br/>• Arquitectura modular refactorizada<br/>• Gestión automática de memoria<br/>• APIs actualizadas (Weaviate v4)<br/>• Métricas RAG especializadas<br/>• Extracción directa desde Weaviate<br/>• Exportación completa multi-formato]:::output
    
    CACHE_MANAGEMENT --> TECH_BENEFITS
```

## Detalles Técnicos de Implementación

### 🔧 **Arquitectura Refactorizada**
- **cumulative_metrics_page.py**: 279 líneas (reducido 90%)
- **utils/data_processing.py**: 140 líneas - Extracción Weaviate
- **utils/cumulative_evaluation.py**: 216 líneas - Lógica evaluación
- **utils/metrics_display.py**: 653 líneas - Visualización
- **utils/file_utils.py**: 198 líneas - Exportación
- **utils/memory_utils.py**: 17 líneas - Gestión memoria

### 🌐 **Integración Weaviate**
```python
# Weaviate v4 API
questions_collection = weaviate_wrapper.client.collections.get(questions_class)
results = questions_collection.query.fetch_objects(
    limit=sample_size,
    return_properties=["title", "question_content", "accepted_answer", "tags"]
)
```

### 📊 **Métricas Implementadas**
- **Tradicionales**: Precision@k, Recall@k, F1@k, MRR, nDCG
- **RAG Especializadas**: BinaryAccuracy@k, RankingAccuracy@k
- **Estadísticas**: ground_truth_links_count, docs_before_count, docs_after_count

### 🔄 **Procesamiento por Lotes**
```python
num_batches = (actual_questions + batch_size - 1) // batch_size
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, actual_questions)
    batch_questions = selected_questions[start_idx:end_idx]
    # Procesar lote + cleanup_memory()
```

### 💾 **Gestión de Memoria**
- **Monitoreo continuo**: get_memory_usage()
- **Cleanup automático**: cleanup_memory() + gc.collect()
- **Gestión de caché**: Múltiples resultados simultáneos
- **Optimización**: Procesamiento por lotes con límites

### 📈 **Flujo de Métricas**
1. **Extracción**: Preguntas aleatorias con filtros MS Learn
2. **Procesamiento**: Pipeline RAG completo por pregunta
3. **Cálculo**: Métricas antes/después reranking
4. **Agregación**: Promedios y estadísticas finales
5. **Visualización**: Gráficos interactivos y tablas
6. **Exportación**: PDF, CSV, JSON completos

### 🎯 **Optimizaciones Implementadas**
- **Extracción directa**: Sin dependencia de archivos JSON
- **Muestreo inteligente**: Factor de seguridad para filtrado
- **Gestión memoria**: Cleanup automático entre lotes
- **Caché de resultados**: Reutilización para análisis iterativo
- **Validación datos**: Integridad automática de métricas

---

*Este diagrama detallado muestra la implementación técnica completa de la página de métricas acumulativas, incluyendo la arquitectura refactorizada y las optimizaciones implementadas.*