# 📊 Diagrama de Alto Nivel: Página de Métricas Acumulativas
## Flujo General del Sistema de Evaluación Masiva

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
    
    %% ENTRADA DEL USUARIO
    USER[👤 Usuario<br/>Evaluación Masiva RAG]:::userInput
    
    %% CONFIGURACIÓN
    CONFIG[⚙️ Configuración<br/>• Número de preguntas<br/>• Modelos a evaluar<br/>• Parámetros RAG]:::userInput
    
    %% EXTRACCIÓN DE DATOS
    DATA_EXTRACTION[🔍 Extracción desde Weaviate<br/>• Preguntas aleatorias<br/>• Filtrado por MS Learn<br/>• Muestreo inteligente]:::processing
    
    %% DECISIÓN DE EVALUACIÓN
    EVAL_DECISION{🎯 Tipo de Evaluación<br/>¿Un modelo o múltiples?}:::decision
    
    %% EVALUACIÓN INDIVIDUAL
    SINGLE_EVAL[🔧 Evaluación Individual<br/>Un modelo específico<br/>Pipeline RAG completo]:::evaluation
    
    %% EVALUACIÓN MÚLTIPLE
    MULTI_EVAL[🔄 Evaluación Múltiple<br/>Todos los modelos<br/>Bucle secuencial]:::evaluation
    
    %% PROCESAMIENTO EN LOTES
    BATCH_PROCESS[📦 Procesamiento por Lotes<br/>• Gestión de memoria<br/>• Progreso incremental<br/>• Cleanup automático]:::processing
    
    %% PIPELINE RAG PARA CADA PREGUNTA
    RAG_PIPELINE[🤖 Pipeline RAG<br/>• Embedding query<br/>• Búsqueda vectorial<br/>• Reranking LLM<br/>• Cálculo métricas]:::processing
    
    %% CÁLCULO DE MÉTRICAS
    METRICS_CALC[📈 Cálculo de Métricas<br/>• Precision@k, Recall@k<br/>• F1@k, MRR, nDCG<br/>• BinaryAccuracy@k<br/>• RankingAccuracy@k]:::evaluation
    
    %% AGREGACIÓN DE RESULTADOS
    AGGREGATION[📊 Agregación de Resultados<br/>• Promedios por modelo<br/>• Estadísticas RAG<br/>• Métricas de memoria]:::processing
    
    %% VISUALIZACIÓN
    VISUALIZATION[📈 Visualización<br/>• Métricas por modelo<br/>• Comparación gráfica<br/>• Tablas detalladas]:::output
    
    %% EXPORTACIÓN
    EXPORT[📥 Exportación<br/>• Reportes PDF<br/>• Datos CSV<br/>• Archivos JSON]:::output
    
    %% ALMACENAMIENTO EN CACHÉ
    CACHE[💾 Caché de Resultados<br/>• Persistencia en sesión<br/>• Reutilización<br/>• Gestión automática]:::storage
    
    %% FLUJO PRINCIPAL
    USER --> CONFIG
    CONFIG --> DATA_EXTRACTION
    DATA_EXTRACTION --> EVAL_DECISION
    
    EVAL_DECISION -->|Un modelo| SINGLE_EVAL
    EVAL_DECISION -->|Múltiples| MULTI_EVAL
    
    SINGLE_EVAL --> BATCH_PROCESS
    MULTI_EVAL --> BATCH_PROCESS
    
    BATCH_PROCESS --> RAG_PIPELINE
    RAG_PIPELINE --> METRICS_CALC
    METRICS_CALC --> AGGREGATION
    
    AGGREGATION --> CACHE
    CACHE --> VISUALIZATION
    VISUALIZATION --> EXPORT
    
    %% COMENTARIOS EXPLICATIVOS
    DATA_EXTRACTION -.- COMMENT1["🔍 INNOVACIÓN:<br/>Extracción inteligente desde Weaviate<br/>sin dependencia de archivos JSON"]
    
    BATCH_PROCESS -.- COMMENT2["📦 OPTIMIZACIÓN:<br/>Procesamiento por lotes con<br/>gestión avanzada de memoria"]
    
    METRICS_CALC -.- COMMENT3["📈 MÉTRICAS AVANZADAS:<br/>Incluye BinaryAccuracy@k y<br/>RankingAccuracy@k para RAG"]
    
    CACHE -.- COMMENT4["💾 EFICIENCIA:<br/>Caché inteligente para<br/>reutilización de resultados"]
    
    %% BENEFICIOS
    BENEFITS[🏆 BENEFICIOS CLAVE<br/>• Evaluación masiva eficiente<br/>• Métricas RAG especializadas<br/>• Comparación multi-modelo<br/>• Gestión automática memoria<br/>• Exportación completa]:::output
    
    EXPORT --> BENEFITS
```

## Elementos Clave del Diagrama

### 🔵 **Entrada del Usuario**
- **Configuración flexible**: Número de preguntas, modelos, parámetros
- **Interfaz intuitiva**: Controles simples para evaluación compleja

### 🟣 **Procesamiento Inteligente**
- **Extracción desde Weaviate**: Preguntas aleatorias con filtros
- **Procesamiento por lotes**: Gestión automática de memoria
- **Pipeline RAG completo**: Embedding, búsqueda, reranking, métricas

### 🟡 **Decisiones Clave**
- **Evaluación individual vs múltiple**: Adaptación automática del flujo
- **Gestión de recursos**: Balanceo entre velocidad y memoria

### 🟢 **Almacenamiento**
- **Caché de sesión**: Persistencia inteligente de resultados
- **Gestión automática**: Cleanup y optimización de memoria

### 🔴 **Salidas del Sistema**
- **Visualización rica**: Métricas, gráficos, tablas
- **Exportación completa**: PDF, CSV, JSON
- **Comparación multi-modelo**: Análisis detallado

### 🏆 **Beneficios Destacados**
- Evaluación masiva eficiente
- Métricas RAG especializadas
- Comparación multi-modelo
- Gestión automática de memoria
- Exportación completa

## Características Técnicas

### 🔧 **Optimizaciones Implementadas**
- **Muestreo inteligente**: Selección eficiente de preguntas
- **Procesamiento por lotes**: Gestión de memoria con límites
- **Caché de resultados**: Reutilización para análisis iterativo
- **Cleanup automático**: Liberación de recursos entre lotes

### 📊 **Métricas Avanzadas**
- **Tradicionales**: Precision@k, Recall@k, F1@k, MRR, nDCG
- **RAG Especializadas**: BinaryAccuracy@k, RankingAccuracy@k
- **Estadísticas**: Conteo de enlaces, documentos, memoria

### 🚀 **Escalabilidad**
- **Evaluación masiva**: Hasta 2000 preguntas
- **Multi-modelo**: Todos los modelos de embedding
- **Gestión recursos**: Procesamiento eficiente y controlado

---

*Este diagrama representa la arquitectura de alto nivel de la página de métricas acumulativas, enfocándose en el flujo general y los beneficios del sistema.*