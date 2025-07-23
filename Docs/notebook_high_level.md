# 📓 Flujo de Alto Nivel - Notebook de Evaluación Multi-Modelo

## 🎯 Flujo Principal

```mermaid
graph TD
    A[📋 Cargar Configuración] --> B[🎯 Seleccionar Modelos]
    B --> C[🔄 Loop: Para cada Modelo]
    C --> D[🔍 Loop: Para cada Pregunta]
    D --> E[📊 Calcular Métricas]
    E --> F[💾 Guardar Resultados]
    
    C -.-> G[Ada<br/>E5-Large<br/>MPNet<br/>MiniLM]
    D -.-> H[Title + Question<br/>NO Answer]
    E -.-> I[Retrieval Metrics<br/>LLM Reranking<br/>RAG Metrics]
    
    style A fill:#e1f5fe
    style C fill:#fff3e0
    style E fill:#e8f5e8
    style F fill:#f3e5f5
```

## 🔄 Proceso de Evaluación por Modelo

```mermaid
flowchart LR
    subgraph "Para cada Modelo"
        A[Cargar Embeddings<br/>Parquet] --> B{Verificar<br/>Dimensiones}
        B -->|✅ Compatible| C[Procesar Preguntas]
        B -->|❌ Error| D[Guardar Error<br/>Siguiente Modelo]
        
        C --> E[Generar Query<br/>Embedding]
        E --> F[Búsqueda de<br/>Documentos]
        F --> G[Métricas BEFORE]
        G --> H{LLM<br/>Reranking?}
        H -->|Sí| I[Reordenar Docs]
        H -->|No| J[Mantener Original]
        I --> K[Métricas AFTER]
        J --> L{RAG<br/>Metrics?}
        K --> L
        L -->|Sí| M[Generar RAG]
        L -->|No| N[Continuar]
        M --> N
        N --> O[Siguiente Pregunta]
    end
    
    style A fill:#e1f5fe
    style E fill:#fff3e0
    style G fill:#e8f5e8
    style M fill:#f3e5f5
```

## 📊 Estructura de Métricas

```mermaid
graph TD
    A[Métricas por Modelo] --> B[Retrieval Metrics]
    A --> C[LLM Reranking]
    A --> D[RAG Metrics]
    
    B --> B1[Precision@k]
    B --> B2[Recall@k]
    B --> B3[F1@k]
    B --> B4[NDCG@k]
    B --> B5[MAP@k]
    B --> B6[MRR]
    
    C --> C1[Métricas BEFORE]
    C --> C2[Métricas AFTER]
    C --> C3[Mejora %]
    
    D --> D1[avg_faithfulness]
    D --> D2[avg_answer_relevance]
    D --> D3[avg_answer_correctness]
    D --> D4[avg_answer_similarity]
    
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#f3e5f5
```

## 🔧 Manejo de Modelos y Dimensiones

```mermaid
graph LR
    subgraph "Configuración de Modelos"
        A[Ada<br/>1536 dims] --> A1[OpenAI API]
        B[E5-Large<br/>1024 dims] --> B1[HuggingFace<br/>GPU→CPU]
        C[MPNet<br/>768 dims] --> C1[HuggingFace]
        D[MiniLM<br/>384 dims] --> D1[HuggingFace]
    end
    
    A1 --> E[Query Embedding]
    B1 --> E
    C1 --> E
    D1 --> E
    
    E --> F{Dimension<br/>Match?}
    F -->|✅| G[Continuar Evaluación]
    F -->|❌| H[Error + Skip Model]
    
    style A fill:#ffebee
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#e1f5fe
```

## 💾 Estructura de Resultados JSON

```mermaid
graph TD
    A[cumulative_results.json] --> B[config]
    A --> C[evaluation_info]
    A --> D[results]
    
    B --> B1[num_questions]
    B --> B2[selected_models]
    B --> B3[use_llm_reranker]
    B --> B4[generate_rag_metrics]
    
    C --> C1[timestamp]
    C --> C2[models_evaluated]
    C --> C3[data_verification]
    
    D --> E[Por cada Modelo]
    E --> E1[avg_before_metrics]
    E --> E2[avg_after_metrics]
    E --> E3[rag_metrics]
    E --> E4[individual_metrics]
    
    E3 --> F[avg_faithfulness]
    E3 --> G[avg_answer_relevance]
    E3 --> H[rag_available]
    
    style A fill:#f3e5f5
    style D fill:#e8f5e8
    style E3 fill:#fff3e0
```

## 🚀 Flujo de Ejecución Temporal

```mermaid
sequenceDiagram
    participant Config as Configuración
    participant Models as Modelos
    participant Questions as Preguntas
    participant Metrics as Métricas
    participant Storage as Almacenamiento
    
    Config->>Models: Cargar lista de modelos
    
    loop Para cada Modelo
        Models->>Models: Verificar dimensiones
        
        loop Para cada Pregunta
            Models->>Questions: Extraer title + question_content
            Questions->>Questions: Generar embedding
            Questions->>Metrics: Calcular retrieval metrics
            
            opt LLM Reranking Enabled
                Metrics->>Metrics: Reordenar documentos
                Metrics->>Metrics: Calcular AFTER metrics
            end
            
            opt RAG Metrics Enabled
                Metrics->>Metrics: Generar respuesta
                Metrics->>Metrics: Calcular RAG metrics
            end
        end
        
        Models->>Storage: Guardar resultados del modelo
    end
    
    Storage->>Storage: Generar JSON final
    Storage->>Storage: Guardar en Google Drive
```

## ⚡ Optimizaciones y Fallbacks

```mermaid
flowchart TD
    A[Proceso de Embedding] --> B{GPU Disponible?}
    B -->|Sí| C[Cargar en GPU]
    B -->|No| D[Cargar en CPU]
    
    C --> E{CUDA Error?}
    E -->|Sí| F[Limpiar GPU]
    E -->|No| G[Continuar]
    
    F --> H[Fallback a CPU]
    D --> G
    H --> G
    
    G --> I[Generar Embedding]
    
    style C fill:#e8f5e8
    style F fill:#ffebee
    style H fill:#fff3e0
```

## 📊 Resumen de Salida

```mermaid
graph LR
    A[Evaluación Completa] --> B[4 Modelos]
    B --> C[100 Preguntas c/u]
    C --> D[Métricas Calculadas]
    
    D --> E[20 Retrieval Metrics]
    D --> F[20 LLM Reranking]
    D --> G[4 RAG Metrics]
    
    E --> H[JSON Estructurado]
    F --> H
    G --> H
    
    H --> I[Compatible con<br/>Streamlit]
    
    style A fill:#e1f5fe
    style D fill:#e8f5e8
    style H fill:#f3e5f5
    style I fill:#c8e6c9
```