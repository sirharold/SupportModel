# DIAGRAMA DE FLUJO DE NIVEL INTERMEDIO - SISTEMA RAG AZURE
## Arquitectura Detallada para Defensa de Título

```mermaid
flowchart TD
    %% Definición de estilos
    classDef userInput fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef models fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    classDef decision fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#000
    
    %% ENTRADA DEL USUARIO
    USER[👤 Usuario<br/>Consulta Técnica Azure]:::userInput
    
    %% INTERFAZ WEB
    WEB[🌐 Interfaz Web Streamlit<br/>3 Páginas Principales:<br/>• Búsqueda Individual<br/>• Comparación Modelos<br/>• Métricas Acumulativas]:::userInput
    
    %% SISTEMA RAG PRINCIPAL
    RAG[🤖 SISTEMA RAG<br/>Pipeline de 6 Etapas<br/>Procesamiento Inteligente]:::processing
    
    %% ETAPA 1: REFINAMIENTO
    REFINE[📝 Etapa 1: Refinamiento<br/>Análisis de Intención<br/>Expansión Contextual<br/>Modelo: Mistral 7B Local]:::processing
    
    %% ETAPA 2: EMBEDDINGS
    EMBED[🔤 Etapa 2: Embeddings<br/>Generación Multi-Modelo<br/>• mpnet-base-dot-v1<br/>• all-MiniLM-L6-v2<br/>• text-embedding-ada-002]:::processing
    
    %% BASE DE DATOS VECTORIAL
    VECTOR_DB[🗄️ Base de Datos Vectorial<br/>Weaviate Cloud Service<br/>Colecciones:<br/>• Documentos Azure<br/>• Preguntas Comunidad]:::storage
    
    %% ETAPA 3: BÚSQUEDA
    SEARCH[🔍 Etapa 3: Búsqueda<br/>Búsqueda Vectorial Híbrida<br/>Múltiples Colecciones<br/>Fusión Inteligente]:::processing
    
    %% ETAPA 4: RERANKING
    RERANK[📊 Etapa 4: Reranking<br/>CrossEncoder Local<br/>ms-marco-MiniLM-L-6-v2<br/>Scoring Contextual]:::processing
    
    %% DECISIÓN DE MODELO
    MODEL_CHOICE{🎯 Selección de Modelo<br/>Generativo<br/>Local vs Remoto}:::decision
    
    %% MODELOS LOCALES
    LOCAL_MODELS[💻 Modelos Locales<br/>Costo Zero<br/>• TinyLlama 1.1B<br/>• Mistral 7B]:::models
    
    %% MODELOS REMOTOS
    REMOTE_MODELS[☁️ Modelos Remotos<br/>APIs Comerciales<br/>• GPT-4<br/>• Gemini Pro<br/>• Llama 3.3 70B (OpenRouter)<br/>• DeepSeek V3 (OpenRouter)]:::models
    
    %% ETAPA 5: GENERACIÓN
    GENERATE[✨ Etapa 5: Generación<br/>Respuesta Fundamentada<br/>Contexto + Pregunta<br/>Sistema de Fallback]:::processing
    
    %% ETAPA 6: EVALUACIÓN
    EVALUATE[📈 Etapa 6: Evaluación<br/>Métricas Tradicionales:<br/>• BERTScore, ROUGE, MRR<br/>Métricas RAG Avanzadas:<br/>• Detección Alucinaciones<br/>• Utilización Contexto<br/>• Completitud Respuesta<br/>• Satisfacción Usuario]:::processing
    
    %% SALIDAS
    INDIVIDUAL_OUTPUT[📋 Respuesta Individual<br/>Respuesta Completa<br/>Documentos Fuente<br/>Métricas Calidad]:::output
    
    COMPARISON_OUTPUT[📊 Comparación Modelos<br/>Análisis Comparativo<br/>Métricas por Modelo<br/>Reporte PDF]:::output
    
    CUMULATIVE_OUTPUT[📈 Métricas Acumulativas<br/>Evaluación Masiva<br/>Estadísticas Agregadas<br/>Análisis Multi-Modelo]:::output
    
    %% DATOS DE ENTRADA
    DATA_SOURCES[📚 Fuentes de Datos<br/>Microsoft Learn<br/>Microsoft Q&A<br/>GitHub Issues<br/>Stack Overflow]:::storage
    
    %% CONEXIONES PRINCIPALES
    USER --> WEB
    WEB --> RAG
    
    %% PIPELINE RAG
    RAG --> REFINE
    REFINE --> EMBED
    EMBED --> VECTOR_DB
    VECTOR_DB --> SEARCH
    SEARCH --> RERANK
    RERANK --> MODEL_CHOICE
    
    %% MODELOS
    MODEL_CHOICE -->|Eficiencia/Costo| LOCAL_MODELS
    MODEL_CHOICE -->|Máxima Calidad| REMOTE_MODELS
    
    LOCAL_MODELS --> GENERATE
    REMOTE_MODELS --> GENERATE
    
    GENERATE --> EVALUATE
    
    %% TIPOS DE SALIDA
    EVALUATE --> INDIVIDUAL_OUTPUT
    EVALUATE --> COMPARISON_OUTPUT
    EVALUATE --> CUMULATIVE_OUTPUT
    
    %% DATOS
    DATA_SOURCES --> VECTOR_DB
    
    %% COMENTARIOS EXPLICATIVOS
    RAG -.- COMMENT1["💡 INNOVACIÓN PRINCIPAL:<br/>Pipeline RAG de 6 etapas con<br/>métricas especializadas y<br/>modelos locales para<br/>optimización de costos"]
    
    MODEL_CHOICE -.- COMMENT2["⚡ ARQUITECTURA HÍBRIDA:<br/>Combina eficiencia de modelos<br/>locales con calidad de APIs<br/>comerciales según necesidad"]
    
    EVALUATE -.- COMMENT3["🎯 CONTRIBUCIÓN CLAVE:<br/>4 métricas RAG especializadas<br/>para evaluación integral de<br/>calidad en contextos técnicos"]
    
    %% BENEFICIOS DEL SISTEMA
    BENEFITS[🏆 BENEFICIOS CLAVE<br/>• Reducción significativa costos<br/>• Respuestas fundamentadas<br/>• Métricas especializadas<br/>• Arquitectura escalable<br/>• Interfaz profesional]:::output
    
    INDIVIDUAL_OUTPUT --> BENEFITS
    COMPARISON_OUTPUT --> BENEFITS
    CUMULATIVE_OUTPUT --> BENEFITS
```

## Elementos Clave del Diagrama

### 🔵 **Componentes Principales**
- **Usuario**: Punto de entrada con consultas técnicas
- **Interfaz Web**: 3 páginas especializadas en Streamlit
- **Sistema RAG**: Pipeline de 6 etapas de procesamiento
- **Base de Datos**: Weaviate con múltiples colecciones
- **Modelos**: Arquitectura híbrida local/remota

### 🟣 **Pipeline de Procesamiento**
1. **Refinamiento**: Análisis y expansión de consulta
2. **Embeddings**: Generación multi-modelo vectorial
3. **Búsqueda**: Recuperación híbrida inteligente
4. **Reranking**: Scoring contextual avanzado
5. **Generación**: Respuesta fundamentada
6. **Evaluación**: Métricas tradicionales y especializadas

### 🟢 **Almacenamiento**
- **Fuentes de datos**: Microsoft Learn, Q&A, GitHub, Stack Overflow
- **Base vectorial**: Weaviate Cloud con colecciones especializadas

### 🟡 **Decisiones Clave**
- **Selección de modelo**: Local vs Remoto según necesidad
- **Tipos de salida**: Individual, Comparativa, Lotes

### 🔴 **Salidas del Sistema**
- **Respuestas individuales**: Completas con fuentes
- **Comparaciones**: Análisis multi-modelo
- **Métricas acumulativas**: Estadísticas agregadas y evaluación masiva

### 🏆 **Beneficios Destacados**
- Optimización de costos
- Respuestas fundamentadas
- Métricas especializadas
- Arquitectura escalable

---

*Este diagrama está diseñado para presentación de defensa de título, enfocándose en claridad conceptual y flujo lógico del sistema RAG implementado.*