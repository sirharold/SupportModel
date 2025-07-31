# DIAGRAMA ULTRA-ALTO NIVEL - COMPARACIÓN DE MODELOS
## Overview Conceptual para Defensa de Título

```mermaid
flowchart TD
    %% Definición de estilos
    classDef user fill:#e3f2fd,stroke:#0d47a1,stroke-width:3px,color:#000
    classDef system fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,color:#000
    classDef innovation fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    classDef results fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px,color:#000
    
    %% COMPONENTES PRINCIPALES
    USER[👤 USUARIO<br/>Consulta + Dataset<br/>de Preguntas]:::user
    
    COMPARISON_SYSTEM[⚖️ SISTEMA DE COMPARACIÓN<br/>Evaluación Simultánea<br/>3 Modelos de Embedding]:::system
    
    INNOVATION[💡 INNOVACIÓN CLAVE<br/>• Evaluación Multi-Modelo<br/>• Métricas RAG Especializadas<br/>• Análisis Comparativo]:::innovation
    
    RESULTS[📊 RESULTADOS<br/>• Ranking de Modelos<br/>• Métricas por Modelo<br/>• Reporte Comparativo]:::results
    
    %% FLUJO PRINCIPAL
    USER --> COMPARISON_SYSTEM
    COMPARISON_SYSTEM --> RESULTS
    
    %% INNOVACIÓN
    INNOVATION --> COMPARISON_SYSTEM
    
    %% ETIQUETAS EXPLICATIVAS
    USER -.- LABEL1["Entrada:<br/>Conjunto de preguntas técnicas<br/>para evaluación comparativa"]
    
    COMPARISON_SYSTEM -.- LABEL2["Procesamiento:<br/>Mismo pipeline RAG ejecutado<br/>con 3 modelos embedding diferentes"]
    
    INNOVATION -.- LABEL3["Contribución:<br/>Framework de evaluación<br/>comparativa especializada"]
    
    RESULTS -.- LABEL4["Salida:<br/>Ranking objetivo + métricas<br/>+ reporte PDF profesional"]
    
    %% TÍTULO PRINCIPAL
    TITLE[🏆 EVALUACIÓN COMPARATIVA DE MODELOS EMBEDDING<br/>PARA SISTEMAS RAG EN CONTEXTO TÉCNICO]
    
    TITLE --> USER
    
    %% BENEFICIOS DESTACADOS
    BENEFITS[✨ BENEFICIOS PRINCIPALES<br/>• Selección objetiva de modelos<br/>• Métricas especializadas RAG<br/>• Reportes automatizados]:::results
    
    RESULTS --> BENEFITS
    
    %% MODELOS EVALUADOS
    MODELS[🔤 MODELOS EVALUADOS<br/>• multi-qa-mpnet-base-dot-v1<br/>• all-MiniLM-L6-v2<br/>• text-embedding-ada-002]:::innovation
    
    MODELS --> COMPARISON_SYSTEM
    
    style TITLE fill:#fff9c4,stroke:#f57f17,stroke-width:4px,color:#000
```

## Conceptos Clave del Sistema de Comparación

### 🔵 **INPUT (Entrada)**
- Dataset de consultas técnicas Azure
- Mismo conjunto para todos los modelos
- Evaluación objetiva y controlada

### 🟣 **PROCESSING (Procesamiento)**
- Sistema de comparación paralela
- 3 modelos de embedding evaluados simultáneamente
- Pipeline RAG idéntico para cada modelo

### 🟠 **INNOVATION (Innovación)**
- Framework de evaluación comparativa
- Métricas RAG especializadas para cada modelo
- Análisis estadístico automatizado

### 🟢 **OUTPUT (Salida)**
- Ranking objetivo de modelos
- Métricas detalladas por modelo
- Reporte PDF profesional con conclusiones

### 🔤 **MODELOS EVALUADOS**
- **mpnet**: Especializado en Q&A
- **MiniLM**: Balance eficiencia/calidad
- **ada-002**: Referencia comercial

---

*Este diagrama ultra-simplificado introduce el concepto de evaluación comparativa antes de mostrar detalles técnicos.*