# DIAGRAMA ULTRA-ALTO NIVEL - SISTEMA RAG AZURE
## Overview Conceptual para Defensa de Título

```mermaid
flowchart TD
    %% Definición de estilos
    classDef user fill:#e3f2fd,stroke:#0d47a1,stroke-width:3px,color:#000
    classDef system fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,color:#000
    classDef innovation fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    classDef results fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px,color:#000
    classDef arrow fill:none,stroke:#333,stroke-width:2px
    
    %% COMPONENTES PRINCIPALES
    USER[👤 USUARIO<br/>Consulta Técnica Azure]:::user
    
    RAG_SYSTEM[🤖 SISTEMA RAG<br/>Pipeline Inteligente<br/>6 Etapas de Procesamiento]:::system
    
    INNOVATION[💡 INNOVACIONES CLAVE<br/>• Métricas RAG Especializadas<br/>• Modelos Locales + Remotos<br/>• Evaluación Avanzada]:::innovation
    
    RESULTS[🎯 RESULTADOS<br/>• Respuestas Fundamentadas<br/>• Reducción de Costos<br/>• Métricas de Calidad]:::results
    
    %% FLUJO PRINCIPAL
    USER --> RAG_SYSTEM
    RAG_SYSTEM --> RESULTS
    
    %% INNOVACIÓN
    INNOVATION --> RAG_SYSTEM
    
    %% ETIQUETAS EXPLICATIVAS
    USER -.- LABEL1["Entrada:<br/>Preguntas en lenguaje natural<br/>sobre servicios Azure"]
    
    RAG_SYSTEM -.- LABEL2["Procesamiento:<br/>Refinamiento → Embedding → Búsqueda<br/>→ Reranking → Generación → Evaluación"]
    
    INNOVATION -.- LABEL3["Contribuciones:<br/>4 métricas RAG + arquitectura<br/>híbrida + optimización costos"]
    
    RESULTS -.- LABEL4["Salida:<br/>Respuestas precisas con fuentes<br/>+ análisis comparativo + métricas"]
    
    %% TÍTULO PRINCIPAL
    TITLE[🏆 SISTEMA EXPERTO RAG PARA CONSULTAS TÉCNICAS AZURE<br/>CON MÉTRICAS AVANZADAS DE EVALUACIÓN]
    
    TITLE --> USER
    
    %% BENEFICIOS DESTACADOS
    BENEFITS[✨ BENEFICIOS PRINCIPALES<br/>• 85% reducción costos<br/>• Evaluación especializada<br/>• Arquitectura escalable]:::results
    
    RESULTS --> BENEFITS
    
    style TITLE fill:#fff9c4,stroke:#f57f17,stroke-width:4px,color:#000
```

## Conceptos Clave del Sistema

### 🔵 **INPUT (Entrada)**
- Usuario con consulta técnica sobre Azure
- Lenguaje natural, sin restricciones

### 🟣 **PROCESSING (Procesamiento)**
- Sistema RAG con pipeline de 6 etapas
- Procesamiento inteligente y automatizado

### 🟠 **INNOVATION (Innovación)**
- Métricas RAG especializadas (contribución principal)
- Arquitectura híbrida local/remota
- Evaluación avanzada multi-dimensional

### 🟢 **OUTPUT (Salida)**
- Respuestas fundamentadas en documentación
- Análisis comparativo de modelos
- Métricas de calidad especializadas

---

*Este diagrama ultra-simplificado es ideal para introducir el concepto general del sistema antes de profundizar en detalles técnicos.*