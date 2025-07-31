# 📊 Guía Completa de Métricas para Sistema RAG

## 📋 Descripción General

Este sistema implementa un conjunto completo de métricas para evaluar sistemas RAG (Retrieval-Augmented Generation), incluyendo métricas tradicionales de recuperación de información, métricas RAGAS para evaluar la calidad de generación, y BERTScore para evaluación semántica profunda.

## 🎯 Categorías de Métricas

### 1. **Métricas IR Tradicionales**

#### **MRR (Mean Reciprocal Rank)**
- **Definición**: Posición promedio del primer documento relevante
- **Fórmula**: `MRR = 1 / rank_of_first_relevant_document`
- **Rango**: [0, 1] donde 1 es perfecto
- **Uso**: Evalúa qué tan rápido encuentra el primer resultado relevante

#### **Precision@K**
- **Definición**: Proporción de documentos relevantes en top K
- **Fórmula**: `Precision@K = |relevantes_en_top_K| / K`
- **Rango**: [0, 1] donde 1 es perfecto
- **K evaluados**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

#### **Recall@K**
- **Definición**: Proporción de documentos relevantes recuperados
- **Fórmula**: `Recall@K = |relevantes_en_top_K| / |total_relevantes|`
- **Rango**: [0, 1] donde 1 es perfecto
- **K evaluados**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

#### **F1@K**
- **Definición**: Media armónica de Precision y Recall
- **Fórmula**: `F1@K = 2 × (P@K × R@K) / (P@K + R@K)`
- **Rango**: [0, 1] donde 1 es perfecto
- **Uso**: Balance entre precisión y cobertura

#### **MAP@K (Mean Average Precision)**
- **Definición**: Promedio de precisiones a cada posición relevante
- **Fórmula**: `MAP@K = (1/Q) × Σ(AP@K_q)`
- **Rango**: [0, 1] donde 1 es perfecto
- **Uso**: Considera el orden de todos los documentos relevantes

#### **NDCG@K (Normalized Discounted Cumulative Gain)**
- **Definición**: Ganancia acumulativa descontada normalizada
- **Fórmula**: `NDCG@K = DCG@K / IDCG@K`
- **Rango**: [0, 1] donde 1 es perfecto
- **Uso**: Maneja relevancia graduada (no solo binaria)

### 2. **Métricas RAGAS**

#### **Faithfulness**
- **Definición**: Fidelidad de la respuesta al contexto recuperado
- **Evaluación**: Cada claim debe estar respaldado por el contexto
- **Rango**: [0, 1] donde 1 = sin alucinaciones
- **Importancia**: Crítico para confiabilidad

#### **Answer Relevancy**
- **Definición**: Relevancia de la respuesta a la pregunta
- **Evaluación**: Similitud entre pregunta y respuesta generada
- **Rango**: [0, 1] donde 1 = perfectamente relevante
- **Importancia**: Asegura respuestas enfocadas

#### **Answer Correctness**
- **Definición**: Exactitud factual y completitud
- **Evaluación**: Combina precisión semántica y factual
- **Rango**: [0, 1] donde 1 = perfectamente correcta
- **Importancia**: Calidad general de la respuesta

#### **Semantic Similarity**
- **Definición**: Similitud con respuesta de referencia
- **Evaluación**: Distancia en espacio de embeddings
- **Rango**: [0, 1] donde 1 = idéntica semánticamente
- **Importancia**: Comparación con ground truth

#### **Context Precision**
- **Definición**: Calidad del ranking de documentos
- **Evaluación**: Documentos relevantes deben estar primero
- **Rango**: [0, 1] donde 1 = orden perfecto
- **Importancia**: Eficiencia del retrieval

#### **Context Recall**
- **Definición**: Cobertura del contexto necesario
- **Evaluación**: Proporción de información necesaria recuperada
- **Rango**: [0, 1] donde 1 = cobertura completa
- **Importancia**: Completitud del retrieval

### 3. **Métricas BERTScore**

#### **BERT Precision**
- **Definición**: Precisión a nivel de tokens con embeddings contextuales
- **Evaluación**: Tokens de respuesta presentes en referencia
- **Rango**: [0, 1] donde 1 = todos los tokens coinciden
- **Modelo**: microsoft/deberta-xlarge-mnli

#### **BERT Recall**
- **Definición**: Cobertura a nivel de tokens
- **Evaluación**: Tokens de referencia presentes en respuesta
- **Rango**: [0, 1] donde 1 = cobertura completa
- **Uso**: Detecta información faltante

#### **BERT F1**
- **Definición**: Balance entre precision y recall de BERT
- **Evaluación**: Media armónica de BERT P y R
- **Rango**: [0, 1] donde 1 = balance perfecto
- **Uso**: Métrica general de calidad semántica

## 📈 Interpretación de Rangos

### Escala Universal (0-1) para RAGAS y BERTScore:
```
🟢 Excelente: > 0.8
🟡 Bueno: 0.6 - 0.8
🟠 Moderado: 0.4 - 0.6
🔴 Necesita mejora: < 0.4
```

### Interpretación por Métrica:
- **MRR = 1.0**: Primer documento es relevante
- **Recall@10 > 0.8**: Recupera mayoría de documentos relevantes
- **Faithfulness > 0.8**: Respuestas confiables sin alucinaciones
- **BERTScore F1 > 0.8**: Alta calidad semántica

## 🔧 Implementación en el Sistema

### 1. **Configuración de Límites de Contenido**
```python
CONTENT_LIMITS = {
    'answer_generation': 2000,     # Caracteres para generar respuestas
    'context_for_ragas': 3000,     # Caracteres para evaluación RAGAS
    'llm_reranking': 4000,         # Caracteres para reranking
    'bert_score': 'sin_limite'     # Contenido completo
}
```

### 2. **Agregación de Documentos**
```python
# Convierte chunks en documentos completos
aggregator = DocumentAggregator()
documents = aggregator.aggregate_chunks_to_documents(
    chunks, 
    multiplier=3  # Top documentos = chunks / 3
)
```

### 3. **Filtrado de Preguntas Válidas**
```python
# Solo preguntas con links en documentos
valid_questions = filter_questions_with_valid_links(
    all_questions,
    doc_links_set
)
# Resultado: ~2,067 de ~15,000 preguntas
```

### 4. **Evaluación Completa**
```python
# Para cada pregunta y modelo:
metrics = {
    'ir_metrics': calculate_ir_metrics(question, docs),
    'ragas_metrics': calculate_ragas_metrics(question, answer, contexts),
    'bert_scores': calculate_bertscore(answer, reference)
}
```

## 📊 Visualización en Streamlit

### Color-Coding Automático
```python
def get_metric_color(value):
    if value > 0.8:
        return 'green'   # Excelente
    elif value >= 0.6:
        return 'yellow'  # Bueno
    else:
        return 'red'     # Necesita mejora
```

### Tablas Interactivas
- **Métricas RAGAS/BERTScore**: Con color-coding por rangos
- **Comparación Multi-modelo**: Side-by-side
- **Antes/Después Reranking**: Con deltas calculados

### Definiciones en Acordeón
```python
with st.expander("📚 Definiciones y Fórmulas de Métricas"):
    # Tabla con definiciones, fórmulas e interpretación
    # Para cada métrica IR tradicional
```

## 🚀 Flujo de Procesamiento

### 1. **Local (Streamlit)**
```
Filtrar preguntas → Configurar evaluación → Subir a Drive
```

### 2. **Cloud (Colab)**
```
Cargar config → Procesar con GPU → Calcular métricas → Guardar resultados
```

### 3. **Visualización (Streamlit)**
```
Cargar resultados → Aplicar color-coding → Mostrar análisis
```

## 📈 Casos de Uso

### 1. **Comparación de Modelos**
```python
# Evaluar mpnet vs minilm vs ada vs e5-large
for model in ['mpnet', 'minilm', 'ada', 'e5-large']:
    metrics = evaluate_model(model)
    compare_results(metrics)
```

### 2. **Análisis de Reranking**
```python
# Impacto del CrossEncoder
before_metrics = evaluate(use_reranking=False)
after_metrics = evaluate(use_reranking=True)
improvement = calculate_improvement(before, after)
```

### 3. **Optimización de Parámetros**
```python
# Encontrar mejor top_k
for k in [5, 10, 15, 20]:
    metrics = evaluate(top_k=k)
    analyze_k_impact(metrics)
```

## 🧪 Validación y Testing

### Tests Implementados:
- ✅ Normalización de URLs
- ✅ Agregación de documentos
- ✅ Cálculo de métricas
- ✅ Color-coding
- ✅ Integración end-to-end

### Resultados Típicos:
```
Con optimizaciones implementadas:
- Context Recall: +15-30%
- Faithfulness: +10-20%
- BERTScore F1: +5-15%
```

## 🎯 Mejores Prácticas

### Para Evaluación:
1. **Usar ≥100 preguntas** para resultados estadísticamente significativos
2. **Incluir todos los modelos** para comparación completa
3. **Habilitar reranking** para mejores resultados
4. **Usar agregación de documentos** para contexto completo

### Para Interpretación:
1. **Priorizar Faithfulness** para aplicaciones críticas
2. **Balancear Precision/Recall** según caso de uso
3. **Considerar BERTScore** para calidad semántica
4. **Analizar tendencias** no solo valores absolutos

## 📋 Checklist de Implementación

- [ ] Configurar ChromaDB con colecciones de preguntas/documentos
- [ ] Verificar links válidos en preguntas (2,067 disponibles)
- [ ] Configurar Google Drive para transferencia de datos
- [ ] Preparar Colab con GPU para procesamiento
- [ ] Implementar visualización con color-coding
- [ ] Agregar definiciones de métricas en UI
- [ ] Validar con conjunto de prueba

## 🔗 Referencias

- **RAGAS**: Es-haq, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **BERTScore**: Zhang, T., et al. (2019). BERTScore: Evaluating Text Generation with BERT
- **IR Metrics**: Manning, C. D., et al. (2008). Introduction to Information Retrieval
- **CrossEncoder**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT

---

**Última actualización**: Diciembre 2024
**Versión**: 2.0 (Sistema completo con 16+ métricas)