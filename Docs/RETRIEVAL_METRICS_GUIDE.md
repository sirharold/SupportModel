# 📊 Guía de Métricas de Recuperación para Sistema RAG

## 📋 Descripción General

Este sistema incluye métricas especializadas para evaluar la calidad de recuperación de documentos antes y después del reranking. Las métricas implementadas son estándar en sistemas de recuperación de información y permiten una evaluación objetiva del rendimiento del sistema RAG.

## 🎯 Métricas Implementadas

### 1. **MRR (Mean Reciprocal Rank)**
- **Definición**: Posición promedio del primer documento relevante
- **Fórmula**: `MRR = 1 / rank_of_first_relevant_document`
- **Rango**: [0, 1] donde 1 es perfecto (primer documento es relevante)
- **Interpretación**: Mayor es mejor

### 2. **Recall@k**
- **Definición**: Fracción de documentos relevantes recuperados en los top k
- **Fórmula**: `Recall@k = |documentos_relevantes_en_top_k| / |total_documentos_relevantes|`
- **Rango**: [0, 1] donde 1 es perfecto (todos los relevantes recuperados)
- **Interpretación**: Mayor es mejor

### 3. **Precision@k**
- **Definición**: Fracción de documentos recuperados que son relevantes en los top k
- **Fórmula**: `Precision@k = |documentos_relevantes_en_top_k| / k`
- **Rango**: [0, 1] donde 1 es perfecto (todos los recuperados son relevantes)
- **Interpretación**: Mayor es mejor

### 4. **F1@k**
- **Definición**: Media armónica de Precision@k y Recall@k
- **Fórmula**: `F1@k = 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)`
- **Rango**: [0, 1] donde 1 es perfecto
- **Interpretación**: Mayor es mejor

### 5. **Accuracy@k**
- **Definición**: Proporción de documentos correctamente clasificados (relevantes/no relevantes) en los top k
- **Fórmula**: `Accuracy@k = (TP + TN) / (TP + TN + FP + FN)`
- **Rango**: [0, 1] donde 1 es perfecto
- **Interpretación**: Mayor es mejor

### 6. **BinaryAccuracy@k**
- **Definición**: Proporción de predicciones correctas en los top k (equivalente a Precision@k)
- **Fórmula**: `BinaryAccuracy@k = documentos_relevantes_en_top_k / k`
- **Rango**: [0, 1] donde 1 es perfecto
- **Interpretación**: Mayor es mejor

### 7. **RankingAccuracy@k**
- **Definición**: Qué tan bien el sistema rankea documentos relevantes vs no relevantes
- **Fórmula**: Proporción de pares (relevante, no_relevante) donde relevante aparece antes
- **Rango**: [0, 1] donde 1 es perfecto
- **Interpretación**: Mayor es mejor

## 📈 Valores de k Evaluados

El sistema evalúa automáticamente las métricas para:
- **k=1**: Solo el primer documento (más estricto)
- **k=3**: Top 3 documentos (uso típico)
- **k=5**: Top 5 documentos (balanceado)
- **k=10**: Top 10 documentos (más permisivo)

## 🔧 Uso Básico

### 1. Métricas para Una Pregunta Individual

```python
from utils.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from utils.clients import initialize_clients

# Inicializar clientes
weaviate_wrapper, embedding_client, openai_client, gemini_client, local_llama_client, local_mistral_client, _ = initialize_clients("multi-qa-mpnet-base-dot-v1")

# Ejecutar pipeline con métricas
result = answer_question_with_retrieval_metrics(
    question="¿Cómo configurar Azure Blob Storage?",
    weaviate_wrapper=weaviate_wrapper,
    embedding_client=embedding_client,
    openai_client=openai_client,
    gemini_client=gemini_client,
    local_llama_client=local_llama_client,
    local_mistral_client=local_mistral_client,
    top_k=10,
    use_llm_reranker=True,
    generate_answer=False,  # Solo documentos para métricas
    calculate_metrics=True,
    ground_truth_answer=ground_truth_answer,
    ms_links=ms_links
)

# Extraer métricas
docs, debug_info, retrieval_metrics = result
```

### 2. Mostrar Métricas Formateadas

```python
from utils.retrieval_metrics import format_metrics_for_display

# Formatear para display
formatted_output = format_metrics_for_display(retrieval_metrics)
print(formatted_output)
```

### 3. Evaluación en Lotes

```python
from utils.qa_pipeline_with_metrics import batch_calculate_retrieval_metrics

# Preparar datos
questions_and_answers = [
    {
        'question': "¿Cómo crear una VM en Azure?",
        'accepted_answer': "Para crear una VM...",
        'ms_links': ["https://learn.microsoft.com/azure/virtual-machines/..."]
    },
    # ... más preguntas
]

# Calcular métricas para todas las preguntas
all_metrics = batch_calculate_retrieval_metrics(
    questions_and_answers=questions_and_answers,
    weaviate_wrapper=weaviate_wrapper,
    embedding_client=embedding_client,
    openai_client=openai_client,
    top_k=10,
    use_llm_reranker=True
)

# Mostrar resumen
from utils.qa_pipeline_with_metrics import print_batch_metrics_summary
print_batch_metrics_summary(all_metrics)
```

## 📊 Interpretación de Resultados

### Ejemplo de Salida

```
📊 MÉTRICAS DE RECUPERACIÓN - COMPARACIÓN BEFORE/AFTER RERANKING
================================================================================
Ground Truth Links: 3
Docs Before: 10, Docs After: 10
--------------------------------------------------------------------------------
Métrica         Before     After      Mejora     % Mejora  
--------------------------------------------------------------------------------
MRR             0.3333     1.0000     0.6667     200.00    %
Recall@1        0.0000     0.3333     0.3333     0.00      %
Precision@1     0.0000     1.0000     1.0000     0.00      %
F1@1            0.0000     0.5000     0.5000     0.00      %
--------------------------------------------------
Recall@5        0.6667     1.0000     0.3333     50.00     %
Precision@5     0.4000     0.6000     0.2000     50.00     %
F1@5            0.5000     0.7500     0.2500     50.00     %
```

### Análisis de Ejemplo

- **MRR mejoró 200%**: El reranking movió el primer documento relevante de la posición 3 a la posición 1
- **Recall@5 mejoró 50%**: Se recuperaron más documentos relevantes en el top 5
- **Precision@5 mejoró 50%**: Mayor proporción de documentos relevantes en el top 5
- **F1@5 balanceó ambos**: Mejora combinada en precisión y recall

## 🎯 Casos de Uso en el Proyecto

### 1. **Evaluación de Modelos de Embedding**
```python
# Comparar diferentes modelos
for model_key in ["multi-qa-mpnet-base-dot-v1", "all-MiniLM-L6-v2", "ada"]:
    # Calcular métricas para cada modelo
    # Comparar resultados
```

### 2. **Impacto del Reranking**
```python
# Evaluar con y sin reranking
metrics_with_reranking = calculate_metrics(use_llm_reranker=True)
metrics_without_reranking = calculate_metrics(use_llm_reranker=False)
```

### 3. **Optimización de Hiperparámetros**
```python
# Probar diferentes valores de top_k
for top_k in [5, 10, 15, 20]:
    # Calcular métricas para cada top_k
    # Encontrar valor óptimo
```

## 🔍 Validación de Ground Truth

### Extracción Automática de Enlaces

El sistema extrae automáticamente enlaces de Microsoft Learn de las respuestas aceptadas:

```python
from utils.retrieval_metrics import extract_ground_truth_links

# Extraer ground truth de respuesta
ground_truth_links = extract_ground_truth_links(
    ground_truth_answer="Para configurar Azure Storage... https://learn.microsoft.com/azure/storage/...",
    ms_links=None  # Se extraen automáticamente
)
```

### Validación Manual

También puedes proporcionar enlaces manualmente:

```python
ms_links = [
    "https://learn.microsoft.com/azure/storage/common/storage-account-create",
    "https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction"
]
```

## 📈 Integración con Streamlit

Las métricas se pueden mostrar en la interfaz web:

```python
from utils.comparison_with_retrieval_metrics import show_retrieval_metrics_comparison

# En tu página de Streamlit
show_retrieval_metrics_comparison(
    question=question,
    selected_question=selected_question,
    top_k=10,
    use_reranker=True
)
```

## 🧪 Testing y Validación

### Ejecutar Tests

```bash
# Tests unitarios
python test_retrieval_metrics.py

# Demo completa
python demo_retrieval_metrics.py
```

### Casos de Prueba

Los tests incluyen:
- ✅ Métricas básicas con datos sintéticos
- ✅ Comparación before/after reranking
- ✅ Casos edge (sin documentos, sin ground truth, etc.)
- ✅ Validación de fórmulas matemáticas

## 📊 Métricas Agregadas

Para evaluaciones en lotes, el sistema calcula:

```python
from utils.retrieval_metrics import calculate_aggregated_metrics

# Calcular estadísticas agregadas
aggregated = calculate_aggregated_metrics(all_metrics)

# Disponible:
# - mean: Promedio
# - median: Mediana
# - std: Desviación estándar
# - min/max: Valores extremos
```

## 🎨 Visualización

El sistema incluye gráficos interactivos:

- **Heatmaps**: Mejoras por modelo y métrica
- **Barras comparativas**: Before vs After
- **Líneas de tendencia**: Evolución por valor de k
- **Rankings**: Mejor modelo por métrica

## 📋 Checklist de Implementación

Para usar las métricas en tu proyecto:

- [ ] Configura conexión a Weaviate
- [ ] Prepara datos de ground truth (respuestas aceptadas)
- [ ] Importa `answer_question_with_retrieval_metrics`
- [ ] Ejecuta con `calculate_metrics=True`
- [ ] Usa `format_metrics_for_display` para mostrar resultados
- [ ] Implementa en interfaz web con Streamlit
- [ ] Ejecuta tests para validar funcionamiento

## 🔗 Referencias

- Voorhees, E. M. (1999). The TREC-8 question answering track evaluation. TREC.
- Manning, C. D., et al. (2008). Introduction to Information Retrieval. Cambridge University Press.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.

## 🆘 Troubleshooting

### Problemas Comunes

1. **Error: No ground truth links found**
   - Verifica que la respuesta aceptada contenga enlaces de Microsoft Learn
   - Usa el parámetro `ms_links` para proporcionar enlaces manualmente

2. **Métricas todas en 0**
   - Revisa que los documentos recuperados contengan el campo `link`
   - Verifica que los enlaces coincidan exactamente con el ground truth

3. **Error de importación**
   - Asegúrate de que todos los módulos estén en el PYTHONPATH
   - Instala dependencias: `pandas`, `numpy`, `plotly`

4. **Rendimiento lento**
   - Reduce `top_k` para evaluaciones rápidas
   - Usa `generate_answer=False` para solo calcular métricas de recuperación
   - Implementa caché para resultados repetidos