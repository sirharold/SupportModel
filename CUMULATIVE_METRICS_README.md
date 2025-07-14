# 📈 Métricas Acumulativas - Documentación

## Descripción

La página de **Métricas Acumulativas** permite evaluar múltiples preguntas de forma automática y calcular métricas promedio para obtener una visión estadística del rendimiento del sistema de recuperación de información.

## Características Principales

### 🎯 **Filtrado Inteligente**
- **Solo preguntas con links**: Se evalúan únicamente preguntas que contienen enlaces de Microsoft Learn en la respuesta aceptada
- **Extracción automática**: Los links se extraen automáticamente usando regex pattern: `https://learn\.microsoft\.com[\w/\-\?=&%\.]+`
- **Validación de calidad**: Asegura que cada pregunta tenga ground truth válido

### 📊 **Métricas Calculadas**
- **Precision@5**: Precisión en los primeros 5 documentos
- **Recall@5**: Cobertura en los primeros 5 documentos  
- **F1@5**: Balance entre precisión y cobertura
- **MRR@5**: Mean Reciprocal Rank en top-5
- **nDCG@5**: Normalized Discounted Cumulative Gain

### 🔄 **Antes y Después del Reranking**
- **Métricas base**: Resultados usando solo similarity search
- **Métricas post-LLM**: Resultados después del reranking con GPT-4
- **Comparación visual**: Gráficos que muestran la mejora/deterioro
- **Cálculo de delta**: Diferencia entre antes y después

## Configuración

### Parámetros Principales

| Parámetro | Valor Por Defecto | Descripción |
|-----------|-------------------|-------------|
| **Número de preguntas** | 5 | Cantidad de preguntas a evaluar (rango: 1-50) |
| **Modelo de Embedding** | multi-qa-mpnet-base-dot-v1 | Modelo para generar embeddings |
| **Top-K documentos** | 10 | Número de documentos a recuperar |
| **LLM Reranking** | Habilitado | Usar GPT-4 para reordenar documentos |

### Fuente de Datos
- **Archivo**: `data/val_set.json` (1,035 preguntas)
- **Formato**: JSON con structure `{title, question_content, accepted_answer, tags, url}`
- **Filtrado**: Solo preguntas con enlaces de Microsoft Learn

## Uso Paso a Paso

### 1. **Configuración**
```python
# En la interfaz de Streamlit:
- Seleccionar número de preguntas (1-50)
- Elegir modelo de embedding
- Configurar Top-K documentos
- Habilitar/deshabilitar LLM reranking
```

### 2. **Ejecución**
```python
# Al hacer clic en "🚀 Ejecutar Evaluación":
1. Carga preguntas desde val_set.json
2. Filtra preguntas con links de MS Learn
3. Selecciona N preguntas aleatoriamente
4. Evalúa cada pregunta individualmente
5. Calcula métricas promedio
```

### 3. **Resultados**
```python
# Visualización de resultados:
- Métricas promedio en columnas
- Gráfico comparativo (antes vs después)
- Tabla detallada por pregunta
- Estadísticas de evaluación
```

## Algoritmo de Cálculo

### Métricas Promedio
```python
def calculate_average_metrics(all_metrics):
    metric_sums = {}
    metric_counts = {}
    
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_sums[key] = metric_sums.get(key, 0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
    
    return {key: metric_sums[key] / metric_counts[key] 
            for key in metric_sums if metric_counts[key] > 0}
```

### Ejemplo de Cálculo
```python
# Si F1@5 en 3 preguntas es: [0.3, 0.4, 0.5]
# Promedio = (0.3 + 0.4 + 0.5) / 3 = 0.4

# Si una pregunta falla y no tiene F1@5:
# F1@5 en 3 preguntas: [0.3, NaN, 0.5]
# Promedio = (0.3 + 0.5) / 2 = 0.4
```

## Interpretación de Resultados

### 📊 **Métricas Principales**
- **Precision@5 > 0.5**: Buena precisión en top-5
- **Recall@5 > 0.3**: Buena cobertura del ground truth
- **F1@5 > 0.4**: Buen balance precision/recall
- **MRR@5 > 0.6**: Primer resultado relevante aparece temprano
- **nDCG@5 > 0.5**: Buen ranking general

### 🔄 **Impacto del Reranking**
- **Delta positivo**: El reranking LLM mejora la métrica
- **Delta negativo**: El reranking LLM empeora la métrica
- **Delta ~0**: El reranking no tiene impacto significativo

## Exportación de Datos

### 📋 **CSV Detallado**
```csv
question_num,ground_truth_links,docs_retrieved,before_precision_5,after_precision_5,...
1,3,10,0.400,0.600,...
2,2,10,0.200,0.400,...
```

### 📊 **CSV Promedio**
```csv
Metric,Before_Reranking,After_Reranking
Precision@5,0.350,0.450
Recall@5,0.280,0.380
F1@5,0.310,0.410
```

## Casos de Uso

### 🔬 **Evaluación de Modelos**
```python
# Comparar diferentes modelos de embedding:
1. Ejecutar con multi-qa-mpnet-base-dot-v1
2. Ejecutar con all-MiniLM-L6-v2
3. Comparar métricas promedio
```

### 📈 **Análisis de Rendimiento**
```python
# Evaluar impacto del reranking:
1. Ejecutar con reranking habilitado
2. Ejecutar con reranking deshabilitado
3. Analizar diferencias en métricas
```

### 🎯 **Optimización de Parámetros**
```python
# Encontrar mejor configuración:
1. Probar diferentes valores de Top-K
2. Evaluar con diferentes números de preguntas
3. Seleccionar configuración óptima
```

## Limitaciones

### ⚠️ **Consideraciones**
- **Tiempo de ejecución**: ~2-5 segundos por pregunta con reranking
- **Dependencia de GPT-4**: Reranking requiere acceso a OpenAI API
- **Selección aleatoria**: Resultados pueden variar entre ejecuciones
- **Sesgo del dataset**: Solo preguntas con links de MS Learn

### 🔧 **Recomendaciones**
- **Usar 5-10 preguntas** para pruebas rápidas
- **Usar 20-50 preguntas** para evaluaciones más robustas
- **Ejecutar múltiples veces** para obtener intervalos de confianza
- **Comparar métricas** antes y después del reranking

## Integración

### 📁 **Archivos Principales**
- `cumulative_metrics_page.py`: Lógica principal de la página
- `utils/qa_pipeline_with_metrics.py`: Pipeline con cálculo de métricas
- `data/val_set.json`: Dataset de preguntas de validación

### 🔗 **Dependencias**
- `streamlit`: Interfaz de usuario
- `pandas`: Manipulación de datos
- `numpy`: Cálculos numéricos
- `plotly`: Visualización de datos
- `json`: Carga de dataset
- `re`: Extracción de links

---

## Próximas Mejoras

1. **Intervalos de confianza**: Calcular IC para métricas promedio
2. **Análisis estadístico**: Pruebas de significancia entre configuraciones
3. **Más métricas**: MAP, NDCG@10, Hit Rate
4. **Cache de resultados**: Evitar re-evaluación de preguntas
5. **Exportación avanzada**: Informes PDF con visualizaciones