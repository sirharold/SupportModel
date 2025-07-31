# 📈 Métricas Acumulativas - Documentación Actualizada

## Descripción

El sistema de **Métricas Acumulativas** permite evaluar múltiples preguntas de forma automática, calcular métricas promedio y analizar el rendimiento del sistema RAG a gran escala. Incluye integración con Google Colab para procesamiento con GPU y análisis avanzado con métricas RAGAS y BERTScore.

## Arquitectura del Sistema

### 🏗️ **Componentes Principales**

1. **Configuración Local (Streamlit)**
   - `cumulative_n_questions_config.py`: Configuración y selección de preguntas
   - `cumulative_metrics_results.py`: Visualización de resultados
   - Integración con Google Drive para transferencia de datos

2. **Procesamiento en Colab**
   - `Colab_Modular_Embeddings_Evaluation.ipynb`: Notebook principal
   - Procesamiento con GPU para acelerar cálculos
   - Evaluación con múltiples modelos de embedding

3. **Biblioteca Externa**
   - `colab_data/lib/rag_evaluation.py`: Clases y funciones reutilizables
   - Implementaciones optimizadas de métricas

## Características Principales

### 🎯 **Filtrado Inteligente Mejorado**
- **Validación de links en documentos**: Solo se seleccionan preguntas cuyos links existen en la colección de documentos
- **Normalización de URLs**: Elimina parámetros y anchors para comparación precisa
- **~2,067 preguntas válidas**: De ~15,000 totales, solo estas tienen links verificados
- **Selección reproducible**: Seed fijo (42) para resultados consistentes

### 📊 **Métricas Calculadas**

#### Métricas IR Tradicionales
- **Precision@K**: Proporción de documentos relevantes en top-K (K=1,2,3,4,5,6,7,8,9,10)
- **Recall@K**: Cobertura de documentos relevantes
- **F1@K**: Balance entre precisión y recall
- **MAP@K**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain

#### Métricas RAGAS (0-1 scale)
- **Faithfulness**: Fidelidad de la respuesta al contexto (sin alucinaciones)
- **Answer Relevancy**: Relevancia de la respuesta a la pregunta
- **Answer Correctness**: Exactitud factual y completitud
- **Semantic Similarity**: Similitud semántica con respuesta esperada
- **Context Precision**: Calidad del ranking de documentos relevantes
- **Context Recall**: Cobertura del contexto necesario

#### Métricas BERTScore (0-1 scale)
- **BERT Precision**: Precisión a nivel de tokens usando embeddings contextuales
- **BERT Recall**: Cobertura a nivel de tokens
- **BERT F1**: Media armónica de precision y recall

### 🔄 **Agregación de Documentos**
- **Conversión chunks → documentos**: Combina chunks del mismo documento
- **Multiplicador configurable**: Por defecto 3x chunks para asegurar cobertura
- **Preservación de metadatos**: Mantiene título, link y contenido original

### 📊 **Límites de Contenido Optimizados**
- **Generación de Respuestas**: 2000 caracteres por documento (antes 500)
- **Contexto RAGAS**: 3000 caracteres por documento (antes 1000)
- **Reranking LLM**: 4000 caracteres por documento (antes 3000)
- **Evaluación BERTScore**: Sin límite - contenido completo

## Configuración Mejorada

### Parámetros de Evaluación

| Parámetro | Valor Por Defecto | Descripción |
|-----------|-------------------|-------------|
| **Número de preguntas** | 100 | Cantidad a evaluar (máx: 2,067 con links válidos) |
| **Modelos de Embedding** | Múltiples | mpnet, minilm, ada, e5-large |
| **Top-K documentos** | 10 | Documentos a recuperar |
| **LLM Reranking** | Habilitado | CrossEncoder MS-MARCO |
| **Modelo Generativo** | tinyllama-1.1b | Para evaluación RAGAS |
| **Agregación Documentos** | Habilitada | Chunks → documentos completos |

### Proceso de Selección de Preguntas

```python
# 1. Cargar links de documentos
doc_links = obtener_links_normalizados(docs_collection)

# 2. Filtrar preguntas con links válidos
preguntas_validas = []
for pregunta in todas_las_preguntas:
    if tiene_link_en_documentos(pregunta, doc_links):
        preguntas_validas.append(pregunta)

# 3. Selección aleatoria reproducible
random.seed(42)
preguntas_seleccionadas = random.sample(preguntas_validas, n)
```

## Flujo de Trabajo Actualizado

### 1. **Configuración en Streamlit**
```
1. Filtrar preguntas con links válidos (~2,067)
2. Seleccionar N preguntas aleatoriamente
3. Configurar modelos y parámetros
4. Generar archivo de configuración JSON
5. Subir a Google Drive
```

### 2. **Procesamiento en Colab**
```
1. Cargar configuración desde Google Drive
2. Inicializar modelos con GPU
3. Para cada pregunta y modelo:
   - Recuperar documentos (con agregación)
   - Aplicar reranking si está habilitado
   - Generar respuesta
   - Calcular todas las métricas
4. Guardar resultados en Drive
```

### 3. **Visualización de Resultados**
```
1. Cargar resultados desde Google Drive
2. Mostrar resumen de evaluación
3. Visualizar comparación entre modelos
4. Aplicar color-coding a métricas:
   - Verde: >0.8 (Excelente)
   - Amarillo: 0.6-0.8 (Bueno)
   - Rojo: <0.6 (Necesita mejora)
```

## Interpretación de Resultados

### 📊 **Rangos de Interpretación Unificados**

Para RAGAS y BERTScore (escala 0-1):
- **0.8-1.0**: 🟢 Excelente rendimiento
- **0.6-0.8**: 🟡 Buen rendimiento
- **0.4-0.6**: 🟠 Rendimiento moderado
- **< 0.4**: 🔴 Necesita mejoras

### 🔍 **Análisis de Métricas Específicas**

**Context Precision/Recall**:
- Evalúan la calidad del sistema de recuperación
- Valores bajos indican problemas en el retrieval

**Faithfulness**:
- Mide alucinaciones en las respuestas
- Crítico para aplicaciones de alta confiabilidad

**BERTScore**:
- Evaluación semántica profunda
- Más robusto que métricas léxicas tradicionales

## Mejoras Implementadas

### ✅ **Calidad de Datos**
- Filtrado inteligente de preguntas con validación de links
- Normalización de URLs para comparación precisa
- Solo preguntas con ground truth verificado

### ✅ **Procesamiento Optimizado**
- Agregación de chunks en documentos completos
- Límites de contenido aumentados para mejor contexto
- Procesamiento paralelo en Colab con GPU

### ✅ **Visualización Mejorada**
- Color-coding automático para interpretación rápida
- Tablas interactivas con definiciones de métricas
- Gráficos comparativos multi-modelo

### ✅ **Evaluación Completa**
- 16 métricas diferentes por pregunta
- Análisis antes/después del reranking
- Métricas tanto de recuperación como de generación

## Archivos Clave del Sistema

### 📁 **Configuración y UI**
- `src/apps/cumulative_n_questions_config.py`: Configuración y filtrado
- `src/apps/cumulative_metrics_results.py`: Visualización de resultados
- `src/ui/enhanced_metrics_display.py`: Funciones de display mejoradas

### 🔧 **Procesamiento**
- `colab_data/Colab_Modular_Embeddings_Evaluation.ipynb`: Notebook principal
- `colab_data/lib/rag_evaluation.py`: Biblioteca de evaluación
- `src/core/document_processor.py`: Agregación de documentos

### 📊 **Datos**
- ChromaDB: Colecciones de preguntas y documentos
- Google Drive: Almacenamiento de configuraciones y resultados
- JSON: Formato de intercambio de datos

## Ejemplo de Uso Completo

```python
# 1. En Streamlit - Configuración
- Seleccionar 500 preguntas
- Habilitar todos los modelos (mpnet, minilm, ada, e5-large)
- Activar reranking y agregación de documentos
- Crear configuración y subir a Drive

# 2. En Colab - Procesamiento
!python -m pip install -r requirements.txt
# Ejecutar notebook con la configuración
# Proceso toma ~45-60 minutos para 500 preguntas

# 3. En Streamlit - Resultados
- Ver resumen: 4 modelos evaluados
- Comparar métricas con color-coding
- Analizar mejoras por reranking
- Exportar resultados
```

## Consideraciones de Rendimiento

### ⏱️ **Tiempos Estimados**
- **100 preguntas**: ~10-15 minutos
- **500 preguntas**: ~45-60 minutos
- **1000 preguntas**: ~90-120 minutos

### 🚀 **Optimizaciones**
- Uso de GPU en Colab (10x más rápido)
- Batch processing para embeddings
- Cache de modelos pre-cargados
- Procesamiento paralelo cuando es posible

## Troubleshooting

### ❌ **Problemas Comunes**

1. **"No se encontraron preguntas con links válidos"**
   - Verificar que la colección de documentos tenga datos
   - Revisar que los links estén normalizados correctamente

2. **"Error de memoria en Colab"**
   - Reducir batch_size en la configuración
   - Procesar menos preguntas por vez

3. **"Métricas faltantes o en cero"**
   - Verificar que los documentos tengan contenido
   - Revisar logs de errores en Colab

## Próximas Mejoras Planificadas

1. **Análisis Estadístico**
   - Intervalos de confianza para métricas
   - Tests de significancia entre modelos
   - Análisis de varianza

2. **Optimizaciones**
   - Cache distribuido de embeddings
   - Procesamiento incremental
   - Paralelización mejorada

3. **Nuevas Métricas**
   - Diversidad de resultados
   - Latencia de respuesta
   - Costo computacional

---

**Última actualización**: Diciembre 2024
**Versión**: 2.0 (con agregación de documentos y filtrado inteligente)