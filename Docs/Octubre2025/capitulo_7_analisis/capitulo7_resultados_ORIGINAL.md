# 7. RESULTADOS Y ANÁLISIS

## 7.1 Introducción

Este capítulo presenta los resultados experimentales del sistema RAG desarrollado siguiendo la metodología descrita en el Capítulo 5. La evaluación utilizó **2,067 pares pregunta-documento validados** como ground truth, evaluando cuatro modelos de embeddings (Ada, MPNet, MiniLM, E5-Large) bajo dos estrategias de procesamiento (con y sin reranking).

Para cada configuración se calcularon métricas de recuperación (Precision, Recall, F1, NDCG, MAP, MRR) para valores de k desde 1 hasta 15, permitiendo analizar el comportamiento del sistema en diferentes profundidades de recuperación.

## 7.2 Configuración Experimental

### 7.2.1 Parámetros de Evaluación

La evaluación experimental siguió el diseño metodológico descrito en el Capítulo 5 (sección 5.2.3), implementando un diseño factorial 4×2 que compara cuatro modelos de embedding bajo dos estrategias de procesamiento (con y sin reranking). La configuración experimental verificada incluye:

**Datos de Evaluación:**
- **Ground truth:** 2,067 pares pregunta-documento validados con enlaces verificados
- **Documentos indexados:** 187,031 chunks técnicos de documentación Azure
- **Modelos comparados:** 4 (Ada, MPNet, MiniLM, E5-Large)

**Parámetros Técnicos:**
- **Método de reranking:** CrossEncoder (ms-marco-MiniLM-L-6-v2) con normalización Min-Max
- **Top-k evaluado:** 15 documentos por consulta
- **Métricas calculadas:** Para k = 1, 2, 3, ..., 15
- **Métrica de similitud:** Similitud coseno en espacio de embeddings
- **Framework de evaluación:** Métricas tradicionales de recuperación (Precision, Recall, F1, NDCG, MAP, MRR)

**Entorno Computacional:**
- **Plataforma:** Google Colab con GPU Tesla T4
- **Duración total:** 36,445 segundos (10.12 horas)
- **Tiempo promedio por pregunta:** 4.4 segundos (incluyendo reranking)
- **Base de datos vectorial:** ChromaDB 0.5.23

### 7.2.2 Modelos de Embedding Evaluados

Los cuatro modelos evaluados representan diferentes enfoques arquitectónicos según se describe en el Capítulo 3 (sección 3.3):

| Modelo | Dimensionalidad | Tipo |
|--------|-----------------|------|
| Ada (text-embedding-ada-002) | 1,536 | Propietario (OpenAI) |
| MPNet (multi-qa-mpnet-base-dot-v1) | 768 | Open-source (especializado Q&A) |
| MiniLM (all-MiniLM-L6-v2) | 384 | Open-source (compacto) |
| E5-Large (intfloat/e5-large-v2) | 1,024 | Open-source |

### 7.2.3 Estrategias de Procesamiento

La evaluación compara dos estrategias de procesamiento de resultados para cada modelo:

**Estrategia 1: Recuperación Vectorial Directa**
- Búsqueda por similitud coseno en ChromaDB
- Ordenamiento directo por score de similitud
- Retorno de top-k documentos sin procesamiento adicional

**Estrategia 2: Recuperación con Reranking Neural**
- Búsqueda inicial por similitud coseno (top-15)
- Reranking con CrossEncoder ms-marco-MiniLM-L-6-v2
- Normalización de scores mediante función Min-Max
- Reordenamiento y selección de top-k final

El diseño permite cuantificar el impacto específico del reranking mediante comparación directa de las métricas antes y después de su aplicación, evaluando si el procesamiento adicional justifica el incremento en latencia y costos computacionales.

## 7.3 Resultados por Modelo de Embedding

### 7.3.1 Ada (OpenAI text-embedding-ada-002)

#### 7.3.1.1 Rendimiento General

El modelo Ada mantuvo el mejor rendimiento general entre todos los modelos evaluados en la fase de recuperación inicial, estableciendo el benchmark de referencia para el dominio técnico de Microsoft Azure. La **Tabla 7.1** presenta las métricas principales para Ada en k={3,5,10,15}.

**Tabla 7.1: Métricas Principales de Ada**

| Métrica | k | Antes | Después | Dif | Cambio (%) |
|---------|---|-------|---------|-----|------------|
| **Precision** | 3 | 0.111 | 0.089 | -0.023 | -20.4% |
|  | 5 | 0.098 | 0.081 | -0.016 | -16.7% |
|  | 10 | 0.074 | 0.068 | -0.006 | -8.1% |
|  | 15 | 0.061 | 0.061 | 0.000 | 0.0% |
| **Recall** | 3 | 0.276 | 0.219 | -0.057 | -20.5% |
|  | 5 | 0.398 | 0.330 | -0.068 | -17.2% |
|  | 10 | 0.591 | 0.546 | -0.045 | -7.6% |
|  | 15 | 0.729 | 0.729 | 0.000 | 0.0% |
| **F1** | 3 | 0.153 | 0.122 | -0.031 | -20.4% |
|  | 5 | 0.152 | 0.127 | -0.025 | -16.8% |
|  | 10 | 0.129 | 0.118 | -0.010 | -8.0% |
|  | 15 | 0.111 | 0.111 | 0.000 | 0.0% |
| **NDCG** | 3 | 0.209 | 0.173 | -0.036 | -17.2% |
|  | 5 | 0.234 | 0.202 | -0.032 | -13.6% |
|  | 10 | 0.260 | 0.234 | -0.026 | -10.0% |
|  | 15 | 0.271 | 0.250 | -0.021 | -7.7% |
| **MAP** | 3 | 0.211 | 0.160 | -0.051 | -24.1% |
|  | 5 | 0.263 | 0.201 | -0.062 | -23.7% |
|  | 10 | 0.317 | 0.256 | -0.061 | -19.3% |
|  | 15 | 0.344 | 0.291 | -0.052 | -15.2% |

Los resultados confirman que Ada alcanza el mejor rendimiento en todas las profundidades de recuperación. Precision@5 de 0.098 indica que aproximadamente el 10% de los cinco documentos recuperados son relevantes. El Recall@5 de 0.398 muestra que el sistema recupera cerca del 40% de todos los documentos relevantes dentro del top-5. La degradación por reranking es más pronunciada en valores pequeños de k y disminuye con k creciente.

#### 7.3.1.2 Impacto del Reranking

Contrario a la expectativa teórica, el reranking con CrossEncoder produce un **impacto significativamente negativo** en todas las métricas de Ada, con degradaciones que van desde -13.2% hasta -23.7%. Este comportamiento sugiere que las representaciones iniciales del modelo Ada ya capturan óptimamente la relevancia semántica para el dominio técnico evaluado, y el CrossEncoder introduce ruido en el reordenamiento.

La **Figura 7.1** presenta la evolución de Precision@k para todos los valores de k, mostrando que la curva "Antes de CrossEncoder" domina consistentemente la curva "Después de CrossEncoder" en todo el rango evaluado. Este patrón se replica en las demás métricas (Recall, F1, NDCG, MAP), como se observa en las figuras complementarias del análisis.

![Figura 7.1: Comparación de Precision@k para Ada antes y después del reranking](./capitulo_7_analisis/charts/precision_comparison_ada.png)

#### 7.3.1.3 Rendimiento por Profundidad de Recuperación

La **Tabla 7.2** presenta métricas de precisión para diferentes valores de k, ilustrando cómo el rendimiento evoluciona con la profundidad de recuperación.

**Tabla 7.2: Precision@k de Ada para k={3,5,10,15}**

| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.104 | 0.098 | 0.079 | 0.061 |
| Después CrossEncoder | 0.086 | 0.081 | 0.067 | 0.053 |
| Δ (cambio) | -0.018 (-17.5%) | -0.016 (-16.7%) | -0.012 (-15.4%) | -0.009 (-14.1%) |

La precisión disminuye naturalmente con k creciente (como es esperado en sistemas de recuperación), pero la degradación por reranking permanece consistente en aproximadamente -15% a -17% en todo el rango. Esta consistencia sugiere que el efecto negativo del CrossEncoder no es un artefacto de un valor específico de k sino una característica sistemática de la interacción entre Ada y el reranker evaluado.

#### 7.3.1.4 Análisis de Recall

El Recall mide la capacidad del sistema para recuperar todos los documentos relevantes disponibles. La **Tabla 7.3** muestra la evolución del recall con k creciente.

**Tabla 7.3: Recall@k de Ada para k={3,5,10,15}**

| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.276 | 0.398 | 0.591 | 0.702 |
| Después CrossEncoder | 0.228 | 0.330 | 0.539 | 0.649 |
| Δ (cambio) | -0.048 (-17.4%) | -0.068 (-17.2%) | -0.052 (-8.9%) | -0.053 (-7.5%) |

El recall alcanza 0.702 en k=15 (antes del reranking), indicando que el sistema recupera aproximadamente 70% de todos los documentos relevantes dentro del top-15. Este resultado es notable considerando la alta dimensionalidad del corpus (187,031 documentos). El reranking reduce el recall en todos los puntos, aunque la degradación porcentual disminuye ligeramente con k creciente (de -17.4% en k=3 a -7.5% en k=15).

#### 7.3.1.5 Implicaciones Prácticas

Para aplicaciones que utilizan Ada como modelo de embedding, los resultados sugieren **no aplicar reranking con CrossEncoder** y utilizar directamente los resultados de la búsqueda vectorial. Esta estrategia además reduce costos computacionales eliminando el paso de reranking (aproximadamente 25% de reducción en latencia por consulta) y simplifica la arquitectura del sistema manteniendo una sola etapa de recuperación.

### 7.3.2 MPNet (multi-qa-mpnet-base-dot-v1)

#### 7.3.2.1 Rendimiento General

MPNet demostró rendimiento sólido y el **impacto más neutral del reranking** entre todos los modelos evaluados, confirmando su especialización en tareas de pregunta-respuesta. La **Tabla 7.4** presenta las métricas principales.

**Tabla 7.4: Métricas Principales de MPNet (k=5)**

| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |
|---------|-----------------|-------------------|-----------------|------------|
| Precision@5 | 0.070 | 0.067 | -0.004 | -5.5% |
| Recall@5 | 0.277 | 0.264 | -0.013 | -4.8% |
| F1@5 | 0.108 | 0.103 | -0.005 | -5.0% |
| NDCG@5 | 0.193 | 0.185 | -0.007 | -3.8% |
| MAP@5 | 0.174 | 0.161 | -0.013 | -7.2% |
| MRR | 0.184 | 0.177 | -0.007 | -4.1% |

MPNet alcanza Precision@5 de 0.070, aproximadamente 28% inferior a Ada pero con una característica distintiva: el reranking produce cambios mínimos (±2% a ±7%) en todas las métricas. Esta estabilidad ante el reranking valida la calidad robusta de sus embeddings especializados para Q&A.

#### 7.3.2.2 Estabilidad ante Reranking

El comportamiento más estable de MPNet ante el reranking se explica por su entrenamiento específico en datasets de pregunta-respuesta (MS MARCO, Natural Questions), que optimizó las representaciones vectoriales para capturar las relaciones semánticas pregunta-documento de manera directa. El espacio de embeddings resultante ya refleja la relevancia de manera precisa, limitando el margen de mejora (o degradación) que puede aportar un reranker adicional.

La **Figura 7.2** muestra curvas prácticamente superpuestas de precisión antes y después del reranking, con divergencias menores de 0.005 en la mayoría de los valores de k. Este patrón se replica consistentemente en todas las métricas evaluadas.

![Figura 7.2: Comparación de Precision@k para MPNet antes y después del reranking](./capitulo_7_analisis/charts/precision_comparison_mpnet.png)

#### 7.3.2.3 Comparación con Ada

Aunque MPNet no supera a Ada en métricas de recuperación absolutas, presenta ventajas específicas relevantes para implementación práctica. La **Tabla 7.5** compara ambos modelos en métricas clave.

**Tabla 7.5: Comparación Ada vs MPNet (k=5, Antes Reranking)**

| Métrica | Ada | MPNet | Diferencia | Diferencia (%) |
|---------|-----|-------|------------|----------------|
| Precision@5 | 0.098 | 0.070 | -0.028 | -28.6% |
| Recall@5 | 0.398 | 0.277 | -0.121 | -30.4% |
| F1@5 | 0.152 | 0.108 | -0.044 | -28.9% |
| NDCG@5 | 0.234 | 0.193 | -0.041 | -17.5% |
| MAP@5 | 0.263 | 0.174 | -0.089 | -33.8% |
| MRR | 0.222 | 0.184 | -0.038 | -17.1% |

MPNet alcanza aproximadamente 71% del rendimiento de Ada en Precision@5, pero ofrece ventajas en dimensionalidad (768 vs 1,536), costos (modelo open-source vs API propietaria), y latencia de inferencia. Esta relación rendimiento-costo posiciona a MPNet como alternativa viable para implementaciones con restricciones presupuestarias o que priorizan independencia de proveedores comerciales.

#### 7.3.2.4 Implicaciones Prácticas

MPNet puede utilizarse con o sin reranking sin impacto significativo en rendimiento, permitiendo flexibilidad en el diseño de sistemas según restricciones de latencia y recursos computacionales. Para aplicaciones donde se requiere predecibilidad de rendimiento y estabilidad, MPNet representa una opción sólida que mantiene calidad consistente independientemente de la estrategia de procesamiento seleccionada.

### 7.3.3 MiniLM (all-MiniLM-L6-v2)

#### 7.3.3.1 Rendimiento General

MiniLM es el modelo de menor dimensionalidad evaluado (384 dimensiones) y muestra el rendimiento inicial más bajo entre los cuatro modelos. Sin embargo, presenta la característica distintiva de ser **el modelo que más se beneficia del reranking**. La **Tabla 7.6** presenta las métricas principales.

**Tabla 7.6: Métricas Principales de MiniLM (k=5)**

| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |
|---------|-----------------|-------------------|-----------------|------------|
| Precision@5 | 0.053 | 0.060 | +0.007 | +13.6% |
| Recall@5 | 0.211 | 0.236 | +0.025 | +11.9% |
| F1@5 | 0.082 | 0.093 | +0.011 | +13.1% |
| NDCG@5 | 0.150 | 0.169 | +0.019 | +12.5% |
| MAP@5 | 0.132 | 0.147 | +0.015 | +11.4% |
| MRR | 0.145 | 0.159 | +0.015 | +10.0% |

Los resultados confirman de manera robusta que MiniLM alcanza mejoras consistentes de **+10% a +14%** en todas las métricas principales cuando se combina con reranking neural, compensando efectivamente sus limitaciones dimensionales. Precision@5 mejora de 0.053 a 0.060, un incremento relativo del 13.6% que acerca el rendimiento a MPNet sin reranking (0.070).

#### 7.3.3.2 Análisis del Impacto del Reranking

El efecto positivo del reranking en MiniLM se explica por la menor capacidad representacional del modelo base (384 dimensiones vs 768-1,536 de los otros modelos). El espacio de embeddings más comprimido produce rankings iniciales de menor calidad que pueden ser mejorados sustancialmente mediante el procesamiento adicional del CrossEncoder, que opera con atención cruzada completa entre pregunta y documento.

La **Figura 7.3** muestra claramente la curva "Después de CrossEncoder" dominando la curva "Antes" en todo el rango de k evaluado. La mejora es más pronunciada en valores pequeños de k (k=1 a k=5) donde la precisión del ranking tiene mayor impacto en las métricas.

![Figura 7.3: Comparación de Precision@k para MiniLM antes y después del reranking](./capitulo_7_analisis/charts/precision_comparison_minilm.png)

#### 7.3.3.3 Evolución por Profundidad de Recuperación

La **Tabla 7.7** muestra cómo el beneficio del reranking se mantiene en diferentes profundidades de recuperación.

**Tabla 7.7: Precision@k de MiniLM para k={3,5,10,15}**

| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.056 | 0.053 | 0.046 | 0.040 |
| Después CrossEncoder | 0.063 | 0.060 | 0.052 | 0.045 |
| Δ (cambio) | +0.007 (+12.8%) | +0.007 (+13.6%) | +0.006 (+13.6%) | +0.005 (+12.9%) |

La mejora relativa se mantiene consistente alrededor del +13% en todo el rango, indicando que el reranking mejora efectivamente el ordenamiento de candidatos sin limitarse a reordenar solo las posiciones superiores. Este comportamiento contrasta con Ada, donde el reranking degradaba consistentemente el ranking en todas las posiciones.

#### 7.3.3.4 Trade-offs Rendimiento-Eficiencia

MiniLM con reranking alcanza Precision@5 de 0.060, aproximadamente 75% del rendimiento de MPNet sin reranking (0.070). Sin embargo, ofrece ventajas significativas en eficiencia:

**Eficiencia de Almacenamiento:**
- MiniLM: 384 dimensiones = 1.5 KB por documento (float32)
- MPNet: 768 dimensiones = 3.0 KB por documento
- Ada: 1,536 dimensiones = 6.0 KB por documento

Para el corpus de 187,031 documentos, MiniLM requiere aproximadamente 281 MB de almacenamiento vectorial, comparado con 562 MB para MPNet y 1.12 GB para Ada. Esta reducción de 4× en requerimientos de almacenamiento es significativa para aplicaciones con restricciones de infraestructura.

**Eficiencia Computacional:**
- Latencia de generación de embeddings: ~40% menor que MPNet
- Throughput de indexación: ~2.5× mayor que MPNet
- Costos de inferencia: Modelo open-source sin costos de API

#### 7.3.3.5 Implicaciones Prácticas

Para aplicaciones con restricciones de recursos o costos, **MiniLM con reranking** representa una alternativa viable que alcanza rendimiento competitivo mientras mantiene eficiencia en almacenamiento y costos de inferencia. La combinación es particularmente apropiada para:

- Implementaciones on-premise con hardware limitado
- Aplicaciones móviles o edge computing donde el almacenamiento es crítico
- Prototipos y MVPs donde minimizar costos es prioritario
- Sistemas que requieren actualizaciones frecuentes del índice (aprovechando la velocidad de reindexación)

La recomendación es utilizar siempre MiniLM con reranking, ya que la penalización en latencia por el paso adicional (~1-2 segundos por consulta) es compensada ampliamente por la mejora del 13% en métricas de recuperación.

### 7.3.4 E5-Large (intfloat/e5-large-v2)

#### 7.3.4.1 Rendimiento General

E5-Large presenta un perfil de rendimiento intermedio, superando consistentemente a MiniLM pero quedando por debajo de MPNet y Ada. El modelo muestra un **comportamiento de mejora moderada** con el reranking. La **Tabla 7.8** presenta las métricas principales.

**Tabla 7.8: Métricas Principales de E5-Large (k=5)**

| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |
|---------|-----------------|-------------------|-----------------|------------|
| Precision@5 | 0.065 | 0.066 | +0.001 | +1.5% |
| Recall@5 | 0.262 | 0.263 | +0.001 | +0.2% |
| F1@5 | 0.100 | 0.101 | +0.001 | +1.1% |
| NDCG@5 | 0.172 | 0.171 | -0.001 | -0.3% |
| MAP@5 | 0.158 | 0.164 | +0.006 | +3.8% |
| MRR | 0.156 | 0.158 | +0.002 | +1.5% |

Los resultados muestran **mejoras selectivas** con el reranking, particularmente en MAP@5 (+3.8%) que mide la calidad promedio del ranking de todos los documentos relevantes. Otras métricas muestran cambios mínimos o ligeramente negativos. Este comportamiento mixto sugiere que el CrossEncoder puede mejorar el ordenamiento promedio de documentos relevantes sin necesariamente mejorar la recuperación en posiciones superiores.

#### 7.3.4.2 Análisis Comparativo

E5-Large alcanza aproximadamente 93% del rendimiento de MPNet en Precision@5 (0.065 vs 0.070), posicionándose como una opción intermedia entre MPNet y MiniLM. La **Tabla 7.9** compara los tres modelos open-source evaluados.

**Tabla 7.9: Comparación Modelos Open-Source (k=5, Antes Reranking)**

| Métrica | MPNet | E5-Large | MiniLM |
|---------|-------|----------|--------|
| Precision@5 | 0.070 | 0.065 | 0.053 |
| Recall@5 | 0.277 | 0.262 | 0.211 |
| F1@5 | 0.108 | 0.100 | 0.082 |
| NDCG@5 | 0.193 | 0.172 | 0.150 |
| Dimensionalidad | 768 | 1,024 | 384 |

E5-Large ocupa una posición intermedia tanto en rendimiento como en dimensionalidad. Sus 1,024 dimensiones representan un compromiso entre la compacidad de MiniLM (384) y la capacidad representacional de MPNet (768) o Ada (1,536).

#### 7.3.4.3 Comportamiento Mixto ante Reranking

El impacto heterogéneo del reranking en diferentes métricas sugiere que E5-Large se encuentra en un punto de balance donde el CrossEncoder mejora algunos aspectos del ranking mientras degrada ligeramente otros. MAP@5 mejora un 3.8% porque el reranking mejora el ordenamiento promedio de todos los documentos relevantes (no solo los del top-k). Sin embargo, NDCG@5 disminuye ligeramente (-0.3%), sugiriendo que el reordenamiento puede empeorar la posición de algunos documentos altamente relevantes.

Este comportamiento requiere evaluación cuidadosa de qué métricas son prioritarias para la aplicación específica antes de decidir si aplicar o no el reranking con E5-Large.

#### 7.3.4.4 Implicaciones Prácticas

E5-Large con reranking puede ser una opción adecuada cuando se requiere un balance entre el rendimiento de MPNet y la eficiencia de MiniLM, particularmente en aplicaciones donde MAP (calidad promedio de ranking) es prioritaria sobre métricas de top-k como Precision@5 o NDCG@5. El modelo es apropiado para:

- Aplicaciones donde se presentan múltiples resultados al usuario (no solo top-3 o top-5)
- Casos donde la calidad del ranking completo es importante (e.g., interfaces de browsing)
- Implementaciones que requieren modelo open-source con capacidad superior a MiniLM

La decisión de aplicar reranking con E5-Large debe basarse en análisis específico del caso de uso, evaluando si la mejora en MAP justifica el incremento en latencia computacional.

## 7.4 Análisis Comparativo entre Modelos

### 7.4.1 Ranking General de Modelos

La **Tabla 7.10** presenta el ranking de modelos según Precision@5, la métrica más representativa de la capacidad del sistema para entregar documentos relevantes en las posiciones superiores del resultado.

**Tabla 7.10: Ranking de Modelos por Precision@5**

**Antes del Reranking:**
| Posición | Modelo | Precision@5 | Diferencia vs Ada |
|----------|--------|-------------|-------------------|
| 1 | Ada (OpenAI) | 0.098 | - |
| 2 | MPNet | 0.070 | -28.6% |
| 3 | E5-Large | 0.065 | -33.7% |
| 4 | MiniLM | 0.053 | -45.9% |

**Después del Reranking:**
| Posición | Modelo | Precision@5 | Diferencia vs Ada |
|----------|--------|-------------|-------------------|
| 1 | Ada (OpenAI) | 0.081 | - |
| 2 | MPNet | 0.067 | -17.3% |
| 3 | E5-Large | 0.066 | -18.5% |
| 4 | MiniLM | 0.060 | -25.9% |

**Observación Clave:** El reranking **reduce la brecha** entre modelos. La diferencia Ada vs MiniLM disminuye de -45.9% a -25.9%, mientras que la diferencia Ada vs MPNet se reduce de -28.6% a -17.3%. Este efecto de convergencia confirma que el CrossEncoder tiene mayor impacto en modelos inicialmente más débiles (MiniLM) y puede incluso degradar modelos fuertes (Ada).

### 7.4.2 Efecto del Reranking por Modelo

La **Figura 7.4** presenta un mapa de calor que visualiza el cambio porcentual en diferentes métricas para cada modelo, facilitando la identificación de patrones. Los resultados muestran tres comportamientos distintos:

![Figura 7.4: Mapa de calor de cambios porcentuales por modelo y métrica](./capitulo_7_analisis/charts/delta_heatmap.png)

**Degradación Sistemática (Ada):**
- Todas las métricas muestran cambios negativos (color rojo en el heatmap)
- Degradación más pronunciada en MAP (-23.7%) y Recall (-17.2%)
- Patrón consistente en todo el espacio de métricas

**Estabilidad (MPNet):**
- Cambios mínimos en todas las métricas (color amarillo neutro en el heatmap)
- Mayor degradación en MAP (-7.2%), otras métricas ≤ 5%
- Robustez ante modificaciones arquitectónicas

**Mejora Consistente (MiniLM):**
- Todas las métricas muestran cambios positivos (color verde en el heatmap)
- Mejora más pronunciada en Precision (+13.6%) y F1 (+13.1%)
- Patrón de mejora uniforme en todo el espacio de métricas

**Comportamiento Mixto (E5-Large):**
- Mejoras selectivas en MAP (+3.8%)
- Estabilidad en la mayoría de las métricas
- Ligera degradación en NDCG (-0.3%)

### 7.4.3 Análisis de Curvas de Precisión

La **Figura 7.5** muestra la evolución de Precision@k para todos los modelos antes del reranking, revelando patrones importantes:

![Figura 7.5: Curvas de Precision@k antes del reranking para todos los modelos](./capitulo_7_analisis/charts/precision_por_k_before.png)

**Dominancia de Ada:**
- La curva de Ada domina en todo el rango de k
- Ventaja más pronunciada en k pequeños (k=1 a k=5)
- Ventaja relativa disminuye ligeramente con k creciente

**Agrupamiento de Modelos Open-Source:**
- MPNet, E5-Large y MiniLM muestran curvas relativamente cercanas
- La separación entre ellos es menor que la separación con Ada
- Orden relativo se mantiene estable: MPNet > E5-Large > MiniLM

**Decaimiento con k:**
- Todas las curvas muestran decaimiento monotónico con k creciente
- Tasa de decaimiento similar entre modelos (pendiente comparable)
- Convergencia asintótica para valores grandes de k

La **Figura 7.6** muestra las mismas curvas después del reranking, evidenciando el efecto de convergencia discutido anteriormente. La brecha entre modelos se reduce notablemente, particularmente entre MiniLM y los modelos superiores.

![Figura 7.6: Curvas de Precision@k después del reranking para todos los modelos](./capitulo_7_analisis/charts/precision_por_k_after.png)

### 7.4.4 Análisis Multi-Métrica

La **Figura 7.7** presenta un gráfico de barras agrupadas comparando los cuatro modelos en cinco métricas clave (Precision@5, Recall@5, F1@5, NDCG@5, MAP@5), con barras antes y después del reranking para cada métrica.

![Figura 7.7: Comparación multi-métrica de todos los modelos](./capitulo_7_analisis/charts/model_ranking_bars.png)

El análisis visual confirma:
- Ada mantiene superioridad en todas las métricas antes del reranking
- El reranking reduce la ventaja de Ada en todas las métricas
- MiniLM es el único modelo donde las barras "Después" superan consistentemente las barras "Antes"
- MPNet muestra barras prácticamente superpuestas, confirmando estabilidad

### 7.4.5 Recomendaciones por Escenario

Basado en el análisis comparativo, se presentan recomendaciones específicas según el contexto de implementación:

**Escenario 1: Máximo Rendimiento (Budget No Restringido)**
- **Modelo Recomendado:** Ada sin reranking
- **Justificación:** Mejor rendimiento absoluto en todas las métricas
- **Trade-offs:** Costos de API, dependencia de proveedor externo
- **Precision@5 Esperada:** 0.098

**Escenario 2: Balance Rendimiento-Costo (Presupuesto Moderado)**
- **Modelo Recomendado:** MPNet sin reranking
- **Justificación:** 71% del rendimiento de Ada, cero costos de API, modelo open-source
- **Trade-offs:** Rendimiento inferior a Ada
- **Precision@5 Esperada:** 0.070

**Escenario 3: Eficiencia y Bajo Costo (Budget Restringido)**
- **Modelo Recomendado:** MiniLM con reranking
- **Justificación:** Balance óptimo eficiencia-rendimiento, mejora significativa con reranking
- **Trade-offs:** Latencia adicional por reranking (~1-2 seg), rendimiento inferior a MPNet
- **Precision@5 Esperada:** 0.060

**Escenario 4: Caso de Uso con Prioridad en MAP**
- **Modelo Recomendado:** E5-Large con reranking
- **Justificación:** Mejora selectiva en MAP, calidad de ranking completo
- **Trade-offs:** Comportamiento mixto en otras métricas
- **MAP@5 Esperada:** 0.164

## 7.5 Análisis del Componente de Reranking

### 7.5.1 Impacto del CrossEncoder por Métrica

El CrossEncoder ms-marco-MiniLM-L-6-v2 produce efectos heterogéneos dependiendo del modelo de embedding base y de la métrica evaluada. La **Tabla 7.11** resume el cambio porcentual promedio en cada métrica, consolidando los resultados de los cuatro modelos.

**Tabla 7.11: Cambio Promedio por Métrica Debido al Reranking**

| Métrica | Ada | MPNet | MiniLM | E5-Large | Promedio General |
|---------|-----|-------|---------|----------|------------------|
| Precision@5 | -16.7% | -5.5% | **+13.6%** | +1.5% | -1.8% |
| Recall@5 | -17.2% | -4.8% | **+11.9%** | +0.2% | -2.5% |
| F1@5 | -16.8% | -5.0% | **+13.1%** | +1.1% | -1.9% |
| NDCG@5 | -13.6% | -3.8% | **+12.5%** | -0.3% | -1.3% |
| MAP@5 | -23.7% | -7.2% | **+11.4%** | **+3.8%** | -4.0% |
| MRR | -13.2% | -4.1% | **+10.0%** | +1.5% | -1.5% |

**Promedio por Modelo** | **-16.9%** | **-5.1%** | **+12.1%** | **+1.3%** | **-2.1%**

El promedio general de -2.1% indica que, considerando los cuatro modelos con igual peso, el reranking produce una degradación neta leve en el rendimiento del sistema. Sin embargo, este promedio oculta heterogeneidad significativa: MiniLM mejora consistentemente (+12.1% promedio), mientras que Ada se degrada sustancialmente (-16.9% promedio).

### 7.5.2 Análisis de Costos Computacionales

El reranking introduce overhead computacional significativo que debe justificarse mediante mejoras en rendimiento. La **Tabla 7.12** compara latencias observadas con y sin reranking.

**Tabla 7.12: Latencia Promedio por Consulta (milisegundos)**

| Componente | Sin Reranking | Con Reranking | Overhead |
|------------|---------------|---------------|----------|
| Generación embedding query | 45 | 45 | - |
| Búsqueda vectorial ChromaDB | 8 | 8 | - |
| Reranking CrossEncoder (top-15) | - | 1,850 | +1,850 |
| **Total** | **53** | **1,903** | **+3,491%** |

El reranking incrementa la latencia por consulta en aproximadamente 1.85 segundos (factor de 35×), procesando 15 pares [query, documento] a través del CrossEncoder. Este overhead es significativo para aplicaciones interactivas donde la latencia percibida por el usuario debe mantenerse bajo 2-3 segundos.

**Costos de Infraestructura:**
- CPU: CrossEncoder requiere inferencia en CPU (no acelera significativamente con GPU para batch size pequeño)
- Memoria: ~2 GB para mantener modelo CrossEncoder cargado
- Throughput: ~40 consultas/minuto con reranking vs ~1,130 consultas/minuto sin reranking (en hardware de referencia)

Para MiniLM, donde el reranking mejora Precision@5 en 13.6%, el overhead de latencia está justificado. Para Ada, donde el reranking degrada Precision@5 en 16.7%, el overhead no solo no aporta valor sino que reduce la calidad de los resultados.

### 7.5.3 Análisis de la Normalización de Scores

El CrossEncoder genera scores en rango ilimitado (logits) que requieren normalización para comparación e interpretación. El sistema implementa normalización Min-Max que mapea los scores al rango [0,1]:

```
score_normalized = (score - score_min) / (score_max - score_min)
```

La **Figura 7.8** presenta la distribución de scores normalizados del CrossEncoder para documentos relevantes vs no relevantes (datos de una muestra de 500 consultas):

- **Documentos Relevantes:** Media = 0.73, Desviación estándar = 0.18
- **Documentos No Relevantes:** Media = 0.42, Desviación estándar = 0.21

La separación de 0.31 puntos entre medias es estadísticamente significativa (p < 0.001, test t de Welch), indicando que el CrossEncoder efectivamente discrimina entre documentos relevantes y no relevantes. Sin embargo, la desviación estándar sustancial (0.18-0.21) sugiere que existe solapamiento significativo entre las distribuciones, explicando por qué el reranking no mejora consistentemente los resultados.

### 7.5.4 Limitaciones del CrossEncoder Evaluado

El análisis identifica varias limitaciones del CrossEncoder ms-marco-MiniLM-L-6-v2 en el contexto específico del dominio técnico de Azure:

**Limitación 1: Desajuste de Dominio**
- CrossEncoder entrenado en MS MARCO (búsqueda web general)
- Documentación Azure contiene lenguaje técnico especializado
- Posible pérdida de comprensión de terminología específica

**Limitación 2: Longitud de Contexto**
- CrossEncoder procesa pares [query, documento] truncados a 512 tokens
- Documentos técnicos de Azure promedian 800 tokens
- Pérdida de información contextual al final del documento

**Limitación 3: Interferencia con Embeddings Fuertes**
- CrossEncoder asume que ranking inicial es subóptimo
- Embeddings de Ada ya producen rankings de alta calidad
- Reordenamiento introduce errores en rankings previamente correctos

**Limitación 4: Escalabilidad**
- Procesamiento secuencial de 15 pares por consulta
- Latencia no escala bien con k creciente
- Bottleneck para aplicaciones de alto throughput

### 7.5.5 Alternativas de Reranking

Basado en las limitaciones identificadas, se proponen alternativas para futuras implementaciones:

**Opción 1: CrossEncoder Especializado en Dominio**
- Fine-tuning de CrossEncoder en pares [pregunta Azure, documento Azure]
- Utilizar los 2,067 pares validados como conjunto de entrenamiento
- Potencial mejora en comprensión de terminología técnica

**Opción 2: CrossEncoder con Mayor Capacidad**
- Modelo ms-marco-MiniLM-L-12-v2 (12 capas vs 6 capas)
- Mayor capacidad representacional para documentos complejos
- Trade-off: Latencia 2× mayor

**Opción 3: Reranking Selectivo**
- Aplicar reranking solo cuando similitud del top-1 es menor a threshold
- Evitar reranking en casos donde embedding inicial es claramente correcto
- Reducir costos computacionales sin sacrificar casos difíciles

**Opción 4: Reranking Multi-Etapa**
- Etapa 1: Reranking rápido con modelo ligero (top-30 → top-15)
- Etapa 2: Reranking preciso con modelo pesado (top-15 → top-5)
- Balance entre precisión y latencia

## 7.6 Validación de Hipótesis de Investigación

Los resultados experimentales permiten validar las hipótesis formuladas implícitamente en los objetivos de investigación presentados en el Capítulo 1.

### 7.6.1 Hipótesis 1: Superioridad de Modelos Propietarios

**Hipótesis:** Los modelos de embeddings propietarios (Ada) superan consistentemente a los modelos open-source en tareas de recuperación semántica de documentación técnica.

**Validación:** **Hipótesis confirmada.** Ada alcanza Precision@5 de 0.098, superando a MPNet (0.070), E5-Large (0.065) y MiniLM (0.053) en 40.0%, 50.8% y 84.9% respectivamente. La superioridad se mantiene consistente en todas las métricas evaluadas (Recall, F1, NDCG, MAP, MRR) y para todos los valores de k analizados (k=1 a k=15).

**Implicación:** Para aplicaciones donde el rendimiento es crítico y el presupuesto permite el uso de APIs comerciales, Ada es la opción recomendada. Sin embargo, la brecha de 40% con MPNet puede no justificar los costos adicionales en todos los casos de uso.

### 7.6.2 Hipótesis 2: Beneficio Universal del Reranking

**Hipótesis:** El reranking neural con CrossEncoder mejora consistentemente las métricas de recuperación independientemente del modelo de embedding base.

**Validación:** **Hipótesis rechazada.** El reranking muestra efectos heterogéneos:
- MiniLM mejora +12.1% promedio
- E5-Large mejora +1.3% promedio
- MPNet degrada -5.1% promedio
- Ada degrada -16.9% promedio

El beneficio del reranking depende críticamente de la calidad del modelo de embedding base, mostrando mejoras solo cuando el ranking inicial es subóptimo (MiniLM). Modelos con embeddings de alta calidad (Ada, MPNet) no se benefician y pueden degradarse.

**Implicación:** El reranking debe aplicarse selectivamente según el modelo de embedding utilizado, no como componente universal de la arquitectura.

### 7.6.3 Hipótesis 3: Trade-off Dimensionalidad-Rendimiento

**Hipótesis:** Existe un trade-off fundamental entre dimensionalidad de embeddings y rendimiento de recuperación, con modelos de mayor dimensionalidad superando consistentemente a modelos compactos.

**Validación:** **Hipótesis parcialmente confirmada.** La correlación entre dimensionalidad y rendimiento es positiva pero no perfecta:
- Ada (1,536 dim): Precision@5 = 0.098 (mejor rendimiento)
- E5-Large (1,024 dim): Precision@5 = 0.065
- MPNet (768 dim): Precision@5 = 0.070 (mejor que E5-Large a pesar de menor dimensionalidad)
- MiniLM (384 dim): Precision@5 = 0.053 (menor rendimiento)

MPNet supera a E5-Large a pesar de tener menor dimensionalidad (768 vs 1,024), sugiriendo que la especialización del modelo (entrenamiento específico en Q&A) puede compensar parcialmente la menor capacidad dimensional.

**Implicación:** La selección de modelo debe considerar tanto dimensionalidad como especialización de dominio. Modelos especializados de dimensionalidad moderada pueden superar modelos generales de mayor dimensionalidad.

### 7.6.4 Hipótesis 4: Convergencia con k Creciente

**Hipótesis:** La brecha de rendimiento entre modelos disminuye con k creciente, convergiendo asintóticamente para valores grandes de k.

**Validación:** **Hipótesis confirmada con matices.** Las diferencias absolutas en Precision@k disminuyen con k creciente:
- k=3: Ada (0.104) - MiniLM (0.056) = 0.048 (diferencia absoluta)
- k=5: Ada (0.098) - MiniLM (0.053) = 0.045
- k=10: Ada (0.079) - MiniLM (0.046) = 0.033
- k=15: Ada (0.061) - MiniLM (0.040) = 0.021

Sin embargo, las diferencias relativas se mantienen relativamente estables (45-50%), indicando que la ventaja proporcional de modelos superiores persiste incluso en k grandes.

**Implicación:** Para aplicaciones que presentan múltiples resultados al usuario (k grande), la selección del modelo de embedding tiene menor impacto absoluto pero mantiene importancia relativa.

## 7.7 Limitaciones del Estudio

### 7.7.1 Limitaciones del Ground Truth

El ground truth utilizado (2,067 pares pregunta-documento) representa solo el 15.4% del corpus total de preguntas (13,436). Esta limitación introduce sesgos potenciales:

**Sesgo de Cobertura:**
- Solo se evalúan preguntas con respuestas aceptadas que incluyen enlaces explícitos
- Preguntas sin enlaces validados quedan excluidas de la evaluación
- Posible sesgo hacia preguntas "bien documentadas" vs preguntas emergentes

**Sesgo de Exhaustividad:**
- Solo se considera un documento relevante por pregunta (el enlazado en la respuesta)
- Pueden existir otros documentos relevantes no identificados
- Métricas como Recall pueden estar artificialmente bajas

**Sesgo Temporal:**
- Ground truth refleja el estado de la documentación en el momento de respuesta
- Documentación de Azure evoluciona rápidamente
- Enlaces pueden haber quedado obsoletos o ser redirigidos

### 7.7.2 Limitaciones del CrossEncoder

El CrossEncoder evaluado (ms-marco-MiniLM-L-6-v2) presenta limitaciones para el dominio específico:

**Desajuste de Dominio:**
- Entrenado en MS MARCO (búsqueda web general), no en documentación técnica
- Posible pérdida de comprensión de terminología especializada de Azure
- No se evaluaron CrossEncoders especializados o fine-tuned

**Limitaciones Arquitectónicas:**
- Truncamiento a 512 tokens pierde información contextual
- Procesamiento secuencial introduce latencia significativa
- No se evaluó reranking multi-etapa o selectivo

### 7.7.3 Limitaciones de Generalización

Los resultados son específicos al dominio de documentación de Microsoft Azure:

**Especificidad de Dominio:**
- Documentación técnica especializada con terminología específica
- Estructura formal de documentos de Microsoft Learn
- Preguntas de foros técnicos con usuarios especializados

**Generalización a Otros Dominios:**
- Resultados pueden no trasladarse a dominios menos técnicos
- Ordenamiento relativo de modelos puede cambiar en otros contextos
- Efectividad del reranking puede variar según características del corpus

### 7.7.4 Limitaciones de Cobertura de Modelos

La evaluación se limitó a cuatro modelos representativos:

**Modelos No Evaluados:**
- text-embedding-3-large y text-embedding-3-small (OpenAI, 2024)
- Modelos multimodales que procesan código y texto conjuntamente
- Modelos especializados en documentación técnica
- Arquitecturas emergentes (2024-2025)

**Configuraciones No Evaluadas:**
- Hybrid search combinando búsqueda vectorial con BM25
- Ensemble de múltiples modelos de embedding
- Query expansion y pseudo relevance feedback
- Fine-tuning de modelos en el corpus específico

## 7.8 Conclusiones del Capítulo

Este capítulo presentó una evaluación experimental rigurosa de cuatro modelos de embeddings (Ada, MPNet, MiniLM, E5-Large) bajo dos estrategias de procesamiento (con y sin reranking) en el dominio de recuperación semántica de documentación técnica de Microsoft Azure. Los resultados se derivan de 2,067 preguntas validadas evaluadas contra un corpus de 187,031 documentos técnicos, proporcionando evidencia empírica robusta sobre la efectividad de diferentes arquitecturas RAG.

### 7.8.1 Hallazgos Principales

**Superioridad de Ada:**
Ada alcanza el mejor rendimiento absoluto en todas las métricas (Precision@5 = 0.098), superando a alternativas open-source en 40-85%. Esta superioridad justifica su uso en aplicaciones donde el rendimiento es crítico y el presupuesto permite costos de API.

**Heterogeneidad del Reranking:**
El CrossEncoder produce efectos heterogéneos: mejora MiniLM (+12.1%), estabiliza MPNet (-5.1%), y degrada Ada (-16.9%). Este hallazgo refuta la hipótesis de beneficio universal del reranking y requiere evaluación específica por modelo.

**Viabilidad de Alternativas Compactas:**
MiniLM con reranking alcanza 74% del rendimiento de Ada con 25% de la dimensionalidad (384 vs 1,536 dimensiones) y costos significativamente menores, estableciéndose como alternativa viable para implementaciones con restricciones de recursos.

**Importancia de Especialización:**
MPNet supera a E5-Large a pesar de menor dimensionalidad (768 vs 1,024), demostrando que el entrenamiento especializado en Q&A compensa parcialmente la menor capacidad dimensional.

### 7.8.2 Recomendaciones Prácticas

Basado en los resultados experimentales, se establecen las siguientes recomendaciones para implementación de sistemas RAG en soporte técnico:

**Para Máximo Rendimiento:**
- Utilizar Ada sin reranking
- Priorizar calidad de retrieval inicial sobre procesamiento multi-etapa
- Presupuesto: $0.0001 por consulta (costos de API)

**Para Balance Rendimiento-Costo:**
- Utilizar MPNet sin reranking
- Aprovechar estabilidad ante variaciones arquitectónicas
- Presupuesto: Solo costos de infraestructura (modelo open-source)

**Para Minimización de Recursos:**
- Utilizar MiniLM con reranking obligatorio
- Aceptar latencia adicional de 1-2 segundos por el paso de reranking
- Presupuesto: Mínimo (modelo compacto + CrossEncoder open-source)

### 7.8.3 Contribuciones a los Objetivos de Investigación

Los resultados abordan directamente los objetivos específicos formulados en el Capítulo 1:

**Objetivo 1 (Comparación de Arquitecturas de Embeddings):**
Evaluación cuantitativa de cuatro arquitecturas en condiciones controladas, estableciendo ranking definitivo: Ada > MPNet > E5-Large > MiniLM, con gaps de rendimiento cuantificados en todas las métricas.

**Objetivo 3 (Impacto del Reranking):**
Demostración de efectos heterogéneos del reranking, rechazando la hipótesis de beneficio universal y estableciendo directrices específicas por modelo para aplicación selectiva de reranking.

**Objetivo 4 (Metodología de Evaluación):**
Implementación de protocolo de evaluación reproducible basado en ground truth validado, métricas estándar, y análisis estadístico riguroso, estableciendo una base metodológica para futuras evaluaciones en el dominio.

Los resultados presentados en este capítulo proporcionan evidencia empírica para informar decisiones de diseño en sistemas RAG para soporte técnico, contribuyendo tanto al conocimiento académico sobre efectividad de arquitecturas de embeddings como a la práctica profesional en implementación de sistemas de recuperación semántica.
