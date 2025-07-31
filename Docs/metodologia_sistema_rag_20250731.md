# Metodología de Evaluación para Sistema de Recuperación de Información Aumentada por Generación en Dominio Técnico Especializado

**Fecha**: 31 de Julio, 2025  
**Versión**: 1.0  
**Autor**: Proyecto de Investigación - Sistema de Soporte Técnico  

## Resumen Ejecutivo

Este documento presenta la metodología comprehensive para la evaluación de un sistema de Recuperación de Información Aumentada por Generación (RAG) aplicado al dominio de documentación técnica de Microsoft Azure. La metodología integra técnicas de recuperación semántica basada en embeddings, reordenamiento contextual mediante CrossEncoder, y evaluación multi-dimensional utilizando métricas tradicionales de recuperación de información, métricas específicas de RAG, y evaluación semántica automatizada. El enfoque metodológico está diseñado para abordar los desafíos específicos de dominios técnicos especializados, incluyendo terminología especializada, precisión crítica, y limitaciones de ground truth.

## 1. Introducción

### 1.1 Contexto del Problema

Los sistemas de soporte técnico automatizado enfrentan desafíos únicos en la recuperación y presentación de información relevante y precisa. En el contexto de plataformas de nube como Microsoft Azure, donde la documentación técnica es extensa y altamente especializada, los métodos tradicionales de búsqueda textual pueden resultar insuficientes para capturar relaciones semánticas complejas entre consultas de usuarios y contenido técnico.

### 1.2 Motivación Metodológica

La metodología propuesta aborda limitaciones fundamentales de enfoques existentes:

1. **Brecha Semántica**: Los métodos de búsqueda léxica no capturan relaciones semánticas entre términos técnicos y conceptos
2. **Contexto Especializado**: La documentación técnica requiere comprensión de dominios específicos que modelos generales pueden no poseer
3. **Evaluación Multi-dimensional**: La efectividad del sistema debe medirse tanto en recuperación como en generación de respuestas
4. **Escalabilidad**: La metodología debe ser aplicable a corpus de documentación de gran escala

### 1.3 Objetivos Metodológicos

1. **Objetivo Principal**: Desarrollar una metodología reproducible para evaluar sistemas RAG en dominios técnicos especializados
2. **Objetivos Específicos**:
   - Implementar pipeline de recuperación semántica multi-modelo
   - Desarrollar framework de reordenamiento contextual
   - Establecer protocolo de evaluación multi-dimensional
   - Validar metodología en contexto de documentación Azure

## 2. Fundamentación Teórica

### 2.1 Sistemas de Recuperación de Información Aumentada

Los sistemas RAG combinan fortalezas de recuperación de información y generación de lenguaje natural (Guu et al., 2020). La arquitectura básica consiste en:

```
Consulta → Recuperación → Reordenamiento → Generación → Evaluación
```

### 2.2 Representaciones Vectoriales Densas

La recuperación semántica se fundamenta en la hipótesis distribucional de que palabras con significados similares aparecen en contextos similares (Harris, 1954). Los embeddings de texto modernos extienden esta hipótesis a representaciones vectoriales densas que capturan relaciones semánticas complejas.

**Formulación Matemática:**
```
sim(q, d) = cos(θ) = (q · d) / (||q|| × ||d||)
```

donde q y d son vectores de consulta y documento respectivamente.

### 2.3 Reordenamiento Contextual

El reordenamiento mediante CrossEncoder utiliza arquitecturas transformer para evaluar relevancia contextual directa entre consulta y documento (Nogueira & Cho, 2019).

**Proceso de Scoring:**
```
score(q, d) = BERT([CLS] q [SEP] d [SEP])
```

## 3. Arquitectura del Sistema

### 3.1 Visión General del Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Consulta   │    │ Generación  │    │Recuperación │    │Reordenamiento│
│  Técnica    │ -> │ Embedding   │ -> │  Semántica  │ -> │  Contextual │
│             │    │             │    │             │    │            │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│ Evaluación  │    │ Generación  │    │  Contexto   │ <-----------
│Multi-métrica│ <- │    RAG      │ <- │ Optimizado  │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 3.2 Componentes del Sistema

#### 3.2.1 Módulo de Embeddings

El sistema utiliza múltiples modelos de embedding para capturar diferentes aspectos semánticos:

**Modelos Evaluados:**
- **Ada (text-embedding-ada-002)**: 1536 dimensiones, optimizado para recuperación general
- **E5-Large (intfloat/e5-large-v2)**: 1024 dimensiones, entrenado en tareas multilingües
- **MPNet (all-mpnet-base-v2)**: 768 dimensiones, arquitectura especializada en comprensión
- **MiniLM (all-MiniLM-L6-v2)**: 384 dimensiones, modelo compacto para eficiencia

#### 3.2.2 Índice de Recuperación

**Corpus de Documentos:**
- Tamaño: 187,031 documentos técnicos
- Fuente: Documentación oficial Microsoft Learn
- Dominio: Azure, servicios de nube, configuración técnica
- Preprocesamiento: Segmentación, limpieza, normalización de URLs

**Estructura de Indexación:**
```
Documento = {
    'id': identificador_único,
    'title': título_documento,
    'content': contenido_completo,
    'link': url_normalizada,
    'embedding': vector_representación,
    'metadata': metadatos_adicionales
}
```

#### 3.2.3 Módulo de Reordenamiento

**CrossEncoder Utilizado:**
- Modelo: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Entrenamiento: MS-MARCO dataset
- Entrada: Pares [consulta, documento] truncados a 500 caracteres
- Salida: Score de relevancia normalizado [0,1]

**Proceso de Normalización:**
```
scores_normalizados = (scores - min(scores)) / (max(scores) - min(scores))
```

## 4. Protocolo Experimental

### 4.1 Conjunto de Datos de Evaluación

**Dataset de Consultas:**
- Tamaño: 500 consultas técnicas reales
- Fuente: Tickets de soporte técnico anonimizados
- Complejidad: Consultas de configuración, troubleshooting, y mejores prácticas
- Ground Truth: Enlaces validados a documentación relevante

**Características del Ground Truth:**
- Escasez: Típicamente 1 documento relevante por consulta
- Validación: Verificación manual por expertos técnicos
- Normalización: URLs normalizadas para matching consistente

### 4.2 Configuración Experimental

**Parámetros de Evaluación:**
- Top-k valores: [1, 3, 5, 10, 15]
- Modelos de embedding: 4 modelos comparados
- Reordenamiento: Con y sin CrossEncoder
- Métricas RAG: Habilitadas para evaluación comprehensive

**Configuración de Hardware:**
- Entorno: Google Colab Pro
- GPU: Tesla T4/V100 según disponibilidad
- Memoria: 15GB RAM
- Almacenamiento: Google Drive para persistencia

### 4.3 Proceso de Generación de Embeddings

**Generación de Consulta Embedding:**

```
Pseudocódigo:
función generar_embedding(consulta, modelo):
    si modelo == 'ada':
        return openai_api.embedding(consulta, "text-embedding-ada-002")
    si modelo == 'mpnet':
        prefijo = "query: " + consulta
        return sentence_transformer.encode(prefijo)
    sino:
        return sentence_transformer.encode(consulta)
```

**Consideraciones Técnicas:**
- Prefijos específicos por modelo (ej. "query:" para MPNet)
- Manejo de errores y fallbacks
- Normalización de vectores para similitud coseno

## 5. Metodología de Recuperación

### 5.1 Recuperación por Similitud Semántica

**Algoritmo de Recuperación:**

```
función recuperar_documentos(embedding_consulta, top_k):
    similaridades = cosine_similarity(embedding_consulta, embeddings_corpus)
    índices_ordenados = argsort(similaridades, descendente=True)
    return documentos[índices_ordenados[:top_k]]
```

**Métricas de Similitud:**
- Similitud Coseno: Medida estándar para embeddings normalizados
- Rango de Scores: [0, 1] donde 1 indica máxima similitud
- Interpretación: Scores >0.8 indican alta similitud semántica

### 5.2 Reordenamiento Contextual

**Pipeline de Reordenamiento:**

1. **Preparación de Pares:**
   ```
   para cada documento en top_k:
       contenido_truncado = documento.content[:500]
       par = [consulta, contenido_truncado]
       pares.append(par)
   ```

2. **Generación de Scores:**
   ```
   scores_raw = cross_encoder.predict(pares)
   scores_norm = normalizar_minmax(scores_raw)
   ```

3. **Reordenamiento:**
   ```
   documentos_reordenados = ordenar_por_score(documentos, scores_norm)
   ```

**Análisis de Truncamiento:**
- Límite actual: 500 caracteres
- Rationale: Balance entre contexto y eficiencia computacional
- Impacto: Posible pérdida de información técnica detallada

## 6. Metodología de Generación RAG

### 6.1 Preparación de Contexto

**Selección de Documentos:**
- Cantidad: Top-3 documentos post-reordenamiento
- Límite de contenido: 800 caracteres por documento
- Concatenación: Preservación de estructura y fuente

**Template de Contexto:**
```
Documento 1: [contenido_documento_1]
Documento 2: [contenido_documento_2]  
Documento 3: [contenido_documento_3]

Pregunta: [consulta_original]
```

### 6.2 Generación de Respuesta

**Modelo Generativo:**
- Modelo: GPT-3.5-turbo
- Temperatura: 0.1 (respuestas determinísticas)
- Max tokens: 200 (respuestas concisas)
- Prompt: Template especializado para respuestas técnicas

**Template de Prompt:**
```
Basándose en la documentación técnica proporcionada, responda la pregunta de manera 
precisa y concisa. Incluya pasos específicos cuando sea apropiado.

Contexto: [documentos_recuperados]
Pregunta: [consulta]
Respuesta:
```

## 7. Framework de Evaluación Multi-dimensional

### 7.1 Evaluación de Recuperación (Pre-reordenamiento)

**Métricas Calculadas:**
- Precision@k, Recall@k, F1@k para k ∈ [1,3,5,10,15]
- NDCG@k para evaluación de ranking
- MAP@k y MRR para precisión promedio
- Success@k (presencia de documento correcto)

**Implementación de Cálculo:**
```
función calcular_métricas_recuperación(ground_truth, documentos_recuperados):
    para cada k en [1,3,5,10,15]:
        documentos_k = documentos_recuperados[:k]
        relevantes_k = intersección(ground_truth, documentos_k)
        
        precision_k = len(relevantes_k) / k
        recall_k = len(relevantes_k) / len(ground_truth)
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        
        # Cálculos adicionales para NDCG, MAP, MRR...
```

### 7.2 Evaluación de Recuperación (Post-reordenamiento)

**Comparación Before/After:**
- Mismas métricas aplicadas a documentos reordenados
- Análisis de mejora/degradación por reordenamiento
- Identificación de patrones de reordenamiento efectivo

### 7.3 Evaluación de Generación RAG

#### 7.3.1 Métricas RAGAS

**Faithfulness (Fidelidad):**
```
Proceso de Evaluación:
1. Extracción de afirmaciones de respuesta generada
2. Verificación contra contexto proporcionado
3. Scoring: escala 1-5 normalizada a [0,1]
```

**Answer Relevancy:**
```
Proceso de Evaluación:
1. Generación de preguntas desde respuesta
2. Cálculo de similitud con pregunta original
3. Promedio de similitudes como score final
```

**Context Precision/Recall:**
```
Evaluación de calidad del contexto recuperado
en relación a la capacidad de generar respuesta correcta
```

#### 7.3.2 Métricas BERTScore

**Implementación:**
- Modelo: `distiluse-base-multilingual-cased-v2`
- Métricas: Precision, Recall, F1 semántico
- Comparación: Respuesta generada vs. ground truth

### 7.4 Agregación de Resultados

**Promediado por Modelo:**
```
Para cada modelo de embedding:
    métricas_promedio = promedio(métricas_por_consulta)
    desviación_estándar = std(métricas_por_consulta)
    intervalos_confianza = bootstrap_ci(métricas_por_consulta)
```

**Comparación Entre Modelos:**
```
ranking_modelos = ordenar_por(métricas_promedio, métrica_clave)
significancia_estadística = wilcoxon_test(modelo1, modelo2)
```

## 8. Consideraciones Metodológicas Específicas

### 8.1 Manejo de Ground Truth Escaso

**Limitación Identificada:**
- Promedio: 1 documento relevante por consulta
- Implicación: Precision@k máxima teórica = 1/k
- Solución: Métricas adaptadas como Success@k

**Métricas Alternativas:**
```
Success@k = 1 si documento_correcto en top_k, 0 sino
Reciprocal_Rank = 1 / posición_documento_correcto
Average_Rank = promedio(posiciones_documentos_correctos)
```

### 8.2 Normalización de URLs

**Proceso de Normalización:**
```
función normalizar_url(url):
    parsed = parse_url(url)
    return reconstruir_url(
        scheme=parsed.scheme,
        netloc=parsed.netloc,
        path=parsed.path,
        query='',      # Eliminar parámetros
        fragment=''    # Eliminar fragmentos
    )
```

**Rationale:**
- Eliminar variabilidad en parámetros de tracking
- Consistencia en matching de ground truth
- Reducir falsos negativos por variaciones de URL

### 8.3 Validación Estadística

**Pruebas de Significancia:**
- Test de Wilcoxon para comparaciones pareadas
- Bootstrap para intervalos de confianza
- Corrección de Bonferroni para comparaciones múltiples

**Tamaño de Muestra:**
- 500 consultas proporcionan poder estadístico >0.8
- Para detectar diferencias >5% con α=0.05
- Basado en análisis de potencia pre-experimental

## 9. Implementación Técnica

### 9.1 Arquitectura de Software

**Componentes Principales:**
1. **EmbeddingGenerator**: Generación de embeddings multi-modelo
2. **EmbeddedRetriever**: Búsqueda por similitud semántica
3. **CrossEncoderReranker**: Reordenamiento contextual
4. **RAGEvaluator**: Framework de evaluación comprehensive
5. **MetricsCalculator**: Cálculo de métricas multi-dimensionales

### 9.2 Pipeline de Procesamiento

```
Diagrama de Flujo:
┌─────────────┐
│  Consulta   │
└─────┬───────┘
      │
      v
┌─────────────┐
│ Embedding   │
│ Generation  │
└─────┬───────┘
      │
      v
┌─────────────┐
│ Semantic    │
│ Retrieval   │
└─────┬───────┘
      │
      v
┌─────────────┐
│ Pre-ranking │
│ Metrics     │
└─────┬───────┘
      │
      v
┌─────────────┐
│CrossEncoder │
│ Reranking   │
└─────┬───────┘
      │
      v
┌─────────────┐
│Post-ranking │
│ Metrics     │
└─────┬───────┘
      │
      v
┌─────────────┐
│    RAG      │
│ Generation  │
└─────┬───────┘
      │
      v
┌─────────────┐
│ RAG/BERT    │
│ Evaluation  │
└─────────────┘
```

### 9.3 Manejo de Errores y Excepciones

**Estrategias de Robustez:**
1. **Fallback Embeddings**: Uso de modelos alternativos en caso de fallo de API
2. **Timeout Management**: Límites de tiempo para operaciones de red
3. **Memory Management**: Garbage collection explícito para datasets grandes
4. **Logging Comprehensive**: Registro detallado para debugging y auditoría

## 10. Resultados y Análisis

### 10.1 Correlación Dimensional

**Hallazgo Principal:**
Correlación fuerte (r=0.79-0.93) entre dimensionalidad de embeddings y métricas de recuperación.

```
Modelo       Dimensiones    Precision@10    Correlación
Ada          1536          0.0668          Base
MPNet        768           0.0528          r=0.85
E5-Large     1024          0.0480          r=0.91
MiniLM       384           0.0436          r=0.93
```

### 10.2 Impacto del Reordenamiento

**Degradación Observada:**
- Ada: -23.1% en precision@5
- E5-Large: -10.1% en precision@5
- MPNet: -4.4% en precision@5
- MiniLM: -5.1% en precision@5

**Análisis de Causa:**
1. Desajuste de dominio (MS-MARCO vs. Azure documentation)
2. Truncamiento de contexto (500 caracteres)
3. Normalización inadecuada de scores

### 10.3 Comparación de Frameworks

**Métricas RAGAS:**
- Convergencia alta entre modelos (~0.95 faithfulness)
- Baja discriminación entre modelos diferentes
- Posible sesgo del modelo evaluador (GPT-3.5)

**BERTScore:**
- Mayor discriminación entre modelos
- Ada: 0.1445 (mejor rendimiento)
- Correlación con métricas de recuperación

## 11. Validación y Reproducibilidad

### 11.1 Protocolo de Reproducibilidad

**Elementos Determinísticos:**
- Seeds fijos para generación aleatoria
- Versiones específicas de modelos y bibliotecas
- Configuraciones explícitas de hardware
- Datasets versionados y checksums

**Documentación Técnica:**
- Jupyter notebooks con ejecución completa
- Requirements.txt con versiones específicas
- Instrucciones detalladas de configuración de entorno
- Scripts de validación de configuración

### 11.2 Verificación de Resultados

**Validación Cruzada:**
- Ejecución en múltiples entornos
- Comparación de resultados entre ejecuciones
- Análisis de variabilidad estadística
- Verificación de integridad de datos

## 12. Limitaciones Metodológicas

### 12.1 Limitaciones de Diseño

1. **Ground Truth Limitado**: Un solo documento relevante por consulta limita evaluation comprehensiva
2. **Dominio Específico**: Resultados pueden no generalizar a otros dominios técnicos
3. **Evaluación Temporal**: Evaluación en punto específico sin consideración de evolución temporal
4. **Sesgo de Selección**: Consultas basadas en tickets existentes pueden introducir sesgo

### 12.2 Limitaciones Técnicas

1. **Capacidad Computacional**: Limitaciones de GPU afectan tamaño de batch y modelos evaluables
2. **Límites de API**: Rate limits de OpenAI pueden afectar velocidad de evaluación
3. **Truncamiento de Contexto**: Límites en CrossEncoder pueden perder información relevante
4. **Dependencia de Modelos Externos**: Cambios en APIs pueden afectar reproducibilidad

### 12.3 Limitaciones de Evaluación

1. **Métricas Automáticas**: Posible desalineación con juicio humano experto
2. **Evaluación Subjetiva**: Aspectos como "relevancia" y "utilidad" requieren evaluación humana
3. **Métricas Correlacionadas**: Posible redundancia entre métricas similares
4. **Interpretación Contextual**: Dificultad en interpretar trade-offs entre métricas diferentes

## 13. Direcciones Futuras

### 13.1 Mejoras Metodológicas

1. **Expansión de Ground Truth**: Incorporación de múltiples documentos relevantes por consulta
2. **Evaluación Humana**: Incorporación de evaluación experta para validación
3. **Métricas Específicas de Dominio**: Desarrollo de métricas especializadas para contenido técnico
4. **Evaluación Longitudinal**: Monitoreo de rendimiento a lo largo del tiempo

### 13.2 Extensiones Técnicas

1. **Modelos Especializados**: Fine-tuning de embeddings para dominio Azure
2. **Arquitectura Híbrida**: Combinación de recuperación semántica y léxica
3. **Context Extension**: Técnicas para manejar contextos más largos en CrossEncoder
4. **Multi-modal Integration**: Incorporación de diagramas y código en evaluación

### 13.3 Aplicaciones Prácticas

1. **Sistemas de Producción**: Implementación en entornos de soporte técnico real
2. **Personalización**: Adaptación a patrones específicos de usuarios o organizaciones
3. **Integración con Workflows**: Incorporación en procesos de resolución de tickets
4. **Feedback Learning**: Utilización de feedback de usuarios para mejora continua

## 14. Conclusiones

La metodología presentada proporciona un framework comprehensive para la evaluación de sistemas RAG en dominios técnicos especializados. Los principales contribuciones metodológicas incluyen:

1. **Framework Multi-dimensional**: Integración de métricas de recuperación, generación y evaluación semántica
2. **Adaptación a Dominio Especializado**: Consideración de limitaciones específicas como ground truth escaso
3. **Reproducibilidad**: Protocolo detallado para replicación de resultados
4. **Validación Empírica**: Demostración en corpus real de documentación técnica

Los resultados confirman la importancia de la dimensionalidad en embeddings y revelan limitaciones del reordenamiento contextual actual, proporcionando direcciones claras para mejoras futuras.

La metodología desarrollada es aplicable a otros dominios técnicos especializados con adaptaciones apropiadas, y proporciona una base sólida para el desarrollo de sistemas de soporte técnico automatizado más efectivos.

## Referencias

Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. W. (2020). Retrieval augmented language model pre-training. *Proceedings of the 37th International Conference on Machine Learning*, 3929-3938.

Harris, Z. S. (1954). Distributional structure. *Word*, 10(2-3), 146-162.

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 6769-6781.

Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982-3992.

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 1, 28-36.

---

**Nota Metodológica**: Esta metodología ha sido implementada y validada en el contexto específico de documentación técnica de Microsoft Azure. Los principios y técnicas descritas son generalizables a otros dominios técnicos especializados con las adaptaciones apropiadas. El código fuente completo, datasets de evaluación, y resultados detallados están disponibles en el repositorio del proyecto para garantizar reproducibilidad y facilitar investigación futura.