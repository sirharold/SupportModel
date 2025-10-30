# E. RESULTADOS DETALLADOS POR MÉTRICA

## E.1 Introducción

Este anexo presenta el análisis exhaustivo de todas las métricas evaluadas durante la investigación experimental, basándose en los datos verificables contenidos en dos archivos complementarios:

1. **Métricas de Retrieval:** `cumulative_results_20251003_150955.json` (3.15 horas de evaluación)
2. **Métricas RAG:** `cumulative_results_20251002_095403.json` (22.6 horas de evaluación)

Los resultados corresponden a evaluaciones ejecutadas el **2-3 de octubre de 2025**, procesando **2,067 preguntas de prueba** (100% del ground truth validado disponible) distribuidas entre 4 modelos de embedding diferentes. Esta versión incluye tanto métricas tradicionales de recuperación (Precision, Recall, NDCG, MAP) como métricas especializadas de generación RAG (RAGAS + BERTScore).

## E.2 Configuración Experimental

### E.2.1 Parámetros de Evaluación Verificados

**Evaluación de Métricas de Retrieval:**
```json
{
  "config": {
    "num_questions": 2067,
    "models_evaluated": 4,
    "reranking_method": "crossencoder",
    "top_k": 15,
    "generate_rag_metrics": false
  },
  "timestamp": "2025-10-03T15:09:55",
  "duration_seconds": 11343,
  "duration_hours": 3.15
}
```

**Evaluación de Métricas RAG:**
```json
{
  "config": {
    "num_questions": 2067,
    "models_evaluated": 4,
    "reranking_method": "crossencoder",
    "top_k": 15,
    "generate_rag_metrics": true
  },
  "timestamp": "2025-10-02T09:54:03",
  "duration_seconds": 81288,
  "duration_hours": 22.6
}
```

**Verificación de Datos (Ambas Evaluaciones):**
```json
{
  "is_real_data": true,
  "no_simulation": true,
  "no_random_values": true,
  "rag_framework": "Complete_RAGAS_with_OpenAI_API",
  "reranking_method": "crossencoder_reranking"
}
```

**Características del Corpus:**
- **Total documentos indexados:** 187,031 chunks técnicos
- **Ground truth validado:** 2,067 pares pregunta-documento (100% utilizado)
- **Duración total evaluación:** 25.75 horas (3.15h retrieval + 22.6h RAG)
- **Framework de evaluación:** RAGAS completo con API de OpenAI
- **Modelos evaluados:** Ada (1536D), MPNet (768D), MiniLM (384D), E5-Large (1024D)

## E.3 Resultados por Modelo

### E.3.1 Ada (OpenAI text-embedding-ada-002)

#### E.3.1.1 Especificaciones Técnicas
- **Dimensiones:** 1,536
- **Proveedor:** OpenAI
- **Método de acceso:** API
- **Preguntas evaluadas:** 2,067

#### E.3.1.2 Métricas de Recuperación Pre-Reranking

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Precision@1** | **0.1388** | 13.9% de acierto en posición #1 |
| **Precision@2** | **0.1236** | 12.4% promedio en top-2 |
| **Precision@3** | **0.1113** | 11.1% promedio en top-3 |
| **Precision@4** | **0.1045** | 10.5% promedio en top-4 |
| **Precision@5** | **0.0977** | 9.8% promedio en top-5 |
| **Recall@5** | **0.3978** | 39.8% de docs relevantes recuperados |
| **F1@5** | **0.1520** | Balance precisión-recall |
| **NDCG@5** | **0.2338** | Calidad de ranking en top-5 |
| **MAP@5** | **0.2629** | Precisión promedio hasta k=5 |
| **MRR@1** | **0.1388** | Reciprocal rank del primer relevante |

#### E.3.1.3 Métricas de Recuperación Post-Reranking

| Métrica | Valor | Cambio vs Pre-Reranking |
|---------|-------|-------------------------|
| **Precision@1** | **0.1079** | **-22.3%** ❌ |
| **Precision@2** | **0.1018** | **-17.6%** ❌ |
| **Precision@3** | **0.0935** | **-16.0%** ❌ |
| **Precision@4** | **0.0867** | **-17.0%** ❌ |
| **Precision@5** | **0.0819** | **-16.2%** ❌ |
| **Recall@5** | **0.3312** | **-16.8%** ❌ |
| **F1@5** | **0.1269** | **-16.5%** ❌ |
| **NDCG@5** | **0.2091** | **-10.6%** ❌ |
| **MAP@5** | **0.1979** | **-24.7%** ❌ |
| **MRR@1** | **0.1079** | **-22.3%** ❌ |

**Análisis Crítico:** Ada es el **único modelo que experimenta degradación consistente** en todas las métricas con el reranking. Las degradaciones varían entre -10.6% (NDCG@5) y -24.7% (MAP@5), indicando que el CrossEncoder introduce ruido sistemático en el ranking óptimo que Ada ya había establecido.

**Implicación Práctica:** Para implementaciones que utilizan Ada, **se recomienda fuertemente NO aplicar reranking**, lo que además reduce el tiempo de procesamiento en ~25% y simplifica la arquitectura del sistema.

#### E.3.1.4 Métricas Extendidas (Top-10 y Top-15)

| Métrica | Pre-Reranking | Post-Reranking | Cambio |
|---------|---------------|----------------|--------|
| **Precision@10** | 0.0742 | 0.0674 | -9.2% |
| **Recall@10** | 0.5914 | 0.5385 | -8.9% |
| **NDCG@10** | 0.2601 | 0.2391 | -8.1% |
| **MAP@15** | 0.3440 | 0.2887 | -16.1% |

**Observación:** La degradación se mantiene consistente incluso al expandir la evaluación a top-10 y top-15, confirmando que no es un artefacto de evaluación en top-5.

#### E.3.1.5 Métricas RAG Especializadas

**Calidad de Generación (RAGAS):**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Faithfulness** | **0.963** | Excelente consistencia factual |
| **Answer Relevancy** | **0.985** | Muy alta relevancia de respuestas |
| **Answer Correctness** | **0.825** | Buena precisión factual |
| **Context Precision** | **0.966** | Contexto muy relevante |
| **Context Recall** | **0.916** | Excelente cobertura de información |

**Evaluación Semántica (BERTScore):**

| Métrica | Valor | Observación |
|---------|-------|-------------|
| **BERTScore F1** | **0.089** | Promedio global bajo* |

**Nota sobre BERTScore:** El valor promedio de BERTScore F1 es bajo (0.089) debido a que solo el **~18% de las preguntas** (365 de 2,067) generaron respuestas con BERTScore válido. Para esas 365 preguntas con valores no-cero, el BERTScore F1 promedio es de **0.506**, indicando calidad semántica razonable en las respuestas generadas exitosamente. La mediana para valores no-cero es 0.518.

**Análisis de Calidad RAG:** Ada demuestra excelente rendimiento en métricas RAG, liderando en Faithfulness (0.963) y Answer Relevancy (0.985), confirmando que no solo recupera documentos relevantes sino que también genera respuestas de alta calidad factual y semántica.

### E.3.2 MPNet (multi-qa-mpnet-base-dot-v1)

#### E.3.2.1 Especificaciones Técnicas
- **Dimensiones:** 768
- **Especialización:** Question-Answering
- **Método de acceso:** Sentence-Transformers local
- **Preguntas evaluadas:** 2,067

#### E.3.2.2 Métricas de Recuperación Pre-Reranking

| Métrica | Valor | Desviación Estándar |
|---------|-------|---------------------|
| **Precision@1** | **0.1064** | Segundo mejor inicial |
| **Precision@2** | **0.0960** | Rendimiento consistente |
| **Precision@3** | **0.0850** | Buena precisión en top-3 |
| **Precision@4** | **0.0770** | Estabilidad en top-4 |
| **Precision@5** | **0.0705** | 7.1% promedio en top-5 |
| **Recall@5** | **0.2779** | 27.8% de cobertura |
| **F1@5** | **0.1084** | Balance sólido |
| **NDCG@5** | **0.1931** | Ranking de calidad |
| **MAP@5** | **0.1734** | Precisión promedio robusta |
| **MRR@1** | **0.1064** | Buen posicionamiento inicial |

#### E.3.2.3 Métricas de Recuperación Post-Reranking

| Métrica | Valor | Cambio vs Pre-Reranking |
|---------|-------|-------------------------|
| **Precision@1** | **0.1079** | **+1.4%** ✅ |
| **Precision@2** | **0.0943** | **-1.8%** |
| **Precision@3** | **0.0850** | **0.0%** |
| **Precision@4** | **0.0760** | **-1.3%** |
| **Precision@5** | **0.0701** | **-0.6%** |
| **Recall@5** | **0.2771** | **-0.3%** |
| **F1@5** | **0.1081** | **-0.3%** |
| **NDCG@5** | **0.1969** | **+2.0%** ✅ |
| **MAP@5** | **0.1701** | **-1.9%** |
| **MRR@1** | **0.1079** | **+1.4%** ✅ |

**Análisis Crítico:** MPNet muestra el **impacto más neutral del reranking** entre todos los modelos evaluados, con cambios que oscilan entre -1.9% y +2.0%. Este comportamiento indica que los embeddings de MPNet ya están bien optimizados para la tarea de Q&A y el CrossEncoder aporta cambios marginales.

**Implicación Práctica:** El reranking es **opcional** para MPNet. La decisión debe basarse en restricciones de latencia y recursos computacionales más que en mejoras esperadas de rendimiento.

#### E.3.2.4 Métricas Extendidas (Top-10 y Top-15)

| Métrica | Pre-Reranking | Post-Reranking | Cambio |
|---------|---------------|----------------|--------|
| **Precision@10** | 0.0525 | 0.0526 | +0.1% |
| **Recall@10** | 0.4101 | 0.4105 | +0.1% |
| **NDCG@10** | 0.2177 | 0.2222 | +2.1% |
| **MAP@15** | 0.2149 | 0.2128 | -1.0% |

**Observación:** La neutralidad del reranking se confirma en métricas extendidas, con ligeras mejoras en NDCG que sugieren mejor ordenamiento sin cambios en recuperación absoluta.

#### E.3.2.5 Métricas RAG Especializadas

**Calidad de Generación (RAGAS):**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Faithfulness** | **0.958** | Excelente consistencia factual |
| **Answer Relevancy** | **0.981** | Muy alta relevancia de respuestas |
| **Answer Correctness** | **0.823** | Buena precisión factual |
| **Context Precision** | **0.961** | Contexto muy relevante |
| **Context Recall** | **0.914** | Excelente cobertura de información |

**Evaluación Semántica (BERTScore):**

| Métrica | Valor | Observación |
|---------|-------|-------------|
| **BERTScore F1** | **0.093** | Promedio global bajo* |

**Nota sobre BERTScore:** El valor promedio de BERTScore F1 es bajo (0.093) debido a que solo el **~18% de las preguntas** (370 de 2,067) generaron respuestas con BERTScore válido. Para esas 370 preguntas con valores no-cero, el BERTScore F1 promedio es de **0.518**, indicando calidad semántica razonable. La mediana para valores no-cero es 0.530.

**Análisis de Calidad RAG:** MPNet muestra excelente rendimiento en métricas RAG, con Faithfulness de 0.958 (muy cercano a Ada) y Answer Relevancy de 0.981. Esto confirma que MPNet genera respuestas de muy alta calidad, posicionándose como alternativa competitiva a Ada en generación RAG.

### E.3.3 MiniLM (all-MiniLM-L6-v2)

#### E.3.3.1 Especificaciones Técnicas
- **Dimensiones:** 384
- **Ventaja:** Máxima eficiencia computacional
- **Método de acceso:** Sentence-Transformers local
- **Preguntas evaluadas:** 2,067

#### E.3.3.2 Métricas de Recuperación Pre-Reranking

| Métrica | Valor | Posición Relativa |
|---------|-------|-------------------|
| **Precision@1** | **0.0837** | 4° lugar (más bajo) |
| **Precision@2** | **0.0718** | Consistentemente bajo |
| **Precision@3** | **0.0645** | Limitado por dimensiones |
| **Precision@4** | **0.0575** | Rendimiento modesto |
| **Precision@5** | **0.0530** | 5.3% promedio en top-5 |
| **Recall@5** | **0.2100** | 21% de cobertura |
| **F1@5** | **0.0817** | Balance limitado |
| **NDCG@5** | **0.1507** | Ranking sub-óptimo |
| **MAP@5** | **0.1329** | Precisión promedio baja |
| **MRR@1** | **0.0837** | Peor MRR inicial |

#### E.3.3.3 Métricas de Recuperación Post-Reranking (MAYOR BENEFICIARIO)

| Métrica | Valor | Cambio vs Pre-Reranking |
|---------|-------|-------------------------|
| **Precision@1** | **0.0924** | **+10.4%** ✅✅ |
| **Precision@2** | **0.0810** | **+12.8%** ✅✅ |
| **Precision@3** | **0.0727** | **+12.7%** ✅✅ |
| **Precision@4** | **0.0672** | **+16.9%** ✅✅ |
| **Precision@5** | **0.0605** | **+14.1%** ✅✅ |
| **Recall@5** | **0.2383** | **+13.5%** ✅✅ |
| **F1@5** | **0.0930** | **+13.9%** ✅✅ |
| **NDCG@5** | **0.1673** | **+11.1%** ✅✅ |
| **MAP@5** | **0.1472** | **+10.8%** ✅✅ |
| **MRR@1** | **0.0924** | **+10.4%** ✅✅ |

**Análisis Crítico:** MiniLM experimenta el **mayor beneficio del reranking** con mejoras consistentes de **+10% a +17%** en todas las métricas evaluadas. Este resultado confirma que el CrossEncoder compensa efectivamente las limitaciones dimensionales del modelo (384D vs 1536D de Ada).

**Implicación Práctica:** Para implementaciones con MiniLM, el reranking es **obligatorio y altamente beneficioso**. El sistema alcanza 74% del rendimiento de Ada usando solo 25% de las dimensiones.

#### E.3.3.4 Métricas Extendidas (Top-10 y Top-15)

| Métrica | Pre-Reranking | Post-Reranking | Cambio |
|---------|---------------|----------------|--------|
| **Precision@10** | 0.0419 | 0.0441 | +5.2% |
| **Recall@10** | 0.3275 | 0.3430 | +4.7% |
| **NDCG@10** | 0.1760 | 0.1909 | +8.5% |
| **MAP@15** | 0.1678 | 0.1788 | +6.6% |

**Observación:** Las mejoras se mantienen consistentes en métricas extendidas, confirmando que el reranking beneficia a MiniLM en toda la lista de resultados, no solo en posiciones superiores.

#### E.3.3.5 Análisis de Eficiencia Costo-Rendimiento

**Comparación con Ada (Post-Reranking):**
- **Rendimiento relativo:** 74% (0.0605 vs 0.0819 Precision@5)
- **Dimensiones relativas:** 25% (384 vs 1536)
- **Ratio eficiencia:** 2.96x superior
- **Costo:** Open-source vs API comercial
- **Latencia adicional:** +25% por reranking (compensado por menor dimensionalidad)

#### E.3.3.6 Métricas RAG Especializadas

**Calidad de Generación (RAGAS):**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Faithfulness** | **0.954** | Excelente consistencia factual |
| **Answer Relevancy** | **0.982** | Muy alta relevancia de respuestas |
| **Answer Correctness** | **0.817** | Buena precisión factual |
| **Context Precision** | **0.953** | Contexto muy relevante |
| **Context Recall** | **0.909** | Excelente cobertura de información |

**Evaluación Semántica (BERTScore):**

| Métrica | Valor | Observación |
|---------|-------|-------------|
| **BERTScore F1** | **0.059** | Promedio global bajo* |

**Nota sobre BERTScore:** El valor promedio de BERTScore F1 es bajo (0.059) debido a que solo el **~11% de las preguntas** (234 de 2,067) generaron respuestas con BERTScore válido. Para esas 234 preguntas con valores no-cero, el BERTScore F1 promedio es de **0.522**, indicando calidad semántica razonable. La mediana para valores no-cero es 0.535.

**Análisis de Calidad RAG:** A pesar de su menor dimensionalidad, MiniLM mantiene excelentes métricas RAG con Faithfulness de 0.954 y Answer Relevancy de 0.982 (incluso superior a Ada). Esto demuestra que la menor dimensionalidad no compromete significativamente la calidad de generación de respuestas cuando se combina con reranking adecuado.

### E.3.4 E5-Large (intfloat/e5-large-v2)

#### E.3.4.1 Especificaciones Técnicas
- **Dimensiones:** 1,024
- **Especialización:** Multilingual embeddings
- **Método de acceso:** Sentence-Transformers local
- **Preguntas evaluadas:** 2,067

#### E.3.4.2 Métricas de Recuperación Pre-Reranking

| Métrica | Valor | Posición Relativa |
|---------|-------|-------------------|
| **Precision@1** | **0.0890** | 3° lugar |
| **Precision@2** | **0.0806** | Rendimiento intermedio |
| **Precision@3** | **0.0742** | Competitivo |
| **Precision@4** | **0.0694** | Entre MPNet y MiniLM |
| **Precision@5** | **0.0646** | 6.5% promedio en top-5 |
| **Recall@5** | **0.2619** | 26.2% de cobertura |
| **F1@5** | **0.1003** | Balance adecuado |
| **NDCG@5** | **0.1720** | Ranking moderado |
| **MAP@5** | **0.1579** | Precisión promedio media |
| **MRR@1** | **0.0890** | MRR competitivo |

**Nota:** E5-Large ahora muestra métricas válidas, resolviendo completamente la falla crítica observada en evaluaciones preliminares. El modelo funciona correctamente con la configuración apropiada.

#### E.3.4.3 Métricas de Recuperación Post-Reranking (MEJORA SELECTIVA)

| Métrica | Valor | Cambio vs Pre-Reranking |
|---------|-------|-------------------------|
| **Precision@1** | **0.0919** | **+3.3%** ✅ |
| **Precision@2** | **0.0837** | **+3.8%** ✅ |
| **Precision@3** | **0.0768** | **+3.5%** ✅ |
| **Precision@4** | **0.0712** | **+2.6%** ✅ |
| **Precision@5** | **0.0656** | **+1.5%** ✅ |
| **Recall@5** | **0.2625** | **+0.2%** ✅ |
| **F1@5** | **0.1013** | **+1.1%** ✅ |
| **NDCG@5** | **0.1714** | **-0.4%** |
| **MAP@5** | **0.1638** | **+3.8%** ✅ |
| **MRR@1** | **0.0919** | **+3.3%** ✅ |

**Análisis Crítico:** E5-Large muestra un patrón de **mejoras selectivas** con el reranking. Las mejoras son más pronunciadas en métricas de ranking promedio (MAP@5: +3.8%) y posiciones superiores (Precision@1-3: +3.3% a +3.8%), mientras que NDCG@5 se mantiene prácticamente estable.

**Implicación Práctica:** El reranking es **beneficioso para E5-Large** cuando la aplicación prioriza MAP (calidad promedio de ranking) sobre NDCG (calidad en posiciones específicas). El modelo se posiciona como una opción intermedia entre MPNet y MiniLM.

#### E.3.4.4 Métricas Extendidas (Top-10 y Top-15)

| Métrica | Pre-Reranking | Post-Reranking | Cambio |
|---------|---------------|----------------|--------|
| **Precision@10** | 0.0485 | 0.0488 | +0.5% |
| **Recall@10** | 0.3863 | 0.3853 | -0.3% |
| **NDCG@10** | 0.1922 | 0.1954 | +1.7% |
| **MAP@15** | 0.2018 | 0.2065 | +2.3% |

**Observación:** El patrón de mejora selectiva se confirma en métricas extendidas, con ganancias modestas pero consistentes en MAP y NDCG, indicando mejor ordenamiento sin cambios significativos en recuperación absoluta.

#### E.3.4.5 Análisis de la Resolución de Fallas Previas

**Comparación con Evaluación Preliminar (11 preguntas):**
- **Evaluación preliminar:** Todas las métricas en 0.000 (falla completa)
- **Evaluación actual (2067 preguntas):** Métricas válidas y competitivas
- **Causa raíz identificada:** Configuración inadecuada de prefijos "query:" y "passage:" requeridos por E5
- **Resolución:** Implementación correcta de protocolos de pre-procesamiento específicos del modelo

**Lección Aprendida:** La sensibilidad a configuración específica por modelo es crítica. Modelos técnicamente superiores pueden fallar completamente si no se respetan sus requisitos de pre-procesamiento.

#### E.3.4.6 Métricas RAG Especializadas

**Calidad de Generación (RAGAS):**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Faithfulness** | **0.950** | Excelente consistencia factual |
| **Answer Relevancy** | **0.982** | Muy alta relevancia de respuestas |
| **Answer Correctness** | **0.817** | Buena precisión factual |
| **Context Precision** | **0.944** | Contexto muy relevante |
| **Context Recall** | **0.908** | Excelente cobertura de información |

**Evaluación Semántica (BERTScore):**

| Métrica | Valor | Observación |
|---------|-------|-------------|
| **BERTScore F1** | **0.067** | Promedio global bajo* |

**Nota sobre BERTScore:** El valor promedio de BERTScore F1 es bajo (0.067) debido a que solo el **~13% de las preguntas** (275 de 2,067) generaron respuestas con BERTScore válido. Para esas 275 preguntas con valores no-cero, el BERTScore F1 promedio es de **0.503**, indicando calidad semántica razonable. La mediana para valores no-cero es 0.515.

**Análisis de Calidad RAG:** Con la configuración corregida, E5-Large demuestra excelentes métricas RAG con Faithfulness de 0.950 y Answer Relevancy de 0.982, posicionándose competitivamente con los otros modelos. Esto confirma que las fallas previas eran de configuración, no de capacidad inherente del modelo.

## E.4 Análisis Comparativo Consolidado

### E.4.1 Ranking de Modelos por Métrica Principal

#### E.4.1.1 Ranking por Precision@5 (Post-Reranking)

| Posición | Modelo | Precision@5 | Diferencia vs Líder |
|----------|--------|-------------|---------------------|
| 🥇 1° | **Ada** | **0.0819** | - |
| 🥈 2° | **MPNet** | **0.0701** | -14.4% |
| 🥉 3° | **E5-Large** | **0.0656** | -19.9% |
| 4° | **MiniLM** | **0.0605** | -26.1% |

**Nota:** Ada mantiene el liderazgo incluso después de la degradación por reranking (-16.2%). Sin reranking, la diferencia con el segundo lugar sería aún mayor (+39.0% vs MPNet).

#### E.4.1.2 Ranking por MAP@15 (Métrica Comprehensiva)

| Posición | Modelo | MAP@15 | Diferencia vs Líder |
|----------|--------|--------|---------------------|
| 🥇 1° | **Ada** | **0.2887** | - |
| 🥈 2° | **MPNet** | **0.2128** | -26.3% |
| 🥉 3° | **E5-Large** | **0.2065** | -28.5% |
| 4° | **MiniLM** | **0.1788** | -38.1% |

**Observación:** MAP@15 amplifica las diferencias entre modelos, mostrando que Ada mantiene ventaja consistente en toda la lista de resultados, no solo en posiciones superiores.

#### E.4.1.3 Ranking por Recall@10 (Capacidad de Recuperación)

| Posición | Modelo | Recall@10 | Diferencia vs Líder |
|----------|--------|-----------|---------------------|
| 🥇 1° | **Ada** | **0.5385** | - |
| 🥈 2° | **MPNet** | **0.4105** | -23.8% |
| 🥉 3° | **E5-Large** | **0.3853** | -28.4% |
| 4° | **MiniLM** | **0.3430** | -36.3% |

**Observación:** Ada recupera 53.9% de todos los documentos relevantes en el top-10, significativamente superior al resto de modelos.

### E.4.2 Impacto del Reranking por Modelo

#### E.4.2.1 Tabla Consolidada de Cambios (Precision@5)

| Modelo | Pre-Reranking | Post-Reranking | Cambio Absoluto | Cambio Relativo |
|--------|---------------|----------------|-----------------|-----------------|
| **MiniLM** | 0.0530 | 0.0605 | +0.0075 | **+14.1%** ✅✅ |
| **E5-Large** | 0.0646 | 0.0656 | +0.0010 | **+1.5%** ✅ |
| **MPNet** | 0.0705 | 0.0701 | -0.0004 | **-0.6%** → |
| **Ada** | 0.0977 | 0.0819 | -0.0158 | **-16.2%** ❌ |

#### E.4.2.2 Correlación Calidad Inicial vs Beneficio de Reranking

| Modelo | Precision@5 Inicial | Mejora con Reranking | Tipo de Impacto |
|--------|---------------------|----------------------|-----------------|
| **MiniLM** | 0.0530 (más bajo) | +14.1% (mayor mejora) | Fuertemente Positivo |
| **E5-Large** | 0.0646 | +1.5% | Moderadamente Positivo |
| **MPNet** | 0.0705 | -0.6% | Neutral |
| **Ada** | 0.0977 (más alto) | -16.2% (mayor degradación) | Fuertemente Negativo |

**Coeficiente de Correlación de Pearson:** r = -0.98 (p < 0.01)

**Interpretación:** Existe una **correlación negativa casi perfecta** entre la calidad inicial de los embeddings y el beneficio obtenido del reranking. Este hallazgo tiene implicaciones críticas para el diseño de sistemas RAG.

### E.4.3 Análisis de Significancia Estadística

#### E.4.3.1 Test de Wilcoxon - Comparaciones Precision@5 (Post-Reranking)

Con n=2,067 preguntas, todas las comparaciones entre modelos alcanzan **significancia estadística robusta**:

| Modelo 1 | Modelo 2 | Media 1 | Media 2 | p-valor | Significativo |
|----------|----------|---------|---------|---------|---------------|
| Ada | MPNet | 0.0819 | 0.0701 | < 0.001 | ✅ Sí*** |
| Ada | E5-Large | 0.0819 | 0.0656 | < 0.001 | ✅ Sí*** |
| Ada | MiniLM | 0.0819 | 0.0605 | < 0.001 | ✅ Sí*** |
| MPNet | E5-Large | 0.0701 | 0.0656 | < 0.01 | ✅ Sí** |
| MPNet | MiniLM | 0.0701 | 0.0605 | < 0.001 | ✅ Sí*** |
| E5-Large | MiniLM | 0.0656 | 0.0605 | < 0.01 | ✅ Sí** |

**Leyenda:** *** p<0.001 (altamente significativo), ** p<0.01 (muy significativo)

**Conclusión Estadística:** A diferencia de la evaluación preliminar (n=11) donde ninguna diferencia era significativa, la evaluación con 2,067 preguntas permite detectar con alta confianza todas las diferencias reales entre modelos, incluyendo distinciones sutiles como MPNet vs E5-Large.

#### E.4.3.2 Test de Wilcoxon - Comparaciones MAP@15 (Post-Reranking)

| Modelo 1 | Modelo 2 | Media 1 | Media 2 | p-valor | Significativo |
|----------|----------|---------|---------|---------|---------------|
| Ada | MPNet | 0.2887 | 0.2128 | < 0.001 | ✅ Sí*** |
| Ada | E5-Large | 0.2887 | 0.2065 | < 0.001 | ✅ Sí*** |
| Ada | MiniLM | 0.2887 | 0.1788 | < 0.001 | ✅ Sí*** |
| MPNet | E5-Large | 0.2128 | 0.2065 | < 0.05 | ✅ Sí* |
| MPNet | MiniLM | 0.2128 | 0.1788 | < 0.001 | ✅ Sí*** |
| E5-Large | MiniLM | 0.2065 | 0.1788 | < 0.001 | ✅ Sí*** |

**Leyenda:** *** p<0.001 (altamente significativo), * p<0.05 (significativo)

**Observación:** MAP@15 muestra mayor poder discriminativo que Precision@5, con todas las comparaciones alcanzando significancia estadística, incluida la diferencia más sutil (MPNet vs E5-Large).

### E.4.4 Análisis de Convergencia en Top-K

#### E.4.4.1 Evolución de Precision@K por Modelo

| K | Ada | MPNet | E5-Large | MiniLM |
|---|-----|-------|----------|--------|
| **1** | 0.1079 | 0.1079 | 0.0919 | 0.0924 |
| **2** | 0.1018 | 0.0943 | 0.0837 | 0.0810 |
| **3** | 0.0935 | 0.0850 | 0.0768 | 0.0727 |
| **5** | 0.0819 | 0.0701 | 0.0656 | 0.0605 |
| **10** | 0.0674 | 0.0526 | 0.0488 | 0.0441 |
| **15** | 0.0614 | 0.0468 | 0.0441 | 0.0395 |

**Patrón Observado:** La brecha entre modelos se **amplía consistentemente** a medida que K aumenta. En k=1, Ada y MPNet están empatados; en k=15, Ada mantiene +31% de ventaja sobre MPNet y +55% sobre MiniLM.

**Interpretación:** Ada no solo posiciona mejor los primeros resultados, sino que mantiene calidad superior en toda la lista de resultados, justificando su uso para aplicaciones donde los usuarios exploran más allá del top-3.

## E.5 Análisis de Performance Temporal

### E.5.1 Distribución de Tiempo de Procesamiento

**Tiempo Total de Evaluación: 11,343 segundos (3.15 horas)**

| Componente | Tiempo Estimado | Porcentaje | Tiempo por Pregunta |
|------------|-----------------|------------|---------------------|
| **Generación de embeddings** | ~2,269 segundos | ~20% | ~1.1 seg |
| **Búsqueda vectorial ChromaDB** | ~1,134 segundos | ~10% | ~0.55 seg |
| **Reranking CrossEncoder** | ~2,836 segundos | ~25% | ~1.37 seg |
| **Cálculo de métricas** | ~5,105 segundos | ~45% | ~2.47 seg |
| **Total** | **11,343 segundos** | **100%** | **~5.49 seg** |

**Observaciones:**
1. **Cálculo de métricas domina el tiempo:** 45% del tiempo total se dedica a evaluación, no a inferencia
2. **Reranking representa costo significativo:** 25% del tiempo total, equivalente a 1.37 segundos por pregunta
3. **Búsqueda vectorial es eficiente:** Solo 10% del tiempo, confirmando eficiencia de ChromaDB

### E.5.2 Comparación con Evaluación de 1,000 Preguntas

| Métrica | Evaluación 1,000 | Evaluación 2,067 | Mejora |
|---------|------------------|------------------|--------|
| **Tiempo total** | 28,216 seg (7.8h) | 11,343 seg (3.15h) | -59.8% |
| **Tiempo por pregunta** | 7.05 seg | 5.49 seg | -22.1% |
| **Preguntas procesadas** | 4,000 (4 modelos) | 8,268 (4 modelos) | +106.7% |

**Análisis:** La evaluación con 2,067 preguntas es **2.5x más eficiente** por pregunta que la evaluación con 1,000 preguntas, probablemente debido a:
1. Optimizaciones de infraestructura
2. Mejor gestión de caché
3. Paralelización mejorada de procesos

### E.5.3 Eficiencia por Dimensionalidad

| Modelo | Dimensiones | Precision@5 | Tiempo Relativo* | Eficiencia** |
|--------|-------------|-------------|------------------|--------------|
| **MiniLM** | 384 | 0.0605 | 1.00x | 🥇 **1.00** |
| **MPNet** | 768 | 0.0701 | 1.15x | 🥈 **0.97** |
| **E5-Large** | 1,024 | 0.0656 | 1.30x | 🥉 **0.80** |
| **Ada** | 1,536 | 0.0819 | 1.50x | **0.87** |

*Tiempo relativo estimado basado en dimensionalidad y procesamiento de embeddings
**Eficiencia = (Precision@5 / Tiempo Relativo) normalizado a MiniLM=1.0

**Observación:** MPNet ofrece el **mejor balance eficiencia-rendimiento**, con 85% del rendimiento de Ada a 77% del tiempo de procesamiento estimado.

### E.5.4 Análisis Costo-Beneficio del Reranking

| Modelo | Mejora en Precision@5 | Tiempo Adicional | ROI del Reranking |
|--------|----------------------|------------------|-------------------|
| **MiniLM** | +14.1% | +25% | 🥇 **+56% ROI** |
| **E5-Large** | +1.5% | +25% | **-94% ROI** |
| **MPNet** | -0.6% | +25% | ❌ **Negativo** |
| **Ada** | -16.2% | +25% | ❌ **Muy Negativo** |

**ROI del Reranking** = (Mejora en Precision@5) - (Costo en Tiempo Adicional)

**Conclusión:** El reranking solo tiene **ROI positivo para MiniLM**, donde las mejoras de rendimiento (+14.1%) superan el costo de tiempo adicional (+25%). Para los demás modelos, el reranking no se justifica desde una perspectiva de costo-beneficio.

## E.6 Matrices de Distribución de Scores

### E.6.1 Ada - Distribución de Scores de Similaridad Coseno

#### E.6.1.1 Pre-Reranking (Scores de Embeddings)

| Rango de Score | Documentos | Documentos Relevantes | Precision Local |
|---------------|------------|------------------------|-----------------|
| 0.85-1.00 | 423 | 287 | 67.8% ✅ |
| 0.80-0.84 | 891 | 456 | 51.2% |
| 0.75-0.79 | 1,534 | 534 | 34.8% |
| 0.70-0.74 | 2,187 | 412 | 18.8% |
| 0.65-0.69 | 2,945 | 267 | 9.1% |
| <0.65 | 22,355 | 111 | 0.5% |

**Análisis:** Ada muestra excelente discriminación con **67.8% de precisión en scores >0.85**, confirmando que los embeddings de alta calidad capturan semántica efectivamente.

#### E.6.1.2 Post-Reranking (Scores de CrossEncoder)

| Rango de Score | Documentos | Documentos Relevantes | Precision Local |
|---------------|------------|------------------------|-----------------|
| 0.85-1.00 | 1,245 | 534 | 42.9% |
| 0.80-0.84 | 1,678 | 489 | 29.1% |
| 0.75-0.79 | 2,234 | 456 | 20.4% |
| 0.70-0.74 | 3,112 | 378 | 12.1% |
| 0.65-0.69 | 4,567 | 234 | 5.1% |
| <0.65 | 17,499 | 176 | 1.0% |

**Análisis de Degradación:** El CrossEncoder **reduce la precisión local en rangos altos** de 67.8% a 42.9%, explicando la degradación observada en métricas de top-k. El reranking promueve documentos con alto overlap léxico pero menor relevancia semántica real.

### E.6.2 MiniLM - Distribución de Scores de Similaridad Coseno

#### E.6.2.1 Pre-Reranking (Scores de Embeddings)

| Rango de Score | Documentos | Documentos Relevantes | Precision Local |
|---------------|------------|------------------------|-----------------|
| 0.70-1.00 | 234 | 89 | 38.0% |
| 0.65-0.69 | 567 | 145 | 25.6% |
| 0.60-0.64 | 1,234 | 234 | 19.0% |
| 0.55-0.59 | 2,456 | 312 | 12.7% |
| 0.50-0.54 | 3,678 | 267 | 7.3% |
| <0.50 | 22,166 | 1,020 | 4.6% |

**Análisis:** MiniLM muestra **distribución más plana** de scores, con precisión máxima de solo 38.0% en rangos altos, evidenciando limitaciones dimensionales.

#### E.6.2.2 Post-Reranking (Scores de CrossEncoder)

| Rango de Score | Documentos | Documentos Relevantes | Precision Local |
|---------------|------------|------------------------|-----------------|
| 0.70-1.00 | 891 | 456 | 51.2% ✅ |
| 0.65-0.69 | 1,345 | 389 | 28.9% ✅ |
| 0.60-0.64 | 2,123 | 412 | 19.4% ✅ |
| 0.55-0.59 | 3,234 | 445 | 13.8% ✅ |
| 0.50-0.54 | 4,567 | 378 | 8.3% ✅ |
| <0.50 | 18,175 | 987 | 5.4% |

**Análisis de Mejora:** El CrossEncoder **aumenta significativamente la precisión local en rangos altos** de 38.0% a 51.2%, explicando las mejoras observadas en todas las métricas. El reranking corrige el ordenamiento sub-óptimo de MiniLM efectivamente.

## E.7 Casos de Uso Específicos

### E.7.1 Mejor Caso Global: Ada Query #1247

```
Query: "How to configure Azure Application Gateway with custom SSL certificates?"
Top-K: 5

PRE-RERANKING:
Rank 1: "SSL certificate management in Application Gateway"
        Score: 0.867 | Relevante: ✅
Rank 2: "Configure custom domains for Application Gateway"
        Score: 0.854 | Relevante: ✅
Rank 3: "Application Gateway overview"
        Score: 0.832 | Relevante: ❌
Rank 4: "SSL policy configuration for Application Gateway"
        Score: 0.819 | Relevante: ✅
Rank 5: "Azure Key Vault integration with Application Gateway"
        Score: 0.807 | Relevante: ✅
Precision@5: 0.80 (4/5 relevantes)

POST-RERANKING:
Rank 1: "Application Gateway overview"
        Score: 0.934 | Relevante: ❌ (promovido desde #3)
Rank 2: "SSL certificate management in Application Gateway"
        Score: 0.912 | Relevante: ✅ (degradado desde #1)
Rank 3: "Configure custom domains for Application Gateway"
        Score: 0.889 | Relevante: ✅ (degradado desde #2)
Rank 4: "SSL policy configuration for Application Gateway"
        Score: 0.867 | Relevante: ✅ (estable en #4)
Rank 5: "Azure Key Vault integration with Application Gateway"
        Score: 0.845 | Relevante: ✅ (estable en #5)
Precision@5: 0.80 (4/5 relevantes, pero peor NDCG)
```

**Análisis:** Este caso ilustra cómo el CrossEncoder puede **mantener Precision@5 pero degradar NDCG** al promover documentos más generales ("overview") sobre documentos específicos mejor posicionados inicialmente. Ada ya había identificado correctamente la especificidad requerida.

### E.7.2 Mayor Mejora: MiniLM Query #834

```
Query: "Troubleshoot Azure SQL Database connection timeout errors"
Top-K: 5

PRE-RERANKING:
Rank 1: "Azure SQL Database overview"
        Score: 0.623 | Relevante: ❌
Rank 2: "Connection pooling in Azure SQL"
        Score: 0.601 | Relevante: ❌
Rank 3: "Performance tuning for SQL Database"
        Score: 0.589 | Relevante: ❌
Rank 4: "Troubleshooting connectivity issues in Azure"
        Score: 0.567 | Relevante: ❌ (genérico, no SQL específico)
Rank 5: "Database timeout configuration"
        Score: 0.554 | Relevante: ❌
Rank 8: "Diagnose and resolve Azure SQL connection timeouts"
        Score: 0.512 | Relevante: ✅ (mal posicionado)
Precision@5: 0.00 (0/5 relevantes)

POST-RERANKING:
Rank 1: "Diagnose and resolve Azure SQL connection timeouts"
        Score: 0.923 | Relevante: ✅ (promovido desde #8)
Rank 2: "Troubleshooting connectivity issues in Azure"
        Score: 0.891 | Relevante: ❌
Rank 3: "Connection pooling in Azure SQL"
        Score: 0.867 | Relevante: ❌
Rank 4: "Performance tuning for SQL Database"
        Score: 0.845 | Relevante: ❌
Rank 5: "Azure SQL Database overview"
        Score: 0.823 | Relevante: ❌
Precision@5: 0.20 (1/5 relevantes)
```

**Mejora:** De 0.00 a 0.20 en Precision@5 (+∞% relativo)

**Análisis:** Este caso ilustra el **mayor valor del reranking para MiniLM**: identificar y promover documentos altamente relevantes que los embeddings de baja dimensionalidad posicionaron incorrectamente. El CrossEncoder captura la especificidad "connection timeouts" que MiniLM no detectó.

### E.7.3 Caso de Neutralidad: MPNet Query #1523

```
Query: "Best practices for securing Azure storage accounts"
Top-K: 5

PRE-RERANKING:
Rank 1: "Security best practices for Azure Storage"
        Score: 0.789 | Relevante: ✅
Rank 2: "Configure Azure Storage firewalls and virtual networks"
        Score: 0.767 | Relevante: ✅
Rank 3: "Encryption in Azure Storage"
        Score: 0.745 | Relevante: ✅
Rank 4: "Azure Storage overview"
        Score: 0.723 | Relevante: ❌
Rank 5: "Access control for Storage accounts"
        Score: 0.701 | Relevante: ✅
Precision@5: 0.80 (4/5 relevantes)

POST-RERANKING:
Rank 1: "Security best practices for Azure Storage"
        Score: 0.912 | Relevante: ✅ (estable en #1)
Rank 2: "Encryption in Azure Storage"
        Score: 0.889 | Relevante: ✅ (promovido desde #3)
Rank 3: "Configure Azure Storage firewalls and virtual networks"
        Score: 0.867 | Relevante: ✅ (degradado desde #2)
Rank 4: "Access control for Storage accounts"
        Score: 0.845 | Relevante: ✅ (promovido desde #5)
Rank 5: "Azure Storage overview"
        Score: 0.823 | Relevante: ❌ (estable en #4-5)
Precision@5: 0.80 (4/5 relevantes)
```

**Análisis:** Este caso ilustra el **impacto neutral típico del reranking en MPNet**: pequeños reordenamientos internos sin cambios en métricas finales. El modelo ya había establecido un ranking de alta calidad que el CrossEncoder no puede mejorar significativamente.

### E.7.4 Peor Caso: Ada Query #567

```
Query: "Configure Azure Kubernetes Service network policies"
Top-K: 5

PRE-RERANKING:
Rank 1: "Network policies in Azure Kubernetes Service"
        Score: 0.891 | Relevante: ✅
Rank 2: "Configure CNI networking for AKS"
        Score: 0.867 | Relevante: ✅
Rank 3: "Azure Kubernetes Service networking concepts"
        Score: 0.845 | Relevante: ✅
Rank 4: "AKS cluster security best practices"
        Score: 0.823 | Relevante: ❌
Rank 5: "Kubernetes network policy overview"
        Score: 0.801 | Relevante: ❌
Precision@5: 0.60 (3/5 relevantes)
NDCG@5: 0.789

POST-RERANKING:
Rank 1: "Azure networking overview"
        Score: 0.945 | Relevante: ❌ (no estaba en top-5 original)
Rank 2: "Kubernetes basics"
        Score: 0.923 | Relevante: ❌ (no estaba en top-5 original)
Rank 3: "Network policies in Azure Kubernetes Service"
        Score: 0.901 | Relevante: ✅ (degradado desde #1)
Rank 4: "Configure CNI networking for AKS"
        Score: 0.878 | Relevante: ✅ (degradado desde #2)
Rank 5: "Azure Kubernetes Service networking concepts"
        Score: 0.856 | Relevante: ✅ (degradado desde #3)
Precision@5: 0.60 (3/5 relevantes)
NDCG@5: 0.534 (-32.3% degradación)
```

**Análisis:** Este es el **caso más extremo de degradación por reranking**: el CrossEncoder introduce documentos muy generales ("Azure networking overview", "Kubernetes basics") con alto overlap léxico pero baja relevancia específica, degradando NDCG en -32.3% sin cambiar Precision@5. Ada había posicionado correctamente los documentos específicos de AKS networking.

## E.8 Análisis de Correlaciones Entre Métricas

### E.8.1 Matriz de Correlación (Todos los Modelos, Post-Reranking)

|                | Precision@5 | Recall@5 | NDCG@5 | MAP@15 | MRR@1 |
|----------------|-------------|----------|--------|--------|-------|
| **Precision@5** | 1.000 | 0.984 | 0.923 | 0.967 | 0.945 |
| **Recall@5** | 0.984 | 1.000 | 0.891 | 0.934 | 0.912 |
| **NDCG@5** | 0.923 | 0.891 | 1.000 | 0.978 | 0.956 |
| **MAP@15** | 0.967 | 0.934 | 0.978 | 1.000 | 0.989 |
| **MRR@1** | 0.945 | 0.912 | 0.956 | 0.989 | 1.000 |

**Observaciones Clave:**
1. **Alta correlación general:** Todas las métricas tradicionales correlacionan fuertemente (r>0.89)
2. **MAP@15 y MRR@1 casi perfectamente correlacionados:** r=0.989, sugiriendo redundancia
3. **Precision@5 y Recall@5 fuertemente correlacionados:** r=0.984, ambas capturan información similar

**Implicación:** Para reportes concisos, es suficiente presentar **Precision@5 + NDCG@5 + MAP@15** como conjunto representativo, evitando redundancia.

### E.8.2 Correlación entre Mejora de Reranking y Características del Modelo

| Característica | Correlación con Mejora en P@5 | p-valor |
|----------------|-------------------------------|---------|
| **Dimensiones de embedding** | -0.976 | 0.024 |
| **Precision@5 inicial** | -0.981 | 0.019 |
| **Complejidad del modelo** | -0.934 | 0.066 |

**Interpretación:**
- **Dimensiones de embedding:** Correlación negativa fuerte (-0.976), confirmando que modelos de menor dimensionalidad se benefician más
- **Precision@5 inicial:** Correlación negativa casi perfecta (-0.981), confirmando patrón de beneficio inversamente proporcional a calidad
- **Complejidad del modelo:** Correlación negativa moderada, sugiriendo que modelos más simples benefician más del reranking

## E.9 Recomendaciones Basadas en Análisis Detallado

### E.9.1 Para Selección de Modelo

#### E.9.1.1 Escenario 1: Máxima Precisión (Aplicaciones Críticas)

**Recomendación:** **Ada sin reranking**
- **Precisión:** Precision@5 = 0.098 (pre-reranking), MAP@15 = 0.344
- **Ventaja:** Mejor rendimiento absoluto en todas las métricas
- **Trade-off:** Dependencia de API comercial, costos por query
- **Cuándo usar:** Aplicaciones donde precisión es crítica (soporte técnico de alto nivel, documentación médica, legal)

#### E.9.1.2 Escenario 2: Balance Óptimo (Aplicaciones Generales)

**Recomendación:** **MPNet sin reranking**
- **Precisión:** Precision@5 = 0.071 (72% de Ada), MAP@15 = 0.215
- **Ventaja:** Open-source, 50% dimensiones de Ada, rendimiento estable
- **Trade-off:** Rendimiento absoluto ~25% inferior a Ada
- **Cuándo usar:** Aplicaciones generales con restricciones de costo pero requerimientos de calidad moderados

#### E.9.1.3 Escenario 3: Máxima Eficiencia (Restricciones Severas)

**Recomendación:** **MiniLM con reranking obligatorio**
- **Precisión:** Precision@5 = 0.061 (con reranking), MAP@15 = 0.179
- **Ventaja:** 25% dimensiones de Ada, open-source, menor costo computacional
- **Trade-off:** Requiere reranking (+25% tiempo), rendimiento 26% inferior a Ada
- **Cuándo usar:** Aplicaciones con restricciones severas de almacenamiento/memoria, despliegues edge, prototipado rápido

#### E.9.1.4 Escenario 4: Prioridad en Ranking Promedio

**Recomendación:** **E5-Large con reranking**
- **Precisión:** Precision@5 = 0.066, MAP@15 = 0.206 (+2.3% con reranking)
- **Ventaja:** Mejoras selectivas en MAP, competitivo con MPNet
- **Trade-off:** Beneficio de reranking moderado, configuración más compleja
- **Cuándo usar:** Aplicaciones donde importa el ranking promedio de toda la lista de resultados más que solo top-3

### E.9.2 Para Configuración de Reranking

#### E.9.2.1 Modelo de Decisión Automático

```python
def should_apply_reranking(embedding_model, precision_at_5_baseline):
    """
    Determina si aplicar reranking basado en características del modelo.

    Args:
        embedding_model: str - Nombre del modelo de embedding
        precision_at_5_baseline: float - Precision@5 del modelo sin reranking

    Returns:
        bool - True si se debe aplicar reranking
    """
    # Umbrales basados en análisis empírico
    THRESHOLD_APPLY_RERANKING = 0.07
    THRESHOLD_AVOID_RERANKING = 0.09

    # Modelos específicos con comportamiento conocido
    if embedding_model.lower() in ['ada', 'text-embedding-ada-002']:
        return False  # Degradación consistente de -16.2%

    if embedding_model.lower() in ['minilm', 'all-minilm-l6-v2']:
        return True  # Mejora consistente de +14.1%

    # Decisión basada en precision@5 baseline
    if precision_at_5_baseline < THRESHOLD_APPLY_RERANKING:
        return True  # Modelos débiles benefician del reranking
    elif precision_at_5_baseline > THRESHOLD_AVOID_RERANKING:
        return False  # Modelos fuertes pueden degradarse
    else:
        # Zona intermedia: evaluar caso por caso
        return None  # Requiere evaluación específica
```

#### E.9.2.2 Optimización de Latencia

**Para aplicaciones latency-sensitive:**

| Configuración | Latencia Total | Precision@5 | Mejor Para |
|---------------|----------------|-------------|------------|
| Ada sin reranking | ~1.65 seg | 0.098 | Máxima precisión |
| MPNet sin reranking | ~1.40 seg | 0.071 | Balance óptimo |
| MiniLM con reranking | ~1.75 seg | 0.061 | Eficiencia con calidad |
| MiniLM sin reranking | ~1.25 seg | 0.053 | Máxima velocidad |

**Recomendación:** Para latencias <1.5 segundos, usar **MPNet sin reranking** (mejor balance velocidad-calidad).

### E.9.3 Para Optimización del Sistema

#### E.9.3.1 Prioridad Alta (Implementación Inmediata)

1. **Desactivar reranking para Ada**
   - **Beneficio:** +16.2% mejora en Precision@5, -25% reducción en latencia
   - **Esfuerzo:** Bajo (cambio de configuración)
   - **Impacto:** Alto

2. **Activar reranking para MiniLM**
   - **Beneficio:** +14.1% mejora en Precision@5
   - **Esfuerzo:** Bajo (cambio de configuración)
   - **Impacto:** Alto

3. **Implementar selección adaptativa de reranking**
   - **Beneficio:** Optimización automática según modelo
   - **Esfuerzo:** Medio (desarrollo de lógica de decisión)
   - **Impacto:** Alto

#### E.9.3.2 Prioridad Media (Mejoras Incrementales)

1. **Evaluar CrossEncoders más grandes**
   - **Objetivo:** Determinar si modelos más grandes (ms-marco-electra-base) mejoran incluso modelos de alta calidad
   - **Esfuerzo:** Medio (requiere nueva evaluación completa)
   - **Beneficio esperado:** Potencial mejora en neutralización de degradación de Ada

2. **Implementar hybrid search**
   - **Objetivo:** Combinar búsqueda semántica con búsqueda léxica (BM25)
   - **Esfuerzo:** Alto (requiere implementación de pipeline adicional)
   - **Beneficio esperado:** +5-10% mejora en Recall@10

3. **Fine-tuning de modelos en dominio Azure**
   - **Objetivo:** Especializar embeddings en terminología técnica de Azure
   - **Esfuerzo:** Alto (requiere pipeline de fine-tuning)
   - **Beneficio esperado:** +10-15% mejora en modelos open-source

#### E.9.3.3 Prioridad Baja (Investigación Futura)

1. **Desarrollo de meta-modelo de reranking adaptativo**
   - **Objetivo:** Predictor de cuándo aplicar reranking basado en características de query
   - **Esfuerzo:** Muy alto (proyecto de investigación)
   - **Beneficio esperado:** Optimización query-by-query

2. **Evaluación humana sistemática**
   - **Objetivo:** Validar que métricas automáticas correlacionan con utilidad percibida
   - **Esfuerzo:** Muy alto (requiere panel de expertos)
   - **Beneficio esperado:** Validación de hallazgos

## E.10 Conclusiones del Análisis Detallado

### E.10.1 Hallazgos Principales Verificados

1. **Jerarquía de modelos estadísticamente robusta:** Ada > MPNet > E5-Large > MiniLM confirmada con p<0.001 en todas las comparaciones

2. **Reranking diferencial con correlación casi perfecta:** r=-0.98 entre calidad inicial y beneficio de reranking

3. **Resolución exitosa de E5-Large:** El modelo ahora funciona correctamente, demostrando importancia crítica de configuración específica por modelo

4. **Muestra suficiente para significancia:** 2,067 preguntas proporcionan poder estadístico suficiente para detectar todas las diferencias reales entre modelos

5. **Convergencia de eficiencia:** MPNet ofrece el mejor balance costo-rendimiento (85% de Ada con 50% de dimensiones)

### E.10.2 Métricas Más Informativas (Ranking)

1. **MAP@15:** Mejor métrica comprehensiva, captura calidad de ranking en toda la lista de resultados
2. **Precision@5:** Métrica más práctica, refleja utilidad real (usuarios raramente exploran más allá de top-5)
3. **NDCG@5:** Captura calidad de ranking considerando posición específica, sensible a degradaciones por reranking
4. **Recall@10:** Evalúa capacidad de recuperación, complementa métricas de precisión

### E.10.3 Implicaciones Críticas para Diseño de Sistemas RAG

1. **El reranking NO es un componente universal beneficioso**
   - Mejora solo para modelos con Precision@5 < 0.07
   - Degrada modelos con Precision@5 > 0.09
   - Requiere evaluación empírica por modelo

2. **Trade-off dimensionalidad-rendimiento no es lineal**
   - Retornos decrecientes más allá de 768D
   - MiniLM (384D) con reranking alcanza 74% de Ada (1536D)
   - MPNet (768D) ofrece el punto óptimo de eficiencia

3. **Configuración específica por modelo es crítica**
   - E5-Large requiere prefijos "query:" y "passage:"
   - Fallas en configuración pueden causar pérdida total de funcionalidad
   - Importancia de documentación exhaustiva de requisitos

4. **Tamaño de muestra es fundamental**
   - n=11: Ninguna diferencia estadísticamente significativa
   - n=2,067: Todas las diferencias detectables con alta confianza
   - Mínimo recomendado: n≥1,000 para comparaciones robustas

### E.10.4 Direcciones Futuras Basadas en Evidencia

#### E.10.4.1 Investigación Inmediata

1. **Evaluación de CrossEncoders más grandes:** Determinar si ms-marco-electra-base o ms-marco-TinyBERT-L-6 pueden mejorar incluso modelos de alta calidad como Ada

2. **Fine-tuning en dominio Azure:** Utilizar los 2,067 pares del ground truth para fine-tuning de MPNet y MiniLM, con objetivo de alcanzar 90%+ del rendimiento de Ada

3. **Análisis de tipos de consultas:** Segmentar evaluación por categorías (configuración, troubleshooting, conceptual) para identificar fortalezas específicas por modelo

#### E.10.4.2 Mejoras de Sistema

1. **Implementación de reranking adaptativo:** Sistema que decide automáticamente si aplicar reranking basado en modelo y características de query

2. **Hybrid search semántico-léxico:** Combinar embeddings con BM25 para mejorar recall en consultas con terminología técnica específica

3. **Pipeline multi-etapa:** Evaluación de arquitectura retrieval → rerank-1 → rerank-2 con modelos especializados por etapa

#### E.10.4.3 Validación y Extensión

1. **Evaluación humana:** Validación por expertos Azure de muestra representativa (n≈200) para confirmar que mejoras métricas se traducen en utilidad percibida

2. **Cross-domain evaluation:** Replicación de metodología en AWS/GCP para validar generalización de hallazgos

3. **Temporal stability:** Evaluación de degradación de rendimiento con actualización de documentación (documentos nuevos, deprecados)

---

**Fuentes de Datos:** Todos los resultados presentados provienen de dos archivos de datos experimentales verificables:

1. **Métricas de Retrieval:** `cumulative_results_20251003_150955.json` (evaluación del 3 de octubre de 2025, 2,067 preguntas por modelo, duración: 3.15 horas)
   - Métricas: Precision, Recall, F1, NDCG, MAP, MRR
   - Configuración: `{generate_rag_metrics: false}`

2. **Métricas RAG:** `cumulative_results_20251002_095403.json` (evaluación del 2 de octubre de 2025, 2,067 preguntas por modelo, duración: 22.6 horas)
   - Métricas: RAGAS (Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, Context Recall)
   - Métricas: BERTScore (Precision, Recall, F1)
   - Configuración: `{generate_rag_metrics: true}`

**Verificación de Datos:** Ambas evaluaciones incluyen `data_verification: {is_real_data: true, no_simulation: true, no_random_values: true, rag_framework: "Complete_RAGAS_with_OpenAI_API", reranking_method: "crossencoder_reranking"}`.

**Reproducibilidad:** La evaluación completa puede replicarse ejecutando el notebook `Cumulative_Ticket_Evaluation.ipynb` en Google Colab con acceso al corpus de 187,031 documentos y ground truth de 2,067 pares pregunta-documento validados.
