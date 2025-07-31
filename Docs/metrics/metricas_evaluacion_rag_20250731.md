# Métricas de Evaluación para Sistemas de Recuperación de Información Aumentada por Generación (RAG)

**Fecha**: 31 de Julio, 2025  
**Versión**: 1.0  
**Autor**: Proyecto de Investigación - Sistema de Soporte Técnico  

## Resumen Ejecutivo

Este documento presenta una revisión exhaustiva de las métricas de evaluación utilizadas en sistemas de Recuperación de Información Aumentada por Generación (Retrieval-Augmented Generation, RAG). Se analizan tres categorías principales de métricas: métricas de recuperación de información tradicionales, métricas específicas del framework RAGAS, y métricas de evaluación semántica BERTScore. El análisis incluye fundamentos teóricos, formulaciones matemáticas, interpretación práctica y aplicabilidad en dominios técnicos especializados.

## 1. Introducción

Los sistemas de Recuperación de Información Aumentada por Generación (RAG) combinan técnicas de recuperación de información con modelos de generación de lenguaje natural para producir respuestas contextualmente relevantes y factualmente precisas (Lewis et al., 2020). La evaluación de estos sistemas requiere un conjunto comprehensivo de métricas que capture tanto la efectividad de la recuperación como la calidad de la generación.

### 1.1 Desafíos en la Evaluación de Sistemas RAG

La evaluación de sistemas RAG presenta desafíos únicos que difieren de los sistemas tradicionales de recuperación de información o generación de texto de manera aislada:

1. **Evaluación Multi-dimensional**: Necesidad de evaluar tanto la recuperación de documentos relevantes como la calidad de las respuestas generadas
2. **Dependencia Contextual**: La calidad de la generación depende críticamente de la relevancia y completitud de los documentos recuperados
3. **Evaluación de Fidelidad**: Verificación de que las respuestas generadas se mantengan fieles al contenido recuperado
4. **Escalabilidad de Evaluación**: Necesidad de métricas automatizadas que permitan evaluación a gran escala

## 2. Métricas de Recuperación de Información Tradicionales

Las métricas de recuperación de información tradicionales evalúan la efectividad del componente de recuperación del sistema RAG, determinando qué tan bien el sistema identifica y ordena documentos relevantes para una consulta dada.

### 2.1 Precision@k

La Precision@k mide la proporción de documentos relevantes entre los primeros k documentos recuperados.

**Formulación Matemática:**
```
Precision@k = |{documentos relevantes} ∩ {top-k documentos recuperados}| / k
```

**Interpretación:**
- Rango: [0, 1]
- Valor 1: Todos los documentos en top-k son relevantes
- Valor 0: Ningún documento en top-k es relevante

**Aplicación en Contexto RAG:**
La Precision@k es particularmente relevante en sistemas RAG porque determina la calidad del contexto proporcionado al modelo generativo. Manning et al. (2008) señalan que la precisión en posiciones tempranas del ranking es crítica para aplicaciones donde el número de documentos utilizados como contexto es limitado.

**Ejemplo Práctico:**
Para una consulta sobre "configuración de Azure Storage encryption" con k=5:
- Documentos recuperados: [Doc1: Azure Storage, Doc2: Azure Encryption, Doc3: AWS Storage, Doc4: Azure Keys, Doc5: Google Cloud]
- Documentos relevantes: Doc1, Doc2, Doc4
- Precision@5 = 3/5 = 0.60

### 2.2 Recall@k

El Recall@k mide la proporción de documentos relevantes que son recuperados entre los primeros k documentos.

**Formulación Matemática:**
```
Recall@k = |{documentos relevantes} ∩ {top-k documentos recuperados}| / |{total documentos relevantes}|
```

**Interpretación:**
- Rango: [0, 1]
- Valor 1: Todos los documentos relevantes están en top-k
- Valor 0: Ningún documento relevante está en top-k

**Limitaciones en Contextos Prácticos:**
El cálculo de Recall@k requiere conocimiento completo del conjunto de documentos relevantes, lo cual es raramente disponible en aplicaciones del mundo real. Buckley & Voorhees (2004) argumentan que esta limitación hace que el recall sea más útil como métrica de investigación que como métrica operacional.

### 2.3 F1-Score@k

El F1-Score@k combina Precision@k y Recall@k en una métrica única que representa la media armónica de ambas.

**Formulación Matemática:**
```
F1@k = 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)
```

**Ventajas:**
- Proporciona balance entre precision y recall
- Penaliza desbalances extremos entre precision y recall
- Permite comparación holística entre sistemas

**Aplicación en Sistemas RAG:**
El F1-Score@k es especialmente valioso en sistemas RAG donde tanto la cobertura de información relevante como la precisión del contexto son críticas para la calidad de la respuesta generada.

### 2.4 Normalized Discounted Cumulative Gain (NDCG@k)

NDCG@k evalúa la calidad del ranking considerando tanto la relevancia como la posición de los documentos recuperados.

**Formulación Matemática:**
```
DCG@k = rel₁ + Σᵢ₌₂ᵏ (relᵢ / log₂(i+1))

IDCG@k = DCG ideal ordenando documentos por relevancia descendente

NDCG@k = DCG@k / IDCG@k
```

**Características:**
- Considera relevancia gradual (no binaria)
- Penaliza documentos relevantes en posiciones bajas
- Normalizado para permitir comparaciones entre consultas

**Relevancia en RAG:**
Järvelin & Kekäläinen (2002) demuestran que NDCG es particularmente efectivo para evaluar sistemas donde el orden de presentación afecta la utilidad, como en sistemas RAG donde los primeros documentos tienen mayor influencia en la generación.

### 2.5 Mean Average Precision (MAP@k)

MAP@k calcula la media de la precisión promedio para cada consulta, considerando todas las posiciones donde aparecen documentos relevantes.

**Formulación Matemática:**
```
AP@k = (1/|rel|) × Σᵢ₌₁ᵏ (Precision@i × rel(i))

MAP@k = (1/|Q|) × Σᵩ₌₁|Q| AP@k(q)
```

donde rel(i) = 1 si el documento en posición i es relevante, 0 en caso contrario.

**Interpretación:**
MAP@k proporciona una medida comprehensiva de la calidad del ranking que considera tanto el número como las posiciones de los documentos relevantes recuperados.

### 2.6 Mean Reciprocal Rank (MRR)

MRR evalúa la efectividad del sistema considerando la posición del primer documento relevante recuperado.

**Formulación Matemática:**
```
RR = 1/rank_primer_relevante

MRR = (1/|Q|) × Σᵢ₌₁|Q| RRᵢ
```

**Aplicación Específica:**
MRR es particularmente relevante en sistemas RAG donde la respuesta puede ser efectivamente generada basándose en un único documento altamente relevante, como en sistemas de preguntas y respuestas técnicas.

## 3. Métricas del Framework RAGAS

RAGAS (Retrieval Augmented Generation Assessment) es un framework especializado para la evaluación de sistemas RAG que introduce métricas específicamente diseñadas para evaluar aspectos únicos de estos sistemas (Es et al., 2023).

### 3.1 Faithfulness (Fidelidad)

Faithfulness evalúa el grado en que la respuesta generada se mantiene fiel al contexto proporcionado, evitando alucinaciones o contradicciones.

**Definición Formal:**
```
Faithfulness = |{claims verificables en contexto}| / |{total claims en respuesta}|
```

**Metodología de Evaluación:**
1. Extracción de afirmaciones (claims) de la respuesta generada
2. Verificación de cada afirmación contra el contexto proporcionado
3. Cálculo de la proporción de afirmaciones verificables

**Importancia en Sistemas Técnicos:**
En dominios técnicos como documentación de Azure, la fidelidad es crítica porque información incorrecta puede llevar a configuraciones erróneas o fallos de seguridad.

### 3.2 Answer Relevancy (Relevancia de Respuesta)

Answer Relevancy mide qué tan bien la respuesta generada aborda la pregunta planteada, independientemente de su fidelidad al contexto.

**Formulación Conceptual:**
```
Answer Relevancy = similitud_semántica(pregunta_original, pregunta_reconstruida_desde_respuesta)
```

**Proceso de Evaluación:**
1. Generación de preguntas que podrían ser respondidas por la respuesta generada
2. Cálculo de similitud semántica entre pregunta original y preguntas generadas
3. Promedio de similitudes como medida de relevancia

**Consideraciones Técnicas:**
La métrica utiliza modelos de lenguaje para generar preguntas contrafactuales, lo que introduce dependencia en la calidad del modelo evaluador.

### 3.3 Answer Correctness (Corrección de Respuesta)

Answer Correctness evalúa la precisión factual de la respuesta generada comparándola con una respuesta de referencia (ground truth).

**Componentes:**
```
Answer Correctness = w₁ × Exactitud_Factual + w₂ × Completitud_Semántica
```

donde w₁ + w₂ = 1 son pesos que balancean exactitud y completitud.

**Desafíos de Implementación:**
- Requiere respuestas de referencia de alta calidad
- Evaluación subjetiva de completitud semántica
- Sensibilidad a variaciones lingüísticas en respuestas correctas

### 3.4 Context Precision (Precisión de Contexto)

Context Precision evalúa qué tan relevantes son los documentos recuperados para responder la pregunta específica.

**Formulación:**
```
Context Precision@k = Σᵢ₌₁ᵏ (relevancia(docᵢ) × Πⱼ₌₁ⁱ⁻¹ irrelevancia(docⱼ)) / Σᵢ₌₁ᵏ Πⱼ₌₁ⁱ⁻¹ irrelevancia(docⱼ)
```

**Interpretación:**
La métrica penaliza documentos irrelevantes en posiciones tempranas del ranking, reflejando la importancia del orden en sistemas RAG.

### 3.5 Context Recall (Recall de Contexto)

Context Recall mide qué tan completo es el contexto recuperado en relación a la información necesaria para generar la respuesta ground truth.

**Definición:**
```
Context Recall = |{oraciones en ground truth atribuibles a contexto}| / |{total oraciones en ground truth}|
```

**Proceso de Evaluación:**
1. Segmentación de respuesta ground truth en oraciones
2. Verificación de atribuibilidad de cada oración al contexto recuperado
3. Cálculo de proporción de oraciones atribuibles

### 3.6 Semantic Similarity (Similitud Semántica)

Semantic Similarity cuantifica la similitud semántica entre la respuesta generada y la respuesta ground truth utilizando embeddings de texto.

**Formulación:**
```
Semantic Similarity = cosine_similarity(embedding(respuesta_generada), embedding(ground_truth))
```

**Ventajas:**
- Captura similitudes semánticas más allá de coincidencias léxicas
- Robusta a variaciones sintácticas
- Escalable computacionalmente

## 4. Métricas BERTScore

BERTScore utiliza representaciones contextuales pre-entrenadas para evaluar la calidad del texto generado mediante similitud semántica a nivel de token (Zhang et al., 2019).

### 4.1 Fundamentación Teórica

BERTScore supera limitaciones de métricas tradicionales como BLEU al:
- Utilizar representaciones contextuales en lugar de n-gramas
- Capturar similitudes semánticas profundas
- Ser robusta a variaciones superficiales

### 4.2 BERT Precision

**Formulación:**
```
BERT Precision = (1/|x|) × Σₓᵢ∈ₓ max_{ŷⱼ∈ŷ} cos(𝐱ᵢ, 𝐲̂ⱼ)
```

donde x es la respuesta generada, ŷ es la referencia, y 𝐱ᵢ, 𝐲̂ⱼ son embeddings BERT.

**Interpretación:**
Mide qué proporción de tokens en la respuesta generada tienen correspondencias semánticamente similares en la referencia.

### 4.3 BERT Recall

**Formulación:**
```
BERT Recall = (1/|ŷ|) × Σ_{ŷⱼ∈ŷ} max_{xᵢ∈x} cos(𝐱ᵢ, 𝐲̂ⱼ)
```

**Interpretación:**
Mide qué proporción de tokens en la referencia tienen correspondencias semánticamente similares en la respuesta generada.

### 4.4 BERT F1

**Formulación:**
```
BERT F1 = 2 × (BERT Precision × BERT Recall) / (BERT Precision + BERT Recall)
```

**Ventajas en Evaluación RAG:**
- Evalúa tanto completitud como precisión semántica
- Menos sensible a variaciones estilísticas
- Correlaciona mejor con juicio humano que métricas léxicas

## 5. Consideraciones para Dominios Técnicos Especializados

### 5.1 Desafíos Específicos

La evaluación de sistemas RAG en dominios técnicos presenta desafíos únicos:

1. **Terminología Especializada**: Vocabulario técnico requiere embeddings domain-específicos
2. **Precisión Crítica**: Errores pueden tener consecuencias operacionales severas
3. **Contexto Extenso**: Documentación técnica requiere procesamiento de contextos largos
4. **Evaluación Multi-modal**: Inclusión de diagramas, código, y configuraciones

### 5.2 Adaptaciones Metodológicas

**Ground Truth Escaso:**
En dominios especializados, el ground truth típicamente contiene un único documento relevante por consulta, limitando la precision máxima teórica:

```
Precision@k_max = 1/k
```

**Métricas Adaptadas:**
- Success@k: Presencia del documento correcto en top-k
- Reciprocal Rank: Inverso de la posición del documento correcto
- Coverage: Proporción de consultas con al menos un documento relevante en top-k

### 5.3 Validación Estadística

La evaluación en dominios especializados requiere validación estadística rigurosa debido a:
- Tamaños de muestra limitados
- Variabilidad en complejidad de consultas
- Sesgos de evaluación humana

**Pruebas Recomendadas:**
- Test de Wilcoxon para comparación de distribuciones no paramétricas
- Bootstrap para estimación de intervalos de confianza
- Análisis de potencia estadística para determinar tamaños de muestra

## 6. Integración de Métricas en Pipeline de Evaluación

### 6.1 Arquitectura de Evaluación Multi-dimensional

Un sistema de evaluación robusto para RAG debe integrar las tres categorías de métricas:

1. **Fase de Recuperación**: Métricas IR tradicionales evalúan calidad del ranking
2. **Fase de Generación**: Métricas RAGAS evalúan calidad de respuesta contextual
3. **Fase de Validación**: BERTScore proporciona evaluación semántica automatizada

### 6.2 Ponderación de Métricas

La combinación de múltiples métricas requiere esquemas de ponderación apropiados:

```
Score_Compuesto = α₁ × NDCG@k + α₂ × Faithfulness + α₃ × BERT_F1
```

donde α₁ + α₂ + α₃ = 1 y los pesos reflejan la importancia relativa de cada aspecto.

### 6.3 Interpretación Holística

La interpretación de resultados debe considerar:
- Correlaciones entre métricas de diferentes categorías
- Trade-offs entre precision y recall en recuperación
- Balance entre fidelidad y completitud en generación

## 7. Limitaciones y Direcciones Futuras

### 7.1 Limitaciones Actuales

1. **Dependencia en Ground Truth**: Muchas métricas requieren datos de referencia de alta calidad
2. **Evaluación Subjetiva**: Aspectos como relevancia y calidad requieren juicio humano
3. **Escalabilidad**: Evaluación manual no escala a sistemas de producción
4. **Sesgo de Modelo**: Métricas basadas en LLM heredan sesgos del modelo evaluador

### 7.2 Direcciones de Investigación

1. **Métricas sin Referencia**: Desarrollo de métricas que no requieran ground truth
2. **Evaluación Multimodal**: Extensión a contenido visual y multimedia
3. **Métricas Específicas de Dominio**: Desarrollo de métricas especializadas por dominio
4. **Evaluación Continua**: Sistemas de evaluación en tiempo real para entornos de producción

## 8. Conclusiones

La evaluación de sistemas RAG requiere un enfoque multi-dimensional que capture tanto la efectividad de la recuperación como la calidad de la generación. Las métricas tradicionales de recuperación de información proporcionan fundamentos sólidos para evaluar el componente de recuperación, mientras que frameworks especializados como RAGAS y métricas semánticas como BERTScore abordan aspectos únicos de la generación aumentada.

En dominios técnicos especializados, las limitaciones de ground truth escaso requieren adaptaciones metodológicas y métricas alternativas. La validación estadística rigurosa es esencial para conclusiones confiables, y la integración de múltiples métricas proporciona una evaluación más comprehensiva que métricas individuales.

El desarrollo continuo de métricas de evaluación para sistemas RAG permanece como un área activa de investigación, con necesidades particulares en dominios especializados donde la precisión y fidelidad son críticas para aplicaciones operacionales.

## Referencias

Buckley, C., & Voorhees, E. M. (2004). Retrieval evaluation with incomplete information. *Proceedings of the 27th annual international ACM SIGIR conference on Research and development in information retrieval*, 25-32.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. *arXiv preprint arXiv:2309.15217*.

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). BERTScore: Evaluating text generation with BERT. *arXiv preprint arXiv:1904.09675*.

---

**Nota**: Este documento forma parte de la metodología de evaluación desarrollada para el proyecto de investigación sobre sistemas de soporte técnico automatizado. Las métricas y metodologías descritas han sido implementadas y validadas empíricamente en el contexto de documentación técnica de Microsoft Azure.