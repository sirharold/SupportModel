# 2. ESTADO DEL ARTE

## 2.1 Introducción

El procesamiento de lenguaje natural ha transformado la gestión del conocimiento y el soporte técnico en las organizaciones modernas. La integración de modelos de lenguaje preentrenados, motores de búsqueda vectorial y bases de conocimiento especializadas ha permitido desarrollar sistemas más eficientes y precisos para la recuperación de información técnica. Este capítulo examina el estado actual en la aplicación de NLP al soporte técnico, con énfasis en arquitecturas RAG, bases de datos vectoriales, modelos de embeddings especializados y métricas de evaluación avanzadas.

La evolución desde sistemas tradicionales basados en coincidencia léxica hacia sistemas semánticos capaces de comprender contexto e intención ha marcado un punto de inflexión en la automatización del soporte técnico. Este cambio de paradigma es especialmente relevante en dominios técnicos complejos como Azure, donde la terminología especializada y la interrelación entre servicios requieren un enfoque semántico sofisticado.

## 2.2 NLP Aplicado a Soporte Técnico

El soporte técnico ha sido tradicionalmente un proceso dependiente del conocimiento tácito y la experiencia humana, lo que genera inconsistencias, demoras y errores sistemáticos. Las técnicas avanzadas de NLP permiten automatizar y mejorar tareas críticas como la clasificación de tickets, la identificación del propósito de consultas, la recuperación de respuestas relevantes y la recomendación de soluciones contextualizadas.

### 2.2.1 Evolución de los Modelos de Lenguaje

Los enfoques contemporáneos han evolucionado desde modelos estadísticos simples hacia arquitecturas transformer sofisticadas. BERT (Bidirectional Encoder Representations from Transformers) y sus variantes como RoBERTa, DistilBERT y DeBERTa han demostrado resultados superiores en tareas de clasificación multiclase y multilabel en dominios técnicos (Devlin et al., 2018; Liu et al., 2019; He et al., 2020). Cuando se especializan mediante fine-tuning en corpus técnicos, estos modelos capturan matices semánticos que los modelos generales no logran identificar.

La aparición de modelos especializados como Sentence-BERT (SBERT) revolucionó la generación de embeddings para tareas de recuperación semántica (Reimers & Gurevych, 2019). A diferencia de BERT tradicional, SBERT está optimizado para generar representaciones vectoriales densas que preservan la similitud semántica, fundamental para sistemas de recuperación de información técnica.

### 2.2.2 Arquitecturas RAG en Soporte Técnico

Las arquitecturas RAG (Retrieval-Augmented Generation) han emergido como el estándar para sistemas de soporte técnico que combinan recuperación de información con generación de respuestas (Lewis et al., 2020). Estas arquitecturas permiten que los modelos de lenguaje accedan dinámicamente a bases de conocimiento externas durante la generación, superando las limitaciones de memoria y actualización de los modelos parametrizados.

En el contexto del soporte técnico, los sistemas RAG implementan típicamente un pipeline de dos etapas: recuperación de documentos relevantes mediante búsqueda vectorial, seguida de generación de respuestas contextualizadas utilizando los documentos recuperados. La efectividad de estos sistemas depende críticamente de la calidad de los embeddings y la precisión del mecanismo de recuperación.

### 2.2.3 Aplicaciones Empresariales Actuales

Empresas tecnológicas líderes como IBM, SAP y Microsoft han implementado soluciones NLP para análisis semántico de tickets y generación de respuestas automatizadas (Saxena et al., 2021). Estas implementaciones incluyen módulos de clasificación automática, extracción de entidades, y sistemas de recomendación basados en similitud semántica.

El uso de técnicas de resumen automático extractivo y abstractivo permite procesar tickets extensos y extraer información clave para facilitar la priorización y el enrutamiento inteligente (Gupta & Gupta, 2020). Estos enfoques son valiosos en entornos de alto volumen donde la clasificación manual resulta impracticable.

## 2.3 Bases de Conocimiento como Entrada para Recuperación de Información

### 2.3.1 Transición hacia Recuperación Semántica

Una tendencia dominante en la industria es la utilización de bases de conocimiento estructuradas como corpus semántico para alimentar sistemas de recuperación en tareas de soporte. Estas bases incluyen documentación técnica oficial, artículos de resolución de problemas, FAQ especializadas y respuestas validadas por la comunidad.

Los métodos tradicionales de recuperación basados en TF-IDF o BM25 han dado paso progresivamente a técnicas vectoriales que representan textos como embeddings densos, capturando relaciones semánticas más profundas y contextuales (Johnson et al., 2019). Esta transición ha sido impulsada por el surgimiento de modelos como Sentence-BERT, que permiten generar representaciones vectoriales eficientes y semánticamente coherentes para documentos y consultas.

### 2.3.2 Arquitecturas de Embeddings Especializados

Los sistemas modernos de recuperación implementan arquitecturas de embeddings especializados que van más allá de los modelos generales. Modelos como E5 (Embeddings from bidirectional Encoder representations) han demostrado rendimiento superior en benchmarks de recuperación semántica en dominios técnicos (Wang et al., 2022). Estos modelos utilizan estrategias de preentrenamiento contrastivo que optimizan la tarea de recuperación.

La familia de modelos MPNet (Masked and Permuted Pre-training) combina las ventajas de BERT y XLNet, resultando en representaciones más robustas para tareas de recuperación de información técnica (Song et al., 2020). Por otro lado, modelos como MiniLM ofrecen un balance optimizado entre rendimiento y eficiencia computacional, siendo útiles en aplicaciones de producción con restricciones de recursos (Wang et al., 2020).

### 2.3.3 Integración con Bases de Datos Vectoriales

El uso de retrievers vectoriales se ha vuelto esencial en arquitecturas RAG modernas. Bases de datos vectoriales especializadas como ChromaDB, FAISS, Milvus y Weaviate han surgido como soluciones optimizadas para almacenamiento y recuperación eficiente de vectores de alta dimensión (Johnson et al., 2019; Douze et al., 2024).

Inicialmente, este proyecto utilizó Weaviate como base de datos vectorial por su robustez empresarial, arquitectura distribuida, integración nativa con múltiples modelos de lenguaje (OpenAI, Cohere, Hugging Face), y capacidades avanzadas de consulta mediante GraphQL (Weaviate, 2023). Sin embargo, durante el desarrollo se migró a ChromaDB por consideraciones prácticas del entorno de investigación académica: eliminación de costos de infraestructura cloud, reducción sustancial de latencia al operar localmente, compatibilidad nativa con Google Colab sin configuración adicional, y simplicidad de despliegue sin requerimientos de servicios externos.

ChromaDB mantiene las capacidades esenciales requeridas: filtrado nativo por metadatos, búsqueda híbrida combinando similitud semántica con criterios estructurados, y rendimiento adecuado para conjuntos de datos de escala media (hasta millones de vectores). Esta migración demostró que para aplicaciones de investigación y desarrollo, la simplicidad y control local pueden superar las ventajas de arquitecturas distribuidas más complejas.

## 2.4 Comparación de Enfoques Vectoriales y Clásicos

### 2.4.1 Limitaciones de Sistemas Clásicos

Los sistemas clásicos de recuperación de información, implementados en plataformas como Apache Lucene y Elasticsearch, utilizan modelos estadísticos que representan documentos como bolsas de palabras (bag-of-words). Aunque estos sistemas son computacionalmente eficientes y relativamente simples de implementar y mantener, presentan limitaciones fundamentales en la comprensión semántica profunda, lo cual restringe su capacidad para responder consultas formuladas en lenguaje natural (Manning et al., 2008).

Estas limitaciones son pronunciadas en dominios técnicos donde existe alta variabilidad terminológica, uso de sinónimos especializados, y donde la relevancia depende fuertemente del contexto semántico más que de la coincidencia léxica exacta.

### 2.4.2 Intentos de Búsqueda Semántica con Bases Relacionales

Existen esfuerzos para implementar capacidades de búsqueda semántica utilizando bases de datos relacionales tradicionales mediante extensiones especializadas. PostgreSQL con la extensión pgvector permite almacenar y consultar vectores de embeddings utilizando SQL estándar (PostgreSQL, 2023). De manera similar, sistemas como Azure SQL Database han incorporado capacidades de búsqueda vectorial mediante extensiones propietarias.

Sin embargo, estas soluciones presentan limitaciones significativas comparadas con bases de datos vectoriales especializadas. En términos de rendimiento, los índices están menos optimizados para espacios de alta dimensionalidad, lo que resulta en mayor consumo de memoria, latencias superiores en consultas de similitud a gran escala, y escalabilidad limitada para billones de vectores. Funcionalmente, ofrecen soporte limitado para métricas de distancia especializadas, carecen de optimizaciones para ANN (Approximate Nearest Neighbor), presentan integración compleja con pipelines de ML/NLP, y no tienen funcionalidades nativas para filtrado híbrido semántico-estructurado. Operacionalmente, requieren expertise tanto en SQL como en operaciones vectoriales, presentan configuración y tuning más complejos para cargas de trabajo vectoriales, y tienen procesos de backup y recuperación más complejos para datos de alta dimensionalidad.

Estas limitaciones hacen que, aunque técnicamente posible, el uso de bases relacionales para búsqueda semántica sea subóptimo comparado con soluciones especializadas como ChromaDB, Pinecone o Weaviate en aplicaciones que requieren alto rendimiento y escalabilidad (Li et al., 2023).

### 2.4.3 Ventajas de Sistemas Vectoriales

En contraste, los sistemas vectoriales modernos utilizan embeddings generados por modelos de aprendizaje profundo, permitiendo recuperar documentos basados en similitud semántica en lugar de coincidencia léxica superficial (Malkov & Yashunin, 2018). Estos sistemas pueden identificar relaciones semánticas complejas, manejar sinónimos y variaciones terminológicas, y capturar dependencias contextuales que los sistemas clásicos no pueden procesar.

La implementación de algoritmos de búsqueda aproximada de vecinos más cercanos (Approximate Nearest Neighbor, ANN) como HNSW (Hierarchical Navigable Small World) permite realizar búsquedas vectoriales eficientes incluso en espacios de alta dimensionalidad, manteniendo latencias aceptables para aplicaciones de producción.

### 2.4.4 Enfoques Híbridos y Reranking

Los sistemas más efectivos combinan las fortalezas de ambos enfoques mediante arquitecturas híbridas que utilizan recuperación vectorial para la selección inicial de candidatos, seguida de reranking mediante modelos más sofisticados. Los CrossEncoders, que procesan conjuntamente la consulta y cada documento candidato, pueden proporcionar scores de relevancia más precisos que los bi-encoders utilizados en la fase de recuperación inicial (Reimers & Gurevych, 2019).

Esta estrategia de pipeline multi-etapa permite balancear eficiencia computacional con precisión de recuperación, siendo efectiva en sistemas de soporte técnico donde la precisión en los primeros resultados es crítica para la experiencia del usuario.

## 2.5 Casos Empresariales Relevantes

La industria tecnológica ha implementado diversas soluciones NLP para automatización de soporte técnico. Microsoft ha incorporado extensivamente modelos de NLP en Azure para análisis automático de tickets y sugerencia de respuestas basadas en documentación técnica (Microsoft Learn, 2023). Su implementación utiliza arquitecturas híbridas que combinan embeddings semánticos, sistemas de ranking multi-etapa y técnicas de respuesta generativa. El sistema procesa automáticamente tickets entrantes, los clasifica por servicio y urgencia, y sugiere documentación relevante basada en casos históricos similares, integrando múltiples fuentes de conocimiento mediante técnicas de fusión de rankings.

Zendesk desarrolló "Answer Bot", un sistema de inteligencia artificial que utiliza NLP avanzado para sugerir artículos de ayuda relevantes automáticamente cuando un usuario envía un ticket (Zendesk, 2023). El sistema ha logrado reducir en un 10-30% el volumen de tickets que requieren intervención humana directa, demostrando el impacto de las tecnologías NLP en la eficiencia operacional. Answer Bot implementa técnicas de aprendizaje continuo que mejoran sus recomendaciones basándose en el feedback implícito de usuarios y explícito de agentes.

ServiceNow integra modelos de NLP con su módulo "Predictive Intelligence", que clasifica y enruta tickets automáticamente utilizando modelos entrenados en datos históricos extensos (ServiceNow, 2022). El sistema también implementa funcionalidades de recomendación de artículos y predicción de resolución, utilizando técnicas de aprendizaje automático para optimizar la asignación de recursos. La plataforma incluye capacidades de análisis de sentimiento para priorizar tickets con mayor urgencia emocional y detectar patrones de escalación potencial.

Salesforce Service Cloud ha implementado bots conversacionales que combinan NLP y búsqueda semántica para asistir tanto a clientes como a agentes en tiempo real (Salesforce, 2023). Estas herramientas son alimentadas por bases vectoriales generadas a partir de documentación técnica, casos históricos e interacciones previas, utilizando arquitecturas transformer para generar respuestas contextualizadas. El sistema integra capacidades de procesamiento multimodal que pueden analizar no solo texto sino también imágenes y documentos adjuntos.

## 2.6 Medidas de Evaluación en Recuperación de Información

### 2.6.1 Métricas Tradicionales de Recuperación y Ranking

La evaluación rigurosa de sistemas de recuperación de información es fundamental para validar la efectividad de las soluciones propuestas. Las métricas tradicionales como Precision, Recall y F1-score continúan siendo ampliamente utilizadas, pero requieren adaptación y complementación con métricas específicas para el paradigma de recuperación semántica.

**Precision** mide la proporción de documentos relevantes entre los documentos recuperados, siendo crucial cuando se busca minimizar falsos positivos. En contextos de soporte técnico, recomendar artículos irrelevantes puede generar frustración y pérdida de confianza en el sistema. **Recall** evalúa la proporción de documentos relevantes recuperados sobre el total disponible. Esta métrica es crítica en soporte técnico, donde omitir información relevante puede resultar en resolución inadecuada del problema del usuario. **F1-Score** representa la media armónica entre precision y recall, proporcionando una métrica balanceada útil cuando ambos aspectos son igualmente importantes.

**Mean Reciprocal Rank (MRR)** es fundamental cuando el sistema devuelve listas ordenadas de resultados y se busca evaluar qué tan pronto aparece la respuesta relevante. En soporte técnico, esta métrica es valiosa para evaluar la utilidad de los primeros resultados mostrados al agente, ya que típicamente solo se revisan los primeros 3-5 resultados. **Normalized Discounted Cumulative Gain (NDCG)** considera tanto la relevancia de los resultados como su posición en la lista, aplicando un descuento logarítmico que penaliza resultados relevantes en posiciones inferiores.

**Precision@k y Recall@k** están diseñadas para evaluar la calidad de los primeros k resultados. Precision@k mide la proporción de resultados relevantes entre los primeros k documentos recuperados. Por ejemplo, si entre los primeros 5 artículos sugeridos, 3 son relevantes, entonces Precision@5 = 0.6. Recall@k evalúa cuántos documentos relevantes fueron recuperados entre los primeros k, comparado con el total disponible. Si hay 4 documentos relevantes totales y el sistema recupera 3 dentro de los primeros 5, entonces Recall@5 = 0.75.

### 2.6.2 Métricas Específicas para Sistemas RAG

Las arquitecturas RAG requieren métricas especializadas que evalúen no solo la recuperación sino también la calidad de la generación y la coherencia entre ambas fases. **Answer Relevancy** mide qué tan bien la respuesta generada aborda la pregunta formulada, evaluando la alineación semántica entre consulta y respuesta (Es et al., 2023). **Context Precision** evalúa qué proporción del contexto recuperado es realmente relevante para responder la pregunta, identificando ruido en la fase de recuperación. **Context Recall** mide si toda la información necesaria para responder está presente en el contexto recuperado. **Faithfulness** evalúa si la respuesta generada es factualmente consistente con el contexto proporcionado, detectando alucinaciones o inconsistencias.

### 2.6.3 Métricas de Similitud Semántica y Aplicación al Proyecto

**BERTScore** utiliza representaciones contextuales de BERT para evaluar la similitud semántica entre respuestas generadas y respuestas de referencia, proporcionando una evaluación más matizada que métricas basadas en coincidencia léxica como BLEU o ROUGE (Zhang et al., 2019). En este proyecto se implementó BERTScore utilizando el modelo `distiluse-base-multilingual-cased-v2`, optimizado para evaluación de similitud semántica cross-lingual, aunque se aplicó a contenido en inglés para mantener consistencia con el corpus de documentación técnica.

En este proyecto se implementó un framework de evaluación que incluye métricas de recuperación tradicionales (Precision@k, Recall@k, MRR, NDCG), métricas RAG especializadas (Answer Relevancy, Context Precision, Context Recall, Faithfulness implementadas via RAGAS), evaluación semántica mediante BERTScore, y análisis pre/post reranking para cuantificar el impacto del CrossEncoder. Esta combinación permite evaluar integralmente tanto la efectividad de la recuperación como la calidad de las respuestas generadas, proporcionando insights detallados sobre el rendimiento de cada componente del pipeline RAG en el contexto del soporte técnico de Azure.
