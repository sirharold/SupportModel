# 3. MARCO TEÓRICO

## 3.1 Introducción

Este capítulo establece los fundamentos teóricos que sustentan el sistema RAG desarrollado en este proyecto. La convergencia de múltiples dominios tecnológicos ha habilitado avances significativos en la última década: recuperación de información semántica, modelos de embeddings densos, arquitecturas de Retrieval-Augmented Generation, y bases de datos vectoriales optimizadas. Esta convergencia permite desarrollar sistemas de soporte técnico automatizado que superan las limitaciones de los enfoques tradicionales basados en coincidencia léxica.

La naturaleza técnica y especializada del dominio de Microsoft Azure presenta desafíos únicos que requieren una comprensión profunda de los fundamentos teóricos que sustentan las tecnologías empleadas. Los conceptos presentados proporcionan los cimientos conceptuales necesarios para comprender la arquitectura, implementación y evaluación del sistema RAG, analizando cada componente tecnológico y su contribución al objetivo general de automatización inteligente del soporte técnico.

## 3.2 Fundamentos de Recuperación de Información

### 3.2.1 Evolución de los Paradigmas de Recuperación

La recuperación de información (Information Retrieval, IR) ha evolucionado desde modelos probabilísticos clásicos hacia enfoques semánticos basados en representaciones vectoriales densas. El modelo vectorial tradicional, introducido por Salton et al. (1975), representa documentos y consultas como vectores en un espacio multidimensional donde cada dimensión corresponde a un término del vocabulario. Sin embargo, este enfoque sufre de limitaciones relacionadas con la maldición de la dimensionalidad y la incapacidad de capturar relaciones semánticas implícitas.

El cambio paradigmático hacia recuperación semántica densa ha sido posible gracias al desarrollo de modelos de lenguaje preentrenados capaces de generar representaciones vectoriales que preservan información semántica y contextual (Karpukhin et al., 2020). Estos modelos transforman texto en vectores de baja dimensión (típicamente 256-1536 dimensiones) que capturan similitudes semánticas no evidentes en el nivel léxico.

### 3.2.2 Fundamentos Matemáticos de Similitud Semántica

La similitud semántica en espacios vectoriales densos se cuantifica mediante la similitud coseno, definida como la relación entre el producto punto de dos vectores y el producto de sus normas euclidianas. Esta métrica normaliza los vectores, enfocándose en la orientación angular más que en la magnitud, lo que resulta apropiado para comparaciones semánticas donde la longitud del documento es menos relevante que su contenido conceptual.

La efectividad de esta aproximación depende críticamente de la calidad de las representaciones vectoriales, que deben preservar relaciones semánticas de manera que documentos conceptualmente similares mantengan proximidad en el espacio vectorial (Reimers & Gurevych, 2019).

### 3.2.3 Arquitecturas de Recuperación Multi-Etapa

Los sistemas modernos de recuperación implementan arquitecturas multi-etapa que optimizan el balance entre recall y precisión mediante un proceso de refinamiento progresivo (Qu et al., 2021). El pipeline típico comienza con una recuperación inicial mediante búsqueda vectorial eficiente sobre el corpus completo utilizando similitud coseno. Esta fase identifica un conjunto amplio de candidatos potencialmente relevantes. Posteriormente, un proceso de reranking utiliza modelos más sofisticados (CrossEncoders) que procesan conjuntamente consulta y documento para refinar el ordenamiento de candidatos. Finalmente, se aplican reglas de negocio y thresholding para optimizar la precisión de los resultados finales presentados al usuario.

Esta arquitectura permite escalar a corpus de gran tamaño manteniendo alta precisión en los resultados finales.

## 3.3 Modelos de Embeddings

### 3.3.1 OpenAI Ada (text-embedding-ada-002)

OpenAI Ada representa el estado del arte en modelos de embeddings comerciales, implementando una arquitectura Transformer optimizada para generación de representaciones vectoriales densas (OpenAI, 2023). El modelo genera vectores de 1,536 dimensiones optimizados para tareas de similitud semántica y recuperación de información. Sus características técnicas incluyen una longitud máxima de contexto de 8,191 tokens, arquitectura Transformer con optimizaciones propietarias para embeddings, y normalización que produce vectores unitarios con norma L2 igual a 1.0.

Ada incorpora técnicas avanzadas de preentrenamiento contrastivo que optimizan la representación vectorial para tareas de similitud semántica. El modelo ha sido entrenado en un corpus diverso que incluye documentación técnica, lo que resulta beneficioso para dominios especializados como Microsoft Azure. Sin embargo, presenta limitaciones relacionadas con su naturaleza propietaria, incluyendo dependencia de API externa, costos operacionales variables, y opacidad arquitectónica que impide optimizaciones específicas del dominio. Su rendimiento puede degradarse en terminología altamente especializada no representada en el corpus de entrenamiento.

### 3.3.2 Sentence-BERT: MPNet y MiniLM

MPNet (Masked and Permuted Pre-training) combina las ventajas de BERT y XLNet mediante una estrategia de preentrenamiento que incorpora tanto masked language modeling como permuted language modeling (Song et al., 2020). Esta aproximación híbrida resulta en representaciones más robustas para tareas de recuperación semántica. El modelo multi-qa-mpnet-base-dot-v1 utilizado en este proyecto genera vectores de 768 dimensiones mediante una arquitectura de 12 capas transformer con 12 cabezas de atención, totalizando aproximadamente 110 millones de parámetros. Su especialización proviene de fine-tuning en pares pregunta-respuesta, optimizando el modelo para este tipo de interacciones.

Un aspecto técnico relevante es el uso del prefijo "query:" al procesar consultas, convención establecida durante el entrenamiento para distinguir entre consultas y documentos, lo que resulta crucial para el rendimiento óptimo del modelo.

MiniLM implementa destilación de conocimiento desde modelos BERT más grandes, manteniendo calidad semántica mientras reduce los requerimientos computacionales (Wang et al., 2020). Esta optimización es valiosa en aplicaciones de producción con restricciones de recursos. El modelo all-MiniLM-L6-v2 genera vectores de 384 dimensiones mediante 6 capas transformer con aproximadamente 22 millones de parámetros, operando aproximadamente 5 veces más rápido que BERT-base. La reducción dimensional y arquitectónica se compensa mediante técnicas avanzadas de destilación que preservan información semántica crítica en el espacio vectorial de menor dimensión.

### 3.3.3 E5-Large: Embeddings Especializados en Recuperación

E5-Large (Embeddings from bidirectional Encoder representations) implementa una estrategia de preentrenamiento contrastivo optimizada para tareas de recuperación de información (Wang et al., 2022). El modelo utiliza técnicas de aprendizaje auto-supervisado que maximizan la similitud entre pares relacionados mientras minimizan la similitud entre pares no relacionados. Sus características técnicas incluyen vectores de 1,024 dimensiones generados por una arquitectura Transformer de 24 capas con aproximadamente 335 millones de parámetros, entrenado en un corpus multilingüe con énfasis en pares texto-texto.

E5-Large ha demostrado rendimiento superior en el benchmark MTEB (Massive Text Embedding Benchmark), particularmente en tareas de recuperación semántica y clasificación de similaridad textual (Muennighoff et al., 2023). Su arquitectura optimizada para recuperación lo posiciona como una alternativa competitiva a modelos propietarios en aplicaciones especializadas.

Las evaluaciones del proyecto revelaron que E5-Large presentó desafíos en el dominio de Microsoft Azure, mostrando métricas de recuperación de 0.0 en todas las categorías (Precision@5, Recall@5, NDCG@5), indicando dificultades para recuperar documentos relevantes en los primeros 10 resultados. Sin embargo, mostró el mejor rendimiento en métricas de generación RAG como faithfulness (0.5909), sugiriendo que sus representaciones vectoriales, aunque problemáticas para recuperación, mantienen calidad semántica para tareas de generación de respuestas.

## 3.4 Arquitecturas RAG (Retrieval-Augmented Generation)

### 3.4.1 Fundamentos Teóricos de RAG

Las arquitecturas RAG combinan los beneficios de modelos parametrizados (conocimiento almacenado en parámetros) con acceso dinámico a conocimiento no parametrizado (bases de datos externas). Esta hibridación permite superar limitaciones de los modelos de lenguaje tradicionales, incluyendo obsolescencia de información, alucinaciones factuales, y limitaciones de memoria (Lewis et al., 2020).

El paradigma RAG descompone la generación de respuestas en dos componentes diferenciables: un retriever especializado en recuperación de información relevante, y un generator que sintetiza respuestas utilizando la información recuperada. Esta separación permite optimizar independientemente cada componente y facilita la actualización de la base de conocimiento sin reentrenar el modelo generativo.

### 3.4.2 Taxonomía de Arquitecturas RAG

La arquitectura RAG clásica implementa un pipeline secuencial donde la recuperación precede completamente a la generación. Esta aproximación es computacionalmente eficiente y facilita la interpretabilidad al separar claramente las responsabilidades de cada componente. El proceso comienza con la recuperación de documentos relevantes basándose en la consulta del usuario, seguido de la construcción de un contexto que combina los documentos recuperados, y finalmente la generación de una respuesta que incorpora la información contextual recuperada.

Variantes más sofisticadas incluyen RAG iterativo, donde el proceso de recuperación puede repetirse basándose en la generación parcial, y RAG adaptativo, donde el modelo aprende dinámicamente cuándo y cómo utilizar información externa (Jiang et al., 2023). Estas variantes ofrecen mayor flexibilidad pero requieren recursos computacionales adicionales.

### 3.4.3 Métricas de Evaluación RAG

La evaluación de sistemas RAG requiere métricas especializadas que capturen tanto la calidad de recuperación como la calidad de generación. El framework RAGAS (Retrieval Augmented Generation Assessment) proporciona métricas que abordan diferentes aspectos del sistema. Faithfulness evalúa la consistencia factual entre la respuesta generada y el contexto proporcionado, detectando casos donde el modelo introduce información no soportada por los documentos recuperados. Answer Relevancy mide qué tan bien la respuesta aborda la pregunta formulada, evaluando la alineación semántica entre consulta y respuesta. Context Precision examina qué proporción del contexto recuperado es realmente relevante para responder la pregunta, identificando ruido en la fase de recuperación. Context Recall verifica si toda la información necesaria para responder está presente en el contexto recuperado, comparando contra respuestas de referencia cuando están disponibles.

## 3.5 CrossEncoders y Reranking

### 3.5.1 Fundamentos Teóricos del Reranking Neural

El reranking neural utiliza modelos que procesan conjuntamente consulta y documento, capturando interacciones semánticas más sofisticadas que los enfoques de embedding independientes. Los CrossEncoders representan el estado del arte en esta aproximación, utilizando mecanismos de atención cruzada para modelar relaciones complejas entre consulta y documento (Nogueira & Cho, 2019).

A diferencia de los bi-encoders que generan representaciones independientes para consultas y documentos, los CrossEncoders procesan ambos elementos simultáneamente, permitiendo que el modelo capture dependencias y relaciones contextuales que no son accesibles en aproximaciones de embedding separadas. Esta capacidad resulta en scores de relevancia más precisos, aunque a costa de mayor complejidad computacional.

### 3.5.2 Arquitectura CrossEncoder ms-marco-MiniLM-L-6-v2

El modelo ms-marco-MiniLM-L-6-v2 ha sido fine-tuneado en el dataset MS MARCO, que contiene 8.8 millones de pares pregunta-pasaje derivados de consultas reales de Bing. Esta especialización resulta en un modelo optimizado para escenarios de recuperación de información factual y técnica. La arquitectura base utiliza MiniLM-L6 con 6 capas transformer, ocupando aproximadamente 90MB, con capacidad para procesar hasta 512 tokens por entrada y generando scores de relevancia como salida.

La normalización Min-Max aplicada a los scores del CrossEncoder garantiza comparabilidad entre consultas y estabilidad en las métricas de evaluación. Este proceso convierte los scores originales (logits) a un rango normalizado entre 0 y 1, donde los valores son relativos al conjunto de documentos evaluados. La normalización calcula el mínimo y máximo de los scores para la consulta actual, y reescala linealmente cada score individual dentro de este rango. Esta técnica permite comparaciones justas entre diferentes consultas que podrían tener distribuciones de scores naturalmente diferentes.

### 3.5.3 Teoría de Optimización Multi-Etapa

La combinación de recuperación densa con reranking neural implementa una estrategia de optimización multi-etapa que balancea eficiencia computacional con precisión. La primera etapa (dense retrieval) opera como un filtro eficiente sobre el corpus completo, utilizando búsqueda vectorial rápida para identificar candidatos potencialmente relevantes. La segunda etapa (reranking) aplica un modelo más sofisticado sobre un conjunto reducido de candidatos, típicamente entre 10 y 100 documentos.

Esta aproximación es sólida desde la perspectiva de optimización computacional, ya que permite aplicar modelos costosos únicamente sobre subconjuntos relevantes identificados por heurísticas eficientes (Chen et al., 2022). La estrategia aprovecha el hecho de que la mayoría de documentos en el corpus son claramente irrelevantes y pueden descartarse rápidamente mediante métodos eficientes, reservando el procesamiento intensivo para el refinamiento de candidatos prometedores.

## 3.6 Bases de Datos Vectoriales

### 3.6.1 Fundamentos de Búsqueda de Vectores de Alta Dimensión

La búsqueda eficiente en espacios vectoriales de alta dimensión presenta desafíos computacionales únicos relacionados con la maldición de la dimensionalidad y la necesidad de índices especializados. Los algoritmos de búsqueda exacta como fuerza bruta escalan linealmente con el tamaño del corpus, resultando impracticables para aplicaciones de producción con millones de documentos.

### 3.6.2 Algoritmos de Búsqueda Aproximada: HNSW

HNSW (Hierarchical Navigable Small World) implementa una estructura de grafo multicapa que permite búsqueda logarítmica aproximada en espacios de alta dimensión (Malkov & Yashunin, 2018). El algoritmo construye una jerarquía de grafos donde cada nivel contiene una fracción de los nodos del nivel inferior, permitiendo navegación eficiente desde búsqueda gruesa a refinada. Los niveles superiores contienen menos nodos y permiten saltos largos en el espacio, mientras los niveles inferiores contienen más nodos y refinan la búsqueda localmente.

La estructura HNSW ofrece garantías teóricas de complejidad O(log N) para búsqueda y O(N log N) para construcción del índice, donde N es el número de vectores almacenados. En dominios técnicos especializados como documentación de Microsoft Azure, la distribución de vectores puede presentar características que permiten optimizaciones específicas. La clustering temática natural de documentos relacionados puede explotarse mediante técnicas de particionamiento inteligente del espacio vectorial.

### 3.6.3 ChromaDB: Arquitectura y Decisión Tecnológica

La migración de Weaviate a ChromaDB se fundamentó en criterios de optimización para flujos de investigación y desarrollo. Weaviate ofrece escalabilidad empresarial, API GraphQL sofisticada, y módulos especializados para diferentes tipos de embeddings, siendo óptimo para aplicaciones de producción distribuida. Sin embargo, presenta latencia de red entre 150-300ms por consulta y dependencia de conectividad externa. ChromaDB, por otro lado, proporciona latencia local menor a 10ms, portabilidad de datos mediante formato Parquet, y simplicidad de configuración sin requerimientos de servicios externos, siendo óptimo para investigación y desarrollo iterativo donde la velocidad de experimentación es prioritaria.

El sistema implementa una arquitectura de almacenamiento que mantiene colecciones separadas para cada modelo de embedding, permitiendo comparaciones directas mientras preserva optimizaciones específicas por modelo. Las colecciones de documentos (docs_ada, docs_mpnet, docs_minilm, docs_e5large) y colecciones de preguntas (questions_ada, questions_mpnet, questions_minilm, questions_e5large) permiten evaluaciones independientes. Una colección adicional (questions_withlinks) mantiene 2,067 pares validados como ground truth. Esta arquitectura facilita evaluaciones comparativas rigurosas y permite optimizaciones independientes por modelo sin interferencia cruzada.

### 3.6.4 Consideraciones de Escalabilidad y Rendimiento

La selección de base de datos vectorial debe considerar múltiples factores incluyendo latencia de consulta, throughput, consumo de memoria, y capacidades de actualización incremental. Para corpus de tamaño moderado (aproximadamente 200,000 vectores), soluciones embebidas como ChromaDB ofrecen ventajas en simplicidad operacional y rendimiento de consulta. Para aplicaciones de producción con corpus de mayor escala (más de 1 millón de vectores), bases de datos distribuidas como Weaviate, Pinecone, o Milvus se vuelven necesarias para mantener latencias aceptables y capacidades de escalamiento horizontal.

El proyecto demostró características de rendimiento notables en múltiples dimensiones. El procesamiento de evaluación total alcanzó 774.78 segundos (12.9 minutos) para evaluación multi-modelo de 4 modelos de embedding contra 187,031 documentos. El throughput de procesamiento alcanzó aproximadamente 241 documentos por segundo para generación de embeddings. El procesamiento de consultas mantuvo 0.057 consultas por segundo durante la evaluación. Los requerimientos de almacenamiento variaron según dimensionalidad del modelo: desde 1.05 GB para MiniLM (384 dimensiones) hasta 2.23 GB para Ada (1,536 dimensiones), totalizando 6.48 GB para todos los modelos.

La aceleración GPU proporcionó mejoras de rendimiento de 10-50x cuando estuvo disponible. El sistema utilizó exitosamente hardware Google Colab T4 para operaciones intensivas. La escalabilidad se demostró manejando operaciones vectoriales de gran escala sobre colecciones de más de 187,000 documentos, manteniendo rendimiento consistente mediante operaciones ChromaDB optimizadas y formatos de almacenamiento basados en Parquet eficientes.

## 3.7 Conclusiones del Marco Teórico

Los fundamentos teóricos presentados establecen las bases científicas y técnicas que sustentan la arquitectura RAG desarrollada. La convergencia de modelos de embeddings especializados, arquitecturas de reranking neural, y bases de datos vectoriales optimizadas habilita sistemas de recuperación semántica que superan las capacidades de enfoques tradicionales basados en coincidencia léxica.

La selección de componentes (Ada, MPNet, MiniLM, E5-Large para embeddings; ms-marco-MiniLM-L-6-v2 para reranking; ChromaDB para almacenamiento vectorial) se fundamenta en criterios teóricos relacionados con optimización de rendimiento, eficiencia computacional, y especialización de dominio. Esta arquitectura proporciona una base robusta para evaluación empírica de diferentes aproximaciones a la recuperación de información técnica especializada.

Los principios establecidos en este capítulo guían tanto la implementación técnica como la metodología de evaluación presentadas en capítulos posteriores, asegurando que el desarrollo del sistema se fundamenta en conocimiento científico validado y mejores prácticas de la industria.

## 3.8 Referencias del Capítulo

Chen, X., He, B., Sun, L., & Sun, Y. (2022). Towards better text understanding and retrieval through kernel-based neural models. *Information Sciences*, 588, 70-84.

Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., ... & Neubig, G. (2023). Active retrieval augmented generation. *arXiv preprint arXiv:2305.06983*.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *arXiv preprint arXiv:2004.04906*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive text embedding benchmark. *arXiv preprint arXiv:2210.07316*.

Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

OpenAI. (2023). *Embeddings API documentation*. https://platform.openai.com/docs/guides/embeddings

Qu, Y., Ding, Y., Liu, J., Liu, K., Ren, R., Zhao, W. X., ... & Wen, J. R. (2021). RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. *arXiv preprint arXiv:2010.08191*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *arXiv preprint arXiv:1908.10084*.

Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM*, 18(11), 613-620.

Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *Advances in Neural Information Processing Systems*, 33, 16857-16867.

Wang, L., Yang, N., Huang, J., Chang, M. W., & Wang, W. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv preprint arXiv:2212.03533*.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems*, 33, 5776-5788.
