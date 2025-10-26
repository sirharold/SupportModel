# 1. INTRODUCCIÓN Y FUNDAMENTOS DEL PROYECTO

## 1.1 Formulación del Problema

Los equipos de soporte técnico en plataformas complejas como Microsoft Azure enfrentan un reto diario: atender miles de consultas que requieren conocimiento especializado distribuido en vastas bases documentales. Esta realidad genera tres desafíos críticos que motivaron esta investigación.

**El conocimiento existe, pero no se encuentra.** Durante la fase de recopilación de datos, se extrajeron 62,417 documentos únicos de Microsoft Learn relacionados con Azure, segmentados en 187,031 chunks para su procesamiento. Sin embargo, los agentes de soporte suelen depender de su memoria y experiencia personal para resolver tickets, lo que resulta en respuestas inconsistentes y tiempos de resolución prolongados. La información está disponible, pero su accesibilidad efectiva es limitada.

**Los esfuerzos se duplican constantemente.** Un análisis de cobertura sobre más de 3,000 preguntas del dataset reveló que 2,067 de estas consultas (67%) ya tenían respuestas documentadas en Microsoft Learn. A pesar de esta alta disponibilidad, cada ticket continúa tratándose como un caso único, lo que representa un desperdicio considerable de recursos humanos y tiempo.

**La búsqueda léxica tradicional no funciona en dominios técnicos.** Los usuarios formulan sus consultas en lenguaje natural usando terminología variada que raramente coincide con los términos exactos de la documentación oficial. Esta brecha semántica limita severamente la efectividad de los motores de búsqueda convencionales basados en palabras clave.

Ante estos desafíos, esta investigación propone desarrollar un sistema de recuperación semántica que integre técnicas avanzadas de procesamiento de lenguaje natural para vincular automáticamente tickets de soporte con documentación técnica relevante, mejorando así la eficiencia y consistencia en la resolución de consultas.

## 1.2 Alcances

### 1.2.1 Alcance Temático

Esta investigación se sitúa en la intersección de tres dominios: procesamiento de lenguaje natural aplicado a contenido técnico especializado, sistemas de recuperación de información basados en semántica vectorial, y gestión automatizada de conocimiento para soporte técnico.

El trabajo abarca el diseño, implementación y evaluación de un sistema RAG (Retrieval-Augmented Generation) completo. El sistema procesa el corpus completo de documentación Azure (62,417 documentos segmentados en 187,031 chunks) e implementa una comparación sistemática de cuatro modelos de embeddings: Ada, MPNet, MiniLM y E5-Large. Además, desarrolla arquitecturas de búsqueda híbrida que combinan recuperación vectorial con reranking semántico, y evalúa el rendimiento mediante métricas específicas de recuperación de información.

### 1.2.2 Alcance Temporal

El desarrollo se ejecutó durante el período académico 2024-2025, con una fase de implementación intensiva de 16 semanas. La recolección de datos (documentación técnica y preguntas) se realizó en marzo de 2025, creando una imagen estática del conocimiento disponible en Microsoft Learn y Microsoft Q&A. Esta decisión permite una evaluación consistente y reproducible sin variaciones temporales en el corpus.

## 1.3 Delimitaciones

### 1.3.1 Delimitación Geográfica

Aunque el sistema está diseñado para operar sin restricciones geográficas, la implementación se enfoca exclusivamente en documentación y consultas en idioma inglés. Los datos provienen de fuentes públicas internacionales (Microsoft Learn y Microsoft Q&A), pero el procesamiento lingüístico se optimiza para terminología técnica en inglés.

### 1.3.2 Delimitación de Dominio

La investigación se delimita estrictamente al ecosistema de Microsoft Azure, excluyendo otros productos de Microsoft o plataformas cloud competidoras. Esta delimitación no es arbitraria: permite una especialización profunda en la terminología, arquitectura y patrones de consulta específicos de Azure, lo que mejora significativamente la precisión del sistema.

### 1.3.3 Delimitación Funcional

Este es un proyecto académico centrado en la evaluación de técnicas de recuperación de información, no en el desarrollo de un producto comercial. Aunque se implementa una arquitectura RAG completa, el énfasis está en la medición y análisis de métricas de recuperación (Precision@k, Recall@k, MRR, NDCG) más que en la utilización práctica de las respuestas generadas. El sistema se diseñó específicamente para identificar, rankear y evaluar documentos relevantes.

## 1.4 Limitaciones

### 1.4.1 Limitaciones de Datos

Por consideraciones de privacidad y confidencialidad empresarial, se utilizaron exclusivamente datos públicos. No se tuvo acceso a tickets reales de soporte empresarial, lo que impide validar el sistema con casos de uso industriales confidenciales.

Las 13,436 preguntas del dataset provienen de Microsoft Q&A (foros públicos), de las cuales solo 2,067 contienen enlaces validados a documentación oficial que pudieron usarse como ground truth. Esta fracción relativamente pequeña (15.4%) puede no representar completamente la complejidad y diversidad de tickets corporativos internos.

### 1.4.2 Limitaciones Técnicas

El procesamiento se limita a contenido textual, excluyendo elementos multimedia como imágenes, diagramas arquitectónicos y videos instructivos que forman parte significativa de la documentación técnica moderna. Esta limitación es relevante considerando que muchos conceptos técnicos se explican mejor visualmente.

Los modelos de embeddings tienen restricciones de contexto variables: MiniLM maneja 256 tokens, MPNet 384 tokens, E5-Large 512 tokens, mientras que OpenAI Ada puede procesar hasta 8,191 tokens. Estas limitaciones requirieron estrategias de segmentación que potencialmente pierden información contextual importante al dividir documentos largos.

### 1.4.3 Limitaciones de Evaluación

La validación se basa en enlaces explícitos entre preguntas y documentos identificados en respuestas aceptadas por la comunidad. Este criterio, aunque objetivo y verificable, es más estricto que escenarios reales donde múltiples documentos pueden ser igualmente válidos para resolver una consulta. Un documento podría ser relevante incluso si no fue específicamente citado en la respuesta aceptada.

## 1.5 Objetivos

### 1.5.1 Objetivo General

Desarrollar y evaluar un sistema de recuperación semántica de información basado en técnicas avanzadas de procesamiento de lenguaje natural, utilizando la documentación de Microsoft Azure como caso de estudio, con el fin de medir y comparar la efectividad de diferentes modelos de embeddings y arquitecturas de recuperación en la identificación precisa de documentos relevantes para consultas técnicas especializadas.

### 1.5.2 Objetivos Específicos

1. **Implementar y comparar múltiples arquitecturas de embeddings**, evaluando modelos de código abierto (MiniLM, MPNet, E5-Large) y propietarios (OpenAI Ada) para determinar la representación vectorial óptima del contenido técnico especializado de Azure.

2. **Diseñar un sistema de almacenamiento y recuperación vectorial** utilizando ChromaDB como base de datos especializada, configurando índices optimizados para búsquedas de similitud semántica a escala con más de 800,000 vectores de alta dimensionalidad distribuidos en 8 colecciones especializadas.

3. **Desarrollar mecanismos avanzados de reranking** implementando CrossEncoders especializados y técnicas de normalización (Min-Max) para mejorar la precisión en el ordenamiento final de documentos recuperados, optimizando específicamente para consultas técnicas complejas.

4. **Evaluar sistemáticamente el rendimiento del sistema** mediante un framework de métricas que incluye medidas tradicionales de recuperación (Precision@k, Recall@k, MRR, NDCG) en etapas pre y post reranking, métricas específicas para sistemas RAG (Answer Relevancy, Context Precision, Faithfulness), y validación semántica utilizando RAGAS y BERTScore.

5. **Establecer una metodología reproducible y extensible**, documentando el proceso de implementación, creando pipelines automatizados de evaluación con métricas verificables, y desarrollando herramientas auxiliares (incluyendo una interfaz Streamlit) que faciliten la ejecución de pruebas y la visualización de resultados para futuras investigaciones.

## 1.6 Referencias del Capítulo

Microsoft. (2025). *Microsoft Learn Documentation*. https://learn.microsoft.com/

Microsoft. (2025). *Microsoft Q&A*. https://learn.microsoft.com/en-us/answers/

Hugging Face. (2025). *Sentence Transformers: Model documentation and specifications*. https://huggingface.co/sentence-transformers/

OpenAI. (2025). *Embeddings API documentation*. https://platform.openai.com/docs/guides/embeddings
