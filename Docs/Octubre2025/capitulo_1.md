# 1. INTRODUCCIÓN Y FUNDAMENTOS DEL PROYECTO

## 1.1 Formulación del Problema

Los sistemas de soporte técnico para productos tecnológicos complejos enfrentan desafíos fundamentales en la gestión del conocimiento especializado. Esta investigación aborda el problema de recuperación semántica de información técnica utilizando Microsoft Azure como caso de estudio representativo de plataformas enterprise modernas.

### 1.1.1 El conocimiento existe, pero no se encuentra

La documentación técnica oficial constituye la fuente primaria de información para resolver consultas de soporte. Sin embargo, la recuperación efectiva de esta información presenta dificultades inherentes a la complejidad y especialización del dominio. Para emular este escenario genérico, se construyó un corpus completo de documentación técnica basado en Microsoft Azure, representativo de las características presentes en otros productos tecnológicos enterprise: alta especialización terminológica, arquitecturas multinivel, y documentación distribuida en múltiples formatos y niveles de abstracción.

La brecha entre disponibilidad y accesibilidad del conocimiento motiva la necesidad de sistemas de recuperación semántica que superen las limitaciones de la búsqueda léxica tradicional.

### 1.1.2 Patrones recurrentes en consultas de soporte técnico

El análisis de preguntas de soporte técnico revela patrones característicos: un subconjunto relativamente pequeño de documentos responde a la mayoría de consultas frecuentes, mientras que casos específicos requieren documentación especializada que raramente se consulta de forma proactiva. Esta distribución irregular del conocimiento dificulta la localización de información relevante, particularmente cuando usuarios formulan consultas usando terminología que no coincide exactamente con la documentación oficial.

Para investigar este fenómeno, se recolectó un dataset de consultas reales de usuarios en foros técnicos especializados, permitiendo caracterizar los patrones de correspondencia entre preguntas naturales y documentación técnica formal. Este dataset proporciona ground truth verificable para evaluación sistemática de técnicas de recuperación.

### 1.1.3 Búsqueda léxica versus recuperación semántica

Los sistemas de búsqueda tradicionales basados en coincidencia de palabras clave presentan limitaciones significativas en dominios técnicos especializados. La terminología técnica admite múltiples formas de expresión (sinónimos, acrónimos, variantes regionales), y las consultas en lenguaje natural raramente replican la estructura formal de la documentación oficial.

Esta investigación propone desarrollar y evaluar un sistema de recuperación semántica basado en representaciones vectoriales densas (embeddings) que capture similitud conceptual más allá de coincidencia léxica superficial. El sistema integra técnicas de Retrieval-Augmented Generation (RAG) para vincular automáticamente consultas con documentación relevante mediante comprensión semántica del contenido técnico.

## 1.2 Alcances

### 1.2.1 Alcance Temático

El trabajo abarca el diseño, implementación y evaluación de un sistema RAG (Retrieval-Augmented Generation) completo aplicado a documentación técnica. El sistema implementa comparación sistemática de cuatro modelos de embeddings (Ada, MPNet, MiniLM, E5-Large), desarrolla arquitecturas de búsqueda híbrida que combinan recuperación vectorial con reranking semántico, y evalúa el rendimiento mediante métricas específicas de recuperación de información en etapas pre y post reranking.

### 1.2.2 Alcance Temporal

El desarrollo se ejecutó durante el período académico 2024-2025. La recolección de datos establece un corpus estático del conocimiento disponible en Microsoft Learn y Microsoft Q&A, permitiendo una evaluación consistente y reproducible sin variaciones temporales.

## 1.3 Delimitaciones

### 1.3.1 Delimitación Geográfica

Aunque el sistema está diseñado para operar sin restricciones geográficas, la implementación se enfoca exclusivamente en documentación y consultas en idioma inglés. Los datos provienen de fuentes públicas internacionales (Microsoft Learn y Microsoft Q&A), pero el procesamiento lingüístico se optimiza para terminología técnica en inglés.

### 1.3.2 Delimitación de Dominio

La investigación se delimita al ecosistema de Microsoft Azure, excluyendo otros productos de Microsoft o plataformas cloud competidoras. Esta delimitación permite especialización profunda en la terminología, arquitectura y patrones de consulta específicos del dominio Azure.

### 1.3.3 Delimitación Funcional

El proyecto se centra en la evaluación de técnicas de recuperación de información mediante métricas específicas (Precision@k, Recall@k, MRR, NDCG) en etapas pre y post reranking, más que en la implementación de un sistema de producción completo.

## 1.4 Limitaciones

### 1.4.1 Limitaciones de Datos

Se utilizaron exclusivamente datos públicos de foros técnicos especializados, sin acceso a tickets corporativos internos. El dataset de evaluación comprende un subconjunto de preguntas con enlaces validados a documentación oficial que sirven como ground truth verificable, lo cual representa un escenario de evaluación más estricto que casos reales donde múltiples documentos pueden ser igualmente relevantes.

### 1.4.2 Limitaciones Técnicas

El procesamiento se limita a contenido textual, excluyendo elementos multimedia (imágenes, diagramas, videos) presentes en la documentación técnica moderna. Los modelos de embeddings tienen restricciones de contexto que requieren segmentación de documentos extensos, potencialmente perdiendo información contextual al dividir el contenido.

### 1.4.3 Limitaciones de Evaluación

La validación se basa en enlaces explícitos entre preguntas y documentos en respuestas validadas por la comunidad. Este criterio, aunque objetivo y reproducible, puede subestimar la relevancia de documentos alternativos igualmente válidos que no fueron citados en la respuesta aceptada.

## 1.5 Objetivos

### 1.5.1 Objetivo General

Desarrollar y evaluar un sistema de recuperación semántica de información basado en técnicas de procesamiento de lenguaje natural, utilizando documentación técnica de Microsoft Azure como caso de estudio. El objetivo es medir y comparar la efectividad de diferentes modelos de embeddings y arquitecturas de recuperación en la identificación de documentos relevantes para consultas técnicas especializadas.

### 1.5.2 Objetivos Específicos

1. **Implementar y comparar múltiples arquitecturas de embeddings**, evaluando modelos de código abierto (MiniLM, MPNet, E5-Large) y propietarios (OpenAI Ada) para determinar la representación vectorial óptima del contenido técnico especializado de Azure.

2. **Diseñar un sistema de almacenamiento y recuperación vectorial** utilizando ChromaDB como base de datos especializada, configurando índices optimizados para búsquedas de similitud semántica a escala con más de 800,000 vectores de alta dimensionalidad distribuidos en 8 colecciones especializadas.

3. **Desarrollar mecanismos avanzados de reranking** implementando CrossEncoders especializados y técnicas de normalización (Min-Max) para mejorar la precisión en el ordenamiento final de documentos recuperados, optimizando específicamente para consultas técnicas complejas.

4. **Evaluar sistemáticamente el rendimiento del sistema** mediante un framework de métricas que incluye medidas tradicionales de recuperación (Precision@k, Recall@k, MRR, NDCG) en etapas pre y post reranking, métricas específicas para sistemas RAG (Answer Relevancy, Context Precision, Faithfulness), y validación semántica utilizando RAGAS y BERTScore.

5. **Establecer una metodología reproducible y extensible**, documentando el proceso de implementación, creando pipelines automatizados de evaluación con métricas verificables, y desarrollando herramientas auxiliares (incluyendo una interfaz Streamlit) que faciliten la ejecución de pruebas y la visualización de resultados para futuras investigaciones.
