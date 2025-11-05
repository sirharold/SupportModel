# 6. IMPLEMENTACIÓN

## 6.1 Introducción

Este capítulo describe la implementación del sistema RAG (Retrieval-Augmented Generation) desarrollado para mejorar la gestión de tickets de soporte técnico mediante recuperación semántica de documentación de Microsoft Azure. La implementación siguió las fases 3 y 4 de la metodología Design Science Research (DSR) presentada en el Capítulo 5: Design and Development, y Demonstration.

La **Fase 3: Design and Development** (secciones 6.3 a 6.6) abarcó la construcción completa del artefacto tecnológico. Se inició con la extracción automatizada de datos desde Microsoft Learn y Microsoft Q&A (sección 6.3), estableciendo el corpus de documentación técnica y preguntas validadas. Posteriormente se implementó ChromaDB como base de datos vectorial (sección 6.4), tras evaluar y migrar desde Weaviate por consideraciones de latencia y costos. La arquitectura del sistema RAG (sección 6.5) integró cuatro modelos de embeddings (Ada, MPNet, MiniLM, E5-Large) con componentes de búsqueda vectorial y evaluación. Finalmente, el pipeline de procesamiento RAG (sección 6.6) implementó recuperación híbrida, reranking con CrossEncoder, y generación de respuestas multi-modal.

La **Fase 4: Demonstration** (sección 6.7) desarrolló la interfaz de usuario mediante Streamlit, demostrando el uso práctico del artefacto con funcionalidades de consulta Q&A y visualización de resultados experimentales. Esta fase materializó la aplicabilidad del sistema tanto para usuarios finales como para análisis académico de rendimiento.

La arquitectura técnica priorizó la separación de responsabilidades, la extensibilidad y la reproducibilidad científica (McConnell, 2004). El diseño soportó evaluación experimental rigurosa mientras mantuvo la flexibilidad necesaria para futuras optimizaciones. Las optimizaciones y mejoras implementadas (sección 6.8) complementaron el desarrollo con consideraciones de rendimiento, calidad y extensibilidad arquitectónica.

## 6.2 Tecnologías Utilizadas

### 6.2.1 Stack Tecnológico Principal

El sistema usó Python 3.12.2 como lenguaje principal por su ecosistema maduro en machine learning y procesamiento de lenguaje natural (Van Rossum & Drake, 2009). Para la interfaz de usuario se adoptó Streamlit 1.46.1, que permitió desarrollo rápido de aplicaciones web interactivas con capacidades de visualización (Streamlit Team, 2023). Como motor de almacenamiento vectorial se seleccionó ChromaDB 0.5.23 por su simplicidad operacional y rendimiento en entornos de investigación (ChromaDB Team, 2024).

### 6.2.2 Librerías Especializadas en NLP

Para los modelos de embeddings se utilizaron sentence-transformers 5.0.0 (MPNet, MiniLM, E5-large) y openai 1.93.0 (Ada). El procesamiento de texto requirió transformers 4.44.0 para el CrossEncoder ms-marco-MiniLM-L-6-v2, torch 2.2.2 como backend para modelos PyTorch, y bert-score 0.3.13 para métricas de evaluación semántica.

### 6.2.3 Infraestructura de Evaluación

El entorno de cómputo incluyó Google Colab con GPU Tesla T4 para aceleración en evaluaciones masivas, Jupyter Notebooks para prototipado y análisis exploratorio, y ejecución local con CPU Intel Core i7 y 16GB RAM para desarrollo iterativo. Los datos se almacenaron en formato Parquet para embeddings pre-computados, JSON para metadatos y resultados de evaluación, y Google Drive para sincronización automática de resultados experimentales.

## 6.3 Fase 3 - Extracción Automatizada de Datos desde Microsoft Learn

### 6.3.1 Herramientas y Técnicas de Web Scraping

La extracción de datos constituyó la base fundamental del sistema RAG. El desarrollo combinó Selenium para navegación dinámica y BeautifulSoup para parsing de contenido, estableciendo un método confiable para la recolección de datos técnicos especializados.

La arquitectura de scraping utilizó Selenium WebDriver con ChromeDriver para manejar JavaScript y contenido dinámico, BeautifulSoup 4 para parsing estructurado de HTML renderizado, estrategias de espera adaptativa para carga asíncrona, y manejo robusto de errores con reintentos automáticos.

Los desafíos técnicos principales incluyeron la carga asíncrona del contenido en Microsoft Learn, que requirió WebDriverWait con condiciones específicas. La estructura HTML variable entre páginas necesitó selectores CSS robustos y flexibles. El volumen de datos superior a 20,000 preguntas requirió un sistema incremental con checkpoints para prevenir pérdida de progreso.

### 6.3.2 Proceso de Extracción de Documentación

La extracción de documentación técnica de Microsoft Learn siguió cuatro pasos principales. Primero, la identificación de puntos de entrada navegó desde los índices principales de Azure. Segundo, el crawling recursivo siguió enlaces internos con filtrado de relevancia para evitar contenido tangencial. Tercero, la extracción de contenido procesó elementos estructurales específicos como títulos, contenido principal y metadatos. Cuarto, la normalización de datos limpió el HTML, normalizó las URLs, y estructuró la información en formato JSON.

La estructura de datos capturó para cada documento el título, la URL normalizada, un resumen extraído del encabezado, el contenido textual completo, y enlaces relacionados a otros documentos de Microsoft Learn. Este formato estructurado facilitó la generación de embeddings.

Los resultados verificados incluyen 62,417 documentos únicos relacionados con Azure, segmentados en 187,031 chunks procesables. La extracción logró cobertura completa de los servicios principales de Azure, preservando metadatos ricos que incluyen títulos, URLs, y contenido textual íntegro.

### 6.3.3 Proceso de Extracción de Preguntas y Respuestas

La extracción de preguntas desde Microsoft Q&A capturó no solo el contenido textual sino también las relaciones semánticas y la validación comunitaria. La metodología incluyó navegación sistemática de páginas indexadas bajo el tag "Azure", extracción de metadatos como fecha y etiquetas, identificación de respuestas aceptadas validadas por la comunidad, y extracción de enlaces a documentación oficial presentes en las respuestas.

La estructura de datos Q&A preservó el título de la pregunta, la URL original, el contenido completo de la pregunta, la respuesta aceptada por la comunidad, las etiquetas temáticas, y la fecha de publicación en formato ISO 8601.

El dataset resultante contiene 13,436 preguntas técnicas con contenido completo, de las cuales 2,067 incluyen enlaces validados a documentación oficial que sirven como ground truth. La distribución temporal muestra concentración en 2023-2024 con 77.3% del total. La longitud promedio de pregunta alcanza 119.9 tokens, mientras que las respuestas promedian 221.6 tokens.

## 6.4 Fase 3 - Implementación de ChromaDB

### 6.4.1 Arquitectura de Base de Datos Vectorial

El proyecto inicialmente utilizó Weaviate como base de datos vectorial por su escalabilidad empresarial, API GraphQL y módulos especializados. Sin embargo, durante las pruebas preliminares se identificaron dos limitaciones críticas para el contexto de investigación académica: latencia de red de 150-300ms por consulta (impactando significativamente la velocidad de experimentación iterativa) y costos de infraestructura para mantener instancias en nube. Estas limitaciones motivaron una migración completa a ChromaDB.

ChromaDB proporcionó latencia local menor a 10ms, portabilidad de datos sin dependencia de servicios externos, y simplicidad de configuración sin costos de infraestructura, resultando óptimo para el desarrollo iterativo y evaluación experimental que requería el proyecto. Esta decisión permitió ejecutar miles de consultas experimentales con tiempos de respuesta predecibles y sin costos adicionales.

### 6.4.2 Configuración e Inicialización

ChromaDB utilizó un patrón de cliente singleton con manejo de conexiones persistentes. El wrapper del cliente manejó la inicialización con path absoluto para consistencia, implementó carga diferida (lazy loading) del cliente para optimizar memoria, y proporcionó acceso cacheado a colecciones con validación de existencia para prevenir errores en tiempo de ejecución.

### 6.4.3 Gestión de Colecciones Multi-Modelo

La arquitectura de almacenamiento usó colecciones separadas para cada modelo de embedding, permitiendo comparaciones directas sin interferencia cruzada. Para multi-qa-mpnet-base-dot-v1 se crearon colecciones de documentos (docs_mpnet con 187,031 documentos en 768 dimensiones), preguntas (questions_mpnet con 13,436 preguntas en 768D), y preguntas validadas (questions_withlinks con 2,067 preguntas). El mismo patrón se replicó para all-MiniLM-L6-v2 (384D), Ada (1536D), y e5-large-v2 (1024D).

### 6.4.4 Optimizaciones de Rendimiento

El almacenamiento eficiente utilizó formato Parquet para embeddings pre-computados, compresión adaptativa basada en dimensionalidad, e indexación optimizada para similitud coseno. La gestión de memoria incluyó carga diferida de colecciones para minimizar footprint, cache de resultados frecuentes con política LRU de desalojo, y procesamiento por lotes para operaciones masivas.

Las métricas de rendimiento observadas mostraron latencia promedio de consulta menor a 10ms para top-k=10, throughput de aproximadamente 241 documentos por segundo para generación de embeddings, y almacenamiento total de 6.48 GB para todas las colecciones.

## 6.5 Fase 3 - Arquitectura del Sistema RAG

### 6.5.1 Componente de Indexación y Embeddings

El sistema permitió comparación directa entre cuatro modelos de representación vectorial mediante una arquitectura modular de generación de embeddings. El cliente de embeddings usó inicialización diferida para prevenir problemas de memoria, soportó modelos distintos para consultas y documentos, y proporcionó métodos separados para generar embeddings de preguntas (modelo optimizado para queries) y embeddings de documentos (modelo optimizado para contenido largo).

### 6.5.2 Componente de Búsqueda Vectorial

#### Búsqueda Vectorial con Filtrado de Diversidad

El componente de búsqueda usó similitud coseno con filtrado de diversidad para evitar resultados redundantes. La búsqueda inicial realizó sobremuestreo recuperando hasta tres veces el número solicitado de documentos (balanceando calidad versus rendimiento). Los resultados se filtraron mediante comparación de similitud coseno entre documentos candidatos, excluyendo aquellos con similitud superior al threshold de 0.85 respecto a documentos ya seleccionados. Este proceso garantizó que los resultados finales fueran semánticamente diversos.

#### Búsqueda Híbrida por Enlaces Validados

El sistema combinó recuperación por enlaces directos con búsqueda vectorial. La búsqueda por lotes optimizada normalizó las URLs para coincidencia robusta, procesó enlaces en batches de 50 para mantener rendimiento, consultó ChromaDB con límite de 5,000 documentos por razones de eficiencia, y filtró resultados comparando enlaces normalizados. Esta estrategia híbrida aprovechó tanto la información estructurada (enlaces explícitos) como la similitud semántica (embeddings vectoriales).

### 6.5.3 Componente de Evaluación

El cálculo de métricas de recuperación siguió estándares establecidos en literatura especializada. El sistema normalizó enlaces para comparación robusta, calculó Mean Reciprocal Rank (MRR) para evaluar la posición del primer documento relevante, y generó métricas @k para diferentes valores de k (1, 3, 5, 10, 15). Para cada k se calcularon Precision@k (proporción de relevantes en top-k), Recall@k (proporción de relevantes totales capturados), F1@k (media armónica de precision y recall), NDCG@k (ganancia acumulada descontada normalizada), y MAP@k (precisión promedio).

## 6.6 Fase 3 - Pipeline de Procesamiento RAG

### 6.6.1 Pipeline End-to-End

El pipeline de procesamiento integró todos los componentes en una arquitectura de ocho etapas. La Etapa 1 refinó la consulta mejorando su claridad y especificidad. La Etapa 2 generó el embedding vectorial de la consulta usando el modelo seleccionado. La Etapa 3 buscó preguntas similares en la base de datos, recuperando las top-30 más cercanas semánticamente. La Etapa 4 extrajo enlaces a documentación oficial desde las respuestas de las top-5 preguntas más similares.

La Etapa 5 ejecutó recuperación híbrida combinando búsqueda por enlaces directos (cuando estaban disponibles) con búsqueda vectorial de documentos por similitud semántica, aplicando threshold de diversidad de 0.85. La Etapa 6 deduplicó y fusionó documentos de ambas fuentes, eliminando duplicados por URL normalizada. La Etapa 7 aplicó opcionalmente reranking neural usando CrossEncoder cuando había múltiples documentos. La Etapa 8 generó la respuesta final usando los top-3 documentos mejor rankeados como contexto.

El pipeline registró detalladamente cada etapa, calculó métricas de tiempo de procesamiento, y retornó la pregunta original, la respuesta generada, los documentos recuperados, las preguntas similares encontradas, el log completo de procesamiento, y métricas de rendimiento incluyendo tiempo total y cantidad de documentos.

### 6.6.2 Reranking con CrossEncoder

El componente de reranking usó el modelo ms-marco-MiniLM-L-6-v2 con normalización de scores. El CrossEncoder procesó pares [pregunta, documento] generando scores de relevancia mediante atención cruzada. El sistema aplicó normalización sigmoid que mapeó logits del CrossEncoder a probabilidades en rango [0,1] mediante la función 1/(1+e^(-x)). Como respaldo, si la normalización sigmoid fallaba por overflow, se aplicó normalización min-max. Los documentos se ordenaron por los scores finales y se retornaron los top-k.

### 6.6.3 Generación de Respuestas Multi-Modal

El sistema soportó múltiples backends de generación. Para modelos locales, se preparó contexto optimizado concatenando hasta tres documentos (limitando cada uno a 800 caracteres), se construyó un prompt estructurado con contexto y pregunta, y se generó la respuesta usando modelos como TinyLlama-1.1B con temperatura baja (0.1) para respuestas deterministas y máximo de 200 tokens.

## 6.7 Fase 4 - Interfaz de Usuario (Streamlit)

### 6.7.1 Arquitectura Multi-Página

La aplicación Streamlit usó arquitectura multi-página que integró todos los componentes del sistema. La navegación incluyó página de Consulta Q&A para interacción principal con el sistema RAG, dashboard de Métricas Cumulativas para visualización de resultados experimentales, y panel de Configuración para ajustes del sistema. La interfaz usó layout amplio (wide) con sidebar expandido por defecto para facilitar navegación.

### 6.7.2 Interfaz de Consulta Q&A

La interfaz principal presentó un área de texto para formular preguntas sobre Azure, con placeholder de ejemplo para guiar al usuario. Los controles de configuración incluyeron selector de modelo de embedding (mpnet, ada, minilm, e5large), slider para especificar top-k documentos (rango 5-20), checkbox para activar CrossEncoder, y checkbox para mostrar fuentes. Al presionar el botón de búsqueda, el sistema ejecutó el pipeline RAG completo y renderizó los resultados con la respuesta generada y opcionalmente las fuentes utilizadas.

### 6.7.3 Dashboard de Métricas

El dashboard de evaluación experimental presentó selector de archivo de resultados, visualizó información general del experimento en cuatro métricas principales (preguntas evaluadas, modelos comparados, top-k, método reranking), generó gráficos comparativos entre modelos, y desplegó tabla detallada de métricas con todas las medidas calculadas para cada configuración experimental.

## 6.8 Optimizaciones y Mejoras

### 6.8.1 Optimizaciones de Rendimiento

El sistema de cache inteligente usó política LRU (Least Recently Used) para almacenar temporalmente modelos de embeddings cargados, eliminando automáticamente los menos utilizados al alcanzar el límite de memoria. El cache persistente almacenó en disco resultados de consultas frecuentes, evitando recálculos. La carga diferida (lazy loading) aplicó a componentes computacionalmente pesados como CrossEncoder y modelos locales, cargándolos solo cuando eran necesarios.

El procesamiento por lotes incluyó búsquedas por enlaces con batch_size de 50, vectorización masiva para generación de embeddings, y paralelización de evaluaciones experimentales. La gestión de memoria automatizó la liberación después de evaluaciones grandes, usó generators para procesar datasets extensos sin cargar todo en memoria, y monitoreó activamente el uso con alertas cuando se aproximaba a límites.

### 6.8.2 Mejoras de Calidad

El filtrado de diversidad usó un algoritmo que evitó documentos redundantes mediante threshold adaptativo basado en distribución de similitudes, preservando documentos altamente relevantes independiente de diversidad. La normalización robusta incluyó estandarización de URLs para matching preciso, limpieza adaptativa de texto para diferentes fuentes, y manejo consistente de encoding y caracteres especiales.

La validación de calidad verificó automáticamente la integridad de embeddings, detectó documentos corrompidos o incompletos, e integró métricas de calidad de datos en el pipeline de procesamiento.

El desarrollo siguió el flujo natural desde la extracción inicial de datos, pasando por la infraestructura de base de datos vectorial y la generación de embeddings, hasta culminar en un pipeline RAG completo con interfaz de usuario integral. Esta arquitectura modular y las optimizaciones proporcionaron una base sólida tanto para investigación académica como para potencial despliegue en producción.
