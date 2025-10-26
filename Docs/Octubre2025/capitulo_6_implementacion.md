# 6. IMPLEMENTACIÓN

## 6.1 Introducción

El sistema RAG (Retrieval-Augmented Generation) desarrollado mejora la gestión de tickets de soporte técnico mediante recuperación semántica de documentación de Microsoft Azure. El desarrollo siguió un flujo natural: primero se extrajeron los datos, luego se estableció la infraestructura de base de datos vectorial, después se generaron los embeddings, y finalmente se construyó el pipeline de recuperación y generación de respuestas.

La arquitectura técnica prioriza la separación de responsabilidades, la extensibilidad y la reproducibilidad científica (McConnell, 2004). El diseño soporta evaluación experimental rigurosa mientras mantiene la flexibilidad necesaria para futuras optimizaciones.

## 6.2 Tecnologías Utilizadas

### 6.2.1 Stack Tecnológico Principal

El sistema usa Python 3.12.2 como lenguaje principal por su ecosistema maduro en machine learning y procesamiento de lenguaje natural (Van Rossum & Drake, 2009). Para la interfaz de usuario se adoptó Streamlit 1.46.1, que permite desarrollo rápido de aplicaciones web interactivas con capacidades de visualización (Streamlit Team, 2023). Como motor de almacenamiento vectorial se seleccionó ChromaDB 0.5.23 por su simplicidad operacional y rendimiento en entornos de investigación (ChromaDB Team, 2024).

### 6.2.2 Librerías Especializadas en NLP

Para los modelos de embeddings se utilizaron sentence-transformers 5.0.0 (MPNet, MiniLM, E5-large) y openai 1.93.0 (Ada). El procesamiento de texto requirió transformers 4.44.0 para el CrossEncoder ms-marco-MiniLM-L-6-v2, torch 2.2.2 como backend para modelos PyTorch, y bert-score 0.3.13 para métricas de evaluación semántica.

### 6.2.3 Infraestructura de Evaluación

El entorno de cómputo incluyó Google Colab con GPU Tesla T4 para aceleración en evaluaciones masivas, Jupyter Notebooks para prototipado y análisis exploratorio, y ejecución local con CPU Intel Core i7 y 16GB RAM para desarrollo iterativo. Los datos se almacenaron en formato Parquet para embeddings pre-computados, JSON para metadatos y resultados de evaluación, y Google Drive para sincronización automática de resultados experimentales.

## 6.3 Extracción Automatizada de Datos desde Microsoft Learn

### 6.3.1 Herramientas y Técnicas de Web Scraping

La extracción de datos constituye la base fundamental del sistema RAG. El desarrollo combinó Selenium para navegación dinámica y BeautifulSoup para parsing de contenido, estableciendo un método confiable para la recolección de datos técnicos especializados.

La arquitectura de scraping utilizó Selenium WebDriver con ChromeDriver para manejar JavaScript y contenido dinámico, BeautifulSoup 4 para parsing estructurado de HTML renderizado, estrategias de espera adaptativa para carga asíncrona, y manejo robusto de errores con reintentos automáticos.

Los desafíos técnicos principales incluyeron la carga asíncrona del contenido en Microsoft Learn, que requirió WebDriverWait con condiciones específicas. La estructura HTML variable entre páginas necesitó selectores CSS robustos y flexibles. El volumen de datos superior a 20,000 preguntas requirió un sistema incremental con checkpoints para prevenir pérdida de progreso.

### 6.3.2 Proceso de Extracción de Documentación

La extracción de documentación técnica de Microsoft Learn siguió cuatro pasos principales. Primero, la identificación de puntos de entrada navegó desde los índices principales de Azure. Segundo, el crawling recursivo siguió enlaces internos con filtrado de relevancia para evitar contenido tangencial. Tercero, la extracción de contenido procesó elementos estructurales específicos como títulos, contenido principal y metadatos. Cuarto, la normalización de datos limpió el HTML, normalizó las URLs, y estructuró la información en formato JSON.

La estructura de datos capturó para cada documento el título, la URL normalizada, un resumen extraído del encabezado, el contenido textual completo, y enlaces relacionados a otros documentos de Microsoft Learn. Este formato estructurado facilitó el procesamiento posterior y la generación de embeddings.

Los resultados verificados incluyen 62,417 documentos únicos relacionados con Azure, segmentados en 187,031 chunks procesables. La extracción logró cobertura completa de los servicios principales de Azure, preservando metadatos ricos que incluyen títulos, URLs, y contenido textual íntegro.

### 6.3.3 Proceso de Extracción de Preguntas y Respuestas

La extracción de preguntas desde Microsoft Q&A capturó no solo el contenido textual sino también las relaciones semánticas y la validación comunitaria. La metodología incluyó navegación sistemática de páginas indexadas bajo el tag "Azure", extracción de metadatos como fecha y etiquetas, identificación de respuestas aceptadas validadas por la comunidad, y extracción de enlaces a documentación oficial presentes en las respuestas.

La estructura de datos Q&A preservó el título de la pregunta, la URL original, el contenido completo de la pregunta, la respuesta aceptada por la comunidad, las etiquetas temáticas, y la fecha de publicación en formato ISO 8601.

El dataset resultante contiene 13,436 preguntas técnicas con contenido completo, de las cuales 2,067 incluyen enlaces validados a documentación oficial que sirven como ground truth. La distribución temporal muestra concentración en 2023-2024 con 77.3% del total. La longitud promedio de pregunta alcanza 119.9 tokens, mientras que las respuestas promedian 221.6 tokens.

### 6.3.4 Consideraciones Éticas y Legales

#### Marco Legal y Licenciamiento

La documentación de Microsoft Learn está licenciada bajo Creative Commons Attribution 4.0 International (CC BY 4.0), excepto donde se indique lo contrario (Microsoft Corporation, 2024). El proyecto cumplió estrictamente estas condiciones mediante atribución completa reconociendo a Microsoft como autor del material original, uso exclusivamente académico para fines de investigación y educación superior, ausencia de redistribución del contenido textual íntegro, y transformación académica utilizando el contenido como insumo para modelos de recuperación semántica.

#### Buenas Prácticas Aplicadas

El respeto por los recursos del servidor se garantizó mediante delays adaptativos entre requests para evitar sobrecarga, respeto por directivas robots.txt y headers de rate limiting, y uso de User-Agent identificativo para transparencia del propósito académico. La protección de datos excluyó información personal o identificadores de usuarios, anonimizó metadatos no esenciales, y garantizó almacenamiento seguro con acceso restringido. La transparencia se mantuvo documentando completamente las fuentes y metodologías, preservando trazabilidad mediante URLs originales, y disponibilizando scripts para validación independiente.

#### Limitaciones y Salvaguardas

El proyecto adoptó voluntariamente limitaciones como la exclusión de contenido marcado como confidencial o beta, respeto por contenido con restricciones específicas de licenciamiento, y limitación temporal de datos para evitar obsolescencia. Las salvaguardas incluyeron monitoreo regular de cambios en términos de uso, procedimientos documentados de eliminación de datos si requerido, y contacto establecido con Microsoft para transparencia de la investigación. Estas medidas garantizan la integridad académica del proyecto.

**Nota sobre Implementación:** El código específico de scraping no se incluye en el sistema actual porque la extracción de datos se realizó en una fase previa. Los datos extraídos se almacenaron en formato estructurado (JSON y Parquet) y se utilizan directamente desde ChromaDB.

## 6.4 Implementación de ChromaDB

### 6.4.1 Arquitectura de Base de Datos Vectorial

ChromaDB se seleccionó como base de datos vectorial principal después de una migración desde Weaviate. La decisión se basó en optimizar el flujo de investigación académica. Weaviate ofrecía escalabilidad empresarial, API GraphQL y módulos especializados, pero presentaba latencia de red de 150-300ms por consulta y dependencia de conectividad externa, siendo más apropiado para aplicaciones de producción distribuida. ChromaDB proporcionó latencia local menor a 10ms, portabilidad de datos y simplicidad de configuración, resultando óptimo para investigación y desarrollo iterativo.

### 6.4.2 Configuración e Inicialización

ChromaDB usa un patrón de cliente singleton con manejo de conexiones persistentes. El wrapper del cliente maneja la inicialización con path absoluto para consistencia, implementa carga diferida (lazy loading) del cliente para optimizar memoria, y proporciona acceso cacheado a colecciones con validación de existencia para prevenir errores en tiempo de ejecución.

### 6.4.3 Gestión de Colecciones Multi-Modelo

La arquitectura de almacenamiento usa colecciones separadas para cada modelo de embedding, permitiendo comparaciones directas sin interferencia cruzada. Para multi-qa-mpnet-base-dot-v1 se crearon colecciones de documentos (docs_mpnet con 187,031 documentos en 768 dimensiones), preguntas (questions_mpnet con 13,436 preguntas en 768D), y preguntas validadas (questions_withlinks con 2,067 preguntas). El mismo patrón se replicó para all-MiniLM-L6-v2 (384D), Ada (1536D), y e5-large-v2 (1024D).

### 6.4.4 Optimizaciones de Rendimiento

El almacenamiento eficiente utiliza formato Parquet para embeddings pre-computados, compresión adaptativa basada en dimensionalidad, e indexación optimizada para similitud coseno. La gestión de memoria incluye carga diferida de colecciones para minimizar footprint, cache de resultados frecuentes con política LRU de desalojo, y procesamiento por lotes para operaciones masivas.

Las métricas de rendimiento observadas muestran latencia promedio de consulta menor a 10ms para top-k=10, throughput de aproximadamente 241 documentos por segundo para generación de embeddings, y almacenamiento total de 6.48 GB para todas las colecciones.

## 6.5 Arquitectura del Sistema RAG

### 6.5.1 Componente de Indexación y Embeddings

El sistema permite comparación directa entre cuatro modelos de representación vectorial mediante una arquitectura modular de generación de embeddings. El cliente de embeddings usa inicialización diferida para prevenir problemas de memoria, soporta modelos distintos para consultas y documentos, y proporciona métodos separados para generar embeddings de preguntas (modelo optimizado para queries) y embeddings de documentos (modelo optimizado para contenido largo).

### 6.5.2 Componente de Búsqueda Vectorial

#### Búsqueda Vectorial con Filtrado de Diversidad

El componente de búsqueda usa similitud coseno con filtrado de diversidad para evitar resultados redundantes. La búsqueda inicial realiza sobremuestreo recuperando hasta tres veces el número solicitado de documentos (balanceando calidad versus rendimiento). Los resultados se filtran mediante comparación de similitud coseno entre documentos candidatos, excluyendo aquellos con similitud superior al threshold de 0.85 respecto a documentos ya seleccionados. Este proceso garantiza que los resultados finales sean semánticamente diversos.

#### Búsqueda Híbrida por Enlaces Validados

El sistema combina recuperación por enlaces directos con búsqueda vectorial. La búsqueda por lotes optimizada normaliza las URLs para coincidencia robusta, procesa enlaces en batches de 50 para mantener rendimiento, consulta ChromaDB con límite de 5,000 documentos por razones de eficiencia, y filtra resultados comparando enlaces normalizados. Esta estrategia híbrida aprovecha tanto la información estructurada (enlaces explícitos) como la similitud semántica (embeddings vectoriales).

### 6.5.3 Componente de Evaluación

El cálculo de métricas de recuperación sigue estándares establecidos en literatura especializada. El sistema normaliza enlaces para comparación robusta, calcula Mean Reciprocal Rank (MRR) para evaluar la posición del primer documento relevante, y genera métricas @k para diferentes valores de k (1, 3, 5, 10, 15). Para cada k se calculan Precision@k (proporción de relevantes en top-k), Recall@k (proporción de relevantes totales capturados), F1@k (media armónica de precision y recall), NDCG@k (ganancia acumulada descontada normalizada), y MAP@k (precisión promedio).

## 6.6 Pipeline de Procesamiento RAG

### 6.6.1 Pipeline End-to-End

El pipeline de procesamiento integra todos los componentes en una arquitectura de ocho etapas. La Etapa 1 refina la consulta mejorando su claridad y especificidad. La Etapa 2 genera el embedding vectorial de la consulta usando el modelo seleccionado. La Etapa 3 busca preguntas similares en la base de datos, recuperando las top-30 más cercanas semánticamente. La Etapa 4 extrae enlaces a documentación oficial desde las respuestas de las top-5 preguntas más similares.

La Etapa 5 ejecuta recuperación híbrida combinando búsqueda por enlaces directos (cuando están disponibles) con búsqueda vectorial de documentos por similitud semántica, aplicando threshold de diversidad de 0.85. La Etapa 6 deduplica y fusiona documentos de ambas fuentes, eliminando duplicados por URL normalizada. La Etapa 7 aplica opcionalmente reranking neural usando CrossEncoder cuando hay múltiples documentos. La Etapa 8 genera la respuesta final usando los top-3 documentos mejor rankeados como contexto.

El pipeline registra detalladamente cada etapa, calcula métricas de tiempo de procesamiento, y retorna la pregunta original, la respuesta generada, los documentos recuperados, las preguntas similares encontradas, el log completo de procesamiento, y métricas de rendimiento incluyendo tiempo total y cantidad de documentos.

### 6.6.2 Reranking con CrossEncoder

El componente de reranking usa el modelo ms-marco-MiniLM-L-6-v2 con normalización de scores. El CrossEncoder procesa pares [pregunta, documento] generando scores de relevancia mediante atención cruzada. El sistema aplica normalización sigmoid que mapea logits del CrossEncoder a probabilidades en rango [0,1] mediante la función 1/(1+e^(-x)). Como respaldo, si la normalización sigmoid falla por overflow, se aplica normalización min-max. Los documentos se ordenan por los scores finales y se retornan los top-k.

### 6.6.3 Generación de Respuestas Multi-Modal

El sistema soporta múltiples backends de generación. Para modelos locales, se prepara contexto optimizado concatenando hasta tres documentos (limitando cada uno a 800 caracteres), se construye un prompt estructurado con contexto y pregunta, y se genera la respuesta usando modelos como TinyLlama-1.1B con temperatura baja (0.1) para respuestas deterministas y máximo de 200 tokens.

## 6.7 Interfaz de Usuario (Streamlit)

### 6.7.1 Arquitectura Multi-Página

La aplicación Streamlit usa arquitectura multi-página que integra todos los componentes del sistema. La navegación incluye página de Consulta Q&A para interacción principal con el sistema RAG, dashboard de Métricas Cumulativas para visualización de resultados experimentales, y panel de Configuración para ajustes del sistema. La interfaz usa layout amplio (wide) con sidebar expandido por defecto para facilitar navegación.

### 6.7.2 Interfaz de Consulta Q&A

La interfaz principal presenta un área de texto para formular preguntas sobre Azure, con placeholder de ejemplo para guiar al usuario. Los controles de configuración incluyen selector de modelo de embedding (mpnet, ada, minilm, e5large), slider para especificar top-k documentos (rango 5-20), checkbox para activar CrossEncoder, y checkbox para mostrar fuentes. Al presionar el botón de búsqueda, el sistema ejecuta el pipeline RAG completo y renderiza los resultados con la respuesta generada y opcionalmente las fuentes utilizadas.

### 6.7.3 Dashboard de Métricas

El dashboard de evaluación experimental presenta selector de archivo de resultados, visualiza información general del experimento en cuatro métricas principales (preguntas evaluadas, modelos comparados, top-k, método reranking), genera gráficos comparativos entre modelos, y despliega tabla detallada de métricas con todas las medidas calculadas para cada configuración experimental.

## 6.8 Optimizaciones y Mejoras

### 6.8.1 Optimizaciones de Rendimiento

El sistema de cache inteligente usa política LRU (Least Recently Used) para almacenar temporalmente modelos de embeddings cargados, eliminando automáticamente los menos utilizados al alcanzar el límite de memoria. El cache persistente almacena en disco resultados de consultas frecuentes, evitando recálculos. La carga diferida (lazy loading) aplica a componentes computacionalmente pesados como CrossEncoder y modelos locales, cargándolos solo cuando son necesarios.

El procesamiento por lotes incluye búsquedas por enlaces con batch_size de 50, vectorización masiva para generación de embeddings, y paralelización de evaluaciones experimentales. La gestión de memoria automatiza la liberación después de evaluaciones grandes, usa generators para procesar datasets extensos sin cargar todo en memoria, y monitorea activamente el uso con alertas cuando se aproxima a límites.

### 6.8.2 Mejoras de Calidad

El filtrado de diversidad usa un algoritmo que evita documentos redundantes mediante threshold adaptativo basado en distribución de similitudes, preservando documentos altamente relevantes independiente de diversidad. La normalización robusta incluye estandarización de URLs para matching preciso, limpieza adaptativa de texto para diferentes fuentes, y manejo consistente de encoding y caracteres especiales.

La validación de calidad verifica automáticamente la integridad de embeddings, detecta documentos corrompidos o incompletos, e integra métricas de calidad de datos en el pipeline de procesamiento.

### 6.8.3 Extensibilidad Arquitectónica

Las interfaces modulares separan claramente capas de datos, lógica y presentación. Los estándares definidos facilitan incorporación de nuevos modelos de embedding. La arquitectura de plugins permite métricas de evaluación customizadas sin modificar el código base.

La configuración flexible usa archivos JSON para parámetros del sistema, variables de entorno para secrets y paths, y permite override dinámico de configuraciones via interfaz web. El logging estructurado proporciona niveles configurables, métricas de rendimiento integradas, y trazabilidad completa de requests y resultados.

El desarrollo siguió el flujo natural desde la extracción inicial de datos, pasando por la infraestructura de base de datos vectorial y la generación de embeddings, hasta culminar en un pipeline RAG completo con interfaz de usuario integral. Esta arquitectura modular y las optimizaciones proporcionan una base sólida tanto para investigación académica como para potencial despliegue en producción.

## 6.9 Nota sobre Implementación

El código completo de todos los componentes descritos en este capítulo está disponible en el repositorio público del proyecto en los directorios `src/` y `colab_data/`. La implementación incluye módulos de extracción de datos, generación de embeddings, búsqueda vectorial, reranking, generación de respuestas, evaluación de métricas, y la aplicación Streamlit completa.

## 6.10 Referencias del Capítulo

ChromaDB Team. (2024). *ChromaDB: The AI-native open-source embedding database*. https://www.trychroma.com/

McConnell, S. (2004). *Code Complete: A Practical Handbook of Software Construction* (2nd ed.). Microsoft Press.

Microsoft Corporation. (2024). *Microsoft Learn Terms of Use*. https://learn.microsoft.com/en-us/legal/

Streamlit Team. (2023). *Streamlit: The fastest way to build and share data apps*. https://streamlit.io/

Van Rossum, G., & Drake, F. L. (2009). *Python 3 Reference Manual*. CreateSpace Independent Publishing Platform.
