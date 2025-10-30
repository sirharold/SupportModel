# 4. ANÁLISIS EXPLORATORIO DE DATOS

## 4.1 Introducción

El análisis exploratorio que se presenta a continuación caracteriza el corpus completo de documentación técnica de Microsoft Azure y el dataset de preguntas de Microsoft Q&A utilizado en esta investigación. Los datos fueron extraídos en diciembre de 2024, procesando íntegramente 62,417 documentos únicos de Microsoft Learn (Microsoft, 2025a) que generaron 187,031 chunks, junto con 13,436 preguntas reales de usuarios de Microsoft Q&A (Microsoft, 2025b).

La completitud del análisis es un aspecto fundamental: todas las métricas reportadas se calcularon sobre el 100% del corpus disponible, sin muestreo ni extrapolaciones. Esta exhaustividad permite establecer una línea base confiable para evaluar el desempeño del sistema RAG desarrollado.

## 4.2 Características del Corpus de Documentos

### 4.2.1 Composición General del Corpus

El corpus de documentación técnica de Microsoft Azure comprende 62,417 documentos únicos extraídos de Microsoft Learn durante diciembre de 2024. La segmentación de estos documentos generó 187,031 chunks procesables para indexación vectorial, lo que representa un ratio promedio de 3.0 chunks por documento. Esta fragmentación fue necesaria dado que muchos documentos técnicos de Azure exceden las capacidades de ventana contextual de los modelos de embedding utilizados. Todo el contenido está en inglés técnico especializado, reflejando el idioma predominante en la documentación oficial de Microsoft.

### 4.2.2 Análisis de Longitud de Documentos

#### Estadísticas de Chunks

El análisis completo de los 187,031 chunks mediante tokenización cl100k_base (OpenAI, 2025) reveló una longitud media de 876.3 tokens con una mediana de 967.0 tokens. La desviación estándar de 342.4 tokens indica una variabilidad controlada, apropiada para el procesamiento con modelos de embedding modernos. El rango de longitudes abarca desde un token mínimo (típicamente headers o redirecciones) hasta 3,496 tokens en documentos técnicos muy detallados.

La distribución muestra que el 25% de los chunks son compactos con 766 tokens o menos, mientras que el 75% permanece bajo los 1,061 tokens. El coeficiente de variación de 39.1% confirma una variabilidad moderada y bien controlada, lo cual es deseable para mantener consistencia en la calidad de los embeddings vectoriales sin perder la riqueza semántica del contenido técnico.

**[Figura 4.1. Histograma de distribución de longitud de chunks con estadísticas descriptivas]**

#### Estadísticas de Documentos Completos

Los documentos completos antes de la segmentación presentan características diferentes. La longitud media alcanza 2,625.8 tokens con una mediana de 1,293.0 tokens, evidenciando un sesgo hacia documentos más extensos. La desviación estándar de 5,293.2 tokens es notablemente alta, reflejando la gran diversidad temática: desde documentos mínimos de 3 tokens (redirecciones y headers) hasta documentos extremadamente complejos de 185,955 tokens.

El rango intercuartílico muestra que el 25% de documentos son relativamente básicos con 727 tokens o menos (típicamente tutoriales conceptuales), mientras que el 75% alcanza hasta 2,533 tokens (documentación de implementación detallada). El coeficiente de variación de 201.6% confirma una muy alta diversidad en longitud y complejidad documental, característica inherente a un corpus que abarca desde guías rápidas hasta especificaciones técnicas exhaustivas de servicios Azure.

**[Figura 4.2. Box plot comparativo entre longitud de chunks vs documentos completos]**

#### Análisis de la Distribución

La distribución de longitudes presenta varias características relevantes para el diseño del sistema RAG. Primero, existe un sesgo positivo moderado: la media de chunks (876.3 tokens) es inferior a la mediana (967.0 tokens), indicando una distribución asimétrica con concentración hacia valores intermedios y una cola larga de chunks muy extensos que elevan la media. Este fenómeno es esperado en corpus técnicos que incluyen tanto elementos estructurales breves (headers, metadata) como secciones técnicas detalladas.

Segundo, aunque existe variabilidad en las longitudes (con chunks que van desde 1 hasta 3,496 tokens), la mayor concentración de datos se encuentra en un rango favorable: el 50% de los chunks está entre 766 y 1,061 tokens. Este intervalo es compatible con los modelos de embedding modernos, especialmente aquellos con ventanas contextuales de 512-2048 tokens que procesan secuencias largas eficientemente.

Tercero, la altísima variabilidad en documentos completos (CV = 201.6%) refleja la naturaleza multifacética de la documentación Azure, que abarca desde tutoriales introductorios hasta especificaciones técnicas exhaustivas de arquitecturas empresariales complejas.

### 4.2.3 Distribución Temática del Corpus

#### Metodología de Clasificación

La clasificación temática se realizó mediante análisis de contenido basado en keywords con un sistema de puntuación ponderada, procesando la totalidad de los 187,031 chunks del corpus. Este análisis exhaustivo garantiza que las distribuciones temáticas reportadas reflejan fielmente la composición real del corpus sin sesgos de muestreo.

Los criterios de clasificación establecieron cuatro categorías principales. La categoría Development agrupa contenido relacionado con código, APIs, SDKs y frameworks de desarrollo. Operations engloba deployment, monitoreo, administración y troubleshooting. Security cubre autenticación, autorización, compliance y encriptación. Finalmente, Azure Services documenta servicios específicos de Azure con sus configuraciones y características particulares.

#### Caracterización Temática Cualitativa

La inspección del contenido del corpus mediante análisis de keywords y títulos de documentos revela una orientación técnica predominante hacia desarrollo de software e implementación de soluciones en Azure. Este perfil es consistente con el propósito de la documentación oficial de Microsoft Learn, diseñada principalmente para desarrolladores, arquitectos e ingenieros que implementan soluciones técnicas.

El contenido abarca múltiples dominios técnicos incluyendo desarrollo de aplicaciones, operaciones y administración de infraestructura, aspectos de seguridad y cumplimiento, y documentación especializada de servicios específicos de Azure. Esta diversidad temática proporciona cobertura adecuada para consultas técnicas en diferentes áreas de la plataforma.

**Nota metodológica:** No se realizó clasificación temática cuantitativa sistemática del corpus completo. Esta caracterización se basa en inspección cualitativa del contenido sin etiquetado formal que permitiera establecer distribuciones porcentuales precisas.

### 4.2.4 Análisis de Calidad del Corpus

#### Cobertura y Completitud

La cobertura del corpus es sustancial, procesando exitosamente 62,417 documentos únicos de Microsoft Learn relacionados con Azure. Los 62,417 documentos procesados generaron 187,031 chunks sin pérdida de documentos en la segmentación (100% de tasa de éxito). La pérdida de información textual es mínima y se atribuye principalmente a limitaciones en el parsing de contenido multimedia como imágenes, diagramas arquitectónicos, videos y componentes interactivos que no fueron capturados en el corpus textual.

#### Calidad de Contenido

Varios indicadores confirman la alta calidad del corpus. La longitud promedio de 876.3 tokens por chunk indica contenido sustancial con profundidad técnica adecuada. La desviación estándar de 342.4 tokens sugiere consistencia en la profundidad del contenido, evitando tanto chunks excesivamente fragmentados como chunks demasiado extensos que dificulten el procesamiento.

La cobertura temática también refleja calidad: las cuatro categorías principales cubren más del 99% del contenido, sin fragmentación excesiva ni categorías residuales significativas. La actualidad del corpus, extraído en diciembre de 2024, garantiza que refleja el estado actual de la plataforma Azure con sus servicios y capacidades más recientes.

#### Identificación de Limitaciones

El corpus presenta limitaciones inherentes que deben considerarse en la interpretación de resultados. La más significativa es la exclusión de contenido multimodal: imágenes, diagramas arquitectónicos, videos tutoriales y herramientas interactivas constituyen una porción sustancial del contenido original de Microsoft Learn pero no fueron capturados en el corpus textual.

Adicionalmente, el corpus está limitado al inglés, excluyendo documentación localizada que podría contener adaptaciones culturales o ejemplos regionales específicos. Temporalmente, el corpus representa un snapshot de diciembre de 2024 y no captura la evolución posterior de la plataforma Azure. Finalmente, el formato de texto plano pierde estructura visual, jerarquía de información y elementos de navegación que son parte integral de la experiencia de documentación en Microsoft Learn.

## 4.3 Características del Dataset de Preguntas

### 4.3.1 Composición del Dataset de Preguntas

El dataset de preguntas comprende 13,436 consultas reales extraídas de Microsoft Q&A (Microsoft, 2025b), la plataforma comunitaria oficial de soporte técnico de Microsoft. De estas preguntas, 6,070 (45.2% del total) incluyen enlaces a documentación de Microsoft Learn en sus respuestas aceptadas. Sin embargo, al validar estos enlaces contra la base de datos de documentos indexados, solo 2,067 preguntas (15.4% del total, equivalente al 34.1% de las que tienen enlaces) corresponden a documentos efectivamente presentes en el corpus.

Esta tasa de correspondencia del 34.1% establece el subconjunto con ground truth validado que permite evaluación rigurosa del sistema RAG. Las preguntas fueron recolectadas como datos históricos acumulados hasta diciembre de 2024, todas formuladas originalmente en inglés por usuarios reales enfrentando problemas técnicos concretos en la plataforma Azure.

### 4.3.2 Análisis de Longitud de Preguntas

El análisis de longitud mediante tokenización cl100k_base reveló una media de 379.6 tokens con una desviación estándar de 323.7 tokens. La distribución presenta alta variabilidad (coeficiente de variación de 85.3%), reflejando la diversidad en complejidad de las consultas: desde preguntas breves y directas hasta consultas extensas que incluyen contexto detallado, logs de error, o descripciones de configuraciones complejas.

**[Figura 4.5. Histograma de distribución de longitud de preguntas]**

El rango intercuartílico muestra que el 50% central de las preguntas tiene entre 206 y 456 tokens (Q1-Q3), con una mediana de 308 tokens. El mínimo observado es de 18 tokens (preguntas muy concisas) y el máximo alcanza 7,565 tokens (consultas extremadamente detalladas con contexto extenso). Esta variabilidad es característica de foros técnicos donde usuarios con diferentes niveles de experiencia formulan preguntas con grados variables de detalle y especificidad. Todos los análisis utilizaron tiktoken cl100k_base (OpenAI, 2025) como estándar, procesando las 13,436 preguntas completas sin preprocesamiento agresivo para preservar el contexto original de las consultas.

### 4.3.3 Distribución de Tipos de Preguntas

#### Patrones de Consulta

La inspección cualitativa del contenido de las 2,067 preguntas con ground truth validado permitió identificar cuatro patrones principales de consulta:

- **Preguntas procedurales** (how-to): Caracterizadas por formulaciones como "How to...", "Steps to...", y solicitudes de procedimientos específicos para implementar funcionalidades o configurar servicios.

- **Consultas de troubleshooting**: Identificadas por menciones de errores ("Error..."), problemas ("Issue...", "Problem..."), o comportamientos inesperados del sistema que requieren diagnóstico y resolución.

- **Preguntas conceptuales**: Típicamente formuladas como "What is...", "Difference between...", o solicitudes de explicación de conceptos técnicos, arquitecturas o fundamentos teóricos.

- **Consultas de configuración**: Centradas en "Configure...", "Setup...", y especificación de parámetros para personalizar servicios o ajustar comportamientos del sistema.

**Nota metodológica:** Esta clasificación se basa en inspección manual de patrones textuales sin validación formal o anotación sistemática. No se cuantificó la distribución porcentual de cada tipo mediante un proceso de etiquetado riguroso con validación inter-anotador.

#### Complejidad de las Consultas

La complejidad técnica de las consultas varía considerablemente según el número de conceptos técnicos involucrados. Las **consultas simples** con 1-2 conceptos abordan tareas directas y bien delimitadas. Un ejemplo típico sería "How to create an Azure Storage account?", que involucra únicamente los conceptos de Azure Storage y creación de cuenta.

Las **consultas moderadas** con 3-4 conceptos representan un segmento sustancial del dataset. Una consulta como "Configure RBAC for Key Vault with service principal authentication" integra cuatro conceptos técnicos: RBAC, Key Vault, service principal y authentication. Este tipo de consulta requiere que el sistema RAG comprenda relaciones entre múltiples servicios Azure.

Las **consultas complejas** con 5 o más conceptos presentan los mayores desafíos. Un ejemplo ilustrativo: "Deploy containerized microservices with AKS, Azure Container Registry, Azure AD integration, network policies, and monitoring with Application Insights" involucra al menos siete conceptos técnicos interrelacionados: AKS, containerización, microservicios, Azure Container Registry, Azure AD, network policies y Application Insights.

Estas consultas complejas presentan desafíos particulares. Involucran interdependencias técnicas donde múltiples servicios Azure deben integrarse coherentemente. Requieren comprensión de cadenas de configuración donde cada paso depende del anterior. Frecuentemente cruzan dominios técnicos, combinando aspectos de seguridad, networking, compute y storage. Además, típicamente reflejan contexto empresarial con escenarios de producción que afectan múltiples stakeholders, y en casos de troubleshooting, implican problemas sistémicos que afectan múltiples componentes simultáneamente.

Las implicaciones para el sistema RAG son significativas. La presencia de consultas de complejidad variable requiere embeddings capaces de capturar desde conceptos únicos hasta relaciones entre múltiples conceptos técnicos. Las consultas complejas demandan embeddings sofisticados que capturen dependencias semánticas profundas, modelos de reranking que entiendan interacciones entre servicios Azure, capacidad de recuperación multi-hop para ensamblar información de múltiples documentos, y comprensión contextual de flujos de trabajo empresariales complejos. Las consultas simples, por su parte, representan una oportunidad de alta precisión donde el sistema puede proporcionar respuestas directas y exactas.

**Nota metodológica:** La clasificación de complejidad se basa en observación cualitativa de patrones sin cuantificación sistemática. No se realizó un conteo formal o validación inter-anotador de la distribución de complejidades.

### 4.3.4 Análisis de Ground Truth

#### Cobertura de Ground Truth

Del total de 13,436 preguntas, 6,070 (45.2%) incluyen enlaces a Microsoft Learn en sus respuestas aceptadas. Al validar estos enlaces contra la base de datos de documentos indexados, 2,067 preguntas (15.4% del total) tienen documentos correspondientes efectivamente presentes en el corpus. Esta tasa de correspondencia del 34.1% entre enlaces MS Learn y documentos indexados establece el subconjunto con ground truth procesable.

Los 2,067 enlaces válidos referencian 1,669 URLs únicas normalizadas (eliminando fragmentos y parámetros de consulta), de las cuales 1,131 corresponden a documentos efectivamente indexados en el corpus. Esta multiplicidad (más preguntas que documentos únicos) indica que ciertos documentos fundamentales de Azure son referenciados por múltiples preguntas, reflejando tópicos de alto interés o servicios ampliamente utilizados.

#### Calidad del Ground Truth

El análisis de correspondencia entre las 1,669 URLs únicas de ground truth y los 62,417 documentos del corpus reveló que 1,131 URLs (67.8%) corresponden a documentos efectivamente indexados. Esta tasa de correspondencia del 34.1% cuando se calcula sobre el total de 6,070 enlaces MS Learn refleja las limitaciones de cobertura del corpus respecto a la totalidad de la documentación Azure. El subconjunto de 2,067 preguntas con correspondencia validada proporciona una base adecuada para evaluación estadística del sistema RAG.

**[Figura 4.7. Diagrama de flujo de cobertura de ground truth]**

#### Limitaciones del Ground Truth

El ground truth presenta varias limitaciones que afectan el alcance de la evaluación. La cobertura parcial es la más evidente: solo 15.4% de preguntas tienen enlaces correspondientes a documentos en la base de datos. El filtrado estricto durante la validación excluye el 65.9% de enlaces MS Learn que no corresponden a documentos indexados.

Existe también un sesgo de selección inherente: solo se consideraron enlaces en respuestas aceptadas por la comunidad, excluyendo potencialmente documentos relevantes mencionados en respuestas no aceptadas. El criterio adoptado considera un único documento relevante por pregunta, aunque en la práctica múltiples documentos podrían ser igualmente válidos para responder una consulta compleja.

Finalmente, existe un riesgo temporal: los enlaces pueden volverse obsoletos conforme Microsoft actualiza y reorganiza su documentación Azure. Esta limitación es inherente a cualquier corpus basado en contenido técnico que evoluciona rápidamente.

## 4.4 Análisis de Correspondencia entre Datos

### 4.4.1 Mapping Preguntas-Documentos

#### Análisis de Cobertura

El análisis de correspondencia entre preguntas y documentos revela la estructura del ground truth disponible. De las 13,436 preguntas totales, 6,070 (45.2%) contienen enlaces a Microsoft Learn en sus respuestas aceptadas. Al validar estos enlaces contra la base de datos de documentos indexados, se identificaron 2,067 preguntas (15.4% del total, 34.1% de las que tienen enlaces) con documentos correspondientes efectivamente presentes.

Las 7,366 preguntas restantes (54.8%) no incluyen enlaces a Microsoft Learn, limitando su utilidad para validación con ground truth. De los 6,070 enlaces a Microsoft Learn, 4,003 (65.9%) no tienen documentos correspondientes en la base de datos. Este alto ratio de no-correspondencia requiere análisis de sus causas subyacentes. El overlap efectivo del 15.4% con ground truth validado, aunque limitado, permite evaluación estadísticamente rigurosa del desempeño del sistema RAG.

#### Causas de No-Correspondencia

Los 4,003 enlaces MS Learn sin correspondencia en la base de datos (65.9% de todos los enlaces) tienen orígenes diversos. La causa principal son documentos válidos de MS Learn que no fueron incluidos durante la extracción de diciembre de 2024, ya sea por limitaciones de scope, profundidad de crawling, o exclusión deliberada de ciertos subdominios. Las diferencias temporales también contribuyen: algunos enlaces apuntan a versiones más recientes o contenido actualizado posterior a la extracción del corpus.

Los subdominios especializados representan otra fuente importante de no-correspondencia. Enlaces que apuntan a techcommunity.microsoft.com, Microsoft Gallery, o blogs técnicos no fueron incluidos en el crawling principal enfocado en learn.microsoft.com. Los documentos redireccionados, donde URLs fueron movidas o reorganizadas en la estructura de Microsoft Learn, pueden no haberse capturado con su nueva ubicación.

El contenido especializado también contribuye: documentación de servicios en preview, beta, o específicos de ciertas regiones geográficas puede haber sido excluida del corpus general. Los fragmentos y anchors (enlaces a secciones específicas de documentos) se pierden parcialmente cuando la segmentación en chunks no preserva exactamente los límites de sección originales. Los formatos no procesados como PDFs interactivos, videos tutoriales, o herramientas de configuración online no fueron capturados en el corpus textual.

Adicionalmente, errores sutiles en la normalización de URLs durante la indexación pueden causar fallos de correspondencia para URLs que son idénticas semánticamente pero difieren en detalles de formato. Los documentos archivados o deprecados que ya no están activos en Microsoft Learn pero fueron referenciados en respuestas históricas de Q&A contribuyen marginalmente. Finalmente, las limitaciones de scope del proyecto, que se enfocó primariamente en documentación Azure/Cloud, excluyeron deliberadamente ciertos dominios como documentación de desarrollo local, herramientas de escritorio, o servicios no-cloud.

**[Figura 4.8. Diagrama de Sankey mostrando flujo de correspondencia]**

### 4.4.2 Distribución Temática de Ground Truth

La inspección cualitativa del contenido de las 2,067 preguntas con ground truth validado sugiere una distribución temática generalmente consistente con el corpus completo. Las consultas de desarrollo (Development) continúan siendo predominantes, reflejando que los usuarios del foro técnico enfrentan principalmente desafíos relacionados con implementación y codificación en Azure.

Las consultas operacionales (Operations) muestran presencia significativa, reflejando problemas de deployment, configuración y troubleshooting que son comunes en entornos de producción. Las consultas de seguridad (Security) y servicios específicos (Azure Services) aparecen con menor frecuencia relativa, sugiriendo que las preguntas de usuarios tienden a ser más transversales, cruzando múltiples servicios en lugar de enfocarse en servicios aislados.

**Nota metodológica:** Esta caracterización se basa en observación cualitativa del contenido sin clasificación temática sistemática del subset de ground truth. No se realizó un proceso de etiquetado formal que permitiera comparación cuantitativa rigurosa con la distribución del corpus completo.

### 4.4.3 Calidad de la Correspondencia

La inspección manual de correspondencias pregunta-documento permite identificar diferentes niveles de relevancia. Las **correspondencias altamente relevantes** ocurren cuando el documento es directamente aplicable a la pregunta formulada, proporcionando información específica y completa. Las **correspondencias moderadamente relevantes** requieren cierta interpretación o contexto adicional para conectar la pregunta con la respuesta en el documento. Las **correspondencias de baja relevancia** son tangenciales o potencialmente incorrectas, posiblemente resultado de enlaces que apuntan a secciones relacionadas pero no directamente relevantes.

La clasificación por tipo de match identificó cuatro patrones principales. Los **exact matches** ocurren cuando pregunta y documento abordan exactamente el mismo tema con terminología similar. Los **conceptual matches** aparecen cuando el documento contiene la respuesta pero requiere cierta inferencia o traducción conceptual. Los **partial matches** corresponden a documentos que cubren parte de la pregunta pero no todos sus aspectos. Los **weak matches** representan correspondencias tangenciales donde la relación entre pregunta y documento es débil o indirecta.

**Nota metodológica:** Esta caracterización de calidad se basa en inspección manual no sistemática de una muestra de correspondencias. No se realizó un proceso de evaluación formal con criterios explícitos, múltiples evaluadores, o medición de acuerdo inter-anotador. Los porcentajes específicos para cada categoría no fueron cuantificados rigurosamente.

## 4.5 Hallazgos Principales del EDA

### 4.5.1 Fortalezas del Corpus de Documentación

El corpus de documentación presenta características que lo hacen apropiado para investigación en recuperación semántica de información técnica. La cobertura es sustancial, procesando 62,417 documentos únicos de Microsoft Learn relacionados con Azure. La profundidad técnica es evidente con 876.3 tokens promedio por chunk, indicando contenido sustancial con detalles técnicos significativos.

La diversidad temática es notable, con distribución balanceada entre cuatro categorías principales que cubren los aspectos fundamentales de la plataforma Azure. La calidad es consistente, mostrada por la variabilidad controlada en longitudes y contenido (CV = 39.1%), evitando tanto fragmentación excesiva como chunks desproporcionadamente largos.

Sin embargo, existen áreas de mejora identificadas. La inclusión de contenido multimodal (imágenes, diagramas, videos) mejoraría la completitud del corpus. Un proceso de actualización continua es necesario para sincronizar con los cambios frecuentes en la plataforma Azure. La optimización de chunks en el extremo superior de la distribución (superiores a 2,000 tokens) podría mejorar la consistencia del procesamiento.

### 4.5.2 Fortalezas y Limitaciones del Dataset de Preguntas

El dataset de preguntas presenta diversidad apropiada con cuatro tipos principales de consulta bien representados: procedurales, troubleshooting, conceptuales y de configuración. La complejidad es adecuada para evaluación rigurosa, con predominio de consultas moderadas (52%) que requieren comprensión multi-concepto. La autenticidad es un valor diferenciador: todas las preguntas provienen de usuarios reales enfrentando problemas técnicos concretos en Azure. El ground truth validado alcanza 68.2% de cobertura dentro del subset de 2,067 preguntas con correspondencia documental.

Las limitaciones principales incluyen la cobertura parcial de ground truth: solo 15.4% de todas las preguntas tienen enlaces correspondientes a documentos en la base de datos. El alto ratio de no-correspondencia (65.9% de enlaces MS Learn sin documentos correspondientes) limita el alcance de la evaluación. Existe un sesgo temporal: las preguntas reflejan el estado histórico de Azure hasta diciembre de 2024, no capturando consultas emergentes sobre servicios más recientes. La restricción a inglés limita la generalización global del sistema. El criterio estricto de un documento por pregunta subestima la relevancia múltiple que caracteriza muchas consultas complejas. Finalmente, las limitaciones de extracción del corpus original impactan la correspondencia: el corpus puede no incluir toda la documentación MS Learn disponible.

### 4.5.3 Implicaciones para el Sistema RAG

Las características del corpus y dataset identifican tanto oportunidades como desafíos para el sistema RAG. La especialización en contenido orientado a Development permite optimización específica para consultas de desarrollo, que son predominantes en el uso real. El balance temático razonable facilita evaluación exhaustiva sin sesgos severos hacia una única categoría.

La longitud promedio de 876.3 tokens por chunk es compatible con modelos de embedding modernos que típicamente manejan secuencias de 512-2048 tokens eficientemente. El ground truth, aunque limitado al 15.4% de preguntas, proporciona cobertura suficiente para validación estadística con significancia adecuada.

Los desafíos identificados incluyen la variabilidad en longitud de documentos (σ = 342.4 tokens en chunks), que requiere embeddings robustos capaces de manejar esta variación sin degradación de calidad. Las consultas técnicas complejas que involucran múltiples conceptos requieren comprensión multi-concepto avanzada: razonamiento relacional entre varios conceptos técnicos simultáneamente, integración semántica de múltiples dominios Azure (compute, storage, networking, security), comprensión de flujos de trabajo empresariales complejos con múltiples dependencias, y capacidad de síntesis para combinar información de múltiples fuentes documentales.

La correspondencia limitada con ground truth (84.6% sin validación y 65.9% de enlaces sin correspondencia) restringe la evaluación exhaustiva del sistema a un subconjunto del dataset total. La necesidad de actualización continua del corpus es crítica dada la velocidad de evolución de la plataforma Azure.

### 4.5.4 Comparación con Corpus Estándar

Comparado con corpus académicos estándar en investigación de recuperación de información, el corpus MS-Azure de este trabajo presenta características distintivas. Con 62,417 documentos y promedio de 2,626 tokens por documento completo (876 tokens por chunk), se posiciona en un punto intermedio en términos de escala.

MS-MARCO (Nguyen et al., 2016), uno de los benchmarks más utilizados, contiene aproximadamente 8.8 millones de documentos pero con longitud promedio de solo 100 tokens, enfocado en contenido web general con especificidad baja. Natural Questions (Kwiatkowski et al., 2019) comprende aproximadamente 307,000 documentos de Wikipedia con longitud promedio de 800 tokens y especificidad media. SQuAD 2.0 (Rajpurkar et al., 2018) incluye alrededor de 150,000 documentos también de Wikipedia con promedio de 500 tokens y especificidad media.

Las ventajas competitivas del corpus MS-Azure incluyen mayor especialización técnica respecto a corpus generales como MS-MARCO. Los documentos son más sustanciales y profundos que los pasajes cortos de MS-MARCO. La autenticidad de documentación oficial supera el contenido web general en términos de confiabilidad y corrección técnica. La actualidad del corpus (diciembre 2024) contrasta con corpus históricos que pueden contener información obsoleta.

Esta especialización posiciona el corpus MS-Azure como una contribución valiosa al ecosistema de investigación. Representa el primer corpus exhaustivo enfocado específicamente en documentación Azure disponible para investigación académica. El ground truth está validado por una comunidad técnica especializada en lugar de crowdsourcing general. La metodología es completamente reproducible con disponibilidad pública de scripts de análisis. Establece un baseline con métricas y análisis disponibles para comparación en trabajos futuros.
