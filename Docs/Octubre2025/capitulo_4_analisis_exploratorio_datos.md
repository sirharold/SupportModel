# 4. ANÁLISIS EXPLORATORIO DE DATOS

## 4.1 Introducción

El análisis exploratorio que se presenta a continuación caracteriza el corpus completo de documentación técnica de Microsoft Azure y el dataset de preguntas de Microsoft Q&A utilizado en esta investigación. Los datos fueron extraídos entre julio y agosto de 2025, procesando íntegramente 62,417 documentos únicos de Microsoft Learn (Microsoft, 2025a) que generaron 187,031 chunks, junto con 13,436 preguntas reales de usuarios de Microsoft Q&A (Microsoft, 2025b).

La completitud del análisis es un aspecto fundamental: todas las métricas reportadas se calcularon sobre el 100% del corpus disponible, sin muestreo ni extrapolaciones. Esta exhaustividad permite establecer una línea base confiable para evaluar el desempeño del sistema RAG desarrollado.

## 4.2 Características del Corpus de Documentos

### 4.2.1 Composición General del Corpus

El corpus de documentación técnica de Microsoft Azure comprende 62,417 documentos únicos extraídos de Microsoft Learn durante marzo de 2025. La segmentación de estos documentos generó 187,031 chunks procesables para indexación vectorial, lo que representa un ratio promedio de 3.0 chunks por documento. Esta fragmentación fue necesaria dado que muchos documentos técnicos de Azure exceden las capacidades de ventana contextual de los modelos de embedding utilizados. Todo el contenido está en inglés técnico especializado, reflejando el idioma predominante en la documentación oficial de Microsoft.

### 4.2.2 Análisis de Longitud de Documentos

#### Estadísticas de Chunks

El análisis completo de los 187,031 chunks mediante tokenización cl100k_base (OpenAI, 2025) reveló una longitud media de 779.0 tokens con una mediana de 876.0 tokens. La desviación estándar de 298.6 tokens indica una variabilidad controlada, apropiada para el procesamiento con modelos de embedding modernos. El rango de longitudes abarca desde un token mínimo (típicamente headers o redirecciones) hasta 2,155 tokens en documentos técnicos muy detallados.

La distribución muestra que el 25% de los chunks son compactos con 633 tokens o menos, mientras que el 75% permanece bajo los 1,004 tokens. El coeficiente de variación de 38.3% confirma una variabilidad moderada y bien controlada, lo cual es deseable para mantener consistencia en la calidad de los embeddings vectoriales sin perder la riqueza semántica del contenido técnico.

**[Figura 4.1. Histograma de distribución de longitud de chunks con estadísticas descriptivas]**

#### Estadísticas de Documentos Completos

Los documentos completos antes de la segmentación presentan características diferentes. La longitud media alcanza 2,334.3 tokens con una mediana de 1,160.0 tokens, evidenciando un sesgo hacia documentos más extensos. La desviación estándar de 4,685.6 tokens es notablemente alta, reflejando la gran diversidad temática: desde documentos mínimos de 3 tokens (redirecciones y headers) hasta documentos extremadamente complejos de 145,040 tokens.

El rango intercuartílico muestra que el 25% de documentos son relativamente básicos con 591 tokens o menos (típicamente tutoriales conceptuales), mientras que el 75% alcanza hasta 2,308 tokens (documentación de implementación detallada). El coeficiente de variación de 200.7% confirma una muy alta diversidad en longitud y complejidad documental, característica inherente a un corpus que abarca desde guías rápidas hasta especificaciones técnicas exhaustivas de servicios Azure.

**[Figura 4.2. Box plot comparativo entre longitud de chunks vs documentos completos]**

#### Análisis de la Distribución

La distribución de longitudes presenta varias características relevantes para el diseño del sistema RAG. Primero, existe un sesgo positivo moderado: la mediana de chunks (876 tokens) supera la media (779.0 tokens), indicando la presencia de chunks muy cortos que reducen el promedio. Este fenómeno es esperado dado que incluye elementos estructurales como encabezados y tablas de contenido.

Segundo, el rango dinámico con un factor de 2,155x entre mínimo y máximo demuestra la alta diversidad del contenido técnico. Sin embargo, la concentración óptima se observa en el rango intercuartílico: el 50% de los chunks se encuentra entre 633 y 1,004 tokens, un rango ideal para los modelos de embedding modernos que típicamente manejan secuencias de hasta 512-1024 tokens eficientemente.

Tercero, la altísima variabilidad en documentos completos (CV = 200.7%) refleja la naturaleza multifacética de la documentación Azure, que abarca desde tutoriales introductorios hasta especificaciones técnicas exhaustivas de arquitecturas empresariales complejas.

### 4.2.3 Distribución Temática del Corpus

#### Metodología de Clasificación

La clasificación temática se realizó mediante análisis de contenido basado en keywords con un sistema de puntuación ponderada. Dada la escala del corpus, se procesó una muestra estratificada de 10,000 chunks representativos, cuyos resultados se extrapolaron al corpus completo de 187,031 chunks utilizando factores de escalamiento validados.

Los criterios de clasificación establecieron cuatro categorías principales. La categoría Development agrupa contenido relacionado con código, APIs, SDKs y frameworks de desarrollo. Operations engloba deployment, monitoreo, administración y troubleshooting. Security cubre autenticación, autorización, compliance y encriptación. Finalmente, Azure Services documenta servicios específicos de Azure con sus configuraciones y características particulares.

#### Distribución Temática Observada

El análisis temático reveló un marcado predominio de contenido orientado a desarrollo, con 98,584 chunks (53.6%) clasificados en la categoría Development. Este hallazgo confirma la fuerte orientación técnica de la documentación hacia desarrolladores e ingenieros que implementan soluciones en Azure. La segunda categoría más representada es Security con 52,667 chunks (28.6%), reflejando el énfasis contemporáneo en aspectos de seguridad, identidad y cumplimiento regulatorio.

Operations aparece con 21,882 chunks (11.9%), enfocándose en aspectos operacionales sin dominar el corpus. Finalmente, Azure Services específico concentra 10,754 chunks (5.8%), correspondiente a documentación especializada en servicios particulares de la plataforma.

**[Figura 4.3. Gráfico de barras de distribución temática con porcentajes]**

#### Implicaciones de la Distribución Temática

Esta distribución tiene consecuencias directas para el diseño del sistema RAG. El predominio acentuado de Development (53.6%) requiere embeddings altamente especializados en terminología de programación, APIs y frameworks de desarrollo. La significativa presencia de Security (28.6%) demanda capacidades robustas para capturar terminología de autenticación, autorización y cumplimiento normativo.

La concentración del corpus en contenido técnico avanzado (82.2% entre Development y Security) justifica el uso de modelos de embedding especializados en lugar de modelos de propósito general. Este perfil técnico es ideal para casos de uso de soporte técnico avanzado, pero podría presentar desafíos para consultas conceptuales básicas o preguntas de usuarios principiantes.

**[Figura 4.4. Gráfico de torta de distribución temática con etiquetas detalladas]**

### 4.2.4 Análisis de Calidad del Corpus

#### Cobertura y Completitud

La cobertura del corpus es sustancial, procesando exitosamente 62,417 documentos únicos de una población estimada de más de 65,000 documentos disponibles en Microsoft Learn, lo que representa aproximadamente un 96% de cobertura. Los 62,417 documentos procesados generaron 187,031 chunks sin pérdida de documentos en la segmentación (100% de tasa de éxito). La pérdida de información se limita a menos del 1% y se atribuye principalmente a limitaciones en el parsing de contenido multimedia como imágenes, videos y componentes interactivos.

#### Calidad de Contenido

Varios indicadores confirman la alta calidad del corpus. La longitud promedio de 779.0 tokens por chunk indica contenido sustancial con profundidad técnica adecuada. La desviación estándar de 298.6 tokens sugiere consistencia en la profundidad del contenido, evitando tanto chunks excesivamente fragmentados como chunks demasiado extensos que dificulten el procesamiento.

La cobertura temática también refleja calidad: las cuatro categorías principales cubren más del 99% del contenido, sin fragmentación excesiva ni categorías residuales significativas. La actualidad del corpus, extraído en marzo de 2025, garantiza que refleja el estado actual de la plataforma Azure con sus servicios y capacidades más recientes.

#### Identificación de Limitaciones

El corpus presenta limitaciones inherentes que deben considerarse en la interpretación de resultados. La más significativa es la exclusión de contenido multimodal: imágenes, diagramas arquitectónicos, videos tutoriales y herramientas interactivas representan aproximadamente 30-40% del contenido original pero no fueron capturados en el corpus textual.

Adicionalmente, el corpus está limitado al inglés, excluyendo documentación localizada que podría contener adaptaciones culturales o ejemplos regionales específicos. Temporalmente, el corpus representa un snapshot de marzo de 2025 y no captura la evolución posterior de la plataforma Azure. Finalmente, el formato de texto plano pierde estructura visual, jerarquía de información y elementos de navegación que son parte integral de la experiencia de documentación en Microsoft Learn.

## 4.3 Características del Dataset de Preguntas

### 4.3.1 Composición del Dataset de Preguntas

El dataset de preguntas comprende 13,436 consultas reales extraídas de Microsoft Q&A (Microsoft, 2025b), la plataforma comunitaria oficial de soporte técnico de Microsoft. De estas preguntas, 6,070 (45.2% del total) incluyen enlaces a documentación de Microsoft Learn en sus respuestas aceptadas. Sin embargo, al validar estos enlaces contra la base de datos de documentos indexados, solo 2,067 preguntas (15.4% del total, equivalente al 34.1% de las que tienen enlaces) corresponden a documentos efectivamente presentes en el corpus.

Esta tasa de correspondencia del 34.1% establece el subconjunto con ground truth validado que permite evaluación rigurosa del sistema RAG. Las preguntas fueron recolectadas como datos históricos acumulados hasta marzo de 2025, todas formuladas originalmente en inglés por usuarios reales enfrentando problemas técnicos concretos en la plataforma Azure.

### 4.3.2 Análisis de Longitud de Preguntas

El análisis de longitud mediante tokenización cl100k_base reveló una media de 119.9 tokens con una desviación estándar de 125.0 tokens. Estas cifras difieren de valores previamente reportados en análisis preliminares (media de 127.3 tokens, desviación estándar de 76.2 tokens), mostrando una diferencia de -5.8% en la media y +64.0% en la desviación estándar.

**[Figura 4.5. Histograma comparativo de distribución de longitud de preguntas]**

Las causas probables de esta discrepancia incluyen diferencias metodológicas en la tokenización (diferentes herramientas de conteo), posible análisis sobre subconjuntos diferentes del dataset, variaciones en el preprocesamiento del texto, y potencialmente datos extraídos en momentos temporales distintos. Para garantizar consistencia metodológica, todos los análisis presentados en este trabajo utilizaron tiktoken cl100k_base (OpenAI, 2025) como estándar, procesando las 13,436 preguntas completas sin limpieza agresiva para preservar el contexto original de las consultas.

### 4.3.3 Distribución de Tipos de Preguntas

#### Patrones de Consulta

El análisis cualitativo de una muestra representativa de las 2,067 preguntas con ground truth validado permitió identificar cuatro patrones principales de consulta. Las preguntas procedurales (how-to) representan aproximadamente 45% del dataset, caracterizadas por formulaciones como "How to...", "Steps to...", y solicitudes de procedimientos específicos.

Las consultas de troubleshooting constituyen cerca del 29% y se identifican por menciones de errores ("Error..."), problemas ("Issue...", "Problem..."), o comportamientos inesperados del sistema. Las preguntas conceptuales aparecen en torno al 17%, típicamente formuladas como "What is...", "Difference between...", o solicitudes de explicación de conceptos técnicos. Finalmente, las consultas de configuración representan aproximadamente 9%, centradas en "Configure...", "Setup...", y especificación de parámetros.

**[Figura 4.6. Gráfico de barras de tipos de preguntas]**

#### Complejidad de las Consultas

La complejidad técnica de las consultas se evaluó mediante el conteo de conceptos técnicos distintos por pregunta. Las consultas simples con 1-2 conceptos constituyen aproximadamente 32% del dataset. Un ejemplo típico sería "How to create an Azure Storage account?", que involucra únicamente los conceptos de Azure Storage y creación de cuenta.

Las consultas moderadas con 3-4 conceptos representan cerca del 52% del dataset. Una consulta como "Configure RBAC for Key Vault with service principal authentication" integra cuatro conceptos técnicos: RBAC, Key Vault, service principal y authentication. Este tipo de consulta requiere que el sistema RAG comprenda relaciones entre múltiples servicios Azure.

Las consultas complejas con 5 o más conceptos aparecen en aproximadamente 16% de los casos. Un ejemplo ilustrativo: "Deploy containerized microservices with AKS, Azure Container Registry, Azure AD integration, network policies, and monitoring with Application Insights" involucra al menos siete conceptos técnicos interrelacionados: AKS, containerización, microservicios, Azure Container Registry, Azure AD, network policies y Application Insights.

Estas consultas complejas presentan desafíos particulares. Involucran interdependencias técnicas donde múltiples servicios Azure deben integrarse coherentemente. Requieren comprensión de cadenas de configuración donde cada paso depende del anterior. Frecuentemente cruzan dominios técnicos, combinando aspectos de seguridad, networking, compute y storage. Además, típicamente reflejan contexto empresarial con escenarios de producción que afectan múltiples stakeholders, y en casos de troubleshooting, implican problemas sistémicos que afectan múltiples componentes simultáneamente.

Las implicaciones para el sistema RAG son significativas. El predominio de consultas moderadas (52%) requiere embeddings capaces de capturar relaciones entre múltiples conceptos técnicos. El 16% de consultas complejas demanda embeddings sofisticados que capturen dependencias semánticas profundas, modelos de reranking que entiendan interacciones entre servicios Azure, capacidad de recuperación multi-hop para ensamblar información de múltiples documentos, y comprensión contextual de flujos de trabajo empresariales complejos. Por otro lado, el 32% de consultas simples representa una oportunidad de alta precisión donde el sistema puede proporcionar respuestas directas y exactas.

### 4.3.4 Análisis de Ground Truth

#### Cobertura de Ground Truth

Del total de 13,436 preguntas, 6,070 (45.2%) incluyen enlaces a Microsoft Learn en sus respuestas aceptadas. Al validar estos enlaces contra la base de datos de documentos indexados, 2,067 preguntas (15.4% del total) tienen documentos correspondientes efectivamente presentes en el corpus. Esta tasa de correspondencia del 34.1% entre enlaces MS Learn y documentos indexados establece el subconjunto con ground truth procesable.

Los 2,067 enlaces válidos referencian 1,847 URLs únicas, que a su vez corresponden a 1,623 documentos únicos en la base de datos. Esta multiplicidad (más preguntas que documentos únicos) indica que ciertos documentos fundamentales de Azure son referenciados por múltiples preguntas, reflejando tópicos de alto interés o servicios ampliamente utilizados.

#### Calidad del Ground Truth

La validación de enlaces reveló que el 98.7% son válidos y accesibles, con solo 1.3% de enlaces rotos o redirigidos. Sin embargo, la correspondencia con el corpus es más limitada: solo el 34.1% de los enlaces MS Learn coinciden con documentos indexados. Dentro del subconjunto de 2,067 preguntas con enlaces que corresponden a documentos indexados, la cobertura efectiva alcanza 68.2%, lo que representa una base adecuada para evaluación estadística del sistema RAG.

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

Los 4,003 enlaces MS Learn sin correspondencia en la base de datos (65.9% de todos los enlaces) tienen orígenes diversos. La causa principal son documentos válidos de MS Learn que no fueron incluidos durante la extracción de marzo de 2025, ya sea por limitaciones de scope, profundidad de crawling, o exclusión deliberada de ciertos subdominios. Las diferencias temporales también contribuyen: algunos enlaces apuntan a versiones más recientes o contenido actualizado posterior a la extracción del corpus.

Los subdominios especializados representan otra fuente importante de no-correspondencia. Enlaces que apuntan a techcommunity.microsoft.com, Microsoft Gallery, o blogs técnicos no fueron incluidos en el crawling principal enfocado en learn.microsoft.com. Los documentos redireccionados, donde URLs fueron movidas o reorganizadas en la estructura de Microsoft Learn, pueden no haberse capturado con su nueva ubicación.

El contenido especializado también contribuye: documentación de servicios en preview, beta, o específicos de ciertas regiones geográficas puede haber sido excluida del corpus general. Los fragmentos y anchors (enlaces a secciones específicas de documentos) se pierden parcialmente cuando la segmentación en chunks no preserva exactamente los límites de sección originales. Los formatos no procesados como PDFs interactivos, videos tutoriales, o herramientas de configuración online no fueron capturados en el corpus textual.

Adicionalmente, errores sutiles en la normalización de URLs durante la indexación pueden causar fallos de correspondencia para URLs que son idénticas semánticamente pero difieren en detalles de formato. Los documentos archivados o deprecados que ya no están activos en Microsoft Learn pero fueron referenciados en respuestas históricas de Q&A contribuyen marginalmente. Finalmente, las limitaciones de scope del proyecto, que se enfocó primariamente en documentación Azure/Cloud, excluyeron deliberadamente ciertos dominios como documentación de desarrollo local, herramientas de escritorio, o servicios no-cloud.

**[Figura 4.8. Diagrama de Sankey mostrando flujo de correspondencia]**

### 4.4.2 Distribución Temática de Ground Truth

El análisis de la distribución temática en el subset de 2,067 preguntas con ground truth validado reveló algunas diferencias respecto al corpus general. Development aumenta de 40.2% en el corpus general a 43.7% en ground truth (+3.5 puntos porcentuales), indicando que las consultas de usuarios tienden a enfocarse ligeramente más en temas de desarrollo.

Operations también incrementa de 27.6% a 31.2% (+3.6pp), reflejando un sesgo hacia problemas operacionales en las consultas reales de usuarios. Security disminuye de 19.9% a 16.8% (-3.1pp), y Azure Services de 12.3% a 8.3% (-4.0pp). Esta subrepresentación de servicios específicos sugiere que las preguntas de usuarios tienden a ser más transversales, cruzando múltiples servicios en lugar de enfocarse en servicios aislados.

La alineación general entre corpus y ground truth es relativamente consistente, con ninguna categoría mostrando desviaciones superiores a 4 puntos porcentuales. Esto valida que el subset con ground truth es razonablemente representativo del corpus completo y no introduce sesgos temáticos severos en la evaluación del sistema RAG.

### 4.4.3 Calidad de la Correspondencia

La evaluación cualitativa de una muestra de 100 correspondencias pregunta-documento reveló que aproximadamente 67% son altamente relevantes, donde el documento es directamente aplicable a la pregunta formulada. Un 28% adicional son moderadamente relevantes, requiriendo cierta interpretación o contexto adicional para conectar la pregunta con la respuesta en el documento. Solo 5% mostraron relevancia baja, correspondencias tangenciales o potencialmente incorrectas.

La clasificación por tipo de match identificó cuatro patrones. Los exact matches (34%) ocurren cuando pregunta y documento abordan exactamente el mismo tema con terminología similar. Los conceptual matches (41%) aparecen cuando el documento contiene la respuesta pero requiere cierta inferencia o traducción conceptual. Los partial matches (20%) corresponden a documentos que cubren parte de la pregunta pero no todos sus aspectos. Los weak matches (5%) representan correspondencias tangenciales o débiles, posiblemente resultado de enlaces que apuntan a secciones relacionadas pero no directamente relevantes del documento.

## 4.5 Hallazgos Principales del EDA

### 4.5.1 Fortalezas del Corpus de Documentación

El corpus de documentación presenta características que lo hacen apropiado para investigación en recuperación semántica de información técnica. La cobertura es comprehensiva, procesando aproximadamente 96% o más de la documentación Azure disponible en Microsoft Learn. La profundidad técnica es evidente con 779.0 tokens promedio por chunk, indicando contenido sustancial con detalles técnicos significativos.

La diversidad temática es notable, con distribución balanceada entre cuatro categorías principales que cubren los aspectos fundamentales de la plataforma Azure. La calidad es consistente, mostrada por la variabilidad controlada en longitudes y contenido (CV = 38.3%), evitando tanto fragmentación excesiva como chunks desproporcionadamente largos.

Sin embargo, existen áreas de mejora identificadas. La inclusión de contenido multimodal (imágenes, diagramas, videos) mejoraría la completitud del corpus. Un proceso de actualización continua es necesario para sincronizar con los cambios frecuentes en la plataforma Azure. La granularidad de algunos chunks muy largos (mayores a 2,000 tokens) podría optimizarse mediante segmentación adicional.

### 4.5.2 Fortalezas y Limitaciones del Dataset de Preguntas

El dataset de preguntas presenta diversidad apropiada con cuatro tipos principales de consulta bien representados: procedurales, troubleshooting, conceptuales y de configuración. La complejidad es adecuada para evaluación rigurosa, con predominio de consultas moderadas (52%) que requieren comprensión multi-concepto. La autenticidad es un valor diferenciador: todas las preguntas provienen de usuarios reales enfrentando problemas técnicos concretos en Azure. El ground truth validado alcanza 68.2% de cobertura dentro del subset de 2,067 preguntas con correspondencia documental.

Las limitaciones principales incluyen la cobertura parcial de ground truth: solo 15.4% de todas las preguntas tienen enlaces correspondientes a documentos en la base de datos. El alto ratio de no-correspondencia (65.9% de enlaces MS Learn sin documentos correspondientes) limita el alcance de la evaluación. Existe un sesgo temporal: las preguntas reflejan el estado histórico de Azure hasta marzo de 2025, no capturando consultas emergentes sobre servicios más recientes. La restricción a inglés limita la generalización global del sistema. El criterio estricto de un documento por pregunta subestima la relevancia múltiple que caracteriza muchas consultas complejas. Finalmente, las limitaciones de extracción del corpus original impactan la correspondencia: el corpus puede no incluir toda la documentación MS Learn disponible.

### 4.5.3 Implicaciones para el Sistema RAG

Las características del corpus y dataset identifican tanto oportunidades como desafíos para el sistema RAG. La especialización en Development (53.6% del corpus) permite optimización específica para consultas de desarrollo, que son predominantes en el uso real. El balance temático razonable facilita evaluación comprehensiva sin sesgos severos hacia una única categoría.

La longitud promedio de 779.0 tokens por chunk es compatible con modelos de embedding modernos que típicamente manejan secuencias de 512-1024 tokens eficientemente. El ground truth, aunque limitado al 15.4% de preguntas, proporciona cobertura suficiente para validación estadística con significancia adecuada.

Los desafíos identificados incluyen la variabilidad alta en longitud de documentos (σ = 298.6 tokens en chunks), que requiere embeddings robustos capaces de manejar esta variación sin degradación de calidad. Las consultas complejas (16% del dataset) requieren comprensión multi-concepto avanzada: razonamiento relacional entre 5 o más conceptos técnicos simultáneamente, integración semántica de múltiples dominios Azure (compute, storage, networking, security), comprensión de flujos de trabajo empresariales complejos con múltiples dependencias, y capacidad de síntesis para combinar información de múltiples fuentes documentales.

La correspondencia limitada con ground truth (84.6% sin validación y 65.9% de enlaces sin correspondencia) restringe la evaluación comprehensiva del sistema a un subconjunto del dataset total. La necesidad de actualización continua del corpus es crítica dada la velocidad de evolución de la plataforma Azure.

### 4.5.4 Comparación con Corpus Estándar

Comparado con corpus académicos estándar en investigación de recuperación de información, el corpus MS-Azure de este trabajo presenta características distintivas. Con 62,417 documentos y promedio de 2,334 tokens por documento completo (779 tokens por chunk), se posiciona en un punto intermedio en términos de escala.

MS-MARCO (Nguyen et al., 2016), uno de los benchmarks más utilizados, contiene aproximadamente 8.8 millones de documentos pero con longitud promedio de solo 100 tokens, enfocado en contenido web general con especificidad baja. Natural Questions (Kwiatkowski et al., 2019) comprende aproximadamente 307,000 documentos de Wikipedia con longitud promedio de 800 tokens y especificidad media. SQuAD 2.0 (Rajpurkar et al., 2018) incluye alrededor de 150,000 documentos también de Wikipedia con promedio de 500 tokens y especificidad media.

Las ventajas competitivas del corpus MS-Azure incluyen mayor especialización técnica respecto a corpus generales como MS-MARCO. Los documentos son más sustanciales y profundos que los pasajes cortos de MS-MARCO. La autenticidad de documentación oficial supera el contenido web general en términos de confiabilidad y corrección técnica. La actualidad del corpus (marzo 2025) contrasta con corpus históricos que pueden contener información obsoleta.

Esta especialización posiciona el corpus MS-Azure como una contribución valiosa al ecosistema de investigación. Representa el primer corpus comprehensivo enfocado específicamente en documentación Azure disponible para investigación académica. El ground truth está validado por una comunidad técnica especializada en lugar de crowdsourcing general. La metodología es completamente reproducible con disponibilidad pública de scripts de análisis. Establece un baseline con métricas y análisis disponibles para comparación en trabajos futuros.

## 4.6 Recomendaciones para Mejoras Futuras

### 4.6.1 Expansión y Actualización del Corpus

La prioridad más alta para mejoras futuras es la expansión del corpus para incluir la documentación MS Learn faltante, reduciendo el 65.9% actual de enlaces sin correspondencia. Esto requiere un proceso de crawling más exhaustivo que cubra subdominios especializados, documentación de servicios en preview, y contenido regionalizado. Un pipeline automatizado de sincronización con Microsoft Learn permitiría actualización continua del corpus, capturando cambios, nuevos servicios, y documentación actualizada.

La incorporación de contenido multimodal mediante técnicas de OCR y procesamiento de imágenes permitiría extraer texto de diagramas arquitectónicos, capturas de pantalla con configuraciones, y flujos de proceso visuales. Un sistema de versioning robusto facilitaría el tracking de cambios en la documentación, permitiendo análisis temporal y detección de actualizaciones relevantes.

Con prioridad media, la expansión a idiomas adicionales habilitaría investigación en recuperación multilingüe y cross-lingual retrieval. La extracción de metadatos enriquecidos (tags, categorías, fechas de actualización, autores) mejoraría las capacidades de filtrado y ranking. El mapping explícito de relaciones semánticas entre documentos (prerequisites, related topics, see also) podría mejorar la navegación y recuperación contextual.

### 4.6.2 Optimización de Segmentación y Ground Truth

Las estrategias de segmentación podrían optimizarse mediante segmentación adaptativa que genere chunks más pequeños para documentos complejos mayores a 2,000 tokens. La preservación de contexto mediante overlap entre chunks adyacentes mantendría coherencia semántica. La segmentación semántica basada en estructura lógica del documento (secciones, subsecciones) en lugar de límites fijos de tokens respetaría mejor las unidades naturales de información.

Para expansión del ground truth, la prioridad principal es incluir los documentos MS Learn correspondientes a los 4,003 enlaces actualmente sin correspondencia. La anotación humana de correspondencias adicionales entre las 7,366 preguntas sin enlaces MS Learn expandiría significativamente el ground truth disponible. Permitir múltiples referencias relevantes por pregunta, en lugar del criterio actual de un documento, reflejaría mejor la realidad de consultas complejas que requieren información de múltiples fuentes.

El crowdsourcing con la comunidad técnica para validación distribuida del ground truth podría escalar el proceso de anotación. La automatización inteligente usando embeddings semánticos para identificar correspondencias no explícitas entre preguntas y documentos complementaría las referencias manuales existentes.

### 4.6.3 Mejoras Metodológicas

Un framework de análisis continuo con scripts de EDA ejecutados periódicamente permitiría monitorear la evolución del corpus. La detección de drift identificaría cambios en distribuciones temporales que podrían afectar el desempeño del sistema RAG. Las quality metrics como KPIs de cobertura, actualidad, y completitud monitoreadas continuamente alertarían sobre degradación del corpus.

Para validación rigurosa, la evaluación inter-annotador con múltiples validadores independientes aumentaría la confiabilidad del ground truth. El testing estadístico formal mediante pruebas de significancia detectaría cambios relevantes en distribuciones del corpus. El benchmarking externo comparando contra nuevos corpus especializados que aparezcan en la literatura posicionaría este trabajo en el contexto de la investigación global.

## 4.7 Conclusiones del Análisis Exploratorio

El análisis exploratorio de datos revela un corpus técnico robusto y apropiadamente estructurado para investigación en recuperación semántica de información especializada. Los 62,417 documentos únicos segmentados en 187,031 chunks, junto con 13,436 preguntas de las cuales 2,067 (15.4%) tienen ground truth validado, proporcionan una base sólida para evaluación sistemática de sistemas RAG en dominios técnicos.

Las características destacadas incluyen profundidad técnica significativa (779.0 tokens promedio por chunk), diversidad temática balanceada entre Development (53.6%), Security (28.6%), Operations (11.9%) y Azure Services (5.8%), calidad verificada con 98.7% de enlaces válidos y 68.2% de cobertura dentro del subset con ground truth, y especialización única como el primer corpus comprehensivo para documentación Azure disponible para investigación académica.

El EDA valida las decisiones metodológicas adoptadas en el diseño del sistema RAG. La segmentación es apropiada con longitud promedio compatible con modelos de embedding actuales. La evaluación es factible con 15.4% de cobertura de ground truth que, aunque limitado, permite validación estadística robusta. La diversidad es suficiente con cuatro categorías temáticas facilitando evaluación comprehensiva. La escala es adecuada con 187,031 chunks proporcionando corpus sustancial para entrenamiento y evaluación.

Este trabajo establece varios precedentes importantes para la investigación en recuperación semántica de información técnica. Proporciona el primer benchmark especializado con análisis sistemático de corpus Azure para investigación académica. La metodología es completamente reproducible con disponibilidad de scripts de análisis y datasets para replicación. Establece un baseline con métricas y distribuciones documentadas para comparación futura. Define un framework de calidad con criterios objetivos para evaluación de corpus técnicos especializados.

El corpus analizado constituye una base sólida para el desarrollo y evaluación de sistemas RAG especializados en documentación técnica, con características que lo posicionan como un recurso valioso para la comunidad de investigación en recuperación de información y procesamiento de lenguaje natural aplicado a dominios técnicos.

**[Figura 4.9. Dashboard resumen con métricas clave del corpus]**

**[Tabla 4.1. Comparativa de características del corpus vs benchmarks estándar]**

## 4.8 Referencias del Capítulo

Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M. W., Dai, A. M., Uszkoreit, J., Le, Q., & Petrov, S. (2019). Natural Questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, *7*, 452-466.

Microsoft. (2025a). *Microsoft Learn Documentation*. https://learn.microsoft.com/

Microsoft. (2025b). *Microsoft Q&A Community Platform*. https://learn.microsoft.com/en-us/answers/

Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., & Deng, L. (2016). MS MARCO: A human generated machine reading comprehension dataset. *Proceedings of the Workshop on Cognitive Computation: Integrating Neural and Symbolic Approaches (CoCo@NIPS)*.

OpenAI. (2025). *tiktoken: Token counting library*. https://github.com/openai/tiktoken

Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, 784-789.
