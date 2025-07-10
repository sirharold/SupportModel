# BATERÍA DE PREGUNTAS PARA DEFENSA DE TÍTULO
## Sistema RAG Azure con Métricas Avanzadas

---

## 📊 **ARQUITECTURA Y DECISIONES TÉCNICAS**

### **1. ¿Por qué usar Weaviate como base de datos vectorial?**
**Respuesta:** Weaviate ofrece búsqueda vectorial nativa, soporte para metadatos híbridos, escalabilidad cloud y integración directa con modelos de ML. Alternativas como FAISS son solo índices, no bases de datos completas.
**Código:** `utils/weaviate_utils_improved.py` implementa WeaviateClientWrapper, configuración de esquemas y operaciones CRUD optimizadas.

### **2. ¿Qué es una base de datos vectorial?**
**Respuesta:** Sistema especializado para almacenar y consultar vectores de alta dimensión usando algoritmos como HNSW. Optimizado para búsqueda por similitud semántica vs. consultas exactas SQL.
**Código:** `config.py:22-36` define WEAVIATE_CLASS_CONFIG con esquemas para diferentes modelos, operaciones en `weaviate_utils_improved.py`.

### **3. ¿Por qué no usar una base de datos relacional?**
**Respuesta:** Las BD relacionales no pueden manejar eficientemente búsquedas de similitud vectorial en espacios de alta dimensión. Requieren índices especializados (HNSW, LSH) que no están disponibles en SQL estándar.

### **4. ¿Por qué seleccionaste esos modelos de embedding específicos?**
**Respuesta:** mpnet: especializado Q&A, MiniLM: balance eficiencia/calidad, ada-002: referencia comercial. Cubren diferentes trade-offs de dimensionalidad, costo y rendimiento.
**Código:** `config.py:4-8` define EMBEDDING_MODELS, `config.py:39-55` contiene MODEL_DESCRIPTIONS con justificación técnica.

### **5. ¿Cómo garantizas la reproducibilidad de los experimentos?**
**Respuesta:** Versiones fijas de modelos, seeds deterministas, dataset controlado, pipeline idéntico para todos los modelos, métricas estandarizadas y documentación completa.

### **6. ¿Por qué usar modelos locales en lugar de solo APIs?**
**Respuesta:** Reducción de costos (85%), control total del proceso, privacidad de datos, independencia de conectividad y capacidad de experimentación ilimitada.
**Código:** `config.py:20` DEFAULT_GENERATIVE_MODEL usa llama local, `utils/local_models.py` implementa gestión completa, `config.py:57-71` LOCAL_MODEL_DESCRIPTIONS.

### **7. ¿Qué ventajas tiene tu pipeline de 6 etapas vs. RAG tradicional?**
**Respuesta:** Refinamiento inteligente de consulta, evaluación multi-modelo, reranking contextual, sistema de fallback, métricas especializadas y optimización de costos.
**Código:** `utils/qa_pipeline.py` implementa pipeline completo, método `process_query_comprehensive()` ejecuta las 6 etapas secuencialmente.

### **8. ¿Cómo validas que el reranking mejora los resultados?**
**Respuesta:** Métricas MRR, nDCG@k, evaluación human-in-the-loop, comparación con ranking por similitud coseno pura y análisis de relevancia contextual.
**Código:** `utils/reranker.py` implementa reranking, `utils/metrics.py` calcula MRR/nDCG, comparación en `comparison_page.py`.

### **9. ¿Por qué usar CrossEncoder para reranking?**
**Respuesta:** Los CrossEncoders procesan pares (query, document) conjuntamente, capturando interacciones contextuales que los bi-encoders no pueden. Mejoran relevancia significativamente.
**Código:** `utils/reranker.py:rerank_documents()` usa ms-marco-MiniLM-L-6-v2 CrossEncoder, inicialización en líneas 10-15.

### **10. ¿Cómo manejas la latencia del sistema?**
**Respuesta:** Modelos locales optimizados, caché de embeddings, procesamiento por lotes, cuantización 4-bit, GPU cuando disponible y sistema de fallback rápido.
**Código:** `utils/local_models.py:30-37` configura cuantización 4-bit, `utils/embedding_safe.py` implementa caché, detección GPU en `local_models.py:25`.

---

## 🤖 **MODELOS Y PROCESAMIENTO**

### **11. ¿Qué criterios usaste para seleccionar Llama 3.1 8B?**
**Respuesta:** Balance calidad/eficiencia, especialización en tareas técnicas, disponibilidad open-source, compatibilidad con hardware estándar y soporte para cuantización.
**Código:** `config.py:16,20` define llama como modelo principal, `config.py:59-64` justificación en LOCAL_MODEL_DESCRIPTIONS.

### **12. ¿Cómo comparas el rendimiento de modelos locales vs. remotos?**
**Respuesta:** Métricas de calidad (BERTScore, ROUGE), latencia, costo por consulta, disponibilidad, privacidad y control del proceso.

### **13. ¿Qué es la cuantización 4-bit y por qué la usas?**
**Respuesta:** Técnica de compresión que reduce precisión numérica manteniendo calidad. Permite ejecutar modelos grandes en hardware limitado con 75% menos memoria.
**Código:** `utils/local_models.py:30-37` método `get_quantization_config()`, usa BitsAndBytesConfig con parámetros nf4 y double_quant.

### **14. ¿Cómo garantizas la calidad de respuestas con modelos locales?**
**Respuesta:** Evaluación con métricas especializadas, comparación con modelos comerciales, validación humana, sistema de fallback y calibración de prompts.

### **15. ¿Por qué no usar solo el modelo más grande disponible?**
**Respuesta:** Trade-off entre calidad, costo, latencia y recursos. Modelos más grandes no siempre mejoran proporcionalmente en tareas específicas.

### **16. ¿Cómo manejas el contexto limitado de los modelos?**
**Respuesta:** Truncamiento inteligente, priorización de contenido relevante, segmentación de documentos largos y refinamiento de consultas.

### **17. ¿Qué estrategia usas para el prompt engineering?**
**Respuesta:** Instrucciones explícitas, formato estructurado, ejemplos few-shot, contexto técnico específico y validación iterativa de prompts.
**Código:** Templates en `utils/local_answer_generator.py:generate_final_answer_local()`, `utils/answer_generator.py` define prompts estructurados.

### **18. ¿Cómo evalúas la efectividad del refinamiento de consulta?**
**Respuesta:** Mejora en Recall@k, reducción de ambigüedad, análisis de expansión terminológica y evaluación de relevancia contextual.
**Código:** `utils/qa_pipeline.py:12-33` función `refine_and_prepare_query()`, `utils/local_answer_generator.py:refine_query_local()` implementa refinamiento.

### **19. ¿Por qué usar temperatura 0.7 en generación?**
**Respuesta:** Balance entre determinismo y creatividad. Valores muy bajos generan respuestas repetitivas, muy altos introducen incoherencia.
**Código:** Configurado en `utils/local_answer_generator.py` parámetro `temperature=0.7` en métodos de generación, ajustable por modelo.

### **20. ¿Cómo manejas consultas fuera del dominio Azure?**
**Respuesta:** Detección de dominio, respuestas de rechazo elegante, sugerencias de reformulación y logging para análisis posterior.
**Código:** Lógica de detección en `utils/qa_pipeline.py`, filtros en `utils/weaviate_utils_improved.py`, manejo en interfaz `EnhancedStreamlit_qa_app.py`.

---

## 📈 **MÉTRICAS Y EVALUACIÓN**

### **21. ¿Por qué desarrollar métricas RAG especializadas?**
**Respuesta:** Las métricas tradicionales no capturan aspectos únicos de RAG: uso de contexto, alucinaciones, completitud técnica. Necesidad de evaluación específica.
**Código:** `utils/advanced_rag_metrics.py` implementa 4 métricas especializadas, `utils/metrics.py` combina tradicionales y especializadas.

### **22. ¿Qué es la detección de alucinaciones en tu contexto?**
**Respuesta:** Identificación automática de información en respuestas que no está soportada por documentos recuperados. Crítico para contextos técnicos.
**Código:** `utils/advanced_rag_metrics.py:calculate_hallucination_score()` implementa detección usando similitud semántica entre respuesta y contexto.

### **23. ¿Cómo validas tus métricas RAG especializadas?**
**Respuesta:** Correlación con evaluación humana, comparación con métricas tradicionales, validación estadística y calibración en dataset específico.

### **24. ¿Qué significa "utilización de contexto" en tu framework?**
**Respuesta:** Métrica que mide qué tan efectivamente el modelo aprovecha los documentos recuperados para generar respuestas relevantes y completas.
**Código:** `utils/advanced_rag_metrics.py:calculate_context_utilization()` analiza overlap semántico entre documentos recuperados y respuesta generada.

### **25. ¿Por qué usar BERTScore en lugar de BLEU?**
**Respuesta:** BERTScore captura similitud semántica usando embeddings contextuales, mientras BLEU solo mide superposición léxica exacta.
**Código:** `utils/metrics.py:calculate_bert_score()` importa bert_score, configuración en líneas de inicialización del evaluador.

### **26. ¿Cómo defines "completitud de respuesta"?**
**Respuesta:** Evaluación de si la respuesta aborda todos los aspectos relevantes de la consulta según el tipo de pregunta (procedural, factual, comparativa).

### **27. ¿Qué baseline usas para comparar tu sistema?**
**Respuesta:** GPT-4 vanilla, sistemas RAG tradicionales, búsqueda keyword, y embedding models individuales sin pipeline.

### **28. ¿Cómo aseguras la significancia estadística de tus resultados?**
**Respuesta:** Tests t-student, tamaño de muestra adecuado, intervalos de confianza, corrección por múltiples comparaciones y validación cruzada.

### **29. ¿Por qué usar MRR como métrica de recuperación?**
**Respuesta:** MRR (Mean Reciprocal Rank) penaliza cuando el documento relevante no está en top positions, crítico para calidad de respuestas RAG.

### **30. ¿Cómo calibras los umbrales de tus métricas?**
**Respuesta:** Análisis ROC, validación con evaluadores humanos, optimización en dataset desarrollo y ajuste iterativo con feedback.

---

## 🔍 **DATOS Y PROCESAMIENTO**

### **31. ¿Cómo garantizas la calidad de los datos scrapeados?**
**Respuesta:** Validación de fuentes oficiales, deduplicación, filtros de calidad, verificación de enlaces y normalización de contenido.

### **32. ¿Por qué usar múltiples fuentes de datos?**
**Respuesta:** Diversidad de perspectivas, cobertura completa, validación cruzada, diferentes tipos de contenido (oficial vs. comunidad).

### **33. ¿Cómo manejas la actualización de datos?**
**Respuesta:** Pipeline automatizado de scraping, detección de cambios, re-embedding incremental y versionado de corpus.

### **34. ¿Qué estrategia usas para el chunking de documentos?**
**Respuesta:** Segmentación semántica por párrafos, límite de 512 tokens, preservación de contexto y overlap mínimo entre chunks.

### **35. ¿Cómo manejas documentos en múltiples idiomas?**
**Respuesta:** Detección automática de idioma, modelos multilingües, traducción cuando necesario y filtrado por idioma relevante.

### **36. ¿Por qué no usar datos sintéticos para aumentar el dataset?**
**Respuesta:** Riesgo de sesgos, calidad inferior, falta de variabilidad real, problemas de validación y preferencia por datos reales.

### **37. ¿Cómo validas la representatividad de tu dataset?**
**Respuesta:** Análisis de distribución de temas, cobertura de servicios Azure, variedad de tipos de consulta y validación con expertos.

### **38. ¿Qué preprocessing aplicas a los textos?**
**Respuesta:** Limpieza HTML, normalización de encoding, eliminación de ruido, preservación de formato técnico y tokenización consistente.

### **39. ¿Cómo manejas el desbalance en tipos de consulta?**
**Respuesta:** Análisis de distribución, estratificación en evaluación, balanceo por importancia y métricas ponderadas.

### **40. ¿Por qué elegiste ese tamaño específico de embedding?**
**Respuesta:** Trade-off entre capacidad representacional y eficiencia computacional. Evaluación empírica mostró punto óptimo.

---

## 🌐 **INTERFAZ Y EXPERIENCIA**

### **41. ¿Por qué usar Streamlit para la interfaz?**
**Respuesta:** Rapidez de desarrollo, especialización en ML/AI, componentes interactivos nativos, fácil deployment y ecosistema Python.
**Código:** `EnhancedStreamlit_qa_app.py` aplicación principal, `comparison_page.py` y `batch_queries_page.py` implementan páginas especializadas.

### **42. ¿Cómo diseñaste la experiencia de usuario?**
**Respuesta:** Investigación de necesidades, flujo intuitivo, feedback inmediato, visualizaciones claras y documentación integrada.

### **43. ¿Qué consideraciones de usabilidad implementaste?**
**Respuesta:** Tiempos de respuesta visibles, mensajes de error claros, configuración avanzada opcional, export de resultados y ayuda contextual.

### **44. ¿Cómo manejas la escalabilidad de la interfaz?**
**Respuesta:** Caching inteligente, procesamiento asíncrono, paginación de resultados, optimización de queries y deployment cloud.

### **45. ¿Por qué incluir la funcionalidad de comparación?**
**Respuesta:** Necesidad de selección objetiva de modelos, transparencia en decisiones, validación experimental y valor académico.
**Código:** `comparison_page.py` implementa evaluación multi-modelo, `utils/comparison_with_advanced_metrics.py` ejecuta comparación paralela.

### **46. ¿Cómo generas los reportes PDF automáticamente?**
**Respuesta:** Templates HTML/CSS, WeasyPrint para rendering, datos estructurados, gráficos embebidos y formato profesional.
**Código:** `utils/pdf_generator.py` implementa generación completa, templates HTML integrados, configuración WeasyPrint optimizada.

### **47. ¿Qué mecanismos de feedback implementaste?**
**Respuesta:** Ratings de respuestas, reportes de errores, logging de interacciones, métricas de satisfacción y análisis de uso.

### **48. ¿Cómo garantizas la accesibilidad de la interfaz?**
**Respuesta:** Contraste adecuado, navegación por teclado, textos alternativos, estructura semántica y compatibilidad con screen readers.

### **49. ¿Por qué incluir visualizaciones interactivas?**
**Respuesta:** Comprensión intuitiva de métricas, análisis exploratorio, presentación profesional y engagement del usuario.

### **50. ¿Cómo manejas la seguridad de la aplicación?**
**Respuesta:** Sanitización de inputs, validación de archivos, límites de rate, logging de seguridad y deployment seguro.

---

## 💰 **COSTOS Y OPTIMIZACIÓN**

### **51. ¿Cómo calculaste la reducción del 85% en costos?**
**Respuesta:** Análisis de pricing APIs comerciales vs. costo computacional local, proyección de uso escalado y amortización de hardware.

### **52. ¿Qué factores no consideraste en el análisis de costos?**
**Respuesta:** Costo de desarrollo, mantenimiento, electricidad, depreciación hardware, tiempo de setup y training del equipo.

### **53. ¿Cómo optimizaste el uso de memoria?**
**Respuesta:** Cuantización de modelos, garbage collection proactivo, lazy loading, batch processing y liberación de recursos.

### **54. ¿Qué estrategia usas para balancear calidad vs. costo?**
**Respuesta:** Modelo local como primario, APIs para casos complejos, fallback inteligente y optimización por tipo de consulta.

### **55. ¿Cómo proyectas los costos a escala empresarial?**
**Respuesta:** Análisis de throughput, costo por consulta, infraestructura requerida, mantenimiento y total cost of ownership.

### **56. ¿Qué optimizaciones implementaste para GPU?**
**Respuesta:** Detección automática, batch processing, memory pooling, cuantización y fallback a CPU cuando necesario.

### **57. ¿Cómo mides el ROI del sistema?**
**Respuesta:** Reducción tiempo resolución, automatización de tareas, mejora satisfacción usuario, reducción costos soporte.

### **58. ¿Qué consideraciones de escalabilidad incluiste?**
**Respuesta:** Arquitectura modular, caching distribuido, load balancing, procesamiento asíncrono y deployment cloud.

### **59. ¿Cómo optimizaste la latencia del sistema?**
**Respuesta:** Modelos locales, caché de embeddings, procesamiento paralelo, índices optimizados y pre-computación.

### **60. ¿Qué trade-offs identificaste entre eficiencia y calidad?**
**Respuesta:** Tamaño de modelo, número de documentos recuperados, profundidad de reranking, temperatura de generación.

---

## 🔬 **METODOLOGÍA Y VALIDACIÓN**

### **61. ¿Cómo validaste la efectividad del sistema completo?**
**Respuesta:** Evaluación con ground truth, comparación con baselines, métricas múltiples, validación humana y análisis estadístico.

### **62. ¿Qué limitaciones identificaste en tu enfoque?**
**Respuesta:** Dependencia de calidad de datos, sesgo hacia Azure, evaluación offline, recursos computacionales y generalización limitada.

### **63. ¿Cómo garantizas la reproducibilidad de tus experimentos?**
**Respuesta:** Código versionado, datasets fijos, configuración documentada, seeds deterministas y ambiente controlado.

### **64. ¿Qué aspectos éticos consideraste?**
**Respuesta:** Privacidad de datos, sesgos en modelos, transparencia de decisiones, uso responsable de IA y impacto social.

### **65. ¿Cómo manejas el sesgo en tus modelos?**
**Respuesta:** Diversidad en datos, evaluación de fairness, mitigación de sesgos, transparencia en limitaciones y monitoreo continuo.

### **66. ¿Qué protocolo seguiste para la evaluación humana?**
**Respuesta:** Evaluadores independientes, criterios claros, inter-annotator agreement, muestras representativas y análisis estadístico.

### **67. ¿Cómo validaste la generalización a otros dominios?**
**Respuesta:** Evaluación en subdominios Azure, transferencia a contextos similares, análisis de robustez y limitaciones identificadas.

### **68. ¿Qué consideraciones de privacidad implementaste?**
**Respuesta:** Procesamiento local, anonimización de datos, consentimiento informado, minimal data collection y cumplimiento normativo.

### **69. ¿Cómo documentaste tu metodología?**
**Respuesta:** Código comentado, README detallado, documentación API, notebooks explicativos y paper académico.

### **70. ¿Qué impacto esperas que tenga tu investigación?**
**Respuesta:** Avance en métricas RAG, democratización de sistemas inteligentes, reducción costos empresariales y metodología replicable.

---

## 🚀 **TRABAJO FUTURO Y EXTENSIONES**

### **71. ¿Cómo extenderías el sistema a otros dominios?**
**Respuesta:** Adaptación de corpus, fine-tuning de modelos, calibración de métricas, validación específica y pipeline genérico.

### **72. ¿Qué mejoras implementarías con más tiempo?**
**Respuesta:** Modelos más grandes, fine-tuning específico, evaluación en tiempo real, integración empresarial y optimización avanzada.

### **73. ¿Cómo abordarías la actualización continua del conocimiento?**
**Respuesta:** Pipeline automatizado, detección de cambios, re-embedding incremental, versionado de modelos y validación continua.

### **74. ¿Qué otras métricas RAG explorarías?**
**Respuesta:** Coherencia temporal, consistencia lógica, profundidad técnica, adaptabilidad contextual y eficiencia informacional.

### **75. ¿Cómo mejorarías la evaluación del sistema?**
**Respuesta:** Evaluación en tiempo real, feedback continuo, A/B testing, métricas de negocio y monitoreo de deriva.

### **76. ¿Qué aspectos de investigación quedaron pendientes?**
**Respuesta:** Fine-tuning específico, evaluación multimodal, optimización avanzada, estudios longitudinales y validación externa.

### **77. ¿Cómo contribuye tu trabajo al estado del arte?**
**Respuesta:** Métricas RAG especializadas, metodología de comparación, arquitectura híbrida, evaluación específica y sistema completo.

### **78. ¿Qué publicaciones académicas resultarían de este trabajo?**
**Respuesta:** Paper sobre métricas RAG, metodología de evaluación, sistemas híbridos, aplicaciones técnicas y reproducibilidad.

### **79. ¿Cómo transferirías este sistema a producción?**
**Respuesta:** Infraestructura escalable, monitoreo robusto, CI/CD, documentación completa, training del equipo y soporte técnico.

### **80. ¿Qué colaboraciones futuras visualizas?**
**Respuesta:** Empresas tecnológicas, grupos de investigación, comunidad open-source, instituciones académicas y organizaciones estándar.

---

## 🎯 **PREGUNTAS DESAFIANTES**

### **81. ¿Por qué tu sistema es mejor que simplemente usar GPT-4?**
**Respuesta:** Especialización técnica, control total, reducción costos, métricas específicas, transparencia, privacidad y capacidad de mejora continua.

### **82. ¿Cómo justificas la complejidad del sistema de 6 etapas?**
**Respuesta:** Cada etapa aporta valor específico, modularity permite optimización independiente, flexibilidad para diferentes casos de uso y mejora incremental.

### **83. ¿No es esto simplemente una implementación de RAG existente?**
**Respuesta:** Contribuciones: métricas especializadas, evaluación comparativa, arquitectura híbrida, optimización específica y validación rigurosa.

### **84. ¿Cómo aseguras que tus métricas son realmente mejores?**
**Respuesta:** Validación con expertos, correlación con evaluación humana, especificidad para contexto técnico, reproducibilidad y comparación sistemática.

### **85. ¿Por qué no usar técnicas más avanzadas como fine-tuning?**
**Respuesta:** Limitaciones de recursos, generalización, mantenimiento, costo-beneficio y enfoque en metodología de evaluación.

### **86. ¿Cómo demuestras que tu sistema funciona mejor en la práctica?**
**Respuesta:** Evaluación con usuarios reales, métricas de satisfacción, casos de uso específicos, comparación con alternativas y feedback cualitativo.

### **87. ¿No hay riesgo de sobre-ingeniería en tu solución?**
**Respuesta:** Cada componente justificado empíricamente, arquitectura modular, posibilidad de simplificación, balance complejidad-beneficio evaluado.

### **88. ¿Cómo garantizas que tu evaluación no está sesgada?**
**Respuesta:** Evaluadores independientes, métricas múltiples, validación cruzada, datasets diversos y transparencia metodológica.

### **89. ¿Por qué limitarte a Azure y no ser más general?**
**Respuesta:** Especialización permite evaluación rigurosa, casos de uso específicos, validación controlada, expertise del dominio y metodología transferible.

### **90. ¿Cómo justificas el esfuerzo vs. usar soluciones comerciales?**
**Respuesta:** Contribución académica, control total, reducción costos, especialización técnica, transparencia y capacidad de investigación.

---

## 🏆 **PREGUNTAS DE CIERRE**

### **91. ¿Cuál es tu principal contribución al campo?**
**Respuesta:** Framework de evaluación RAG especializado, metodología de comparación rigurosa, arquitectura híbrida optimizada y sistema completo reproducible.

### **92. ¿Qué aprendiste durante este proyecto?**
**Respuesta:** Complejidad de evaluación RAG, importancia de métricas especializadas, trade-offs arquitectónicos, metodología experimental rigurosa.

### **93. ¿Cómo ha evolucionado tu entendimiento del problema?**
**Respuesta:** De simple recuperación a evaluación compleja, importancia de contexto técnico, necesidad de métricas especializadas, valor de comparación sistemática.

### **94. ¿Qué recomendaciones darías a futuros investigadores?**
**Respuesta:** Enfoque en evaluación rigurosa, métricas específicas al dominio, validación múltiple, documentación completa y reproducibilidad.

### **95. ¿Cómo medirías el éxito de tu investigación?**
**Respuesta:** Adopción de métricas, reproducibilidad de resultados, impacto en investigación posterior, aplicaciones prácticas y contribución al conocimiento.

### **96. ¿Qué aspectos fueron más desafiantes?**
**Respuesta:** Desarrollo de métricas válidas, evaluación comparativa objetiva, balance complejidad-utilidad, validación rigurosa y documentación completa.

### **97. ¿Cómo connects tu trabajo con el estado del arte actual?**
**Respuesta:** Extiende evaluación RAG existente, aporta metodología específica, valida técnicas conocidas, identifica limitaciones y propone soluciones.

### **98. ¿Qué impacto práctico visualizas?**
**Respuesta:** Mejora en sistemas de soporte, reducción costos empresariales, democratización de tecnología, metodología replicable y avance académico.

### **99. ¿Cómo validarías la adopción exitosa de tu sistema?**
**Respuesta:** Métricas de uso, satisfacción usuario, eficiencia operacional, reducción costos, casos de éxito y feedback cualitativo.

### **100. ¿Qué mensaje final quieres transmitir sobre tu investigación?**
**Respuesta:** Sistema RAG especializado que combina rigor académico con aplicabilidad práctica, contribuyendo con métricas especializadas, metodología reproducible y arquitectura optimizada para contextos técnicos.

---

## 📋 **CONSEJOS PARA USO**

### **Preparación:**
- Estudia cada respuesta a fondo
- Practica transiciones naturales
- Prepara ejemplos específicos
- Conoce las limitaciones

### **Durante la defensa:**
- Mantén respuestas concisas
- Usa los diagramas como apoyo
- Admite limitaciones cuando sea apropiado
- Conecta con contribuciones principales

### **Estrategia:**
- Agrupa preguntas similares
- Practica con colegas
- Prepara demos específicas
- Mantén confianza en tu trabajo

---

*Total: 100 preguntas organizadas por categorías temáticas con respuestas concisas tipo "pointer" para preparación de defensa.*