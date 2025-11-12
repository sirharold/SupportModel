# RESUMEN

El objetivo de este trabajo es diseñar, implementar y evaluar un sistema de recuperación semántica de información que asista en la resolución de tickets de soporte técnico relacionados con Microsoft Azure, utilizando documentación oficial y preguntas frecuentes como fuente de conocimiento. Se desarrolló una arquitectura en tres bloques principales: extracción automatizada de datos públicos desde Microsoft Learn y Microsoft Q&A mediante web scraping; generación de embeddings utilizando modelos preentrenados (all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1, E5-large) y embeddings de OpenAI; y almacenamiento en ChromaDB para consulta semántica.

El corpus desarrollado comprende 187,031 chunks de documentación segmentada desde 62,417 documentos únicos de Microsoft Learn, y 13,436 preguntas de Microsoft Q&A validadas como ground truth. El análisis exploratorio reveló contenido técnico sustancial con 872.3 tokens promedio por chunk y distribución balanceada entre categorías de Azure.

La evaluación se realizó mediante métricas estándar de recuperación de información: Precision@5, Recall@5, MRR@5, nDCG y F1-score. La mejor configuración de modelos open-source (MiniLM con title+summary+content) alcanza Precision@5 de 0.0256, Recall@5 de 0.0833 y MRR@5 de 0.0573. Los embeddings de OpenAI obtienen métricas superiores: Precision@5 de 0.034, Recall@5 de 0.112 y MRR@5 de 0.072, demostrando mayor capacidad de recuperación.

Este proyecto establece tres contribuciones principales: desarrollo del primer corpus especializado en documentación Azure para investigación académica; provisión de benchmarks reproducibles para evaluación de modelos de embeddings en dominios técnicos; y demostración empírica de la viabilidad de sistemas de recuperación semántica para asistencia en soporte técnico. Las métricas establecen una línea base para investigación futura, identificando oportunidades de mejora mediante fine-tuning, hybrid search y reranking avanzado.

**Palabras clave:** Recuperación de Información, Embeddings, RAG, Soporte Técnico, Azure, ChromaDB, Semantic Search
