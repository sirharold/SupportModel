# RESUMEN

Esta investigación desarrolla y evalúa un sistema de recuperación semántica de información técnica especializada basado en arquitecturas RAG (Retrieval-Augmented Generation), utilizando documentación de Microsoft Azure como caso de estudio. El trabajo aborda el problema de accesibilidad del conocimiento técnico donde los sistemas de búsqueda léxica tradicionales presentan limitaciones significativas en dominios especializados. La investigación compara sistemáticamente cuatro modelos de embeddings —Ada (OpenAI), MPNet, E5-Large y MiniLM— evaluando su rendimiento en recuperación vectorial y tras aplicar reranking neural con CrossEncoder. El sistema almacena más de 800,000 vectores en ChromaDB, procesando un corpus de 187,031 chunks de documentación desde 62,417 documentos únicos de Microsoft Learn, utilizando 2,067 pares pregunta-documento validados como ground truth.

Los resultados establecen una jerarquía de rendimiento: Ada alcanza Precision@5 de 0.062, superando a MPNet (0.052), E5-Large (0.045) y MiniLM (0.041), con diferencias relativas de 19-34%. MPNet alcanza 83.9% del rendimiento de Ada utilizando solo 50% de dimensiones (768 vs 1,536), representando un trade-off favorable para aplicaciones con restricciones de recursos. El análisis de reranking revela un patrón diferencial robusto: CrossEncoder mejora modelos débiles (MiniLM +13.1%) pero degrada modelos optimizados (Ada -15.6%), con mejora leve en E5-Large (+2.2%), estableciendo que la aplicación de reranking debe ser selectiva según el modelo de embedding utilizado.

La evaluación multi-métrica identifica una discrepancia crítica: mientras las métricas de recuperación tradicionales muestran valores bajos (Precision@5 < 0.07), las métricas semánticas revelan rendimiento superior (Faithfulness 0.635-0.649, BERTScore 0.589 con convergencia completa), sugiriendo que todos los modelos producen respuestas de calidad semántica comparable. Esta discrepancia evidencia la principal limitación metodológica: el ground truth basado en enlaces comunitarios no garantiza validez de la correspondencia entre preguntas y documentos, imposibilitando conclusiones sobre rendimiento absoluto aunque permitiendo comparaciones relativas válidas entre modelos.

Las contribuciones metodológicas incluyen la documentación sistemática de limitaciones en construcción automatizada de ground truth, un framework de evaluación multi-métrica combinando métricas tradicionales con RAGAS y BERTScore, y la validación del patrón de reranking diferencial. Las contribuciones técnicas comprenden una implementación de referencia para almacenamiento vectorial con ChromaDB (latencia < 100ms, 800,000+ vectores) y un pipeline automatizado de evaluación reproducible. El trabajo establece como recomendación principal el desarrollo de ground truth validado por expertos del dominio técnico, con extensiones recomendadas en búsqueda híbrida, procesamiento multi-modal, y validación cross-domain en otros ecosistemas cloud (AWS, GCP).

**Palabras clave:** Recuperación de Información Semántica, RAG, Embeddings, Reranking Neural, Soporte Técnico, ChromaDB, RAGAS, BERTScore, Microsoft Azure




```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# ABSTRACT

This research develops and evaluates a semantic retrieval system for specialized technical information based on RAG (Retrieval-Augmented Generation) architectures, using Microsoft Azure documentation as a case study. The work addresses the problem of technical knowledge accessibility where traditional lexical search systems present significant limitations in specialized domains. The research systematically compares four embedding models —Ada (OpenAI), MPNet, E5-Large, and MiniLM— evaluating their performance in vector retrieval and after applying neural reranking with CrossEncoder. The system stores over 800,000 vectors in ChromaDB, processing a corpus of 187,031 documentation chunks from 62,417 unique Microsoft Learn documents, using 2,067 validated question-document pairs as ground truth.

Results establish a performance hierarchy: Ada achieves Precision@5 of 0.062, surpassing MPNet (0.052), E5-Large (0.045), and MiniLM (0.041), with relative differences of 19-34%. MPNet achieves 83.9% of Ada's performance using only 50% of dimensions (768 vs 1,536), representing a favorable trade-off for resource-constrained applications. Reranking analysis reveals a robust differential pattern: CrossEncoder improves weak models (MiniLM +13.1%) but degrades optimized models (Ada -15.6%), with slight improvement in E5-Large (+2.2%), establishing that reranking application must be selective according to the embedding model used.

Multi-metric evaluation identifies a critical discrepancy: while traditional retrieval metrics show low values (Precision@5 < 0.07), semantic metrics reveal superior performance (Faithfulness 0.635-0.649, BERTScore 0.589 with complete convergence), suggesting that all models produce comparable semantic quality responses. This discrepancy evidences the main methodological limitation: ground truth based on community links does not guarantee validity of question-document correspondence, preventing conclusions about absolute performance while allowing valid relative comparisons between models.

Methodological contributions include systematic documentation of limitations in automated ground truth construction, a multi-metric evaluation framework combining traditional metrics with RAGAS and BERTScore, and validation of the differential reranking pattern. Technical contributions comprise a reference implementation for vector storage with ChromaDB (latency < 100ms, 800,000+ vectors) and a reproducible automated evaluation pipeline. The work establishes as main recommendation the development of expert-validated ground truth from the technical domain, with recommended extensions in hybrid search, multi-modal processing, and cross-domain validation in other cloud ecosystems (AWS, GCP).

**Keywords:** Semantic Information Retrieval, RAG, Embeddings, Neural Reranking, Technical Support, ChromaDB, RAGAS, BERTScore, Microsoft Azure


