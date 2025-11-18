# 8. CONCLUSIONES Y TRABAJO FUTURO

## 8.1 Introducción

Este capítulo sintetiza los hallazgos de la investigación sobre recuperación semántica de información técnica especializada. La investigación evaluó experimentalmente un sistema RAG implementado sobre un corpus de 187,031 documentos de Microsoft Azure, procesando 2,067 pares pregunta-documento.

El ground truth utilizado se derivó de enlaces incluidos en respuestas comunitarias. Este enfoque reveló tanto logros técnicos significativos como limitaciones metodológicas importantes que condicionan la interpretación de los resultados. Es crucial señalar desde el inicio que este método no garantiza que los documentos enlazados respondan efectivamente las preguntas planteadas. Esta limitación afecta la validez de las métricas de recuperación tradicionales, aunque permite validar aspectos técnicos de implementación y establecer comparaciones válidas entre modelos.

La evaluación cumplió los cinco objetivos técnicos establecidos al inicio de la investigación. Sin embargo, los resultados absolutos de recuperación (Precision@5 entre 0.041 y 0.062) revelan un rendimiento insuficiente para aplicaciones prácticas. Esta situación probablemente refleja más la calidad limitada del ground truth que fallas intrínsecas de las técnicas de recuperación semántica, un hallazgo relevante para la comunidad científica.

## 8.2 Cumplimiento de Objetivos de Investigación

### 8.2.1 Objetivo 1: Implementación y Comparación de Arquitecturas de Embeddings

El primer objetivo planteaba implementar y comparar múltiples arquitecturas de embeddings, evaluando tanto modelos de código abierto (MiniLM, MPNet, E5-Large) como propietarios (OpenAI Ada). Se implementaron exitosamente cuatro modelos con dimensionalidades variables entre 384 y 1,536 dimensiones. La evaluación documentó diferencias de rendimiento relativas entre 19% y 34% entre los modelos.

La Tabla 8.1 presenta los resultados cuantitativos para cada modelo:

| Ranking | Modelo | Tipo | Dimensionalidad | Precision@5 | Diferencia vs Ada | Eficiencia Relativa* |
|---------|--------|------|----------------|-------------|-------------------|---------------------|
| 1 | Ada | Propietario | 1,536 | 0.062 | - | Baseline (100%) |
| 2 | MPNet | Open-source | 768 | 0.052 | -19.2% | 83.9% con 50% dimensiones |
| 3 | E5-Large | Open-source | 1,024 | 0.045 | -27.4% | 72.6% con 67% dimensiones |
| 4 | MiniLM | Open-source | 384 | 0.041 | -33.9% | 66.1% con 25% dimensiones |

**Tabla 8.1**: Rendimiento comparativo y eficiencia relativa de modelos de embeddings evaluados
*Eficiencia relativa = (Precision@5 del modelo / Precision@5 de Ada) × 100

Si bien estos valores absolutos son insuficientes para aplicaciones prácticas, las diferencias relativas entre modelos constituyen hallazgos válidos que permiten establecer una jerarquía de rendimiento. Particularmente notable es la eficiencia relativa de MPNet, que alcanza 84% del rendimiento de Ada utilizando solo la mitad de dimensiones, un trade-off relevante para aplicaciones con restricciones de recursos.

Los resultados indican que el sistema recupera documentos relevantes en solo 4-6% de los casos en el top-5, un rendimiento claramente insuficiente. Sin embargo, la discrepancia con las métricas semánticas (significativamente superiores, ver sección 8.2.4) sugiere limitaciones en la metodología de evaluación más que fallas del sistema.

### 8.2.2 Objetivo 2: Sistema de Almacenamiento y Recuperación Vectorial

El segundo objetivo consistía en diseñar un sistema de almacenamiento y recuperación vectorial escalable. Se implementó ChromaDB 0.5.23 con ocho colecciones especializadas que almacenan 187,031 documentos y 13,436 preguntas por modelo, totalizando más de 800,000 vectores.

El sistema demostró escalabilidad técnica y rendimiento consistente a lo largo de la evaluación. Las especificaciones validadas incluyen almacenamiento eficiente para cuatro modelos, latencia promedio inferior a 100 milisegundos por consulta vectorial, y soporte simultáneo para múltiples dimensionalidades. Un logro importante es que la infraestructura funciona correctamente, confirmando que los resultados bajos en las métricas de recuperación no se deben a fallas del sistema vectorial, sino a otros factores metodológicos.

### 8.2.3 Objetivo 3: Mecanismos Avanzados de Reranking

El tercer objetivo planteaba desarrollar mecanismos avanzados de reranking. Se implementó CrossEncoder ms-marco-MiniLM-L-6-v2 con normalización Min-Max, lo que permitió identificar un patrón de efectividad diferencial particularmente relevante.

El reranking mejora el rendimiento de modelos débiles (MiniLM +13.1%) mientras que degrada modelos ya optimizados (Ada -15.6%, MPNet -3.4%), con impacto leve positivo en E5-Large (+2.2%). Este hallazgo es robusto e independiente de la calidad del ground truth, ya que representa comportamiento comparativo consistente entre configuraciones. El análisis detallado de este patrón se presenta en la sección 8.3.3.

### 8.2.4 Objetivo 4: Evaluación Sistemática del Rendimiento

El cuarto objetivo consistía en evaluar sistemáticamente el rendimiento del sistema. Se calcularon seis métricas tradicionales (Precision, Recall, F1, NDCG, MAP, MRR) para valores de k entre 1 y 15, tanto antes como después del reranking. Estas métricas se complementaron con seis métricas RAGAS (Faithfulness, Answer Relevance, Answer Correctness, Context Precision, Context Recall, Semantic Similarity) y tres métricas BERTScore (Precision, Recall, F1).

Un hallazgo clave emergió del análisis multi-métrico: las métricas RAG mostraron valores sustancialmente superiores (Faithfulness entre 0.635 y 0.649, BERTScore de 0.589 con convergencia entre modelos) en comparación con las métricas de recuperación tradicionales (Precision@5 inferior a 0.07). Esta discrepancia sugiere que el ground truth puede ser demasiado restrictivo y que los sistemas recuperan documentos semánticamente útiles que no son reconocidos como relevantes por la metodología de evaluación (ver sección 8.3.1).

> **Nota Metodológica**: Los valores de BERTScore Precision y Recall reportados provienen de la evaluación completa de 2,067 preguntas presentada en el archivo de resultados final (`cumulative_results_20251114_071914.json`). El valor de F1 fue calculado manualmente a partir de Precision y Recall mediante la fórmula F1 = 2×P×R/(P+R), ya que el campo `avg_bert_f1` se encontraba como `null` en el JSON de resultados.

### 8.2.5 Objetivo 5: Metodología Reproducible y Extensible

El quinto objetivo planteaba establecer una metodología reproducible y extensible. Se desarrolló un pipeline automatizado completo con trazabilidad total de resultados, materializada en un archivo de 135 MB con 2,067 evaluaciones detalladas.

El pipeline es técnicamente robusto y reproducible, independientemente de las limitaciones del ground truth. Esta infraestructura facilita la replicación y extensión de la investigación por otros equipos, y permite la validación independiente de los hallazgos reportados.

## 8.3 Conclusiones Principales

### 8.3.1 Rendimiento Insuficiente Condicionado por Limitaciones del ground truth

La investigación reveló resultados de recuperación insuficientes para aplicaciones prácticas, con una Precision@5 máxima de 0.062 (Ada). Sin embargo, esta conclusión está fuertemente condicionada por la calidad del ground truth utilizado, lo que constituye un hallazgo metodológico importante.

El ground truth basado en enlaces de respuestas comunitarias presenta una limitación fundamental: asume sin validación que dichos documentos efectivamente responden las preguntas. En la práctica, los enlaces pueden ser referencias complementarias más que respuestas directas, las respuestas pueden incluir múltiples enlaces con relevancia variable, y no existe validación experta de la correspondencia real entre preguntas y documentos.

El contraste entre métricas es revelador: mientras la recuperación tradicional muestra Precision@5 entre 0.041 y 0.062 (muy bajo), las métricas semánticas muestran valores significativamente superiores (Faithfulness entre 0.635 y 0.649, BERTScore F1 de 0.589 con convergencia entre modelos). Esta discrepancia sugiere que los sistemas recuperan documentos semánticamente relevantes que no son reconocidos como tales por el ground truth restrictivo. Los valores bajos pueden reflejar más las limitaciones del método de evaluación que fallas reales en las técnicas de recuperación semántica.

La implicación crítica es que los resultados cuantitativos no deben interpretarse como evidencia de inefectividad de la recuperación semántica. Por el contrario, constituyen evidencia de la necesidad de metodologías de evaluación más rigurosas que incorporen validación humana experta.

### 8.3.2 Jerarquía de Modelos Válida en Términos Comparativos

A pesar de las limitaciones del ground truth, las diferencias relativas entre modelos constituyen hallazgos válidos, dado que todos fueron evaluados bajo las mismas condiciones. Como muestra la Tabla 8.1 (sección 8.2.1), el ranking establece a Ada como líder, seguido por MPNet, E5-Large y MiniLM, con diferencias de rendimiento de 19-34% respecto al modelo superior.

Estos hallazgos sobre comportamiento comparativo son válidos independientemente de los valores absolutos. Esto permite concluir con confianza que Ada es superior a los modelos open-source en el contexto evaluado, aunque ninguno alcance rendimiento suficiente para producción según las métricas obtenidas.

### 8.3.3 Patrón de Reranking Diferencial con Implicaciones Arquitectónicas

El patrón de reranking diferencial identificado constituye un hallazgo técnico robusto y reproducible con importantes implicaciones arquitectónicas. La Tabla 8.2 presenta el impacto del reranking con CrossEncoder sobre cada modelo:

| Modelo | Pre-Reranking | Post-Reranking | Cambio Absoluto | Cambio Relativo | Categoría de Impacto |
|--------|---------------|----------------|-----------------|-----------------|---------------------|
| MiniLM | 0.041 | 0.047 | +0.005 | +13.1% | Beneficio significativo |
| E5-Large | 0.045 | 0.046 | +0.001 | +2.2% | Neutro (mejora leve) |
| MPNet | 0.052 | 0.050 | -0.002 | -3.4% | Neutro (degradación leve) |
| Ada | 0.062 | 0.052 | -0.010 | -15.6% | Degradación significativa |

**Tabla 8.2**: Impacto del reranking con CrossEncoder sobre Precision@5 por modelo

Este patrón revela un principio importante: el reranking beneficia modelos de recuperación inicial débil pero puede degradar modelos cuya recuperación ya está optimizada. La validez de este hallazgo es independiente de la calidad del ground truth, ya que representa comportamiento comparativo consistente entre configuraciones. Esto tiene implicaciones prácticas para el diseño de arquitecturas RAG: la aplicación de reranking debe ser selectiva y basada en las características del modelo de embedding.

### 8.3.4 Convergencia Semántica Independiente del Rendimiento de Recuperación

Un hallazgo particularmente interesante es que todos los modelos convergen en métricas semánticas, mostrando valores de Faithfulness entre 0.635 y 0.649, y BERTScore de 0.589 (convergencia completa), independientemente de su rendimiento en recuperación exacta. Modelos con Precision@5 muy diferentes (0.041 versus 0.062) producen respuestas de calidad semántica similar.

Este fenómeno sugiere tres conclusiones importantes:

1. Las métricas de recuperación tradicionales pueden subestimar la utilidad práctica de los sistemas
2. El componente de generación compensa parcialmente las limitaciones en recuperación, produciendo respuestas de calidad comparable incluso con recuperación inicial diferente
3. La evaluación de sistemas RAG requiere métricas multi-dimensionales que capturen tanto recuperación como generación

La implicación práctica es que la calidad de respuesta final puede ser aceptable incluso con recuperación aparentemente deficiente. Esto abre posibilidades interesantes para el uso de modelos más eficientes en escenarios donde la calidad semántica final es más importante que la precisión exacta de recuperación.

### 8.3.5 Necesidad de Validación Humana en Evaluaciones Futuras

La investigación reveló la necesidad fundamental de incorporar evaluación humana experta en la validación de sistemas de recuperación de información técnica especializada. Como se discutió en la sección 8.3.1, el contraste marcado entre métricas de recuperación (bajas) y semánticas (moderado-altas) hace imposible distinguir entre fallas reales del sistema y limitaciones del ground truth.

Para futuras investigaciones, se recomienda validación por expertos del dominio técnico capaces de juzgar la relevancia real de documentos en contextos especializados. Este enfoque, aunque más costoso, es esencial para establecer ground truth verdaderamente confiable.

### 8.3.6 Eficiencia Relativa de Modelos Open-Source

Un hallazgo con implicaciones prácticas importantes emerge del análisis de la relación entre rendimiento y recursos computacionales. MPNet alcanza 83.9% del rendimiento de Ada utilizando solo 50% de dimensiones (768 vs 1,536), lo que representa un trade-off altamente favorable para aplicaciones con restricciones de recursos.

Este resultado es particularmente relevante considerando la convergencia en métricas semánticas discutida en la sección 8.3.4. Si bien Ada supera a MPNet en Precision@5 (0.062 vs 0.052), ambos modelos producen respuestas de calidad semántica comparable (Faithfulness ~0.71, BERTScore 0.589). Esto sugiere que para aplicaciones donde la calidad semántica final es más importante que la precisión exacta de recuperación, MPNet podría ser una alternativa eficiente a modelos propietarios más costosos.

El análisis de eficiencia se extiende a todos los modelos evaluados: E5-Large alcanza 72.6% del rendimiento de Ada con 67% de dimensiones (1,024), mientras que MiniLM, el modelo más ligero evaluado, logra 66.1% del rendimiento con apenas 25% de dimensiones (384). Esta gradación permite seleccionar modelos según las restricciones específicas de cada aplicación, balanceando rendimiento, costo computacional, y requisitos de latencia.

La implicación práctica es que en escenarios con restricciones de recursos (dispositivos edge, alta concurrencia, presupuestos limitados), los modelos open-source evaluados pueden ofrecer soluciones viables que, si bien no alcanzan el rendimiento máximo, proporcionan capacidades de recuperación y generación semántica aceptables a una fracción del costo computacional.

## 8.4 Contribuciones del Trabajo

### 8.4.1 Contribuciones Metodológicas

La principal contribución metodológica de este trabajo es la documentación sistemática de las limitaciones que presenta el uso de enlaces de respuestas comunitarias como ground truth para evaluar sistemas de recuperación técnica. Este enfoque, comúnmente utilizado en investigación debido a su conveniencia y escalabilidad, no garantiza la validez de la correspondencia entre preguntas y documentos, lo que limita significativamente la interpretabilidad de resultados cuantitativos obtenidos. Este hallazgo crítico alerta a futuras investigaciones sobre la necesidad de validación experta adicional.

Una segunda contribución metodológica significativa es el framework de evaluación multi-métrica desarrollado, que combina métricas tradicionales de recuperación, métricas específicas para RAG mediante RAGAS, y evaluación semántica mediante BERTScore. Este enfoque permite detectar discrepancias entre diferentes dimensiones de evaluación, revelando limitaciones metodológicas que enfoques uni-métricos no detectarían. La capacidad de comparar simultáneamente métricas de recuperación exacta y calidad semántica resultó fundamental para identificar las limitaciones del ground truth utilizado.

Finalmente, la validación del patrón de reranking diferencial constituye una contribución metodológica con implicaciones prácticas. El principio de efectividad diferencial del reranking, basado en la calidad de los embeddings iniciales, es un resultado técnico válido e independiente de las limitaciones del ground truth. Este hallazgo contribuye al conocimiento sobre arquitecturas RAG al demostrar que la aplicación efectiva de reranking requiere consideración del modelo de embedding utilizado.

### 8.4.2 Contribuciones Técnicas

Desde el punto de vista técnico, el trabajo establece una implementación de referencia para almacenamiento y recuperación vectorial a escala académica, utilizando ChromaDB con más de 800,000 vectores (ver especificaciones detalladas en sección 8.2.2). Esta infraestructura técnicamente robusta puede ser útil para futuras investigaciones que requieran capacidades similares de almacenamiento y búsqueda vectorial.

Adicionalmente, el pipeline de evaluación automatizado desarrollado constituye un sistema completo y reproducible que abarca desde la configuración inicial hasta la visualización de resultados, con trazabilidad completa en cada etapa. Esta infraestructura facilita la replicación y extensión de la investigación por parte de otros equipos, proporcionando una base sólida para trabajos futuros.

## 8.5 Limitaciones Identificadas

### 8.5.1 Limitaciones Metodológicas

La limitación más significativa de este trabajo radica en que los 2,067 pares pregunta-documento utilizados como ground truth provienen de enlaces incluidos en respuestas comunitarias, sin validación experta que confirme la correspondencia real. Como se detalla en la sección 8.3.1, esta limitación imposibilita distinguir entre fallas reales del sistema y fallas del ground truth restrictivo.

La ausencia de evaluación humana experta habría requerido recursos significativos fuera del alcance de este trabajo, pero habría proporcionado insights fundamentales sobre la efectividad real de los sistemas evaluados.

### 8.5.2 Limitaciones Técnicas

Desde el punto de vista técnico, el procesamiento exclusivamente textual representa una limitación importante, dado que entre 30% y 40% del contenido de documentación técnica moderna incluye elementos multimedia como diagramas, capturas de pantalla, y videos. Los resultados obtenidos son válidos únicamente para el componente textual de la documentación, y no pueden generalizarse al contenido visual.

La especialización estricta en el ecosistema Azure también limita la generalización directa de hallazgos a otros dominios técnicos. Aunque los principios identificados (como el patrón de reranking diferencial) probablemente sean aplicables en otros contextos, la validación empírica de su generalización requeriría evaluaciones adicionales en otros ecosistemas cloud como AWS o GCP.

### 8.5.3 Limitaciones de Alcance

Finalmente, el uso exclusivo de datos públicos, sin acceso a datos corporativos internos, limita la validación de hallazgos con casos de uso industriales reales. Los entornos corporativos presentan complejidades adicionales, como terminología interna, configuraciones personalizadas, y contextos organizacionales específicos, que pueden no estar adecuadamente representados en documentación pública. Los hallazgos obtenidos pueden no reflejar completamente la complejidad de implementaciones en contextos corporativos reales.

## 8.6 Trabajo Futuro

### 8.6.1 Desarrollo de ground truth Validado por Expertos

La recomendación principal para futuras investigaciones es desarrollar ground truth validado por expertos del dominio técnico. Este proceso debería incluir:

**Tabla 8.3: Componentes Recomendados para Validación de ground truth**

| Componente | Descripción | Beneficio Esperado |
|------------|-------------|-------------------|
| Panel de especialistas | Formación de expertos en Azure para validar correspondencia pregunta-documento | Evaluación informada del contexto técnico |
| Criterios graduales | Escalas de relevancia (0-3) en lugar de evaluaciones binarias | Mayor precisión en la evaluación de relevancia |
| Validación multi-evaluador | Múltiples expertos independientes por cada par | Garantía de consenso y reducción de sesgos |
| Documentación de razonamiento | Expertos explican sus criterios de relevancia | Trazabilidad y reproducibilidad de decisiones |

El resultado esperado sería un ground truth verdaderamente confiable que permita evaluar la efectividad real de sistemas de recuperación técnica, proporcionando una base sólida para conclusiones sobre rendimiento absoluto más allá de comparaciones relativas entre modelos.

### 8.6.2 Extensiones Recomendadas

Si se continúa esta línea de investigación, varias direcciones serían relevantes y prometedoras:

**Tabla 8.4: Extensiones Recomendadas para Investigación Futura**

| Extensión | Descripción | Beneficio Esperado |
|-----------|-------------|-------------------|
| Datos corporativos validados | Acceso a tickets de soporte con documentos de solución verificados | Validación con casos de uso reales e industriales |
| Búsqueda híbrida | Combinación de recuperación vectorial semántica con técnicas keyword-based | Mejora de cobertura y precisión de recuperación |
| Contenido multi-modal | Procesamiento de diagramas y elementos visuales | Mayor aplicabilidad a documentación técnica real |
| Validación cross-domain | Evaluación en otros ecosistemas (AWS, GCP) | Establecimiento de robustez de principios identificados |

Es importante notar que estas extensiones requieren el desarrollo previo de ground truth validado para ser verdaderamente efectivas, dado que los problemas metodológicos identificados se reproducirían en cualquier contexto sin ground truth confiable.

## 8.7 Conclusión del Capítulo

Esta investigación cumplió sus objetivos técnicos de implementación y evaluación, desarrollando un sistema RAG completo con un pipeline automatizado de evaluación multi-métrica sobre un corpus sustancial de documentación Azure. Sin embargo, los resultados revelan limitaciones metodológicas fundamentales que condicionan la interpretación de los hallazgos.

Los hallazgos técnicos válidos incluyen la jerarquía relativa entre modelos (Tabla 8.1), el patrón de reranking diferencial (Tabla 8.2), la convergencia en métricas semánticas independiente del rendimiento de recuperación, el análisis de eficiencia relativa de modelos open-source, y la demostración de escalabilidad de la infraestructura ChromaDB. Estos hallazgos son robustos porque se basan en comparaciones relativas bajo condiciones experimentales controladas.

La limitación crítica es que el ground truth basado en enlaces de respuestas comunitarias no garantiza la validez de la correspondencia entre preguntas y documentos, produciendo resultados cuantitativos de validez cuestionable. Esta limitación no invalida los hallazgos comparativos, pero impide conclusiones definitivas sobre rendimiento absoluto.

La contribución principal del trabajo es la documentación honesta de las limitaciones inherentes a metodologías automatizadas de construcción de ground truth para dominios técnicos especializados. Al alertar a la comunidad científica sobre estos desafíos, el trabajo subraya la necesidad de incorporar validación humana experta en futuras investigaciones.

Para la comunidad científica, la implicación más importante es que los resultados cuantitativos reportados no deben interpretarse como evidencia de inefectividad de la recuperación semántica en sí misma. Más bien, constituyen evidencia de las limitaciones de enfoques automatizados de evaluación que carecen de validación experta. La efectividad real de estos sistemas requiere validación con ground truth desarrollado por expertos del dominio capaces de juzgar la relevancia en contextos técnicos especializados.

La recomendación final para futuras investigaciones en recuperación de información técnica especializada es priorizar el desarrollo de ground truth validado por expertos antes de ejecutar evaluaciones cuantitativas a escala. Solo así puede garantizarse que los resultados reflejen las capacidades reales de los sistemas evaluados, permitiendo conclusiones válidas sobre su efectividad en aplicaciones prácticas.
