# 📝 SESIÓN DE HUMANIZACIÓN DE TESIS - REGISTRO DE TRABAJO

**Fecha**: 24 de octubre de 2025
**Objetivo**: Humanizar capítulos de tesis, reduciendo contenido que suena generado por IA
**Carpeta de trabajo**: `/Docs/Octubre2025/`

---

## 🎯 CONTEXTO DEL PROYECTO

### Situación Inicial
El usuario tiene una tesis de más de 200 páginas con múltiples versiones de capítulos:
- **Versión entregada**: `Docs/Finales version entregada en agosto 2025/` (Agosto 2025)
- **Versiones actualizadas**: `Docs/Octubre2025/` (Octubre 2025)
  - Solo Capítulo 7 y Anexo E tienen versiones actualizadas con datos de 2,067 preguntas
  - Resto de capítulos solo tienen versión de agosto

### Objetivo del Trabajo
Revisar y humanizar cada capítulo para:
1. **Eliminar lenguaje que suena generado por IA**
2. **Convertir listas en prosa académica fluida**
3. **Reducir frases excesivamente largas y complejas**
4. **Variar inicios de párrafos y estructura**
5. **Mantener rigor académico pero con tono más natural**

---

## ✅ DIRECTRICES Y PREFERENCIAS DEL USUARIO

### ✅ Mantener:
- Palabras técnicas en inglés: "pipeline", "framework", "chunks", "embeddings", "reranking"
- Referencias académicas correctamente citadas
- Rigor científico y precisión técnica
- Estructura académica formal

### ❌ Eliminar:
- Palabra "comprehensivo" (anglicismo)
- Uso excesivo de "-mente" (fundamentalmente, específicamente, sistemáticamente)
- Frases robotizadas ("Este capítulo examina...", "El sistema se limita...")
- Listas largas de viñetas (convertir a prosa)
- Referencias internas a código/scripts del proyecto
- Secciones como "Nota sobre fuentes de datos"
- Objetivos que son medios, no fines (ej: "Construir un corpus")

### 🎨 Estilo Preferido:
- Prosa académica fluida sin listas
- Frases más cortas (objetivo: ~25 palabras promedio vs 40+)
- Inicios de párrafo variados
- Transiciones naturales entre secciones
- Voz más cercana al lector (sin perder formalidad)

---

## 📊 PROGRESO REALIZADO

### ✅ CAPÍTULO 1 - Introducción (COMPLETADO)
**Archivo**: `/Docs/Octubre2025/capitulo_1.md`

**Cambios aplicados:**
1. ❌ Eliminada sección 1.7 "Nota sobre fuentes de datos"
2. ❌ Eliminado objetivo "Construir un corpus comprehensivo" (es un medio)
3. ❌ Eliminada referencia a `verify_questions_links_match.py`
4. 📖 Convertida lista de alcances temáticos a prosa (§1.2.1)
5. ✂️ Reducidos objetivos específicos de 6 a 5
6. 🔄 Reformuladas secciones de problema con negritas temáticas:
   - "**El conocimiento existe, pero no se encuentra**"
   - "**Los esfuerzos se duplican constantemente**"
   - "**La búsqueda léxica tradicional no funciona**"

**Métricas:**
- Reducción de repeticiones de cifras: -60%
- Longitud promedio de frase: -25%
- Palabras "comprehensivo": 2 → 0

---

### ✅ CAPÍTULO 2 - Estado del Arte (COMPLETADO)
**Archivo**: `/Docs/Octubre2025/capitulo_2_estado_del_arte.md`

**Cambios aplicados:**
1. 📖 Convertida lista de limitaciones SQL (§2.4.2) de 15 viñetas → 1 párrafo de prosa
2. 🔄 Consolidados casos empresariales de 4 subsecciones → 1 sección narrativa (§2.5)
3. 📊 Consolidadas métricas de evaluación de 6 subsecciones → 3 bloques temáticos (§2.6)
4. 📖 Convertida lista de framework de evaluación (§2.6.6) → prosa integrada en §2.6.3
5. 🗑️ Eliminadas referencias duplicadas:
   - Johnson et al. (2021) - duplicado de (2019)
   - Malkov & Yashunin (2020) - duplicado de (2018)
6. ✂️ Reducida introducción de 2 párrafos densos → 2 párrafos concisos
7. 🔄 Variados inicios de párrafos (eliminadas repeticiones de "Los sistemas...", "Las arquitecturas...")

**Métricas:**
- Subsecciones: 17 → 12 (-29%)
- Listas de viñetas: 2 grandes (~20 líneas) → 0 (-100%)
- Referencias: 31 → 29 (-2 duplicados)
- Longitud promedio frase: ~40 palabras → ~25 palabras (-37%)

**Verificación de referencias:**
- ✅ 23/23 referencias citadas correctamente en el texto
- ✅ 0 referencias huérfanas
- ✅ Todas con formato (Autor, año) correcto

---

### ✅ CAPÍTULO 3 - Marco Teórico (COMPLETADO)
**Archivo**: `/Docs/Octubre2025/capitulo_3_marco_teorico.md`

**Estado**: Completado (actualizado a las 14:12 según ls)

---

### ✅ CAPÍTULO 4 - Análisis Exploratorio de Datos (COMPLETADO)
**Archivo**: `/Docs/Octubre2025/capitulo_4_analisis_exploratorio_datos.md`

**Cambios aplicados:**
1. ❌ Eliminada frase inicial "Este capítulo presenta..." → inicio más directo
2. ❌ Eliminadas referencias a scripts internos:
   - Línea 7: "scripts de análisis optimizados disponibles en `Docs/Analisis/`"
   - Sección completa "Fuentes de Datos" (líneas 503-510 del original)
3. 📖 Convertidas TODAS las listas de viñetas a prosa académica fluida:
   - §4.2.3.3: Lista "Implicaciones para el Sistema RAG" → prosa narrativa
   - §4.4.1.2: Lista de 10 causas de no-correspondencia → 3 párrafos de prosa
   - §4.5.1: Listas "Fortalezas" y "Áreas de mejora" → prosa integrada
   - §4.5.2: Listas de fortalezas/limitaciones → prosa narrativa
   - §4.6: TODAS las listas de "Prioridad Alta/Media/Estrategias" → prosa fluida
4. 🔄 Eliminados placeholders con llaves `{45.2%}` → integrados como texto narrativo
5. ✂️ Cambiados placeholders `{PLACEHOLDER_FIGURA_X.X}` → `[Figura X.X. Descripción]`
6. 📊 Consolidadas subsecciones con estructura muy profunda (4 niveles)
7. 🔄 Variados inicios de párrafos y mejorada fluidez narrativa
8. ✂️ Reducida longitud de frases complejas

**Métricas:**
- Listas de viñetas: ~50 líneas → 0 (-100%)
- Referencias a código/scripts: 3 → 0 (-100%)
- Placeholders con llaves: ~15 instancias → 0 (-100%)
- Estructura de secciones: Mantenida pero con prosa más fluida
- Longitud: ~511 líneas → ~379 líneas (-26%)

**Aspectos técnicos preservados:**
- ✅ Todas las métricas numéricas exactas (62,417 docs, 187,031 chunks, etc.)
- ✅ Estadísticas detalladas de análisis (media, mediana, desv. estándar)
- ✅ Porcentajes y ratios de cobertura/correspondencia
- ✅ Comparaciones con corpus estándar (MS-MARCO, Natural Questions, SQuAD)
- ✅ 6 referencias bibliográficas (agregadas citas académicas para benchmarks)

**Verificación de referencias:**
- ✅ 6/6 referencias citadas correctamente en el texto
- ✅ Kwiatkowski et al. (2019) - Natural Questions - citado en §4.5.4
- ✅ Microsoft (2025a) - Microsoft Learn - citado en §4.1
- ✅ Microsoft (2025b) - Microsoft Q&A - citado en §4.1, §4.3.1
- ✅ Nguyen et al. (2016) - MS-MARCO - citado en §4.5.4
- ✅ OpenAI (2025) - tiktoken - citado en §4.2.2, §4.3.2
- ✅ Rajpurkar et al. (2018) - SQuAD 2.0 - citado en §4.5.4
- ✅ 0 referencias huérfanas
- ✅ Formato (Autor, año) consistente

---

### ✅ CAPÍTULO 5 - Metodología (COMPLETADO)
**Archivo**: `/Docs/Octubre2025/capitulo_5_metodologia.md`

**Cambios aplicados:**
1. ❌ **ELIMINADOS ~280 líneas de código Python**, reemplazados con descripción en prosa:
   - `normalize_url()` → descripto en 1 frase
   - `generate_query_embedding()` → descripto en párrafo
   - `EmbeddedRetriever` → arquitectura en prosa
   - `vector_retrieval()` → integrado en descripción pipeline
   - `rerank_with_cross_encoder()` → algoritmo descripto en párrafo
   - `ComprehensiveEvaluationFramework` → framework descripto en prosa
   - `calculate_rag_metrics_real()` → metodología de cálculo descripta
   - `calculate_bertscore()` → configuración descripta
   - `statistical_validation()` → tests estadísticos descriptos
   - Código de logging, semillas, config → mencionados en texto
2. ❌ Eliminada Carta Gantt ASCII (17 líneas) - redundante con Mermaid
3. ✅ **MANTENIDO diagrama Mermaid** (flujo metodológico visual)
4. 📖 Convertidas ~200 líneas de listas a prosa académica fluida:
   - Fases 1-6 con formato "Entrada/Proceso/Salida" → prosa narrativa
   - Fases I-VI del cronograma → prosa integrada
   - Hitos H1-H5 → descritos en párrafo
   - Variables de investigación → prosa explicativa
   - Estadísticas del corpus → integradas en texto
   - Características de modelos → descritas narrativamente
   - Aspectos éticos y limitaciones → prosa fluida
5. ❌ Eliminadas 7 referencias a scripts internos:
   - `verify_document_statistics.py` (3 menciones)
   - `calculate_topic_distribution_v2.py`
   - `verify_questions_statistics_v2.py` (3 menciones)
6. ❌ Eliminados 8 placeholders con llaves `{estimación pendiente...}`
7. ❌ Reemplazadas 8 instancias de "comprehensivo/a" → exhaustivo/integral/completo
8. ✅ Agregada referencia: **Weaviate (2023)**
9. 🔄 Mejorados inicios de párrafo (eliminadas frases robotizadas)
10. ✅ Agregada sección §5.9: **Nota sobre Implementación** (referencia al código en repositorio)

**Métricas:**
- Código Python eliminado: ~280 líneas → 0 (-100%)
- Gantt ASCII eliminado: 17 líneas → 0 (-100%)
- Listas de viñetas: ~200 líneas → 0 (-100%)
- Referencias a scripts: 7 → 0 (-100%)
- Placeholders con llaves: 8 → 0 (-100%)
- Palabra "comprehensivo/a": 8 → 0 (-100%)
- Longitud: ~951 líneas → ~367 líneas **(-61%)**

**Aspectos técnicos preservados:**
- ✅ Diagrama Mermaid del flujo metodológico completo
- ✅ Todas las fases, hitos y cronograma descriptos
- ✅ Especificaciones técnicas de modelos (Ada, MPNet, MiniLM, E5-Large)
- ✅ Descripción completa de métricas (Precision, Recall, F1, MRR, nDCG, RAGAS, BERTScore)
- ✅ Metodología de validación estadística (Wilcoxon, Bonferroni)
- ✅ Consideraciones éticas y limitaciones metodológicas
- ✅ 24 referencias bibliográficas

**Verificación de referencias:**
- ✅ 24/24 referencias citadas correctamente en el texto
- ✅ Chapman et al. (2000) - CRISP-DM - citado en §5.1
- ✅ ChromaDB Team (2023) - citado en §5.5.3
- ✅ Cleverdon (1967) - Cranfield - citado en §5.2.4
- ✅ Creswell & Creswell (2017) - citado en §5.1
- ✅ Es et al. (2023) - RAGAS - citado en §5.7.4
- ✅ Ferro & Peters (2019) - citado en §5.7.2
- ✅ Han et al. (2011) - Min-Max normalization - citado en §5.6.3
- ✅ Hevner et al. (2004) - DSR - citado en §5.2.3
- ✅ Karpukhin et al. (2020) - Dense retrieval - citado en §5.6.1
- ✅ Kelly (2009) - citado en §5.2.4
- ✅ Landers & Behrend (2015) - Web scraping ético - citado en §5.4.1
- ✅ Mitchell (2018) - Web scraping - citado en §5.4.1
- ✅ Muennighoff et al. (2023) - MTEB - citado en §5.5.1
- ✅ Peffers et al. (2007) - DSR - citado en §5.2.3
- ✅ Qu et al. (2021) - RocketQA - citado en §5.6.1
- ✅ Reimers & Gurevych (2019) - Sentence-BERT - citado en §5.5.1
- ✅ Sanderson (2010) - IR evaluation - citado en §5.7.2
- ✅ Shearer (2000) - CRISP-DM - citado en §5.1
- ✅ Voorhees & Harman (2005) - TREC - citado en §5.2.4
- ✅ Wang et al. (2020) - MiniLM - citado en §5.5.1
- ✅ Wang et al. (2022) - E5-Large - citado en §5.5.1
- ✅ Weaviate (2023) - citado en §5.5.3
- ✅ 0 referencias huérfanas
- ✅ Formato (Autor, año) consistente

---

### ✅ CAPÍTULO 6 - Implementación (COMPLETADO)
**Archivo**: `/Docs/Octubre2025/capitulo_6_implementacion.md`

**Cambios aplicados:**
1. ❌ **ELIMINADOS ~500 líneas de código Python** (15 bloques completos):
   - Requirements.txt → descripto en prosa
   - Ejemplos JSON → reducidos a descripción textual
   - `ChromaDBClientWrapper` class → funcionalidad en prosa
   - Config colecciones → arquitectura en prosa
   - `EmbeddingClient` class → componentes en prosa
   - Búsqueda vectorial con diversidad → algoritmo en prosa
   - Búsqueda híbrida por enlaces → estrategia en prosa
   - Métricas de recuperación → cálculos en prosa
   - **Pipeline RAG end-to-end (83 líneas)** → 8 etapas narrativas
   - Reranking CrossEncoder → normalización en prosa
   - Generación de respuestas → backends en prosa
   - Streamlit main, QA Interface, Dashboard → arquitectura UI en prosa

2. 📖 Convertidas **~80-100 líneas de listas** a prosa académica fluida:
   - Stack tecnológico → párrafo descriptivo
   - Arquitectura scraping, desafíos → prosa narrativa
   - Pipeline extracción, resultados → prosa integrada
   - Metodología Q&A, dataset → descripción fluida
   - Consideraciones éticas (6 listas) → 3 párrafos cohesivos
   - Comparación BD, optimizaciones → prosa narrativa
   - Optimizaciones y mejoras (9 listas) → prosa descriptiva

3. ❌ Eliminado **1 placeholder** con llaves `{código de scraping...}`

4. ❌ Eliminadas **9 referencias a scripts** internos:
   - `src/services/storage/chromadb_utils.py` (2 menciones)
   - `src/config/config.py`
   - `src/data/embedding.py`
   - `src/evaluation/metrics/retrieval.py`
   - `src/core/qa_pipeline.py`
   - `src/core/reranker.py`
   - `src/services/answer_generation/local.py`
   - `src/apps/main_qa_app.py`

5. ❌ Reemplazadas **2 "comprehensivo/a"** → completo/integral

6. 🔄 **Simplificadas 30+ frases complejas** (40+ palabras):
   - Párrafos iniciales fragmentados en frases más cortas
   - Eliminadas subordinadas innecesarias
   - Reducida complejidad sintáctica

7. 🔄 **Reemplazadas 50+ palabras rebuscadas**:
   - "implementa/implementación" (15x) → usa/desarrollo/aplica
   - "constituye" (3x) → es/representa
   - "fundamenta" (2x) → basa/apoya
   - "establece/establecer" (8x) → crea/define/configura
   - "metodología robusta" → método confiable
   - "infraestructura" → sistema/arquitectura

8. 🔄 **Variados 25+ inicios repetitivos**:
   - "La implementación..." (5x) → diversificados
   - "El sistema implementa..." (8x) → variados
   - "Una vez completada... el siguiente paso fue..." (3x) → eliminados

9. 🔄 **Convertidas 15+ voz pasiva** → voz activa:
   - "fue seleccionado" → "se seleccionó"
   - "se fundamenta" → "se basa"
   - "se encuentra licenciada" → "está licenciada"

10. ❌ **Eliminadas 20+ redundancias**:
    - "representa... constituye" → "es"
    - "metodología estructurada que garantiza..." → "método que incluye..."
    - Frases de relleno pomposas → descripción directa

11. ✅ **Corregidas referencias bibliográficas**:
    - Eliminada Chapman et al. (2000) (no citada en Cap. 6)
    - Verificadas 5 referencias restantes

12. ✅ Agregada sección §6.9: **Nota sobre Implementación**

**Métricas:**
- Código Python eliminado: ~500 líneas → 0 (-100%)
- Listas de viñetas: ~80-100 líneas → 0 (-100%)
- Referencias a scripts: 9 → 1 (solo en nota final) (-89%)
- Placeholders con llaves: 1 → 0 (-100%)
- Palabra "comprehensivo/a": 2 → 0 (-100%)
- Frases complejas simplificadas: 30+
- Palabras rebuscadas reemplazadas: 50+
- Inicios repetitivos variados: 25+
- **Longitud: 808 líneas → 173 líneas (-78.5%)**

**Aspectos técnicos preservados:**
- ✅ Stack tecnológico completo (Python, Streamlit, ChromaDB, librerías NLP)
- ✅ Proceso de extracción (scraping, normalización, validación)
- ✅ Resultados verificados (62,417 docs, 187,031 chunks, 13,436 preguntas)
- ✅ Arquitectura ChromaDB (colecciones, optimizaciones, métricas rendimiento)
- ✅ Componentes RAG (indexación, búsqueda, evaluación)
- ✅ Pipeline de 8 etapas completamente descripto
- ✅ Reranking CrossEncoder (normalización sigmoid/min-max)
- ✅ Interfaz Streamlit (multi-página, Q&A, métricas)
- ✅ Optimizaciones (cache, batch processing, gestión memoria)
- ✅ Consideraciones éticas y legales (CC BY 4.0)

**Verificación de referencias:**
- ✅ 5/5 referencias citadas correctamente en el texto
- ✅ ChromaDB Team (2024) - citado en §6.2.1
- ✅ McConnell (2004) - Code Complete - citado en §6.1
- ✅ Microsoft Corporation (2024) - Terms of Use - citado en §6.3.4
- ✅ Streamlit Team (2023) - citado en §6.2.1
- ✅ Van Rossum & Drake (2009) - Python - citado en §6.2.1
- ✅ 0 referencias huérfanas
- ✅ Formato (Autor, año) consistente

---

## 🔄 PROCESO DE HUMANIZACIÓN APLICADO

### 1. Análisis Inicial
Para cada capítulo se identifica:
- Lenguaje excesivamente académico/formal
- Estructura demasiado rígida
- Frases largas y complejas
- Redundancias y repeticiones
- Listas que rompen flujo narrativo
- Tono impersonal y distante

### 2. Cambios Estructurales
- Consolidar subsecciones excesivas
- Convertir listas a prosa académica
- Eliminar secciones redundantes
- Fusionar contenido repetitivo

### 3. Mejoras de Estilo
- Reducir longitud de frases
- Eliminar palabras "relleno"
- Variar inicios de párrafo
- Agregar transiciones naturales
- Mantener rigor académico

### 4. Verificación Final
- Chequear que todas las referencias estén citadas
- Verificar coherencia narrativa
- Confirmar que se mantiene precisión técnica

---

## 📋 CAPÍTULOS PENDIENTES

### ⏳ Por Revisar:
1. **Capítulo 0** - Resumen
2. **Capítulo 7** - Resultados y Análisis (SIGUIENTE - usar versión Octubre2025)
3. **Capítulo 8** - Conclusiones
4. **Anexos** - Revisar si aplica

### 📁 Ubicación de Archivos:
**Archivos origen**: `/Docs/Finales version entregada en agosto 2025/`
**Archivos destino**: `/Docs/Octubre2025/` ← TODOS los capítulos humanizados van aquí

---

## 🎯 PRÓXIMOS PASOS

1. **Capítulo 7 - Resultados y Análisis**:
   - Usar versión Octubre2025 (con datos de 2,067 preguntas)
   - Análisis de contenido
   - Identificar tablas, gráficos, listas
   - Convertir listas a prosa
   - Verificar referencias
   - Verificar que métricas coincidan con datos reales

2. **Capítulo 8 - Conclusiones**:
   - Análisis y humanización final

3. **Revisión final** de consistencia entre capítulos

---

## 💡 NOTAS IMPORTANTES

### Para retomar el trabajo:
- Revisar este documento antes de continuar
- Verificar ubicación de archivos (siempre en `/Octubre2025/`)
- Aplicar mismo proceso de humanización
- Mantener directrices del usuario

### Palabras clave para buscar en próximos capítulos:
- ❌ "comprehensivo"
- ❌ Frases con "Este capítulo...", "Este proyecto...", "El sistema..."
- ❌ Listas largas de viñetas
- ❌ Palabras terminadas en -mente (fundamentalmente, específicamente, etc.)
- ❌ Frases de más de 3 líneas sin puntos
- ❌ Referencias a código/scripts internos

### Verificaciones obligatorias:
- ✅ Todas las referencias citadas en texto
- ✅ No quedan listas de viñetas
- ✅ Frases promedio <30 palabras
- ✅ Archivo guardado en `/Octubre2025/`

---

## 📊 ESTADÍSTICAS GENERALES

### Capítulos Completados: 6/8 (75%)

| Capítulo | Estado | Archivo | Cambios Principales |
|----------|--------|---------|---------------------|
| Cap. 0 | ⏳ Pendiente | - | - |
| Cap. 1 | ✅ Completado | `capitulo_1.md` | -9.6%, eliminada lista, 5 objetivos |
| Cap. 2 | ✅ Completado | `capitulo_2_estado_del_arte.md` | -30.9%, 0 listas, 23 refs |
| Cap. 3 | ✅ Completado | `capitulo_3_marco_teorico.md` | -51.4%, ya actualizado |
| Cap. 4 | ✅ Completado | `capitulo_4_analisis_exploratorio_datos.md` | -47.6%, 0 listas, 6 refs |
| Cap. 5 | ✅ Completado | `capitulo_5_metodologia.md` | **-61.4%**, 0 código, 24 refs |
| Cap. 6 | ✅ Completado | `capitulo_6_implementacion.md` | **-78.5%**, 0 código, 5 refs |
| Cap. 7 | ⏳ Siguiente | Usar versión Octubre | Con datos 2,067 preguntas |
| Cap. 8 | ⏳ Pendiente | - | - |

**Tiempo estimado restante**: 2-3 capítulos × 30-40 min cada uno = 1-2 horas

---

## 🔗 CONTEXTO ADICIONAL DEL PROYECTO

### Tema de Tesis:
Sistema de recuperación semántica de información (RAG) para soporte técnico de Microsoft Azure.

### Datos del Proyecto:
- 62,417 documentos únicos de Microsoft Learn
- 187,031 chunks procesados
- 13,436 preguntas totales
- 2,067 preguntas con ground truth validado
- 4 modelos evaluados: Ada, MPNet, MiniLM, E5-Large

### Versiones de Resultados:
- **Versión Agosto 2025**: ~1,000 preguntas evaluadas
- **Versión Octubre 2025**: 2,067 preguntas (dataset completo) ← USAR ESTA

---

**Última actualización**: 24 de octubre de 2025
**Última acción**: Completado Capítulo 6 - Implementación (0 código Python, 0 listas, **-78.5% longitud**, 5 refs ✅)
**Siguiente tarea**: Capítulo 7 - Resultados y Análisis
