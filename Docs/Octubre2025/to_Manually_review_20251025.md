# REVISIÓN DE RIGOR CIENTÍFICO - CAPÍTULOS 1-6
**Fecha:** 25 de octubre de 2025
**Documento:** Análisis de rigurosidad científica en tesis humanizada

---

## 📊 **TABLA DE REVISIÓN POR CAPÍTULO**

---

## **CAPÍTULO 1 - INTRODUCCIÓN**

| Línea | Problema/Observación | Severidad | Acción Recomendada |
|-------|---------------------|-----------|-------------------|
| - | Sin problemas de rigor identificados | - | - |

---

## **CAPÍTULO 2 - ESTADO DEL ARTE**

| Línea | Problema/Observación | Severidad | Acción Recomendada |
|-------|---------------------|-----------|-------------------|
| - | Sin problemas de rigor identificados | - | - |

---

## **CAPÍTULO 3 - MARCO TEÓRICO**

| Línea | Problema/Observación | Severidad | Acción Recomendada |
|-------|---------------------|-----------|-------------------|
| 51 | "E5-Large... métricas de recuperación de **0.0 en todas las categorías**" pero luego "mostró el mejor rendimiento en métricas de generación RAG como faithfulness (0.5909)" | **ALTA** | Matizar la afirmación. Explicar que 0.0 puede indicar error de implementación o incompatibilidad con el dataset, no limitación intrínseca del modelo. Aclarar la contradicción aparente. |
| 113 | "El proyecto demostró características de rendimiento notables en múltiples dimensiones. El procesamiento de evaluación total alcanzó 774.78 segundos (12.9 minutos)" | MEDIA | Clarificar si estos tiempos son en MacBook local o Google Colab con GPU. Inconsistencia con Cap. 5. |

---

## **CAPÍTULO 4 - ANÁLISIS EXPLORATORIO DE DATOS**

| Línea | Problema/Observación | Severidad | Acción Recomendada |
|-------|---------------------|-----------|-------------------|
| 5 vs 13 | **CONTRADICCIÓN TEMPORAL**: Línea 5 dice "julio y agosto de 2025", Línea 13 dice "marzo de 2025" | **CRÍTICA** | Corregir para que ambas mencionen la misma fecha. Verificar cuál es la correcta. |
| 45 | "se procesó una muestra estratificada de 10,000 chunks representativos, cuyos resultados se extrapolaron al corpus completo... utilizando factores de escalamiento validados" | **ALTA** | Documentar metodología de estratificación. Especificar "factores de escalamiento validados". Agregar intervalos de confianza. |
| 51 | "98,584 chunks (53.6%) clasificados en la categoría Development" | **ALTA** | Esta cifra proviene de extrapolación no documentada. Agregar disclaimer o documentar metodología completa. |
| 69 | "de una población estimada de **más de 65,000 documentos** disponibles en Microsoft Learn, lo que representa aproximadamente un **96% de cobertura**" | **ALTA** | Agregar fuente de esta estimación o eliminar/matizar. ¿Cómo se determinó que MS Learn tiene 65,000+ docs? |
| 79 | "imágenes, diagramas... representan aproximadamente **30-40% del contenido original**" | **ALTA** | Agregar fuente o metodología. ¿Cómo se calculó este porcentaje? Sin respaldo es especulación. |
| 103-105 | "Las preguntas procedurales... representan aproximadamente **45%**... consultas de troubleshooting... **29%**... preguntas conceptuales... **17%**... consultas de configuración... **9%**" | MEDIA | Documentar metodología de clasificación. Especificar tamaño de muestra. Agregar validación inter-anotador o reconocer que son estimaciones aproximadas. |
| 111-115 | "consultas simples... **32%**... consultas moderadas... **52%**... consultas complejas... **16%**" | MEDIA | Misma observación: porcentajes muy precisos sin metodología documentada de conteo de "conceptos técnicos". |
| 163 | "La distribución temporal muestra concentración en 2023-2024 con **77.3%** de las preguntas" | MEDIA | Especificar si esto se calculó sobre 13,436 o 2,067 preguntas. Referenciar script de análisis o agregar a nota metodológica. |
| 175 | "La evaluación cualitativa de una muestra de **100 correspondencias**... **67% son altamente relevantes**" | MEDIA | Muestra pequeña (4.8% de 2,067). Especificar: ¿selección aleatoria? ¿Quién evaluó? ¿Hubo validación inter-anotador? |
| 182 | "La cobertura es **comprehensiva**" | BAJA | Anglicismo. Cambiar a "exhaustiva" o "completa". |
| 213 | "el primer corpus **comprehensivo**" | BAJA | Anglicismo. Cambiar a "exhaustivo" o "completo". |

---

## **CAPÍTULO 5 - METODOLOGÍA**

| Línea | Problema/Observación | Severidad | Acción Recomendada |
|-------|---------------------|-----------|-------------------|
| 113 | "La configuración de hardware se mantuvo constante (Intel Core i7, 16GB RAM)" | MEDIA | **INCONSISTENCIA con línea 181** (MacBook específico) y con Cap. 6 línea 21 (Google Colab GPU). Clarificar qué hardware se usó para qué etapas. |
| 163 | "La distribución temporal muestra concentración en 2023-2024 con 77.3% de las preguntas" | MEDIA | Duplicado de Cap. 4. Misma observación: necesita fuente o script de análisis. |
| 181 | "MacBook Pro 16,1 equipado con procesador Intel Core i7 de 6 núcleos a 2.6 GHz, 16 GB de memoria RAM DDR4" | MEDIA | Especificación muy detallada pero **inconsistente con Cap. 6** que menciona Google Colab con GPU T4. ¿Dónde se hizo cada evaluación? |
| 211 | "La selección de k=15... se estableció tras experimentación iterativa, comenzando con k=50 y reduciendo progresivamente... basándose en análisis de métricas de recuperación" | MEDIA | Se menciona experimentación pero no se documenta. Agregar referencia a experimentos preliminares o anexo con resultados. |
| 279 | "Para 6 comparaciones principales entre modelos, el nivel de significancia ajustado es α_adjusted = 0.05/6 = 0.0083" | INFO | Esto es correcto, pero ¿realmente se hacen solo 6 comparaciones? Con 4 modelos hay 6 pares, pero hay múltiples métricas. Verificar que el cálculo sea apropiado. |

---

## **CAPÍTULO 6 - IMPLEMENTACIÓN**

| Línea | Problema/Observación | Severidad | Acción Recomendada |
|-------|---------------------|-----------|-------------------|
| 21 | "Google Colab con GPU Tesla T4 para aceleración en evaluaciones masivas" | MEDIA | **INCONSISTENCIA con Cap. 5** que menciona MacBook local. Clarificar distribución de trabajo entre local y Colab. |
| 47 | "La distribución temporal muestra concentración en 2023-2024 con 77.3% del total" | MEDIA | Tercera mención de esta cifra. Necesita documentación en un solo lugar. |
| 69 | "Weaviate... latencia de red de **150-300ms** por consulta... ChromaDB proporcionó latencia local **menor a 10ms**" | MEDIA | Métricas de rendimiento sin documentar. ¿Se midieron experimentalmente? ¿En qué condiciones? Agregar referencia a benchmarks o matizar como "estimado". |
| 83 | "latencia promedio de consulta menor a 10ms para top-k=10, throughput de aproximadamente 241 documentos por segundo" | MEDIA | Métricas precisas. ¿Dónde están los logs/benchmarks que las respaldan? Agregar referencia o reconocer como "observado durante desarrollo". |
| 95 | "threshold de diversidad de **0.85**" | MEDIA | ¿Por qué 0.85 específicamente? Falta justificación para este valor umbral. Agregar breve explicación o reconocer como "empíricamente determinado". |

---

## 📊 **RESUMEN CONSOLIDADO**

| Capítulo | Problemas Severidad Alta | Problemas Severidad Media | Problemas Severidad Baja | Total |
|----------|-------------------------|--------------------------|-------------------------|-------|
| Cap. 1 | 0 | 0 | 0 | 0 |
| Cap. 2 | 0 | 0 | 0 | 0 |
| Cap. 3 | 1 | 1 | 0 | 2 |
| Cap. 4 | 5 | 5 | 2 | **12** |
| Cap. 5 | 0 | 5 | 0 | 5 |
| Cap. 6 | 0 | 5 | 0 | 5 |
| **TOTAL** | **6** | **16** | **2** | **24** |

---

## 🎯 **PRIORIDADES DE CORRECCIÓN**

### **Prioridad 1 - CRÍTICA (1 problema):**
1. ✅ **Cap. 4, línea 5 vs 13**: Resolver contradicción temporal (julio-agosto vs marzo)
   - **Verificar en datos originales cuál es la fecha correcta**
   - Unificar en todo el documento

### **Prioridad 2 - ALTA (5 problemas):**
1. ✅ **Cap. 4, línea 45**: Documentar metodología de extrapolación
   - Agregar sección metodológica o nota al pie
   - Especificar cómo se realizó la estratificación
   - Incluir intervalos de confianza

2. ✅ **Cap. 4, línea 69**: Fuente para estimación de 65,000 docs en MS Learn
   - Buscar fuente oficial de Microsoft
   - O cambiar a "estimado mediante análisis del sitemap" u otra metodología documentable
   - O eliminar la cifra y solo mencionar "alta cobertura"

3. ✅ **Cap. 4, línea 79**: Fuente para 30-40% multimedia
   - Documentar cómo se calculó
   - O cambiar a "una porción significativa" sin porcentaje específico

4. ✅ **Cap. 3, línea 51**: Matizar afirmación sobre E5-Large métricas 0.0
   - Explicar que puede ser error de configuración/implementación
   - Reconciliar con el alto rendimiento en faithfulness (0.5909)
   - Sugerir que requiere investigación adicional

5. ✅ **Cap. 4, líneas 103-105, 111-115**: Documentar clasificación de tipos de preguntas
   - Agregar nota metodológica sobre cómo se clasificaron
   - Especificar si fue manual o automático
   - Incluir validación inter-anotador si existe

### **Prioridad 3 - MEDIA (16 problemas):**

#### **Grupo A: Inconsistencias de entorno (4 problemas)**
- Cap. 3, línea 113: Aclarar tiempos de procesamiento
- Cap. 5, línea 113, 181: Especificar hardware usado
- Cap. 6, línea 21: Clarificar uso de Colab vs local
- **Solución única**: Agregar sección en Cap. 5 o 6 explicando distribución de tareas

#### **Grupo B: Parámetros sin justificación (2 problemas)**
- Cap. 5, línea 211: k=15 sin documentar experimentos
- Cap. 6, línea 95: threshold=0.85 sin justificar
- **Solución**: Agregar breve justificación o reconocer como "empírico"

#### **Grupo C: Métricas de rendimiento (3 problemas)**
- Cap. 6, línea 69: Latencias Weaviate vs ChromaDB
- Cap. 6, línea 83: Throughput específico
- **Solución**: Agregar "observado durante desarrollo" o referenciar logs

#### **Grupo D: Datos temporales repetidos (3 problemas)**
- Cap. 4, línea 163: 77.3% preguntas 2023-2024
- Cap. 5, línea 163: Duplicado
- Cap. 6, línea 47: Triplicado
- **Solución**: Documentar una sola vez con metodología, referenciar en otros lugares

#### **Grupo E: Metodología de evaluación (4 problemas)**
- Cap. 4, líneas 103-105, 111-115: Clasificación de preguntas
- Cap. 4, línea 175: Muestra de 100 correspondencias
- Cap. 5, línea 279: Verificar corrección Bonferroni
- **Solución**: Agregar sección metodológica o notas al pie

### **Prioridad 4 - BAJA (2 problemas):**
- Cap. 4, líneas 182, 213: Anglicismos "comprehensivo/a"
- **Solución simple**: Buscar y reemplazar con "exhaustivo/a" o "completo/a"

---

## 📝 **NOTAS ADICIONALES**

### **Observaciones Generales:**
1. Los capítulos 1 y 2 están bien documentados sin problemas de rigor
2. El Capítulo 4 concentra la mayoría de problemas (12 de 24)
3. La mayoría de problemas (16 de 24) son de severidad media y solucionables
4. Solo 1 problema es crítico (contradicción temporal)

### **Recomendación Final:**
Priorizar la corrección de los 6 problemas de alta severidad antes de publicación. Los problemas de severidad media pueden abordarse en una revisión posterior o incluirse como "limitaciones reconocidas" en el documento.

---

**Documento generado:** 2025-10-25
**Analista:** Claude (Sonnet 4.5)
**Capítulos revisados:** 1, 2, 3, 4, 5, 6
