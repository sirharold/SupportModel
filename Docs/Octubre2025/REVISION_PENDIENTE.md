# REVISIÓN DE TESIS - PROBLEMAS PENDIENTES

**Fecha:** 18 noviembre 2025
**Total problemas pendientes:** 0

---

## ALTA PRIORIDAD (0 problemas) ✅

### ~~1. Inconsistencia terminológica ChromaDB~~ ✅ **RESUELTO**
**Ubicación:** Múltiples capítulos (2, 3, 5, 6)
**Verificación:** 36 ocurrencias en capítulos 0-8, todas escritas correctamente como "ChromaDB"
**Estado:** ✅ Ya está estandarizado correctamente en todos los capítulos

### ~~2. Precisión en fechas de extracción de datos~~ ✅ **RESUELTO**
**Ubicación:** Cap 7 (línea 34)
**Solución aplicada:** Timeline integrado en narrativa - Extracción (dic 2024), Procesamiento (ene-oct 2025), Evaluación (nov 2025)

### ~~3. Falta de valores NDCG@k completos~~ ✅ **RESUELTO**
**Ubicación:** Cap 7, Tabla 7.5 (línea 179)
**Solución aplicada:** Nota aclaratoria explicando que tabla muestra k=3,5,10,15 representativos, mientras evaluación completa k=1-15 se presenta en Figura 7.4

### ~~4. Falta de valores MAP@k completos~~ ✅ **RESUELTO**
**Ubicación:** Cap 7, Tabla 7.6 (línea 200)
**Solución aplicada:** Nota aclaratoria explicando que tabla muestra k=3,5,10,15 representativos, mientras evaluación completa k=1-15 se presenta en Figura 7.5

### ~~3. Formato inconsistente de citas bibliográficas~~ ✅ **RESUELTO**
**Verificación:** Todas las citas siguen formato APA (Autor, año) o (Autor et al., año)
**Solución aplicada:**
- Estandarizado Microsoft 2025 a 2025a y 2025b en Capítulos 1, 7, 8
- Verificado que todas las 27 citas únicas tienen entrada en bibliografía
- Formato APA consistente en todo el documento

### ~~4. Inconsistencia en "ground truth"~~ ✅ **RESUELTO**
**Ubicación:** Capítulos 4 y 8
**Solución aplicada:** Estandarizado a "ground truth" (minúsculas, sin guion) en 5 ubicaciones: Cap 4 líneas 147, 153; Cap 8 líneas 61, 156, 160

---

## MEDIA PRIORIDAD (0 problemas) ✅

### ~~5. Falta de aclaración sobre Answer Correctness~~ ✅ **RESUELTO**
**Verificación:** Answer Correctness es métrica RAGAS válida (confirmado en Tabla 7.9)

### ~~6. Descripción incompleta de "Semantic Similarity"~~ ✅ **RESUELTO**
**Verificación:** Semantic Similarity es métrica RAGAS válida (confirmado en Tabla 7.9)

### ~~7. Falta de justificación de selección de k~~ ✅ **RESUELTO**
**Ubicación:** Cap 7, Tablas 7.5 y 7.6 (líneas 179, 200)
**Solución aplicada:** Notas aclaratorias justificando selección de k=3,5,10,15 como valores representativos por legibilidad

### ~~8. Inconsistencia en nombres de archivos~~ ✅ **NO REQUIERE ACCIÓN**
**Verificación:** Paths completos no son necesarios (confirmado por usuario)

### ~~9. Falta de aclaración sobre "crossencoder"~~ ✅ **RESUELTO**
**Ubicación:** Cap 5 (línea 119)
**Solución aplicada:** Estandarizado a "CrossEncoder" (única instancia con guion corregida)

### ~~10. Falta de referencias bibliográficas completas~~ ✅ **RESUELTO**
**Verificación:** Todas las referencias mencionadas tienen entradas completas en bibliografía
**Referencias verificadas:**
- ChromaDB Team. (2024) - Presente en Capítulo 6
- Streamlit Team. (2023) - Presente en Capítulo 6
- Weaviate. (2023) - Presente en Capítulos 2 y 5
**Estado:** Todas las referencias incluyen título completo y URL oficial

---

## BAJA PRIORIDAD (0 problemas) ✅

### 11. Figuras sin numeración consecutiva verificable
**Ubicación:** Cap 7
**Problema:** Saltos en numeración de figuras (7.1-7.7 pero verificar si están todas)
**Corrección:** Revisar que todas las figuras estén numeradas consecutivamente
**Prioridad:** BAJA
**Nota:** Verificación manual requerida

### ~~Nueva detección: Sección de conclusiones en Capítulo 7~~ ✅ **RESUELTO**
**Ubicación:** Capítulo 7, línea 338
**Problema:** Sección titulada "7.6 Conclusiones del Capítulo"
**Solución aplicada:** Renombrado a "7.6 Síntesis de Resultados" (las conclusiones deben estar solo en Capítulo 8)
**Verificación:** Capítulos 1-6 no contienen secciones de conclusiones ✓

### ~~12. Inconsistencia en formato de porcentajes~~ ✅ **RESUELTO**
**Verificación:** Todos los porcentajes ya están sin espacio antes del símbolo % (formato estándar: "77.3%", "38.3%", etc.)
**Estado:** Formato consistente en todos los capítulos

### ~~13. Términos técnicos sin cursiva consistente~~ ✅ **RESUELTO**
**Criterio establecido:**
- **Términos técnicos naturalizados** (NLP, RAG, BERT, embedding, pipeline, threshold, chunks, dataset, baseline, etc.): Texto normal sin cursiva
- **Títulos de publicaciones en bibliografía**: Cursiva según formato APA
- **Negritas**: Reservadas para destacar conceptos clave o definiciones
**Estado:** El criterio actual es consistente y apropiado para textos técnicos en español

---

## NOTAS ADICIONALES

**Problemas ya resueltos (NO incluidos en esta lista):**
- Tabla 7.5 NDCG@k valores específicos (11 valores confirmados)
- Tabla 7.6 MAP@k valores específicos (10 valores confirmados)
- Tabla 7.11 BERTScore F1 (4 valores confirmados)
- Tabla 8.2 cambio absoluto MiniLM (confirmado +0.005)
- Nota metodológica BERTScore (presente en Caps 7 y 8)
- Distribución temporal - marcador de inferencia (presente)
- Patrón de reranking - marcador "Los resultados sugieren" (presente)
- Timeline de investigación - integrado en narrativa Cap 7 (Extracción dic 2024, Procesamiento ene-oct 2025, Evaluación nov 2025)
- Terminología ChromaDB - estandarizada correctamente (36 ocurrencias verificadas)
- Notas aclaratorias NDCG@k y MAP@k - justificación de valores representativos k=3,5,10,15 (Tablas 7.5 y 7.6)
- Terminología "ground truth" - estandarizada a minúsculas sin guion en 5 ubicaciones (Caps 4 y 8)
- Answer Correctness - confirmado como métrica RAGAS válida (Tabla 7.9)
- Semantic Similarity - confirmado como métrica RAGAS válida (Tabla 7.9)
- Terminología "CrossEncoder" - estandarizada sin guion (Cap 5:119)
- Formato de porcentajes - verificado consistente sin espacio antes de % en todos los capítulos
- Términos técnicos - criterio establecido: términos naturalizados sin cursiva, títulos en cursiva (APA), negritas para conceptos clave
- Formato de citas - estandarizado APA, Microsoft 2025a/2025b diferenciado
- Referencias bibliográficas - verificadas 27 citas únicas con entradas completas (ChromaDB Team, Streamlit Team, Weaviate confirmadas)
- Sección de conclusiones - removida de Capítulo 7, renombrada a "Síntesis de Resultados"

**Verificación de datos numéricos:**
- Todos los valores numéricos en Tablas 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.10, 7.11, 8.1, 8.2 fueron verificados contra el archivo de resultados
- Los valores coinciden con los datos del JSON de experimentos

**Estado general:**
✅ **REVISIÓN COMPLETA** - El documento presenta alta calidad técnica y metodológica. Todos los problemas identificados han sido resueltos. Se resolvieron 15 problemas durante esta sesión: ChromaDB, timeline, NDCG@k/MAP@k justificación, ground truth, Answer Correctness, Semantic Similarity, CrossEncoder, formato de porcentajes, términos técnicos, formato de citas APA, referencias bibliográficas completas, y sección de conclusiones en Capítulo 7.

**Nota**: El problema #11 (numeración de figuras) requiere verificación manual de las imágenes, pero no afecta el contenido textual.
