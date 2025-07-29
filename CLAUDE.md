# CLAUDE.md - Análisis y Directrices del Proyecto

## 📋 DIRECTRICES PRINCIPALES

### ⚠️ IMPORTANTE - Contracto de Datos
- **NO cambiar el archivo de resultados** generado en el Colab
- El archivo de resultados se usa en la app de Streamlit - es un CONTRATO
- **SIEMPRE usar solo métricas reales**, no aleatorias, simuladas o inventadas
- Si se necesita crear métricas simuladas por alguna razón, debe estar EXPLÍCITO en la app de Streamlit

### 🗂️ Estructura del Proyecto
- **ChromaDB Principal**: `/Users/haroldgomez/chromadb2`
- **Archivos Parquet**: `colab_data/docs_[modelo]_with_embeddings_*.parquet`
- **Archivos de Resultados**: `cumulative_results_*.json`
- **Datos de Entrenamiento**: `data/train_set.json`, `data/val_set.json`

## 🔍 ANÁLISIS ACTUAL - PROBLEMA DE MÉTRICAS CERO

### 📊 Estado de las Colecciones ChromaDB
**Ubicación**: `/Users/haroldgomez/chromadb2`

#### Preguntas:
- `questions_ada`: 13,436 preguntas ✅
- `questions_mpnet`: 13,436 preguntas ✅
- `questions_minilm`: 13,436 preguntas ✅
- `questions_e5large`: 13,436 preguntas ✅
- **`questions_withlinks`: 2,067 preguntas** ✅ (CON LINKS VALIDADOS)

#### Documentos:
- `docs_ada`: 187,031 documentos ✅
- `docs_mpnet`: 187,031 documentos ✅
- `docs_minilm`: 187,031 documentos ✅
- `docs_e5large`: 187,031 documentos ✅

### 🚨 CAUSA RAÍZ DEL PROBLEMA

**PROBLEMA IDENTIFICADO**: Las métricas están en cero NO por falta de datos, sino por **BAJA CALIDAD DE EMBEDDINGS**

#### Evidencia del Debug:
```
Pregunta: "Azure disk encryption with Platform Managed Keys..."
Documento relevante: "Server-side encryption of Azure Disk Storage"

❌ PROBLEMA:
- Documentos irrelevantes: Ranks 1-8 (scores 0.85-0.50)
- Documentos RELEVANTES: Ranks 9-10 (scores 0.45-0.40)

📊 Métricas resultantes:
- Precision@5: 0.000 (ningún relevante en top-5)
- Precision@10: 0.200 (2 relevantes en top-10 pero mal rankeados)
```

#### Cobertura de Datos:
- **✅ 68.2% de cobertura** entre preguntas y documentos
- **✅ 4,138 preguntas tienen ground truth válido**
- **✅ URLs normalizadas funcionan correctamente**
- **❌ Los embeddings no capturan semántica técnica correctamente**

### 📈 Performance por Modelo:
- **Ada**: Métricas promedio > 0 (encuentra algunos relevantes)
- **MPNet**: Métricas promedio > 0 (encuentra algunos relevantes)
- **E5-large**: Todas las métricas = 0 (no encuentra relevantes)
- **MiniLM**: Métricas muy bajas

## 🔧 SOLUCIONES RECOMENDADAS

### 1. 🎯 MODELOS ESPECIALIZADOS
**Reemplazar modelos actuales por:**

#### Premium (Mejores resultados):
- `text-embedding-3-large` (OpenAI) - 3072D
- `text-embedding-3-small` (OpenAI) - 1536D

#### Open Source Especializados:
- `sentence-transformers/multi-qa-mpnet-base-dot-v1` - Q&A especializado
- `microsoft/codebert-base` - Contenido técnico
- `sentence-transformers/all-mpnet-base-v2` - Documentos largos

#### Multilingües:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `intfloat/multilingual-e5-large` (versión mejorada)

### 2. 🎓 FINE-TUNING EN DOMINIO MICROSOFT LEARN
- Usar los 2,067 pares de `questions_withlinks`
- Contrastive Learning con pares positivos/negativos
- Herramientas: `sentence-transformers`, OpenAI Fine-tuning API

### 3. 🔄 HYBRID SEARCH (EMBEDDINGS + KEYWORD)
```python
def hybrid_search(query, top_k=10):
    semantic_results = embedding_search(query, top_k=20)
    keyword_results = bm25_search(query, top_k=20)
    return combine_scores(semantic_results, keyword_results, weights=[0.7, 0.3])
```

### 4. 🧠 RERANKING AVANZADO
- CrossEncoder más grande: `ms-marco-MiniLM-L-12-v2`
- Pipeline multi-etapa: Retrieval → Rerank1 → Rerank2
- Query expansion y pseudo relevance feedback

## 📅 PLAN DE IMPLEMENTACIÓN

### Fase 1 (Rápida - 1-2 días):
1. ✅ Cambiar a `multi-qa-mpnet-base-dot-v1`
2. ✅ Implementar CrossEncoder más grande

### Fase 2 (Media - 1 semana):
3. ⏳ Implementar hybrid search con BM25
4. ⏳ Optimizar pesos de combinación

### Fase 3 (Avanzada - 2-3 semanas):
5. ⏳ Fine-tuning con datos de Microsoft Learn
6. ⏳ Pipeline de reranking multi-etapa

## 🔄 ÚLTIMAS ACTUALIZACIONES

### Colab - Versión 3.5 (Min-Max Normalization):
- ✅ Corregido modelo BERTScore: `distiluse-base-multilingual-cased-v2`
- ✅ Agregada normalización URL completa
- ✅ Expandidas métricas RAG (6 RAGAS + 3 BERTScore)
- ✅ Aplicada normalización Min-Max a CrossEncoder
- ✅ Metodología actualizada en Streamlit

### Streamlit Actualizado:
- ✅ Lógica de scoring statistics adaptada a datos disponibles
- ✅ Tabla dinámica sin columnas N/A
- ✅ Metodología de evaluación actualizada para CrossEncoder Min-Max

## 🎯 PRÓXIMOS PASOS RECOMENDADOS

1. **Implementar modelo especializado**: Cambiar a `multi-qa-mpnet-base-dot-v1`
2. **Mejorar CrossEncoder**: Usar modelo más grande
3. **Validar mejoras**: Ejecutar evaluación y comparar métricas
4. **Considerar hybrid search**: Si las mejoras no son suficientes

## 📝 NOTAS TÉCNICAS

### Normalización URL:
```python
def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
```

### CrossEncoder Normalization:
- **Anterior**: Sigmoid
- **Actual**: Min-Max para scores más interpretables
- **Resultado**: Scores en rango [0,1] más altos

### Archivos de Debug Creados:
- `debug_metrics_zero.py` - Análisis paso a paso
- `verify_questions_links_match.py` - Verificación de cobertura
- `check_chromadb_collections.py` - Estado de colecciones

---

**Fecha de último análisis**: 2025-07-29
**Estado**: Problema identificado, soluciones definidas, listo para implementación