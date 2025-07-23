# 📦 Colab Data - Real Embeddings

Esta carpeta contiene los archivos de embeddings reales exportados desde ChromaDB para usar en Google Colab.

## 📁 Archivos Disponibles

| Archivo | Tamaño | Modelo | Dimensiones | Documentos |
|---------|--------|--------|-------------|------------|
| `docs_ada_with_embeddings_20250721_123712.parquet` | 2.2 GB | Ada (OpenAI) | 1536 | 187,031 |
| `docs_e5large_with_embeddings_20250721_124918.parquet` | 1.7 GB | E5-Large-v2 | 1024 | 187,031 |
| `docs_mpnet_with_embeddings_20250721_125254.parquet` | 1.4 GB | MPNet | 768 | 187,031 |
| `docs_minilm_with_embeddings_20250721_125846.parquet` | 1.0 GB | MiniLM-L6-v2 | 384 | 187,031 |

**Total: 6.3 GB de embeddings reales**

## 🚀 Para usar en Google Colab

### 1. Subir a Google Drive
```
Sube estos archivos a:
/content/drive/MyDrive/RAG_Evaluation/
```

### 2. Usar en Notebook
```python
# En Colab, los archivos estarán en:
BASE_PATH = '/content/drive/MyDrive/RAG_Evaluation/'

EMBEDDING_FILES = {
    'ada': BASE_PATH + 'docs_ada_with_embeddings_20250721_123712.parquet',
    'e5-large': BASE_PATH + 'docs_e5large_with_embeddings_20250721_124918.parquet', 
    'mpnet': BASE_PATH + 'docs_mpnet_with_embeddings_20250721_125254.parquet',
    'minilm': BASE_PATH + 'docs_minilm_with_embeddings_20250721_125846.parquet'
}
```

### 3. Cargar y Usar
```python
# Cargar embeddings reales
retriever = RealEmbeddingRetriever(EMBEDDING_FILES['e5-large'])

# Evaluar con métricas reales
query_embedding = model.encode("How to configure Azure VPN?")
results = retriever.search_documents(query_embedding, top_k=10)
```

## 📊 Contenido de cada archivo

Cada archivo `.parquet` contiene:
- **document**: Contenido del documento
- **embedding**: Vector embedding real (List[float])
- **link**: URL de Microsoft Learn
- **title**: Título del documento  
- **summary**: Resumen del documento
- **content**: Contenido completo
- **chunk_index**: Índice de fragmento

## 🎯 Características

✅ **Embeddings reales** - No simulación
✅ **187,031 documentos** reales de Microsoft Learn
✅ **4 modelos diferentes** para comparación
✅ **Cálculo de coseno preciso** con sklearn
✅ **Métricas exactas** - Precision@k, Recall@k, F1@k, MRR
✅ **GPU optimizado** para Colab

## 📋 Next Steps

1. **Subir archivos** a Google Drive
2. **Abrir notebook**: `../e5_migration_archive/Colab_Real_Embeddings_Evaluation.ipynb`  
3. **Ejecutar evaluación** con embeddings reales
4. **Comparar modelos** usando métricas precisas

---

🎉 **Ya no más simulación - solo embeddings reales y métricas precisas!**