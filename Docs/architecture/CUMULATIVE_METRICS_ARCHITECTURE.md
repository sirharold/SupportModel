# 🏗️ Arquitectura del Sistema de Métricas Acumulativas

## Visión General

El sistema de Métricas Acumulativas es una solución distribuida que combina procesamiento local (Streamlit) con cómputo en la nube (Google Colab) para evaluar sistemas RAG a gran escala.

## Componentes Principales

### 1. **Frontend - Streamlit UI** 📱

#### **Configuración** (`cumulative_n_questions_config.py`)
```python
# Funcionalidades principales:
- Filtrado inteligente de preguntas con links válidos
- Selección de modelos de embedding (mpnet, minilm, ada, e5-large)
- Configuración de parámetros de evaluación
- Generación de archivo JSON de configuración
- Upload automático a Google Drive
```

#### **Visualización de Resultados** (`cumulative_metrics_results.py`)
```python
# Características:
- Carga de resultados desde Google Drive
- Visualización con color-coding automático
- Tablas interactivas con definiciones
- Gráficos comparativos multi-modelo
- Exportación de datos en múltiples formatos
```

### 2. **Backend - Google Colab** 🚀

#### **Notebook Principal** (`Colab_Modular_Embeddings_Evaluation.ipynb`)
```python
# Procesamiento con GPU:
1. Carga de configuración desde Drive
2. Inicialización de modelos:
   - Embeddings: SentenceTransformer
   - Reranking: CrossEncoder MS-MARCO
   - Generación: TinyLlama 1.1B
   - Evaluación: RAGAS + BERTScore
3. Evaluación por lotes
4. Guardado incremental de resultados
```

#### **Biblioteca de Evaluación** (`colab_data/lib/rag_evaluation.py`)
```python
# Clases principales:
- RealLLMReranker: Reranking con CrossEncoder
- RealRAGCalculator: Cálculo de métricas RAGAS
- RealBERTScoreEvaluator: Evaluación BERTScore
- DocumentAggregator: Chunks → Documentos
```

### 3. **Almacenamiento - Google Drive** 💾

```
/SupportModel_Results/
├── configs/
│   ├── n_questions_config_*.json
│   └── evaluation_status.json
├── results/
│   ├── n_questions_results_*.json
│   └── results_summary_*.csv
└── logs/
    └── evaluation_*.log
```

## Flujo de Datos Detallado

### Fase 1: Configuración y Filtrado

```python
# 1. Cargar colecciones de ChromaDB
questions_collection = client.get_collection("questions_ada")
docs_collection = client.get_collection("docs_ada")

# 2. Filtrar preguntas con links válidos
def filter_questions_with_valid_links():
    # Obtener todos los links de documentos
    doc_links = normalize_and_collect_doc_links()
    
    # Filtrar preguntas
    valid_questions = []
    for question in all_questions:
        if has_valid_link_in_docs(question, doc_links):
            valid_questions.append(question)
    
    return valid_questions  # ~2,067 de ~15,000

# 3. Crear configuración
config = {
    "evaluation_type": "n_questions_cumulative_analysis",
    "data_config": {
        "num_questions": 500,
        "top_k": 10,
        "use_reranking": True,
        "use_document_aggregation": True,
        "chunk_multiplier": 3
    },
    "model_config": {
        "embedding_models": ["mpnet", "minilm", "ada", "e5-large"],
        "generative_model": "tinyllama-1.1b"
    },
    "metrics_config": {
        "calculate_traditional_metrics": True,
        "calculate_rag_metrics": True,
        "calculate_llm_quality": True
    },
    "content_limits": {
        "answer_generation": 2000,
        "context_for_ragas": 3000,
        "llm_reranking": 4000,
        "bert_score": "sin_limite"
    },
    "questions_data": filtered_questions
}
```

### Fase 2: Procesamiento en Colab

```python
# 1. Inicialización
config = load_config_from_drive()
models = initialize_models_with_gpu()

# 2. Procesamiento por modelo
for model_name in config['model_config']['embedding_models']:
    model = load_embedding_model(model_name)
    
    for question in config['questions_data']:
        # a. Recuperación
        chunks = retrieve_chunks(question, model, top_k=30)
        
        # b. Agregación
        if config['use_document_aggregation']:
            documents = aggregate_chunks_to_documents(chunks)
        
        # c. Reranking
        if config['use_reranking']:
            documents = rerank_with_crossencoder(question, documents)
        
        # d. Generación
        answer = generate_answer(question, documents[:top_k])
        
        # e. Evaluación
        metrics = {
            'traditional': calculate_ir_metrics(question, documents),
            'ragas': calculate_ragas_metrics(question, answer, documents),
            'bertscore': calculate_bertscore(answer, reference_answer)
        }
        
        # f. Guardar resultado
        save_incremental_result(question_id, model_name, metrics)
```

### Fase 3: Visualización de Resultados

```python
# 1. Cargar y procesar resultados
results = load_results_from_drive()
processed_results = process_results_for_display(results)

# 2. Mostrar resumen
display_evaluation_summary(processed_results)
# - Modelos evaluados: 4
# - Preguntas procesadas: 500
# - Tiempo total: 45.2m
# - GPU utilizada: ✅

# 3. Visualización con color-coding
display_enhanced_models_comparison(processed_results)
# - Tabla RAGAS/BERTScore con colores
# - Gráficos comparativos
# - Análisis automático

# 4. Metodología
display_methodology_section()
# - Definiciones de métricas
# - Proceso de evaluación
# - Garantías científicas
```

## Métricas Calculadas

### Métricas IR Tradicionales
```python
def calculate_ir_metrics(question, retrieved_docs):
    ground_truth = extract_links_from_answer(question['accepted_answer'])
    
    metrics = {}
    for k in range(1, 11):  # k = 1 to 10
        docs_at_k = retrieved_docs[:k]
        relevant_at_k = count_relevant_docs(docs_at_k, ground_truth)
        
        metrics[f'precision@{k}'] = relevant_at_k / k
        metrics[f'recall@{k}'] = relevant_at_k / len(ground_truth)
        metrics[f'f1@{k}'] = harmonic_mean(precision, recall)
    
    metrics['map'] = calculate_map(retrieved_docs, ground_truth)
    metrics['mrr'] = calculate_mrr(retrieved_docs, ground_truth)
    metrics['ndcg'] = calculate_ndcg(retrieved_docs, ground_truth)
    
    return metrics
```

### Métricas RAGAS
```python
def calculate_ragas_metrics(question, answer, contexts):
    dataset = Dataset.from_dict({
        'question': [question['content']],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [question['accepted_answer']]
    })
    
    metrics = ragas.evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
            answer_similarity
        ]
    )
    
    return metrics.to_dict()
```

### Métricas BERTScore
```python
def calculate_bertscore(generated_answer, reference_answer):
    P, R, F1 = bert_score(
        [generated_answer],
        [reference_answer],
        lang='en',
        model_type='microsoft/deberta-xlarge-mnli',
        use_fast_tokenizer=True
    )
    
    return {
        'bert_precision': P.item(),
        'bert_recall': R.item(),
        'bert_f1': F1.item()
    }
```

## Optimizaciones Implementadas

### 1. **Agregación de Documentos**
```python
class DocumentAggregator:
    def aggregate_chunks_to_documents(self, chunks, multiplier=3):
        # Agrupar chunks por documento original
        doc_groups = defaultdict(list)
        for chunk in chunks:
            doc_link = chunk['metadata']['link']
            doc_groups[doc_link].append(chunk)
        
        # Combinar chunks de cada documento
        documents = []
        for link, chunk_list in doc_groups.items():
            combined_content = ' '.join([c['content'] for c in chunk_list])
            documents.append({
                'content': combined_content,
                'metadata': {
                    'link': link,
                    'title': chunk_list[0]['metadata']['title'],
                    'chunk_count': len(chunk_list)
                }
            })
        
        # Retornar top documentos
        return documents[:int(len(chunks) / multiplier)]
```

### 2. **Límites de Contenido Dinámicos**
```python
CONTENT_LIMITS = {
    'answer_generation': 2000,      # Para generar respuestas
    'context_for_ragas': 3000,      # Para evaluación RAGAS
    'llm_reranking': 4000,          # Para reranking
    'bert_score': float('inf')      # Sin límite
}

def truncate_content(text, purpose):
    limit = CONTENT_LIMITS.get(purpose, 1000)
    if limit == float('inf'):
        return text
    return text[:limit]
```

### 3. **Procesamiento por Lotes**
```python
def process_questions_batch(questions, model, batch_size=10):
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        # Procesar embeddings en paralelo
        embeddings = model.encode(
            [q['content'] for q in batch],
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # Evaluar cada pregunta
        for q, emb in zip(batch, embeddings):
            result = evaluate_single_question(q, emb, model)
            results.append(result)
            
        # Guardar progreso incremental
        save_intermediate_results(results)
    
    return results
```

### 4. **Color-Coding Inteligente**
```python
def get_metric_color(value, metric_type='ragas'):
    """Retorna color basado en rangos de interpretación"""
    if pd.isna(value):
        return 'background-color: #f0f0f0'  # Gris para NA
    
    # Rangos universales para RAGAS y BERTScore
    if value > 0.8:
        return 'background-color: #d4edda; color: #155724'  # Verde
    elif value >= 0.6:
        return 'background-color: #fff3cd; color: #856404'  # Amarillo
    else:
        return 'background-color: #f8d7da; color: #721c24'  # Rojo
```

## Manejo de Errores y Resiliencia

### Reintentos Automáticos
```python
@retry(max_attempts=3, backoff_factor=2)
def robust_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise
```

### Guardado Incremental
```python
def save_checkpoint(results, checkpoint_id):
    """Guarda progreso cada N preguntas"""
    checkpoint_path = f"/content/drive/MyDrive/checkpoints/{checkpoint_id}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f)
    logger.info(f"Checkpoint saved: {len(results)} results")
```

### Recuperación de Fallos
```python
def resume_from_checkpoint(config):
    """Continúa desde el último checkpoint"""
    checkpoint = find_latest_checkpoint(config['id'])
    if checkpoint:
        processed_ids = set(checkpoint['processed_questions'])
        remaining = [q for q in config['questions'] 
                    if q['id'] not in processed_ids]
        logger.info(f"Resuming from checkpoint: {len(remaining)} questions left")
        return remaining
    return config['questions']
```

## Monitoreo y Logging

```python
# Logging estructurado
logger.info("Evaluation progress", extra={
    'model': model_name,
    'question_id': question['id'],
    'metrics': {
        'mrr': metrics['mrr'],
        'recall@5': metrics['recall@5'],
        'ragas_faithfulness': metrics['faithfulness']
    },
    'timing': {
        'retrieval_ms': retrieval_time,
        'generation_ms': generation_time,
        'evaluation_ms': evaluation_time
    }
})

# Métricas de rendimiento
performance_metrics = {
    'total_time_seconds': total_time,
    'questions_per_minute': num_questions / (total_time / 60),
    'gpu_memory_used_gb': torch.cuda.memory_allocated() / 1e9,
    'average_time_per_question': total_time / num_questions
}
```

## Consideraciones de Escalabilidad

### Límites del Sistema
- **ChromaDB**: ~200K documentos, ~15K preguntas
- **Colab GPU**: 12-16GB RAM, timeout 12 horas
- **Google Drive**: 15GB gratis, ilimitado con workspace

### Recomendaciones por Escala
- **Pequeña (10-100 preguntas)**: Ejecución directa
- **Mediana (100-500 preguntas)**: Usar checkpoints
- **Grande (500-2000 preguntas)**: Dividir en múltiples sesiones
- **Muy Grande (2000+ preguntas)**: Considerar solución distribuida

---

**Última actualización**: Diciembre 2024
**Versión**: 2.0