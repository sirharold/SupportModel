# 📋 Sugerencias Detalladas para Mejorar el Sistema de Retrieval - 29/07/2025

## 🎯 CONTEXTO DEL PROBLEMA

### Situación Actual:
- **Problema**: Métricas de retrieval muy bajas o cero para la mayoría de modelos
- **Causa Identificada**: Baja calidad de embeddings para el dominio técnico Microsoft Learn
- **Evidencia**: Documentos relevantes aparecen en ranks 9-10 en lugar de top-5
- **Cobertura de Datos**: ✅ 68.2% - Los datos SÍ existen, el problema es la recuperación

## 🔧 SUGERENCIAS DETALLADAS

---

## 1. 🎯 CAMBIO DE MODELOS DE EMBEDDING

### 🥇 **OPCIÓN A: OpenAI Embeddings (Premium)**

#### Modelos Recomendados:
- **`text-embedding-3-large`**: 3072 dimensiones
- **`text-embedding-3-small`**: 1536 dimensiones

#### ✅ **Ventajas:**
- **Calidad Superior**: Entrenados en documentación técnica masiva
- **Dominio Específico**: Mejor comprensión de contextos Microsoft/Azure
- **Soporte Multilingüe**: Maneja español/inglés mezclado
- **Resultados Probados**: Estado del arte en benchmarks de retrieval

#### ❌ **Desventajas:**
- **Costo**: ~$0.02 por 1M tokens de entrada
- **Dependencia Externa**: Requiere API key y conexión a internet
- **Límites de Rate**: Posibles restricciones en uso intensivo

#### 💰 **Análisis de Costos:**
```
Estimación para re-embeddings completos:
- 187,031 documentos × ~800 tokens promedio = ~150M tokens
- 13,436 preguntas × ~50 tokens promedio = ~0.7M tokens
- Total: ~150M tokens × $0.02/1M = ~$3.00 USD
- Costo mensual estimado: <$10 USD para evaluaciones regulares
```

#### 🔨 **Implementación:**
```python
from openai import OpenAI

client = OpenAI(api_key='tu-api-key')

def get_openai_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
```

---

### 🥈 **OPCIÓN B: Modelos Open Source Especializados**

#### **B1: Multi-QA MPNet (Recomendado)**
- **Modelo**: `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- **Dimensiones**: 768
- **Especialización**: Optimizado específicamente para Q&A

#### ✅ **Ventajas:**
- **Gratuito**: Sin costos de API
- **Especializado en Q&A**: Entrenado específicamente para pregunta-respuesta
- **Rendimiento Probado**: Mejor que MPNet base para retrieval
- **Control Total**: Sin dependencias externas

#### ❌ **Desventajas:**
- **Calidad Menor**: No tan bueno como OpenAI embeddings
- **Recursos Computacionales**: Requiere más GPU/CPU para generar embeddings

#### 🔨 **Implementación:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def get_qa_embedding(text):
    return model.encode(text)
```

#### **B2: Microsoft CodeBERT**
- **Modelo**: `microsoft/codebert-base`
- **Especialización**: Contenido técnico y documentación

#### **B3: Multilingual MPNet**
- **Modelo**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Especialización**: Mejor soporte multilingüe

---

### 📊 **COMPARACIÓN DE OPCIONES:**

| Aspecto | OpenAI 3-large | Multi-QA MPNet | MPNet Actual |
|---------|----------------|-----------------|--------------|
| **Calidad Esperada** | 9/10 | 7/10 | 5/10 |
| **Costo** | $3-10/mes | Gratis | Gratis |
| **Especialización Q&A** | 9/10 | 10/10 | 6/10 |
| **Facilidad Implementación** | 8/10 | 9/10 | 10/10 |
| **Soporte Multilingüe** | 9/10 | 6/10 | 8/10 |

### 🎯 **RECOMENDACIÓN:**
**Comenzar con Multi-QA MPNet** (gratis, especializado) y **evaluar si necesitamos OpenAI** para mayor calidad.

---

## 2. 🎓 FINE-TUNING DE EMBEDDINGS

### **ESTRATEGIA: Contrastive Learning**

#### 📊 **Datos Disponibles:**
- **2,067 preguntas** en `questions_withlinks` con ground truth validado
- **Pares positivos**: (pregunta, documento_relevante) 
- **Pares negativos**: (pregunta, documento_irrelevante)

#### 🔨 **Implementación con Sentence-Transformers:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Preparar datos de entrenamiento
train_examples = []

for question_data in questions_withlinks:
    question = question_data['question']
    positive_docs = get_relevant_documents(question_data['validated_links'])
    negative_docs = get_random_irrelevant_documents(sample_size=3)
    
    # Crear pares positivos
    for pos_doc in positive_docs:
        train_examples.append(InputExample(texts=[question, pos_doc], label=1.0))
    
    # Crear pares negativos
    for neg_doc in negative_docs:
        train_examples.append(InputExample(texts=[question, neg_doc], label=0.0))

# 2. Configurar entrenamiento
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 3. Entrenar
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
```

#### ⏱️ **Tiempo Estimado:**
- **Preparación de datos**: 1-2 días
- **Entrenamiento**: 4-8 horas en GPU
- **Evaluación**: 1 día

#### ✅ **Ventajas:**
- **Especialización Específica**: Adaptado a nuestro dominio exacto
- **Datos Reales**: Usa pares pregunta-documento reales
- **Control Total**: Podemos iterar y mejorar

#### ❌ **Desventajas:**
- **Complejidad**: Requiere experiencia en ML
- **Tiempo**: Proceso largo y iterativo
- **Recursos**: Necesita GPU para entrenamiento

---

## 3. 🔄 HYBRID SEARCH (EMBEDDING + KEYWORD)

### **ESTRATEGIA: Combinar Búsqueda Semántica y Textual**

#### 🔨 **Implementación con BM25:**

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearcher:
    def __init__(self, documents, embedding_model):
        self.embedding_model = embedding_model
        self.documents = documents
        
        # Preparar BM25
        tokenized_docs = [doc['content'].split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Pre-calcular embeddings
        self.doc_embeddings = [
            embedding_model.encode(doc['content']) 
            for doc in documents
        ]
    
    def search(self, query, top_k=10, semantic_weight=0.7, keyword_weight=0.3):
        # 1. Búsqueda semántica
        query_embedding = self.embedding_model.encode(query)
        semantic_scores = cosine_similarity(
            [query_embedding], 
            self.doc_embeddings
        )[0]
        
        # 2. Búsqueda por keywords
        tokenized_query = query.split()
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. Normalizar scores
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
        keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min())
        
        # 4. Combinar scores
        final_scores = (semantic_weight * semantic_scores + 
                       keyword_weight * keyword_scores)
        
        # 5. Ranking final
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        return [
            {
                'document': self.documents[idx],
                'score': final_scores[idx],
                'semantic_score': semantic_scores[idx],
                'keyword_score': keyword_scores[idx]
            }
            for idx in top_indices
        ]
```

#### 📊 **Optimización de Pesos:**

```python
# Evaluar diferentes combinaciones de pesos
weight_combinations = [
    (0.9, 0.1),  # Muy semántico
    (0.7, 0.3),  # Balanceado hacia semántico
    (0.5, 0.5),  # Balanceado
    (0.3, 0.7),  # Balanceado hacia keywords
    (0.1, 0.9),  # Muy keyword-based
]

best_weights = None
best_score = 0

for sem_w, key_w in weight_combinations:
    metrics = evaluate_hybrid_search(sem_w, key_w)
    if metrics['f1@5'] > best_score:
        best_score = metrics['f1@5']
        best_weights = (sem_w, key_w)
```

#### ✅ **Ventajas:**
- **Mejor Cobertura**: Captura tanto semántica como términos exactos
- **Robustez**: Funciona bien con consultas técnicas específicas
- **Flexibilidad**: Pesos ajustables según el caso de uso

#### ❌ **Desventajas:**
- **Complejidad**: Sistema más complejo de mantener
- **Optimización**: Requiere tuning de hiperparámetros
- **Recursos**: Doble cómputo (embedding + BM25)

---

## 4. 🧠 RERANKING AVANZADO

### **ESTRATEGIA: Pipeline Multi-Etapa**

#### **Etapa 1: Retrieval Inicial (Top-50)**
```python
# Usar embedding actual para obtener candidatos
initial_docs = embedding_search(query, top_k=50)
```

#### **Etapa 2: CrossEncoder Mejorado**
```python
from sentence_transformers import CrossEncoder

# Modelo más grande y preciso
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_with_crossencoder(query, docs, top_k=20):
    pairs = [[query, doc['content'][:512]] for doc in docs]
    scores = cross_encoder.predict(pairs)
    
    # Aplicar normalización Min-Max
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    # Reordenar por score
    ranked_docs = sorted(
        zip(docs, normalized_scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [doc for doc, score in ranked_docs[:top_k]]
```

#### **Etapa 3: Query Expansion**
```python
def expand_query(original_query, top_docs, expansion_terms=3):
    # Extraer términos relevantes de top documentos
    relevant_terms = extract_key_terms(top_docs[:3])
    
    # Expandir consulta
    expanded_query = f"{original_query} {' '.join(relevant_terms[:expansion_terms])}"
    
    return expanded_query
```

#### **Pipeline Completo:**
```python
def advanced_retrieval_pipeline(query, top_k=10):
    # Etapa 1: Retrieval inicial amplio
    candidates = embedding_search(query, top_k=50)
    
    # Etapa 2: Primer reranking con CrossEncoder
    reranked_20 = rerank_with_crossencoder(query, candidates, top_k=20)
    
    # Etapa 3: Query expansion y segunda búsqueda
    if len(reranked_20) >= 3:
        expanded_query = expand_query(query, reranked_20)
        expanded_candidates = embedding_search(expanded_query, top_k=30)
        
        # Combinar resultados originales y expandidos
        combined_docs = merge_results(reranked_20, expanded_candidates)
        
        # Etapa 4: Reranking final
        final_results = rerank_with_crossencoder(query, combined_docs, top_k=top_k)
    else:
        final_results = reranked_20[:top_k]
    
    return final_results
```

#### ✅ **Ventajas:**
- **Máxima Precisión**: Múltiples capas de refinamiento
- **Adaptabilidad**: Cada etapa puede optimizarse independientemente
- **Robustez**: Múltiples oportunidades de recuperación

#### ❌ **Desventajas:**
- **Latencia**: Proceso más lento por múltiples etapas
- **Complejidad**: Sistema muy complejo de mantener
- **Recursos**: Alto uso computacional

---

## 📅 PLAN DE IMPLEMENTACIÓN RECOMENDADO

### **🎯 FASE 1: Mejoras Rápidas (1-2 días)**

#### Prioridad 1: Cambio de Modelo
```bash
# Instalar modelo especializado
pip install sentence-transformers

# Modificar código para usar multi-qa-mpnet-base-dot-v1
# Re-generar embeddings para documentos y preguntas
```

#### Prioridad 2: CrossEncoder Mejorado
```bash
# Cambiar a modelo más grande
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
```

**Tiempo**: 1-2 días
**Esfuerzo**: Bajo
**Impacto Esperado**: +20-40% en métricas

---

### **🎯 FASE 2: Implementación Híbrida (1 semana)**

#### Implementar Hybrid Search
1. Integrar BM25 con embeddings actuales
2. Optimizar pesos con grid search
3. Evaluar mejoras en subset de datos

**Tiempo**: 3-5 días
**Esfuerzo**: Medio
**Impacto Esperado**: +15-30% adicional

---

### **🎯 FASE 3: Fine-tuning (2-3 semanas)**

#### Entrenar Modelo Especializado
1. Preparar datasets de entrenamiento
2. Fine-tune modelo con contrastive learning
3. Evaluar y iterar

**Tiempo**: 2-3 semanas
**Esfuerzo**: Alto
**Impacto Esperado**: +30-50% adicional

---

## 💰 ANÁLISIS COSTO-BENEFICIO

### **Opción Conservadora (Recomendada):**
- **Fase 1**: Multi-QA MPNet + CrossEncoder mejorado
- **Costo**: $0 (solo tiempo de desarrollo)
- **Riesgo**: Bajo
- **Impacto Esperado**: 40-60% mejora en métricas

### **Opción Premium:**
- **Todas las fases** + OpenAI embeddings
- **Costo**: ~$10-20/mes operacional
- **Riesgo**: Medio (dependencia externa)
- **Impacto Esperado**: 70-90% mejora en métricas

### **Opción Experimental:**
- **Fine-tuning completo** + pipeline avanzado
- **Costo**: 2-3 semanas desarrollo + recursos GPU
- **Riesgo**: Alto (complejidad)
- **Impacto Esperado**: 80-95% mejora en métricas

---

## 🎯 RECOMENDACIÓN FINAL

### **ESTRATEGIA RECOMENDADA:**

1. **Iniciar con Fase 1** (cambio de modelo + CrossEncoder)
2. **Evaluar resultados** después de 2 días
3. **Si mejoras < 50%**: Proceder con Fase 2 (hybrid search)
4. **Si mejoras < 70%**: Considerar OpenAI embeddings
5. **Si mejoras < 80%**: Evaluar fine-tuning (Fase 3)

### **Criterios de Decisión:**
- **Meta Mínima**: Precision@5 > 0.3 (vs actual ~0.05)
- **Meta Objetivo**: Precision@5 > 0.5 
- **Meta Ideal**: Precision@5 > 0.7

### **Plan de Evaluación:**
```python
# Métricas clave a monitorear
key_metrics = [
    'precision@5',
    'recall@5', 
    'f1@5',
    'ndcg@10',
    'mrr'
]

# Evaluación en subset de 100 preguntas para iteración rápida
# Evaluación completa en 2,067 preguntas para validación final
```

---

**Preparado por**: Claude Code Assistant  
**Fecha**: 29 de Julio, 2025  
**Basado en**: Análisis detallado de métricas cero en sistema de retrieval Microsoft Learn  
**Próximo paso recomendado**: Implementar Fase 1 (cambio a Multi-QA MPNet)