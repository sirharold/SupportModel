# 📓 Notebook de Evaluación Multi-Modelo - Diagrama de Flujo

## 🎯 **Propósito**
Evaluación completa de múltiples modelos de embeddings usando datos reales de ChromaDB exportados, con métricas de retrieval, LLM reranking y RAG.

---

## 🔄 **Flujo Principal del Notebook**

```mermaid
graph TD
    A[🔧 Setup & Installation] --> B[📂 Mount Google Drive]
    B --> C[🔑 Load API Keys]
    C --> D[📁 Configure File Paths]
    D --> E[📊 Load Configuration]
    E --> F[🎯 Determine Models to Evaluate]
    F --> G[🔄 Model Evaluation Loop]
    G --> H[📈 Calculate Averages]
    H --> I[📊 Display Results Table]
    I --> J[💾 Save Results to JSON]
    
    style A fill:#e1f5fe
    style G fill:#fff3e0
    style I fill:#e8f5e8
    style J fill:#f3e5f5
```

---

## 📦 **Celda 1-3: Configuración Inicial**

### 🔧 **Setup & Installation**
```mermaid
flowchart LR
    A1[📦 Install Packages] --> A2[🔐 HF Authentication] 
    A2 --> A3[📂 Mount Google Drive]
    A3 --> A4[🔑 Load OpenAI API Key]
    A4 --> A5[📁 Configure Paths]
    
    A1 -.-> P1[sentence-transformers<br/>pandas, numpy<br/>scikit-learn<br/>openai]
    A4 -.-> P2[From Colab userdata<br/>or .env file]
    A5 -.-> P3[BASE_PATH: colab_data/<br/>RESULTS_OUTPUT_PATH<br/>CONFIG_FILE]
```

---

## 🏗️ **Celda 4-9: Clases y Funciones Core**

### 📊 **Core Classes Architecture**
```mermaid
classDiagram
    class RealEmbeddingRetriever {
        +parquet_file: str
        +df: DataFrame
        +embeddings_matrix: ndarray
        +num_docs: int
        +embedding_dim: int
        +search_documents(query_embedding, top_k)
    }
    
    class RAGCalculator {
        +client: OpenAI
        +calculate_rag_metrics(question, docs)
        +faithfulness: float
        +answer_relevance: float
    }
    
    class LLMReranker {
        +client: OpenAI  
        +rerank_documents(question, docs, top_k)
        +GPT-3.5-turbo ranking
    }
    
    RealEmbeddingRetriever --> RAGCalculator : retrieved docs
    RealEmbeddingRetriever --> LLMReranker : rerank docs
```

### 🔢 **Metrics Functions Flow**
```mermaid
graph LR
    A[Retrieved Documents] --> B[normalize_link()]
    B --> C[Ground Truth Links]
    C --> D{Calculate Metrics}
    D --> E[Precision@k]
    D --> F[Recall@k] 
    D --> G[F1@k]
    D --> H[NDCG@k]
    D --> I[MAP@k]
    D --> J[MRR]
    
    E --> K[Return Metrics Dict]
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
```

---

## 🎯 **Celda 10-13: Configuración y Mappings**

### 📋 **Configuration Loading**
```mermaid
sequenceDiagram
    participant N as Notebook
    participant F as Config File
    participant V as Variables
    
    N->>F: Load evaluation_config_*.json
    F->>N: questions_data[]
    F->>N: evaluation_params{}
    N->>V: Store configuration
    
    Note over N,V: Parameters: num_questions, selected_models,<br/>generative_model, top_k, use_llm_reranker
```

### 🗺️ **Model Mappings**
```mermaid
graph TD
    A[Model Selection] --> B{Model Type}
    
    B -->|Ada| C["text-embedding-ada-002<br/>🔸 1536 dims<br/>🔸 OpenAI API"]
    B -->|E5-Large| D["intfloat/e5-large-v2<br/>🔸 1024 dims<br/>🔸 HuggingFace"]
    B -->|MPNet| E["multi-qa-mpnet-base-dot-v1<br/>🔸 768 dims<br/>🔸 HuggingFace"]
    B -->|MiniLM| F["all-MiniLM-L6-v2<br/>🔸 384 dims<br/>🔸 HuggingFace"]
    
    C --> G[Query Generation]
    D --> G
    E --> G
    F --> G
    
    style C fill:#ffebee
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#e1f5fe
```

---

## 🔄 **Celda 14: Evaluation Loop (CORAZÓN DEL NOTEBOOK)**

### 🎯 **Main Evaluation Flow**
```mermaid
flowchart TD
    A[Start Model Loop] --> B[Load Model Files]
    B --> C{Test Dimension<br/>Compatibility}
    
    C -->|❌ Mismatch| D[Store Error Result]
    C -->|✅ Compatible| E[Question Processing Loop]
    
    E --> F[Extract title + question_content]
    F --> G[Generate Query Embedding]
    G --> H{Embedding Type}
    
    H -->|OpenAI| I[OpenAI API Call]
    H -->|HuggingFace| J{GPU Available?}
    
    J -->|✅ Yes| K[Load on GPU]
    J -->|❌ CUDA Error| L[CPU Fallback]
    
    I --> M[Document Retrieval]
    K --> M
    L --> M
    
    M --> N[Calculate BEFORE Metrics]
    N --> O{LLM Reranking<br/>Enabled?}
    
    O -->|✅ Yes| P[GPT Reranking]
    O -->|❌ No| Q[Use Original Docs]
    
    P --> R[Calculate AFTER Metrics]
    Q --> S{RAG Metrics<br/>Enabled?}
    R --> S
    
    S -->|✅ Yes| T[Generate RAG Metrics]
    S -->|❌ No| U[Next Question]
    T --> U
    
    U --> V{More Questions?}
    V -->|✅ Yes| E
    V -->|❌ No| W[Calculate Model Averages]
    
    W --> X{More Models?}
    X -->|✅ Yes| A
    X -->|❌ No| Y[End Evaluation]
    
    D --> Z[Cleanup & GC]
    Y --> Z
    
    style E fill:#fff3e0
    style G fill:#e1f5fe
    style M fill:#e8f5e8
    style P fill:#ffebee
    style T fill:#f3e5f5
```

### 🔍 **Query Construction Detail**
```mermaid
graph LR
    A[QA Item] --> B[Extract title]
    A --> C[Extract question_content]
    A --> D[❌ NOT accepted_answer]
    
    B --> E{Combine Logic}
    C --> E
    
    E --> F["title + question_content"]
    F --> G[Generate Embedding]
    G --> H[Document Search]
    
    style D fill:#ffebee
    style F fill:#e8f5e8
    style H fill:#e1f5fe
```

### 🧠 **Memory Management Flow**
```mermaid
sequenceDiagram
    participant M as Model Loop
    participant GPU as GPU Memory
    participant CPU as CPU Fallback
    participant GC as Garbage Collector
    
    M->>GPU: Try load model on GPU
    
    alt CUDA Success
        GPU->>M: Model loaded
        M->>M: Process questions
    else CUDA Out of Memory
        GPU->>CPU: Fallback to CPU
        CPU->>GC: torch.cuda.empty_cache()
        GC->>CPU: Memory cleared
        CPU->>M: CPU model loaded
        M->>M: Process questions
    end
    
    M->>GC: del retriever, del model
    GC->>GPU: Clear memory
    
    Note over M,GC: Repeat for each model
```

---

## 📊 **Celda 15-16: Results Processing**

### 📈 **Results Summary Generation**
```mermaid
flowchart LR
    A[all_model_results] --> B[Filter Successful Models]
    B --> C[Create Summary Table]
    C --> D[Calculate Performance Insights]
    D --> E[Find Top Performer]
    E --> F[Display Results]
    
    A --> G[Filter Error Models]
    G --> H[Display Error Details]
    
    B --> I[Extract Metrics]
    I --> J[P@5, R@5, F1@5, MRR]
    I --> K[RAG Metrics]
    I --> L[Improvement Deltas]
    
    J --> C
    K --> C
    L --> C
    
    style C fill:#e8f5e8
    style F fill:#e1f5fe
    style H fill:#ffebee
```

### 🔍 **Debug Information Flow**
```mermaid
graph TD
    A[Results Processing] --> B[Query Verification]
    B --> C[Sample Questions Display]
    C --> D[Precision Scores]
    D --> E[Performance Debugging]
    
    B --> F[Construction Method]
    F --> G["✅ title + question_content"]
    F --> H["❌ NOT accepted_answer"]
    
    style G fill:#e8f5e8
    style H fill:#ffebee
```

---

## 💾 **Celda 17-18: Results Storage**

### 🗃️ **JSON Structure Generation**
```mermaid
graph TD
    A[Prepare Results] --> B[Convert NumPy Types]
    B --> C[Build JSON Structure]
    
    C --> D[config section]
    C --> E[evaluation_info section]  
    C --> F[results section]
    
    D --> G[num_questions<br/>selected_models<br/>generative_model<br/>top_k, etc.]
    
    E --> H[timestamp<br/>timezone<br/>evaluation_type<br/>enhanced_display_compatible<br/>data_verification]
    
    F --> I[Model Results]
    I --> J[avg_before_metrics<br/>avg_after_metrics<br/>individual_metrics<br/>rag_metrics]
    
    C --> K[Save to Drive]
    K --> L[cumulative_results_timestamp.json]
    
    style C fill:#e8f5e8
    style K fill:#f3e5f5
```

### 📁 **File Output Structure**
```
📁 /content/drive/MyDrive/TesisMagister/acumulative/
├── 📄 cumulative_results_{timestamp}.json
├── 📊 Results compatible with Streamlit display
├── 🔍 All model metrics (before/after LLM reranking)
├── 🤖 RAG metrics for each model
└── ✅ Data verification flags
```

---

## 🎛️ **Control Flow & Error Handling**

### ⚠️ **Error Handling Strategy**
```mermaid
flowchart TD
    A[Process Step] --> B{Error Occurred?}
    
    B -->|❌ CUDA Memory| C[CPU Fallback]
    B -->|❌ Dimension Mismatch| D[Skip Model + Store Error]
    B -->|❌ API Error| E[Log Error + Continue]
    B -->|❌ Model Load Error| F[Store Error Result]
    B -->|✅ Success| G[Continue Processing]
    
    C --> G
    D --> H[Next Model]
    E --> I[Next Question]
    F --> H
    G --> J[Normal Flow]
    
    H --> K[Cleanup Memory]
    I --> L[Continue Evaluation]
    J --> M[Store Results]
    
    style C fill:#fff3e0
    style D fill:#ffebee
    style G fill:#e8f5e8
```

### 🔄 **Memory Management Pattern**
```mermaid
sequenceDiagram
    participant Loop as Model Loop
    participant Mem as Memory Manager
    participant GPU as GPU
    participant CPU as CPU
    
    Loop->>Mem: Start model evaluation
    Mem->>GPU: Try GPU allocation
    
    alt Success
        GPU->>Loop: Model ready
        Loop->>Loop: Process all questions
    else CUDA OOM
        GPU->>CPU: Switch to CPU
        CPU->>Mem: Clear GPU cache
        Mem->>Loop: CPU model ready
        Loop->>Loop: Process questions (slower)
    end
    
    Loop->>Mem: Model complete
    Mem->>GPU: Delete model
    Mem->>GPU: garbage collect
    Mem->>Loop: Ready for next model
```

---

## 📋 **Key Metrics Calculated**

### 🎯 **Retrieval Metrics (Before/After LLM)**
```
📊 Precision@k (k=1,3,5,10): Relevant docs in top-k / k
📊 Recall@k (k=1,3,5,10): Relevant docs in top-k / total relevant  
📊 F1@k (k=1,3,5,10): Harmonic mean of Precision & Recall
📊 NDCG@k (k=1,3,5,10): Normalized Discounted Cumulative Gain
📊 MAP@k (k=1,3,5,10): Mean Average Precision
📊 MRR: Mean Reciprocal Rank (1/rank of first relevant doc)
```

### 🤖 **RAG Metrics (OpenAI-based)**
```
🤖 Faithfulness: Answer fidelity to source documents
🤖 Answer Relevance: How well answer addresses the question  
🤖 Answer Correctness: Factual accuracy of the answer
🤖 Answer Similarity: Semantic similarity to expected answer
```

---

## ✅ **Success Criteria & Verification**

### 🎯 **Evaluation Success Indicators**
- ✅ All 4 models evaluated without dimension errors
- ✅ 100 questions processed per model (E5-Large may use CPU)
- ✅ RAG metrics generated and stored correctly
- ✅ Results compatible with Streamlit visualization
- ✅ Only title+question_content used (not accepted_answer)
- ✅ JSON serialization successful with numpy type conversion

### 📊 **Expected Output Files**
```
📁 Google Drive: /TesisMagister/acumulative/
└── cumulative_results_{unix_timestamp}.json
    ├── config: {num_questions, models, parameters}
    ├── evaluation_info: {timestamp, verification flags}  
    └── results: {
        ada: {avg_metrics, individual_metrics, rag_metrics},
        e5-large: {...},
        mpnet: {...}, 
        minilm: {...}
    }
```

---

## 🚀 **Performance Optimizations**

### ⚡ **Speed Optimizations**
1. **GPU First, CPU Fallback** - Maximum speed when possible
2. **Memory Cleanup** - Aggressive garbage collection between models  
3. **Batch Processing** - Process questions in sequence efficiently
4. **Model Caching** - Avoid reloading same models

### 💾 **Memory Optimizations** 
1. **Explicit Deletion** - `del retriever, del model`
2. **CUDA Cache Clear** - `torch.cuda.empty_cache()`
3. **Garbage Collection** - `gc.collect()` after each model
4. **CPU Fallback** - Automatic when GPU memory insufficient

---

## 🎉 **Final Output**
Un archivo JSON completo con métricas de evaluación multi-modelo, compatible con el sistema de visualización Streamlit existente, con verificación de datos reales y métricas RAG funcionales.