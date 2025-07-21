# ☁️ Azure Q&A Expert System

> Advanced Retrieval-Augmented Generation (RAG) system for Azure documentation with comprehensive evaluation metrics and cost optimization features.

## 🎯 Overview

The Azure Q&A Expert System is a sophisticated RAG-powered application that helps users find relevant Azure documentation and generates comprehensive answers to technical questions. The system features multiple embedding models, local LLM support for cost optimization, and advanced evaluation metrics including hallucination detection and context utilization analysis.

## ✨ Features

### 🔍 **Individual Search**
- Intelligent document retrieval with semantic search
- RAG-powered answer generation using multiple models
- Real-time evaluation metrics and confidence scoring
- Support for both local and API-based models

### 📊 **Batch Query Processing**
- Process multiple queries simultaneously
- Comprehensive analytics and performance metrics
- Export capabilities with detailed reports
- Scalable processing with progress tracking

### 🔬 **Model Comparison**
- Side-by-side comparison of embedding models
- Advanced RAG metrics evaluation
- Performance benchmarking and analysis
- Visual analytics with interactive charts

### 🧪 **Advanced RAG Evaluation**
- **Hallucination Detection**: Identify unsupported claims
- **Context Utilization**: Measure effective use of retrieved documents
- **Answer Completeness**: Evaluate response completeness by question type
- **User Satisfaction Proxy**: Assess clarity, directness, and actionability

### 💰 **Cost Optimization**
- Local model support (Llama 3.1 8B, Mistral 7B)
- Zero-cost operation mode
- Flexible model selection (local vs API)
- Real-time cost tracking and warnings

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │    │   Weaviate DB    │    │  Local Models   │
│  - Individual   │◄──►│  - Documents     │◄──►│  - Llama 3.1    │
│  - Batch        │    │  - Questions     │    │  - Mistral 7B   │
│  - Comparison   │    │  - Embeddings    │    │  - CrossEncoder │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   External APIs │    │   Evaluation     │    │   Analytics     │
│  - OpenAI       │    │  - BERTScore     │    │  - Metrics      │
│  - Gemini       │    │  - ROUGE         │    │  - Comparisons  │
│  - HuggingFace  │    │  - Advanced RAG  │    │  - Reports      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB+ recommended for local models)
- **Storage**: 20GB free space for local models
- **GPU**: NVIDIA with 6GB+ VRAM (optional but recommended)

### External Services
- **Weaviate Cloud Service**: Vector database access
- **OpenAI Account**: API key for GPT models (optional)
- **Gemini API**: Google AI access (optional)
- **Hugging Face**: Token for model downloads (optional)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/haroldgomez/SupportModel.git
cd SupportModel
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the project root:

```env
# Required
WCS_URL=your_weaviate_cloud_url
WCS_API_KEY=your_weaviate_api_key

# Optional (for API models)
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACE_API_KEY=your_huggingface_token
```

### 4. Launch Application
```bash
PYTHONPATH=. streamlit run src/apps/main_qa_app.py
```

Access the application at `http://localhost:8501`

## ⚙️ Configuration

### Model Selection

#### Embedding Models
- **multi-qa-mpnet-base-dot-v1** (Default): Best for Q&A, local, free
- **all-MiniLM-L6-v2**: Fast and efficient, local, free
- **ada (text-embedding-ada-002)**: OpenAI, high quality, paid

#### Generative Models
- **llama-3.1-8b** (Default): Local, zero cost, good quality
- **mistral-7b**: Local, zero cost, fast generation
- **gpt-4**: OpenAI, highest quality, expensive
- **gemini-pro**: Google, good quality, moderate cost

### Performance vs Cost Trade-offs

| Configuration | Cost | Speed | Quality | Use Case |
|---------------|------|-------|---------|----------|
| **Local Only** | Free | Medium | Good | Development, cost-sensitive |
| **Hybrid** | Low | Fast | High | Production, balanced |
| **Full API** | High | Fastest | Highest | Enterprise, quality-critical |

## 💰 Cost Optimization

### Zero-Cost Operation
For complete cost-free operation, see [LOCAL_MODELS_SETUP.md](LOCAL_MODELS_SETUP.md) for detailed setup instructions.

**Quick Setup for Local Models:**
1. Select `llama-3.1-8b` or `mistral-7b` as generative model
2. Use `multi-qa-mpnet-base-dot-v1` for embeddings
3. Disable "Usar Re-Ranking con LLM" for API-free operation
4. Models will be downloaded automatically on first use

### Cost Monitoring
- Real-time cost warnings for API usage
- Model selection recommendations
- Usage tracking per session

## 📊 Usage Examples

### Individual Search
```python
# Example of programmatic usage
from utils.qa_pipeline import answer_question_with_rag
from utils.clients import initialize_clients

# Initialize clients
weaviate_wrapper, embedding_client, openai_client, gemini_client, \
local_llama_client, local_mistral_client, _ = initialize_clients(
    "multi-qa-mpnet-base-dot-v1", 
    "llama-3.1-8b"
)

# Perform RAG query
results, debug_info, generated_answer, rag_metrics = answer_question_with_rag(
    question="How do I configure Managed Identity in Azure Functions?",
    weaviate_wrapper=weaviate_wrapper,
    embedding_client=embedding_client,
    openai_client=openai_client,
    local_llama_client=local_llama_client,
    top_k=10,
    generative_model_name="llama-3.1-8b"
)

print(f"Generated Answer: {generated_answer}")
print(f"Documents Retrieved: {len(results)}")
print(f"RAG Metrics: {rag_metrics}")
```

### Advanced Metrics Evaluation
```python
from utils.enhanced_evaluation import evaluate_rag_with_advanced_metrics

# Comprehensive evaluation with advanced metrics
eval_result = evaluate_rag_with_advanced_metrics(
    question="How do I create an Azure Storage account?",
    weaviate_wrapper=weaviate_wrapper,
    embedding_client=embedding_client,
    openai_client=openai_client,
    local_llama_client=local_llama_client,
    generative_model_name="llama-3.1-8b"
)

# Access advanced metrics
advanced_metrics = eval_result['advanced_metrics']
hallucination_score = advanced_metrics['hallucination']['hallucination_score']
utilization_score = advanced_metrics['context_utilization']['utilization_score']
completeness_score = advanced_metrics['completeness']['completeness_score']
satisfaction_score = advanced_metrics['satisfaction']['satisfaction_score']

print(f"Hallucination Score: {hallucination_score:.3f}")
print(f"Context Utilization: {utilization_score:.3f}")
print(f"Completeness: {completeness_score:.3f}")
print(f"User Satisfaction: {satisfaction_score:.3f}")
```

### Retrieval Metrics Evaluation
```python
from utils.metrics import calculate_content_metrics, compute_ndcg, compute_mrr

# Evaluate retrieval quality
results, _ = answer_question_documents_only(
    "How do I create a storage account?", 
    weaviate_wrapper, 
    embedding_client, 
    openai_client
)

# Content quality metrics
content_metrics = calculate_content_metrics(results, ground_truth_answer)
print(f"BERTScore F1: {content_metrics['BERT_F1']:.3f}")
print(f"ROUGE-1: {content_metrics['ROUGE1']:.3f}")

# Ranking metrics
relevant_links = ["https://learn.microsoft.com/azure/storage/"]
ndcg = compute_ndcg(results, relevant_links, k=10)
mrr = compute_mrr(results, relevant_links, k=10)
print(f"NDCG@10: {ndcg:.3f}")
print(f"MRR: {mrr:.3f}")
```

## 🧪 Advanced Features

### RAG Metrics Explanation

#### 🚫 Hallucination Detection
- **Purpose**: Identify information not supported by retrieved context
- **Range**: 0.0 (no hallucinations) to 1.0 (fully hallucinated)
- **Calculation**: Entity extraction + fact verification against context
- **Thresholds**: <0.1 excellent, <0.2 good, ≥0.2 needs improvement

#### 🎯 Context Utilization
- **Purpose**: Measure effective use of retrieved documents
- **Range**: 0.0 (no context used) to 1.0 (optimal utilization)
- **Calculation**: Document coverage × phrase utilization
- **Thresholds**: >0.8 excellent, >0.6 good, ≤0.6 needs improvement

#### ✅ Answer Completeness
- **Purpose**: Evaluate response completeness by question type
- **Range**: 0.0 (incomplete) to 1.0 (fully complete)
- **Calculation**: Expected components presence analysis
- **Thresholds**: >0.9 excellent, >0.7 good, ≤0.7 needs improvement

#### 😊 User Satisfaction Proxy
- **Purpose**: Assess response quality (clarity + directness + actionability)
- **Range**: 0.0 (unsatisfactory) to 1.0 (highly satisfactory)
- **Calculation**: Multi-factor quality assessment
- **Thresholds**: >0.8 excellent, >0.6 good, ≤0.6 needs improvement

### PDF Report Generation
- Comprehensive comparison reports
- Visual analytics and charts
- Export functionality for analysis
- Professional formatting

### Process Flow Visualization
See [individual_search_flowchart.md](individual_search_flowchart.md) for detailed system flow diagrams.

## 🗂️ Project Structure

```
SupportModel/
├── 📱 EnhancedStreamlit_qa_app.py     # Main Streamlit application
├── 🔬 comparison_page.py              # Model comparison interface
├── 📊 batch_queries_page.py           # Batch processing interface
├── ⚙️ config.py                       # Configuration settings
├── 📋 requirements.txt                # Python dependencies
├── 📖 LOCAL_MODELS_SETUP.md          # Local models setup guide
├── 🗺️ individual_search_flowchart.md  # System flow documentation
├── 🧪 tests/                          # Test suites
│   ├── test_metrics.py
│   ├── test_chunk_filtering.py
│   └── test_extract_openai_links.py
└── 🛠️ utils/                          # Core utilities
    ├── qa_pipeline.py                 # Main RAG pipeline
    ├── clients.py                     # Client initialization
    ├── embedding_safe.py              # Embedding clients
    ├── local_models.py                # Local model management
    ├── advanced_rag_metrics.py        # Advanced evaluation
    ├── enhanced_evaluation.py         # Evaluation framework
    ├── metrics.py                     # Traditional metrics
    ├── reranker.py                    # Document reranking
    ├── answer_generator.py            # Response generation
    ├── weaviate_utils_improved.py     # Vector DB utilities
    └── pdf_generator.py               # Report generation
```

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_metrics.py

# Run with coverage
python -m pytest tests/ --cov=utils
```

### Test Coverage
- **Metrics evaluation**: Comprehensive testing of NDCG, MRR, precision/recall
- **Content quality**: BERTScore and ROUGE metrics validation
- **Link extraction**: URL parsing and validation
- **Chunk filtering**: Document processing accuracy

## 📈 Performance & Monitoring

### Expected Response Times
| Operation | Local Models | API Models | Hybrid |
|-----------|-------------|------------|--------|
| Document Search | 0.5-1.0s | 0.3-0.8s | 0.4-0.9s |
| Answer Generation | 3-10s | 1-5s | 2-7s |
| Advanced Metrics | 1-3s | 2-6s | 1.5-4s |
| Batch Processing | 10s+ per query | 5s+ per query | 7s+ per query |

### System Monitoring
- Real-time response time tracking
- Session metrics (queries, documents, avg time)
- Model performance comparison
- Cost tracking and optimization suggestions

### Resource Requirements
| Configuration | RAM Usage | GPU Memory | Disk Space |
|---------------|-----------|------------|------------|
| API Only | 2-4GB | None | 5GB |
| Local Embeddings | 4-6GB | 1-2GB | 10GB |
| Full Local | 8-16GB | 4-8GB | 25GB |

## 🔧 Development

### Environment Setup for Development
```bash
# Clone with development dependencies
git clone https://github.com/haroldgomez/SupportModel.git
cd SupportModel

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install pytest coverage black isort

# Pre-commit setup
pre-commit install
```

### Code Style
- **Formatting**: Black formatter
- **Import sorting**: isort
- **Type hints**: Encouraged for new code
- **Documentation**: Docstrings for public functions

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔧 Troubleshooting

### Common Issues

**Model Download Failures**
```bash
# Solution: Set HuggingFace token
export HUGGINGFACE_API_KEY=your_token
# Or ensure stable internet connection
```

**Out of Memory Errors**
```bash
# Solution: Use smaller models or increase system RAM
# Select "all-MiniLM-L6-v2" for lower memory usage
```

**Weaviate Connection Issues**
```bash
# Solution: Verify environment variables
echo $WCS_URL
echo $WCS_API_KEY
# Check network connectivity to Weaviate Cloud
```

**Slow Performance**
```bash
# Solution: Enable GPU acceleration
# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Debug Mode
Enable detailed logging by setting `show_debug_info=True` in the interface or:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Additional Resources

- **📖 Local Models Setup**: [LOCAL_MODELS_SETUP.md](LOCAL_MODELS_SETUP.md)
- **🔄 System Flow**: [individual_search_flowchart.md](individual_search_flowchart.md)
- **📊 RAG Usage Guide**: [RAG_USAGE.md](RAG_USAGE.md)
- **🐛 Issue Tracker**: [GitHub Issues](https://github.com/haroldgomez/SupportModel/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **Weaviate**: Vector database infrastructure
- **Hugging Face**: Model hosting and transformers
- **OpenAI**: Advanced language models and APIs
- **Streamlit**: Web application framework
- **Sentence Transformers**: Embedding models
- **Microsoft Learn**: Azure documentation source

---

*Built with ❤️ for the Azure community*