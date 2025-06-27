# SupportModel

This project provides a simple question answering pipeline using Weaviate and OpenAI embeddings.

## Prerequisites

- Python 3.10 or higher
- Access to a Weaviate instance (e.g. Weaviate Cloud Service)
- OpenAI account with an API key

## Environment variables

Create a `.env` file in the project root containing:

```env
WCS_URL=<your_weaviate_url>
WCS_API_KEY=<your_weaviate_api_key>
OPENAI_API_KEY=<your_openai_api_key>
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/haroldgomez/SupportModel.git
cd SupportModel
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) Database setup

If you use an additional database, configure your connection in `config.py` and run migrations:

```bash
python manage.py migrate
```

4. Run the application

```bash
streamlit run streamlit_qa_app.py
```


## Evaluating retrieval metrics

The `utils.metrics` module provides helpers to measure the quality of the ranked documents returned by `answer_question`.

```python
from utils.qa_pipeline import answer_question
from utils.metrics import compute_ndcg, compute_mrr, compute_precision_recall_f1

# assume weaviate_wrapper and embedding_client are already created
results, _ = answer_question("How do I create a storage account?", weaviate_wrapper, embedding_client)
relevant_links = ["https://learn.microsoft.com/azure/storage/"]

ndcg = compute_ndcg(results, relevant_links, k=10)
mrr = compute_mrr(results, relevant_links, k=10)
precision, recall, f1 = compute_precision_recall_f1(results, relevant_links, k=10)
print(ndcg, mrr, precision, recall, f1)
```

These metrics are useful during training or when comparing the system against an external set of relevant links.
