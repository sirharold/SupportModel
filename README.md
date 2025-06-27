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

The `utils.metrics` module provides helpers to measure the quality of the ranked documents returned by `answer_question` or another source like OpenAI.

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

# If you also have a list of links from OpenAI you can compare both rankings
openai_links = ["https://learn.microsoft.com/...", "https://learn.microsoft.com/...",]

# Our ranking evaluated with OpenAI links
our_p, our_r, our_f1 = compute_precision_recall_f1(results, openai_links, k=10)
our_mrr = compute_mrr(results, openai_links, k=10)

# OpenAI ranking evaluated with our links
openai_docs = [{"link": l} for l in openai_links]
openai_p, openai_r, openai_f1 = compute_precision_recall_f1(openai_docs, [d["link"] for d in results], k=10)
openai_mrr = compute_mrr(openai_docs, [d["link"] for d in results], k=10)
```

These metrics are useful during training or when comparing the system against an external set of relevant links.
