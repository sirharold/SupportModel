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

