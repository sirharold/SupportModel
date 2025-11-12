# ABSTRACT

The objective of this work is to design, implement, and evaluate a semantic information retrieval system to assist in resolving technical support tickets related to Microsoft Azure, using official documentation and frequently asked questions as knowledge sources. A three-block architecture was developed: automated extraction of public data from Microsoft Learn and Microsoft Q&A through web scraping; embedding generation using pretrained models (all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1, E5-large) and OpenAI embeddings; and storage in ChromaDB for semantic querying.

The developed corpus comprises 187,031 documentation chunks segmented from 62,417 unique Microsoft Learn documents, and 13,436 Microsoft Q&A questions validated as ground truth. Exploratory analysis revealed substantial technical content with 872.3 average tokens per chunk and balanced distribution across Azure categories.

Evaluation was performed using standard information retrieval metrics: Precision@5, Recall@5, MRR@5, nDCG, and F1-score. The best open-source model configuration (MiniLM with title+summary+content) achieves Precision@5 of 0.0256, Recall@5 of 0.0833, and MRR@5 of 0.0573. OpenAI embeddings obtain superior metrics: Precision@5 of 0.034, Recall@5 of 0.112, and MRR@5 of 0.072, demonstrating greater retrieval capacity.

This project establishes three main contributions: development of the first specialized corpus in Azure documentation for academic research; provision of reproducible benchmarks for evaluating embedding models in technical domains; and empirical demonstration of the viability of semantic retrieval systems for technical support assistance. The obtained metrics establish a baseline for future research, identifying improvement opportunities through fine-tuning, hybrid search, and advanced reranking.

**Keywords:** Information Retrieval, Embeddings, RAG, Technical Support, Azure, ChromaDB, Semantic Search
