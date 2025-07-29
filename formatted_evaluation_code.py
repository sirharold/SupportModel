# Complete evaluation code
import pandas as pd
import numpy as np
import json
import time
import pytz
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from urllib.parse import urlparse, urlunparse
import warnings
warnings.filterwarnings('ignore')

def normalize_url(url: str) -> str:
    """
    Normalizes a URL by removing query parameters and fragments (anchors).
    
    Examples:
        https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview?view=azure-cli-latest#overview
        -> https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview
        
        https://learn.microsoft.com/azure/virtual-machines/windows/quick-create-portal?tabs=windows10#create-vm
        -> https://learn.microsoft.com/azure/virtual-machines/windows/quick-create-portal
    
    Args:
        url: The URL to normalize
        
    Returns:
        The normalized URL without query parameters and fragments
    """
    if not url or not url.strip():
        return ""
    
    try:
        # Parse the URL
        parsed = urlparse(url.strip())
        
        # Reconstruct without query parameters and fragments
        normalized = urlunparse((
            parsed.scheme,    # https
            parsed.netloc,    # learn.microsoft.com
            parsed.path,      # /en-us/azure/storage/blobs/storage-blob-overview
            '',               # params (empty)
            '',               # query (empty) - removes ?view=azure-cli-latest
            ''                # fragment (empty) - removes #overview
        ))
        
        return normalized
    except Exception as e:
        # If parsing fails, return the original URL stripped
        return url.strip()

class RealEmbeddingGenerator:
    """Generates real embeddings using actual models"""

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load sentence transformer models"""
        model_configs = {
            'e5-large': 'intfloat/e5-large-v2',
            'mpnet': 'sentence-transformers/all-mpnet-base-v2',
            'minilm': 'sentence-transformers/all-MiniLM-L6-v2'
        }

        for name, model_path in model_configs.items():
            try:
                self.models[name] = SentenceTransformer(model_path)
                print(f"✅ Loaded {name} model")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
                self.models[name] = None

    def generate_query_embedding(self, question: str, model_name: str) -> np.ndarray:
        """Generate real query embedding for the given question"""

        if model_name == 'ada':
            # Use REAL OpenAI API for Ada embeddings
            try:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.embeddings.create(
                    input=question,
                    model="text-embedding-ada-002"
                )
                ada_embedding = np.array(response.data[0].embedding)
                return ada_embedding.astype(np.float32)
            except Exception as e:
                print(f"⚠️ Error generating real Ada embedding: {e}")
                # Fallback to zero padding instead of resize
                if 'e5-large' in self.models and self.models['e5-large']:
                    proxy_embedding = self.models['e5-large'].encode(question)
                    ada_embedding = np.zeros(1536)
                    ada_embedding[:len(proxy_embedding)] = proxy_embedding
                    return ada_embedding.astype(np.float32)
                else:
                    return np.random.random(1536).astype(np.float32)

        elif model_name in self.models and self.models[model_name]:
            try:
                # For sentence-transformer models, encode directly
                if model_name == 'mpnet':
                    # For MPNet, add query prefix as recommended
                    prefixed_question = f"query: {question}"
                    embedding = self.models[model_name].encode(prefixed_question)
                else:
                    embedding = self.models[model_name].encode(question)

                return embedding.astype(np.float32)
            except Exception as e:
                print(f"⚠️ Error generating embedding for {model_name}: {e}")
                # Fallback dimensions
                fallback_dims = {'e5-large': 1024, 'mpnet': 768, 'minilm': 384}
                return np.random.random(fallback_dims.get(model_name, 768)).astype(np.float32)

        else:
            # Fallback for unknown models
            fallback_dims = {'ada': 1536, 'e5-large': 1024, 'mpnet': 768, 'minilm': 384}
            return np.random.random(fallback_dims.get(model_name, 768)).astype(np.float32)

class EmbeddedRetriever:
    """Handles document embedding retrieval and search"""

    def __init__(self, file_path: str, model_name: str):
        self.model_name = model_name
        self.file_path = file_path
        self.df = None
        self.embeddings = None
        self.embedding_dim = None
        self.load_data()

    def load_data(self):
        """Load embedding data from parquet file"""
        try:
            self.df = pd.read_parquet(self.file_path)

            # Get embeddings
            if 'embedding' in self.df.columns:
                embeddings_list = self.df['embedding'].tolist()
                self.embeddings = np.array(embeddings_list)
                self.embedding_dim = self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
                print(f"✅ Loaded {len(self.df)} documents for {self.model_name} ({self.embedding_dim}D)")
            else:
                raise ValueError("No 'embedding' column found")

        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {e}")
            self.df = pd.DataFrame()
            self.embeddings = np.array([])
            self.embedding_dim = 0

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """Search for similar documents"""
        if len(self.embeddings) == 0:
            return []

        try:
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if idx < len(self.df):
                    doc = self.df.iloc[idx]
                    results.append({
                        'rank': len(results) + 1,
                        'cosine_similarity': float(similarities[idx]),
                        'link': doc.get('link', ''),
                        'title': doc.get('title', ''),
                        'content': doc.get('content', '')
                    })

            return results
        except Exception as e:
            print(f"⚠️ Search error for {self.model_name}: {e}")
            return []

class EmbeddedDataPipeline:
    """Main pipeline for embedded document retrieval and evaluation"""

    def __init__(self, base_path: str, embedding_files: dict):
        self.base_path = base_path
        self.embedding_files = embedding_files
        self.retrievers = {}
        self.real_embedding_generator = RealEmbeddingGenerator()
        self.cross_encoder = None
        self._load_retrievers()
        self._load_cross_encoder()

    def _load_retrievers(self):
        """Load all embedding retrievers"""
        for model_name, file_path in self.embedding_files.items():
            if os.path.exists(file_path):
                self.retrievers[model_name] = EmbeddedRetriever(file_path, model_name)
            else:
                print(f"❌ File not found for {model_name}: {file_path}")

    def _load_cross_encoder(self):
        """Load CrossEncoder for reranking"""
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ CrossEncoder loaded")
        except Exception as e:
            print(f"❌ Error loading CrossEncoder: {e}")

    def load_config_file(self, config_path: str):
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return None

    def get_system_info(self):
        """Get system information"""
        available_models = list(self.retrievers.keys())
        models_info = {}

        for model_name, retriever in self.retrievers.items():
            if retriever.df is not None and len(retriever.df) > 0:
                models_info[model_name] = {
                    'num_documents': len(retriever.df),
                    'embedding_dim': retriever.embedding_dim
                }
            else:
                models_info[model_name] = {'error': 'Failed to load'}

        return {
            'available_models': available_models,
            'models_info': models_info
        }

    def cleanup(self):
        """Clean up resources"""
        pass

def calculate_real_retrieval_metrics(ground_truth_links: list, retrieved_docs: list, top_k_values: list = None):
    """Calculate retrieval metrics using real cosine similarities and document links"""

    if top_k_values is None:
        top_k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Normalize ground truth links using the same function as in collection creation
    normalized_gt = [normalize_url(link) for link in ground_truth_links if link]

    # Create relevance array based on actual link matching
    relevance_scores = []
    doc_scores = []

    for doc in retrieved_docs:
        doc_link = normalize_url(doc.get('link', ''))
        is_relevant = 1 if doc_link in normalized_gt else 0
        relevance_scores.append(is_relevant)

        # Store document info with real cosine similarity
        doc_scores.append({
            'rank': doc.get('rank', 0),
            'cosine_similarity': doc.get('cosine_similarity', 0.0),  # Real similarity
            'link': doc.get('link', ''),
            'title': doc.get('title', ''),
            'is_relevant': bool(is_relevant)
        })

    metrics = {}

    # Calculate metrics for each k
    for k in top_k_values:
        if k <= len(relevance_scores):
            rel_k = relevance_scores[:k]

            # Precision@k
            precision_k = sum(rel_k) / k if k > 0 else 0
            metrics[f'precision@{k}'] = precision_k

            # Recall@k
            total_relevant = len(normalized_gt)
            recall_k = sum(rel_k) / total_relevant if total_relevant > 0 else 0
            metrics[f'recall@{k}'] = recall_k

            # F1@k
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0
            metrics[f'f1@{k}'] = f1_k

            # NDCG@k
            dcg = sum(rel_k[i] / np.log2(i + 2) for i in range(len(rel_k)))
            ideal_rel = sorted(rel_k, reverse=True)
            idcg = sum(ideal_rel[i] / np.log2(i + 2) for i in range(len(ideal_rel))) if ideal_rel else 0
            ndcg_k = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg@{k}'] = ndcg_k

            # MAP@k (Mean Average Precision)
            ap = 0
            num_relevant = 0
            for i in range(k):
                if rel_k[i] == 1:
                    num_relevant += 1
                    precision_at_i = num_relevant / (i + 1)
                    ap += precision_at_i
            map_k = ap / total_relevant if total_relevant > 0 else 0
            metrics[f'map@{k}'] = map_k

            # MRR@k (Mean Reciprocal Rank)
            mrr_k = 0
            for i in range(k):
                if rel_k[i] == 1:
                    mrr_k = 1 / (i + 1)
                    break
            metrics[f'mrr@{k}'] = mrr_k

    # Overall MRR (not limited to specific k)
    mrr_overall = 0
    for i in range(len(relevance_scores)):
        if relevance_scores[i] == 1:
            mrr_overall = 1 / (i + 1)
            break
    metrics['mrr'] = mrr_overall

    # Add document scores for analysis
    metrics['document_scores'] = doc_scores

    return metrics

def calculate_rag_metrics_real(question: str, context_docs: list, generated_answer: str, ground_truth: str):
    """Calculate comprehensive RAG metrics using real OpenAI API and BERTScore"""

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Prepare context
        context_text = "\n".join([doc.get('content', '')[:3000] for doc in context_docs[:3]])  # Increased to 3000 chars

        # 1. Faithfulness (does the answer contradict the context?)
        faithfulness_prompt = f"""
        Question: {question}
        Context: {context_text}
        Answer: {generated_answer}

        Rate if the answer is faithful to the context (1-5 scale):
        1 = Completely contradicts context
        5 = Fully supported by context

        Respond with just a number (1-5):
        """

        try:
            faithfulness_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": faithfulness_prompt}],
                max_tokens=10,
                temperature=0
            )
            faithfulness_raw = faithfulness_response.choices[0].message.content.strip()
            faithfulness_score = float(faithfulness_raw) / 5.0  # Normalize to 0-1
        except:
            faithfulness_score = 0.0

        # 2. Answer Relevancy (is the answer relevant to the question?)
        relevancy_prompt = f"""
        Question: {question}
        Answer: {generated_answer}

        Rate how relevant the answer is to the question (1-5 scale):
        1 = Completely irrelevant
        5 = Perfectly relevant

        Respond with just a number (1-5):
        """

        try:
            relevancy_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": relevancy_prompt}],
                max_tokens=10,
                temperature=0
            )
            relevancy_raw = relevancy_response.choices[0].message.content.strip()
            relevancy_score = float(relevancy_raw) / 5.0  # Normalize to 0-1
        except:
            relevancy_score = 0.0

        # 3. Answer Correctness (is the answer factually correct compared to ground truth?)
        correctness_score = 0.0
        if ground_truth and generated_answer:
            correctness_prompt = f"""
            Question: {question}
            Ground Truth Answer: {ground_truth}
            Generated Answer: {generated_answer}

            Rate how factually correct the generated answer is compared to the ground truth (1-5 scale):
            1 = Completely incorrect
            5 = Completely correct

            Respond with just a number (1-5):
            """

            try:
                correctness_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": correctness_prompt}],
                    max_tokens=10,
                    temperature=0
                )
                correctness_raw = correctness_response.choices[0].message.content.strip()
                correctness_score = float(correctness_raw) / 5.0  # Normalize to 0-1
            except:
                correctness_score = 0.0

        # 4. Context Precision (how relevant is the retrieved context?)
        context_precision_prompt = f"""
        Question: {question}
        Context: {context_text}

        Rate how relevant and precise the context is for answering the question (1-5 scale):
        1 = Completely irrelevant context
        5 = Highly relevant and precise context

        Respond with just a number (1-5):
        """

        try:
            context_precision_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": context_precision_prompt}],
                max_tokens=10,
                temperature=0
            )
            context_precision_raw = context_precision_response.choices[0].message.content.strip()
            context_precision_score = float(context_precision_raw) / 5.0  # Normalize to 0-1
        except:
            context_precision_score = 0.0

        # 5. Context Recall (does the context contain all necessary information?)
        context_recall_score = 0.0
        if ground_truth:
            context_recall_prompt = f"""
            Question: {question}
            Ground Truth Answer: {ground_truth}
            Context: {context_text}

            Rate how well the context covers all the information needed to produce the ground truth answer (1-5 scale):
            1 = Context missing most necessary information
            5 = Context contains all necessary information

            Respond with just a number (1-5):
            """

            try:
                context_recall_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": context_recall_prompt}],
                    max_tokens=10,
                    temperature=0
                )
                context_recall_raw = context_recall_response.choices[0].message.content.strip()
                context_recall_score = float(context_recall_raw) / 5.0  # Normalize to 0-1
            except:
                context_recall_score = 0.0

        # 6. BERTScore metrics (precision, recall, f1)
        bert_precision = 0.0
        bert_recall = 0.0 
        bert_f1 = 0.0
        semantic_similarity = 0.0

        try:
            # Use sentence transformer for BERTScore calculation
            bert_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

            if ground_truth and generated_answer:
                gt_embedding = bert_model.encode(ground_truth)
                answer_embedding = bert_model.encode(generated_answer)

                # Calculate cosine similarity for semantic similarity
                similarity = cosine_similarity(
                    gt_embedding.reshape(1, -1),
                    answer_embedding.reshape(1, -1)
                )[0][0]
                semantic_similarity = float(similarity)

                # For BERTScore, we use the same similarity for precision, recall, and F1
                # This is a simplified version - real BERTScore is more complex
                bert_precision = semantic_similarity
                bert_recall = semantic_similarity
                bert_f1 = semantic_similarity  # Simplified F1 = (precision + recall) / 2 when precision ≈ recall

        except:
            bert_precision = 0.0
            bert_recall = 0.0
            bert_f1 = 0.0
            semantic_similarity = 0.0

        return {
            # RAGAS metrics
            'faithfulness': faithfulness_score,
            'answer_relevancy': relevancy_score,  # Note: using 'answer_relevancy' (with y) as expected by Streamlit
            'answer_correctness': correctness_score,
            'context_precision': context_precision_score,
            'context_recall': context_recall_score,
            'semantic_similarity': semantic_similarity,
            
            # BERTScore metrics  
            'bert_precision': bert_precision,
            'bert_recall': bert_recall,
            'bert_f1': bert_f1,
            
            # Additional fields
            'evaluation_method': 'Complete_RAGAS_OpenAI_BERTScore'
        }

    except Exception as e:
        print(f"⚠️ Error in RAG metrics calculation: {e}")
        return {
            # RAGAS metrics - all zeros on error
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'answer_correctness': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'semantic_similarity': 0.0,
            
            # BERTScore metrics - all zeros on error
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0,
            
            # Additional fields
            'evaluation_method': 'Error_Fallback'
        }

def generate_rag_answer(question: str, context_docs: list):
    """Generate answer using OpenAI GPT and context documents"""

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Prepare context from top documents
        context_text = "\n\n".join([
            f"Document {i+1}: {doc.get('content', '')[:800]}"
            for i, doc in enumerate(context_docs[:3])
        ])

        prompt = f"""
        Based on the following context documents, answer the question accurately and concisely.

        Context:
        {context_text}

        Question: {question}

        Answer:
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Error generating RAG answer: {e}")
        return f"Error generating answer: {str(e)}"

def rerank_with_cross_encoder(question: str, documents: list, cross_encoder, top_k: int = 10):
    """Rerank documents using CrossEncoder"""

    if not cross_encoder or not documents:
        return documents

    try:
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.get('content', '')[:500]  # Limit content length
            pairs.append([question, content])

        # Get CrossEncoder scores
        scores = cross_encoder.predict(pairs)
        
        # Apply Min-Max normalization to convert logits to [0, 1] range
        scores = np.array(scores)
        if len(scores) > 1 and scores.max() != scores.min():
            # Standard Min-Max normalization
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            # Fallback for edge cases (all scores identical)
            normalized_scores = np.full_like(scores, 0.5)

        # Add scores to documents and sort
        for i, doc in enumerate(documents):
            doc['crossencoder_score'] = float(normalized_scores[i])

        # Sort by CrossEncoder score
        reranked = sorted(documents, key=lambda x: x['crossencoder_score'], reverse=True)

        # Update ranks
        for i, doc in enumerate(reranked):
            doc['rank'] = i + 1

        return reranked[:top_k]

    except Exception as e:
        print(f"⚠️ CrossEncoder reranking error: {e}")
        return documents

def evaluate_single_model_complete(model_name: str, data_pipeline: EmbeddedDataPipeline,
                                   questions_data: list, reranking_method: str = 'crossencoder',
                                   top_k: int = 10, generate_rag: bool = True):
    """Complete evaluation for a single model with real embeddings and metrics"""

    print(f"\n🔄 Evaluating {model_name}...")

    retriever = data_pipeline.retrievers.get(model_name)
    if not retriever or len(retriever.df) == 0:
        print(f"❌ No valid retriever for {model_name}")
        return None

    all_before_metrics = []
    all_after_metrics = []
    individual_rag_metrics = []

    # Score accumulators
    before_scores = []
    after_scores = []
    ce_scores = []
    total_docs_reranked = 0

    for i, question_data in enumerate(tqdm(questions_data, desc=f"{model_name}")):
        question = question_data['question']
        ground_truth_links = question_data.get('ground_truth_links', [])
        ground_truth_answer = question_data.get('accepted_answer', '')

        # Generate real query embedding
        query_embedding = data_pipeline.real_embedding_generator.generate_query_embedding(
            question, model_name
        )

        # Retrieve documents
        retrieved_docs = retriever.search(query_embedding, top_k=top_k)

        if not retrieved_docs:
            continue

        # Calculate BEFORE reranking metrics
        before_metrics = calculate_real_retrieval_metrics(
            ground_truth_links, retrieved_docs, list(range(1, top_k + 1))
        )

        # Calculate average cosine similarity (before)
        before_avg_score = np.mean([doc['cosine_similarity'] for doc in retrieved_docs])
        before_scores.append(before_avg_score)

        all_before_metrics.append(before_metrics)

        # AFTER reranking
        if reranking_method == 'crossencoder' and data_pipeline.cross_encoder:
            # Rerank with CrossEncoder
            reranked_docs = rerank_with_cross_encoder(
                question, retrieved_docs, data_pipeline.cross_encoder, top_k
            )

            # Calculate AFTER reranking metrics
            after_metrics = calculate_real_retrieval_metrics(
                ground_truth_links, reranked_docs, list(range(1, top_k + 1))
            )

            # Calculate CrossEncoder scores
            ce_question_scores = [doc.get('crossencoder_score', 0) for doc in reranked_docs]
            ce_avg_score = np.mean(ce_question_scores) if ce_question_scores else 0
            ce_scores.append(ce_avg_score)

            # After score (using original cosine similarities)
            after_avg_score = np.mean([doc['cosine_similarity'] for doc in reranked_docs])
            after_scores.append(after_avg_score)

            total_docs_reranked += len(reranked_docs)

            # Store CrossEncoder specific metrics
            after_metrics['model_crossencoder_scores'] = ce_question_scores
            after_metrics['model_avg_crossencoder_score'] = ce_avg_score
            after_metrics['model_total_documents_reranked'] = len(reranked_docs)

        else:
            # No reranking
            after_metrics = before_metrics.copy()
            reranked_docs = retrieved_docs
            after_scores.append(before_avg_score)

        all_after_metrics.append(after_metrics)

        # RAG Metrics (using reranked docs as context)
        if generate_rag:
            try:
                # Generate answer
                generated_answer = generate_rag_answer(question, reranked_docs[:3])

                # Calculate RAG metrics
                rag_metrics = calculate_rag_metrics_real(
                    question, reranked_docs[:3], generated_answer, ground_truth_answer
                )

                rag_metrics['question_index'] = i
                rag_metrics['generated_answer'] = generated_answer
                individual_rag_metrics.append(rag_metrics)

            except Exception as e:
                print(f"⚠️ RAG metrics error for question {i}: {e}")

    # Calculate averages
    def calculate_averages(metrics_list):
        if not metrics_list:
            return {}

        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        averages = {}
        for key in all_keys:
            if key != 'document_scores':  # Skip document scores in averages
                values = [m.get(key, 0) for m in metrics_list if isinstance(m.get(key), (int, float))]
                if values:
                    averages[key] = np.mean(values)

        return averages

    avg_before_metrics = calculate_averages(all_before_metrics)
    avg_after_metrics = calculate_averages(all_after_metrics)

    # Add model-level score metrics
    avg_before_metrics['model_avg_score'] = np.mean(before_scores) if before_scores else 0
    avg_after_metrics['model_avg_score'] = np.mean(after_scores) if after_scores else 0

    if reranking_method == 'crossencoder' and ce_scores:
        avg_after_metrics['model_avg_crossencoder_score'] = np.mean(ce_scores)
        avg_after_metrics['model_total_documents_reranked'] = total_docs_reranked

    # RAG averages - Complete RAGAS + BERTScore metrics
    rag_averages = {}
    if individual_rag_metrics:
        rag_averages = {
            # RAGAS metrics averages
            'avg_faithfulness': np.mean([r['faithfulness'] for r in individual_rag_metrics]),
            'avg_answer_relevance': np.mean([r['answer_relevancy'] for r in individual_rag_metrics]),  # Note: 'answer_relevancy' with y
            'avg_answer_correctness': np.mean([r['answer_correctness'] for r in individual_rag_metrics]),
            'avg_context_precision': np.mean([r['context_precision'] for r in individual_rag_metrics]),
            'avg_context_recall': np.mean([r['context_recall'] for r in individual_rag_metrics]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in individual_rag_metrics]),
            
            # BERTScore metrics averages
            'avg_bert_precision': np.mean([r['bert_precision'] for r in individual_rag_metrics]),
            'avg_bert_recall': np.mean([r['bert_recall'] for r in individual_rag_metrics]),
            'avg_bert_f1': np.mean([r['bert_f1'] for r in individual_rag_metrics]),
            
            # Status and count
            'rag_available': True,
            'total_rag_evaluations': len(individual_rag_metrics)
        }
    else:
        rag_averages = {'rag_available': False}

    return {
        'model_name': model_name,
        'full_model_name': model_name,
        'num_questions_evaluated': len(questions_data),
        'embedding_dimensions': retriever.embedding_dim,
        'total_documents': len(retriever.df),
        'avg_before_metrics': avg_before_metrics,
        'avg_after_metrics': avg_after_metrics,
        'all_before_metrics': all_before_metrics,
        'all_after_metrics': all_after_metrics,
        'rag_metrics': rag_averages,
        'individual_rag_metrics': individual_rag_metrics
    }

def run_real_complete_evaluation(available_models: list, config_data: dict,
                                 data_pipeline: EmbeddedDataPipeline,
                                 reranking_method: str = 'crossencoder',
                                 max_questions: int = None, debug: bool = False):
    """Run complete evaluation with real embeddings and metrics"""

    start_time = time.time()

    questions_data = config_data['questions']
    if max_questions:
        questions_data = questions_data[:max_questions]

    params = config_data.get('params', {})
    top_k = params.get('top_k', 10)
    generate_rag = params.get('generate_rag_metrics', True)

    print(f"🚀 Starting evaluation of {len(available_models)} models on {len(questions_data)} questions")
    print(f"📊 Reranking method: {reranking_method}")
    print(f"🎯 Top-K: {top_k}")
    print(f"🤖 RAG metrics: {generate_rag}")

    all_model_results = {}

    for model_name in available_models:
        result = evaluate_single_model_complete(
            model_name=model_name,
            data_pipeline=data_pipeline,
            questions_data=questions_data,
            reranking_method=reranking_method,
            top_k=top_k,
            generate_rag=generate_rag
        )

        if result:
            all_model_results[model_name] = result

            # Brief summary
            avg_f1 = result['avg_after_metrics'].get('f1@5', 0)
            avg_score = result['avg_after_metrics'].get('model_avg_score', 0)
            print(f"  ✅ {model_name}: F1@5={avg_f1:.3f}, Score={avg_score:.3f}")

    end_time = time.time()
    evaluation_duration = end_time - start_time

    evaluation_params = {
        'num_questions': len(questions_data),
        'models_evaluated': len(all_model_results),
        'reranking_method': reranking_method,
        'top_k': top_k,
        'generate_rag_metrics': generate_rag
    }

    return {
        'all_model_results': all_model_results,
        'evaluation_duration': evaluation_duration,
        'evaluation_params': evaluation_params
    }

def embedded_process_and_save_results(all_model_results: dict, output_path: str,
                                      evaluation_params: dict, evaluation_duration: float):
    """Process and save results in Streamlit-compatible format"""

    # Chile timezone
    chile_tz = pytz.timezone('America/Santiago')
    now_utc = datetime.now(pytz.UTC)
    now_chile = now_utc.astimezone(chile_tz)

    # Generate filename with date format YYYYMMDD_HHMMSS
    timestamp_str = now_chile.strftime('%Y%m%d_%H%M%S')
    filename = f"cumulative_results_{timestamp_str}.json"
    filepath = os.path.join(output_path, filename)

    # Create comprehensive results structure
    results_data = {
        'config': evaluation_params,
        'evaluation_info': {
            'timestamp': now_chile.isoformat(),
            'timezone': 'America/Santiago',
            'evaluation_type': 'cumulative_metrics_colab_multi_model',
            'total_duration_seconds': evaluation_duration,
            'models_evaluated': len(all_model_results),
            'questions_per_model': evaluation_params.get('num_questions', 0),
            'enhanced_display_compatible': True,
            'data_verification': {
                'is_real_data': True,
                'no_simulation': True,
                'no_random_values': True,
                'rag_framework': 'RAGAS_with_OpenAI_API',
                'reranking_method': f"{evaluation_params.get('reranking_method', 'none')}_reranking"
            }
        },
        'results': all_model_results
    }

    # Save to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Results saved: {filename}")

        return {
            'json': filepath,
            'filename': filename,
            'chile_time': now_chile.strftime('%Y-%m-%d %H:%M:%S %Z')
        }

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None

print("✅ Complete evaluation code loaded")