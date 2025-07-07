from typing import List, Dict
from bert_score import score as bert_scorer
from rouge_score import rouge_scorer
import numpy as np
from utils.auth import ensure_huggingface_login

def compute_ndcg(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@k)."""
    # Implementation...
    return 0.0

def compute_mrr(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
    """Computes Mean Reciprocal Rank (MRR@k)."""
    # Implementation...
    return 0.0

def compute_precision_recall_f1(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> tuple[float, float, float]:
    """Computes Precision, Recall, and F1-score @k."""
    # Implementation...
    return 0.0, 0.0, 0.0

from utils.weaviate_utils_improved import WeaviateConfig

def calculate_content_metrics(retrieved_docs: List[Dict], ground_truth_answer: str, top_n: int = 3) -> Dict:
    """
    Calculates BERTScore and ROUGE scores based on retrieved content.
    """
    if not retrieved_docs or not ground_truth_answer:
        return {}

    # Concatenate the content of the top N documents
    candidate_text = " ".join([doc.get('content', '') for doc in retrieved_docs[:top_n]])
    
    if not candidate_text.strip():
        return {"error": "No content found in retrieved documents."}

    # BERTScore
    bert_scores = {}
    try:
        # Load config to get the key
        config = WeaviateConfig.from_env()
        hf_api_key = config.huggingface_api_key
        
        print("[DEBUG METRICS] Ensuring Hugging Face login...")
        ensure_huggingface_login(token=hf_api_key)
        
        print(f"[DEBUG METRICS] Calculating BERTScore...")
        P, R, F1 = bert_scorer([candidate_text], [ground_truth_answer], lang="en", verbose=False, model_type='roberta-large-mnli')
        bert_scores = {
            "BERT_P": P.mean().item(),
            "BERT_R": R.mean().item(),
            "BERT_F1": F1.mean().item(),
        }
        print(f"[DEBUG METRICS] BERTScore calculated successfully: {bert_scores}")
    except Exception as e:
        error_message = f"Error calculating BERTScore: {e}"
        print(f"[DEBUG METRICS] {error_message}")
        bert_scores = {"BERT_P": "Error", "BERT_R": "Error", "BERT_F1": error_message} # Make error visible in UI

    # ROUGE Score
    rouge_scores = {}
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth_answer, candidate_text)
        rouge_scores = {
            "ROUGE1": scores['rouge1'].fmeasure,
            "ROUGE2": scores['rouge2'].fmeasure,
            "ROUGE-L": scores['rougeL'].fmeasure,
        }
    except Exception as e:
        print(f"Error calculating ROUGE score: {e}")
        rouge_scores = {}

    return {**bert_scores, **rouge_scores}