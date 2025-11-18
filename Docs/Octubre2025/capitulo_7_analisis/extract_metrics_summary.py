#!/usr/bin/env python3
"""
Script to extract key metrics from cumulative results JSON
"""
import json
import sys

def extract_key_metrics(json_path):
    """Extract key metrics from the JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 80)
    print("KEY METRICS EXTRACTION FROM CUMULATIVE RESULTS")
    print("=" * 80)

    # Config info
    config = data.get('config', {})
    eval_info = data.get('evaluation_info', {})
    results = data.get('results', {})

    print("\n1. CONFIGURATION:")
    print(f"   - Total questions: {config.get('num_questions', 'N/A')}")
    print(f"   - Models evaluated: {config.get('models_evaluated', 'N/A')}")
    print(f"   - Model names: {', '.join(results.keys())}")
    print(f"   - Reranking method: {config.get('reranking_method', 'N/A')}")
    print(f"   - Top K: {config.get('top_k', 'N/A')}")

    print("\n2. EVALUATION INFO:")
    print(f"   - Timestamp: {eval_info.get('timestamp', 'N/A')}")
    print(f"   - Total duration: {eval_info.get('total_duration_seconds', 'N/A'):.2f} seconds")
    print(f"   - Questions per model: {eval_info.get('questions_per_model', 'N/A')}")

    # Retrieval metrics summary (pre-reranking)
    print("\n3. RETRIEVAL METRICS (PRE-RERANKING - avg_before_metrics):")
    for model_name, model_data in results.items():
        print(f"\n   {model_name.upper()}:")
        if 'avg_before_metrics' in model_data:
            retrieval = model_data['avg_before_metrics']
            print(f"      Precision@5: {retrieval.get('precision_at_5', 0):.4f}")
            print(f"      Precision@10: {retrieval.get('precision_at_10', 0):.4f}")
            print(f"      Recall@5: {retrieval.get('recall_at_5', 0):.4f}")
            print(f"      Recall@10: {retrieval.get('recall_at_10', 0):.4f}")
            print(f"      MRR: {retrieval.get('mrr', 0):.4f}")
            print(f"      NDCG@10: {retrieval.get('ndcg_at_10', 0):.4f}")

    # Reranking metrics summary (post-reranking)
    print("\n4. RERANKING METRICS (POST-RERANKING - avg_after_metrics):")
    for model_name, model_data in results.items():
        print(f"\n   {model_name.upper()}:")
        if 'avg_after_metrics' in model_data:
            reranking = model_data['avg_after_metrics']
            print(f"      Precision@5: {reranking.get('precision_at_5', 0):.4f}")
            print(f"      Precision@10: {reranking.get('precision_at_10', 0):.4f}")
            print(f"      Recall@5: {reranking.get('recall_at_5', 0):.4f}")
            print(f"      Recall@10: {reranking.get('recall_at_10', 0):.4f}")
            print(f"      MRR: {reranking.get('mrr', 0):.4f}")
            print(f"      NDCG@10: {reranking.get('ndcg_at_10', 0):.4f}")

    # RAG metrics
    print("\n5. RAG METRICS (RAGAS + BERTScore):")
    for model_name, model_data in results.items():
        print(f"\n   {model_name.upper()}:")
        if 'rag_metrics' in model_data and model_data['rag_metrics']:
            rag = model_data['rag_metrics']
            print(f"      Faithfulness: {rag.get('avg_faithfulness', 'N/A'):.4f}" if rag.get('avg_faithfulness') is not None else "      Faithfulness: N/A")
            print(f"      Answer Relevancy: {rag.get('avg_answer_relevance', 'N/A'):.4f}" if rag.get('avg_answer_relevance') is not None else "      Answer Relevancy: N/A")
            print(f"      Answer Correctness: {rag.get('avg_answer_correctness', 'N/A'):.4f}" if rag.get('avg_answer_correctness') is not None else "      Answer Correctness: N/A")
            print(f"      Context Precision: {rag.get('avg_context_precision', 'N/A'):.4f}" if rag.get('avg_context_precision') is not None else "      Context Precision: N/A")
            print(f"      Context Recall: {rag.get('avg_context_recall', 'N/A'):.4f}" if rag.get('avg_context_recall') is not None else "      Context Recall: N/A")
            print(f"      Semantic Similarity: {rag.get('avg_semantic_similarity', 'N/A'):.4f}" if rag.get('avg_semantic_similarity') is not None else "      Semantic Similarity: N/A")
            print(f"      BERTScore Precision: {rag.get('avg_bert_precision', 'N/A'):.4f}" if rag.get('avg_bert_precision') is not None else "      BERTScore Precision: N/A")
            print(f"      BERTScore Recall: {rag.get('avg_bert_recall', 'N/A'):.4f}" if rag.get('avg_bert_recall') is not None else "      BERTScore Recall: N/A")
            print(f"      BERTScore F1: {rag.get('avg_bert_f1', 'N/A'):.4f}" if rag.get('avg_bert_f1') is not None else "      BERTScore F1: N/A (not calculated)")
            print(f"      Total evaluations: {rag.get('total_rag_evaluations', 'N/A')}")
        else:
            print(f"      (No RAG metrics available)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    json_path = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251114_071914.json"
    extract_key_metrics(json_path)
