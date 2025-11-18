#!/usr/bin/env python3
"""
Script to extract all metrics from JSON for chapter 7 verification
"""
import json

# Load JSON
with open('/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251114_071914.json', 'r') as f:
    data = json.load(f)

print("=" * 100)
print("TABLA 7.1: BEFORE RERANKING (k=5)")
print("=" * 100)
print(f"{'Modelo':<10} {'Precision@5':<12} {'Recall@5':<10} {'F1@5':<8} {'NDCG@5':<8} {'MAP@5':<8} {'MRR':<8}")
print("-" * 100)

for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    m = data['results'][model]['avg_before_metrics']
    print(f"{model:<10} {m['precision@5']:<12.3f} {m['recall@5']:<10.3f} {m['f1@5']:<8.3f} {m['ndcg@5']:<8.3f} {m['map@5']:<8.3f} {m['mrr']:<8.3f}")

print("\n" + "=" * 100)
print("TABLA 7.8: AFTER RERANKING (k=5)")
print("=" * 100)
print(f"{'Modelo':<10} {'Precision@5':<12} {'Recall@5':<10} {'F1@5':<8} {'NDCG@5':<8} {'MAP@5':<8} {'MRR':<8}")
print("-" * 100)

for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    m = data['results'][model]['avg_after_metrics']
    print(f"{model:<10} {m['precision@5']:<12.3f} {m['recall@5']:<10.3f} {m['f1@5']:<8.3f} {m['ndcg@5']:<8.3f} {m['map@5']:<8.3f} {m['mrr']:<8.3f}")

print("\n" + "=" * 100)
print("TABLA 7.12: IMPACTO DEL RERANKING")
print("=" * 100)

for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    before = data['results'][model]['avg_before_metrics']
    after = data['results'][model]['avg_after_metrics']

    print(f"\n{model.upper()}:")
    for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']:
        b = before[metric]
        a = after[metric]
        delta = a - b
        pct = (delta / b) * 100 if b != 0 else 0
        print(f"  {metric:12s}: {b:.3f} → {a:.3f} ({delta:+.3f}, {pct:+.1f}%)")

print("\n" + "=" * 100)
print("TABLA 7.14: RAGAS METRICS")
print("=" * 100)
print(f"{'Modelo':<10} {'Faithful':<10} {'Ans.Rel':<10} {'Ctx.Prec':<10} {'Ctx.Rec':<10} {'Sem.Sim':<10}")
print("-" * 100)

for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    r = data['results'][model]['rag_metrics']
    print(f"{model:<10} {r['avg_faithfulness']:<10.3f} {r['avg_answer_relevance']:<10.3f} {r['avg_context_precision']:<10.3f} {r['avg_context_recall']:<10.3f} {r['avg_semantic_similarity']:<10.3f}")

print("\n" + "=" * 100)
print("TABLA 7.15: BERTSCORE METRICS")
print("=" * 100)
print(f"{'Modelo':<10} {'BERT Prec':<12} {'BERT Rec':<12} {'BERT F1':<12}")
print("-" * 100)

for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    r = data['results'][model]['rag_metrics']
    print(f"{model:<10} {r['avg_bert_precision']:<12.3f} {r['avg_bert_recall']:<12.3f} {r['avg_bert_f1']:<12.3f}")

print("\n" + "=" * 100)
print("PRECISION@k, RECALL@k, F1@k, NDCG@k FOR k={3,5,10,15}")
print("=" * 100)

for k in [3, 5, 10, 15]:
    print(f"\n📊 k={k}")
    print(f"{'Modelo':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'NDCG':<12}")
    print("-" * 60)
    for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
        m = data['results'][model]['avg_before_metrics']
        print(f"{model:<10} {m[f'precision@{k}']:<12.3f} {m[f'recall@{k}']:<12.3f} {m[f'f1@{k}']:<12.3f} {m[f'ndcg@{k}']:<12.3f}")
