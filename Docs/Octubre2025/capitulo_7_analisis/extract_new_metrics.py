#!/usr/bin/env python3
"""
Script to extract all metrics from NEW JSON (20251114) for chapter 7 verification
"""
import json
import sys

# Load JSON
json_path = '../cumulative_results_20251114_071914.json'
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded: {json_path}")
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    sys.exit(1)

print("\n" + "=" * 100)
print("DATOS DEL NUEVO ARCHIVO (20251114_071914)")
print("=" * 100)

print("\n" + "=" * 100)
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
print("PRECISION, RECALL, F1, NDCG, MAP POR k (BEFORE RERANKING)")
print("=" * 100)

for k in [3, 5, 10, 15]:
    print(f"\n📊 k={k}")
    print(f"{'Modelo':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'NDCG':<12} {'MAP':<12}")
    print("-" * 70)
    for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
        m = data['results'][model]['avg_before_metrics']
        prec = m.get(f'precision@{k}', 0)
        rec = m.get(f'recall@{k}', 0)
        f1 = m.get(f'f1@{k}', 0)
        ndcg = m.get(f'ndcg@{k}', 0)
        map_k = m.get(f'map@{k}', 0)
        print(f"{model:<10} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f} {ndcg:<12.3f} {map_k:<12.3f}")

print("\n" + "=" * 100)
print("PRECISION, RECALL, F1, NDCG, MAP POR k (AFTER RERANKING)")
print("=" * 100)

for k in [3, 5, 10, 15]:
    print(f"\n📊 k={k}")
    print(f"{'Modelo':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'NDCG':<12} {'MAP':<12}")
    print("-" * 70)
    for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
        m = data['results'][model]['avg_after_metrics']
        prec = m.get(f'precision@{k}', 0)
        rec = m.get(f'recall@{k}', 0)
        f1 = m.get(f'f1@{k}', 0)
        ndcg = m.get(f'ndcg@{k}', 0)
        map_k = m.get(f'map@{k}', 0)
        print(f"{model:<10} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f} {ndcg:<12.3f} {map_k:<12.3f}")

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
    faithful = r.get('avg_faithfulness', 0) or 0
    ans_rel = r.get('avg_answer_relevance', 0) or 0
    ctx_prec = r.get('avg_context_precision', 0) or 0
    ctx_rec = r.get('avg_context_recall', 0) or 0
    sem_sim = r.get('avg_semantic_similarity', 0) or 0
    print(f"{model:<10} {faithful:<10.3f} {ans_rel:<10.3f} {ctx_prec:<10.3f} {ctx_rec:<10.3f} {sem_sim:<10.3f}")

print("\n" + "=" * 100)
print("TABLA 7.15: BERTSCORE METRICS")
print("=" * 100)
print(f"{'Modelo':<10} {'BERT Prec':<12} {'BERT Rec':<12} {'BERT F1':<12}")
print("-" * 100)

for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    r = data['results'][model]['rag_metrics']
    bert_prec = r.get('avg_bert_precision')
    bert_rec = r.get('avg_bert_recall')
    bert_f1 = r.get('avg_bert_f1')

    # Handle None values
    if bert_prec is not None and bert_rec is not None and bert_f1 is not None:
        print(f"{model:<10} {bert_prec:<12.3f} {bert_rec:<12.3f} {bert_f1:<12.3f}")
    else:
        print(f"{model:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} (valores None en JSON)")

print("\n" + "=" * 100)
print("VERIFICACIÓN: ¿HAY BERT F1 EN EL JSON?")
print("=" * 100)
for model in ['ada', 'mpnet', 'e5-large', 'minilm']:
    r = data['results'][model]['rag_metrics']
    print(f"{model}: avg_bert_f1 = {r.get('avg_bert_f1')}")

print("\n" + "=" * 100)
print("ANÁLISIS COMPLETO - DATOS EXTRAÍDOS CON ÉXITO")
print("=" * 100)
