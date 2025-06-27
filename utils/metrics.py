from typing import List, Dict, Tuple
import math


def compute_answerability_score(ranked_docs: List[Dict], threshold: float = 0.75) -> float:
    """
    Calcula el porcentaje de documentos que superan un umbral de similitud.
    """
    if not ranked_docs:
        return 0.0
    strong_matches = [doc for doc in ranked_docs if doc.get("score", 0) >= threshold]
    return round(len(strong_matches) / len(ranked_docs) * 100, 2)


def summarize_ranking(ranked_docs: List[Dict], threshold: float = 0.75):
    """
    Muestra un resumen simple con un score de respuesta y estadísticas del ranking.
    """
    if not ranked_docs:
        print("❗ No documents to summarize.")
        return

    scores = [doc.get("score", 0) for doc in ranked_docs]
    max_score = round(max(scores), 3)
    avg_score = round(sum(scores) / len(scores), 3)
    answerability = compute_answerability_score(ranked_docs, threshold)

    print("\n📊 Ranking Summary")
    print("------------------")
    print(f"🔹 Top score: {max_score}")
    print(f"🔹 Average score: {avg_score}")
    print(f"✅ Estimated answerability: {answerability}% of documents are highly relevant (>{threshold})\n")


def compute_precision_recall_f1(
    ranked_docs: List[Dict],
    relevant_links: List[str],
    k: int | None = None
) -> Tuple[float, float, float]:
    """Return precision, recall and F1 for the top-k ranked documents."""
    if k is not None:
        ranked_docs = ranked_docs[:k]

    retrieved = [doc.get("link") for doc in ranked_docs if doc.get("link")]
    if not retrieved:
        return 0.0, 0.0, 0.0

    relevant_set = set(relevant_links)
    hits = [l for l in retrieved if l in relevant_set]
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_mrr(ranked_docs: List[Dict], relevant_links: List[str], k: int | None = None) -> float:
    """Compute Mean Reciprocal Rank for a ranked list."""
    if k is None:
        k = len(ranked_docs)

    relevant_set = set(relevant_links)
    for idx, doc in enumerate(ranked_docs[:k], start=1):
        if doc.get("link") in relevant_set:
            return 1.0 / idx
    return 0.0


def compute_ndcg(ranked_docs: List[Dict], relevant_links: List[str], k: int | None = None) -> float:
    """Compute normalized Discounted Cumulative Gain for the ranking."""
    if k is None:
        k = len(ranked_docs)

    relevance = [1 if doc.get("link") in relevant_links else 0 for doc in ranked_docs[:k]]
    dcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance))

    ideal_rels = [1] * min(len(relevant_links), k)
    idcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0
