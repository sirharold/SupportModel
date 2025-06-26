from typing import List, Dict


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
