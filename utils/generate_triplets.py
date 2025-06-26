import json
import re
import random
from typing import List, Dict

AZURE_DOCS_PATTERN = r"https://learn\.microsoft\.com[\w/\-\?=&%.]+"

def extract_links(text: str) -> List[str]:
    return list(set(re.findall(AZURE_DOCS_PATTERN, text or "")))

def build_triplet_dataset(train_set_path: str, output_path: str, max_negatives: int = 3):
    with open(train_set_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Construir triplets
    triplets = []
    for q in train_data:
        question = q.get("question_content", "").strip()
        answer = q.get("accepted_answer", "")
        positive_links = extract_links(answer)

        if not question or not positive_links:
            continue

        # Tripleta positiva: usar primer link válido
        positive = positive_links[0]

        # Seleccionar negativos: otros documentos no referenciados
        other_links = [
            other.get("accepted_answer", "")
            for other in train_data if other != q
        ]
        flat_links = set(link for text in other_links for link in extract_links(text))
        negatives = list(flat_links - set(positive_links))

        sampled_negatives = random.sample(negatives, min(len(negatives), max_negatives))

        for neg_link in sampled_negatives:
            triplets.append({
                "question": question,
                "positive_link": positive,
                "negative_link": neg_link
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)

    print(f"✅ Triplet dataset saved to {output_path} with {len(triplets)} samples.")

if __name__ == "__main__":
    build_triplet_dataset("data/train_set.json", "data/train_triplets.json")
