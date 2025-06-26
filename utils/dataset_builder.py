import re
import random
import json
import os
from typing import List, Tuple

AZURE_DOCS_PATTERN = r"https://learn\.microsoft\.com[\w/\-\?=&%.]+"

def filter_questions_with_links(qna_data: List[dict]) -> List[dict]:
    """
    Filtra las preguntas cuya respuesta aceptada contiene al menos un link a la documentación de Azure.
    """
    return [q for q in qna_data if re.search(AZURE_DOCS_PATTERN, q.get("accepted_answer", ""))]

def split_dataset(questions: List[dict], train_size: int = 2000, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Divide el dataset en conjunto de entrenamiento y validación.
    """
    random.seed(seed)
    random.shuffle(questions)
    return questions[:train_size], questions[train_size:]

def save_to_json(data: List[dict], filename: str):
    """
    Guarda una lista de diccionarios en un archivo JSON con indentación.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_and_save_dataset(weaviate_wrapper, output_dir: str = "data"):
    try:
        print("Fetching questions from 'Questions' collection...")
        questions_collection = weaviate_wrapper.client.collections.get("Questions")
        all_data = questions_collection.query.fetch_objects(limit=15000)
        items = [obj.properties for obj in all_data.objects]

        print(f"Total retrieved: {len(items)}")
        filtered = filter_questions_with_links(items)
        print(f"Questions with Azure links: {len(filtered)}")

        train_set, val_set = split_dataset(filtered, train_size=2000)

        os.makedirs(output_dir, exist_ok=True)
        save_to_json(train_set, f"{output_dir}/train_set.json")
        save_to_json(val_set, f"{output_dir}/val_set.json")

        print(f"Train set saved to {output_dir}/train_set.json ({len(train_set)} items)")
        print(f"Validation set saved to {output_dir}/val_set.json ({len(val_set)} items)")

    except Exception as e:
        print("Error building dataset:", e)

def is_successful_match(retrieved_docs: List[dict], ground_truth_links: List[str]) -> bool:
    """
    Verifica si alguno de los documentos recuperados contiene un link que aparece en la respuesta aceptada.
    """
    retrieved_links = {doc.get("link", "").strip() for doc in retrieved_docs}
    return any(link.strip() in retrieved_links for link in ground_truth_links)

def generate_classification_examples(train_data: List[dict], weaviate_wrapper, embedding_client, top_k: int = 10) -> List[dict]:
    """
    Genera ejemplos para entrenamiento supervisado binario: cada par pregunta-doc tendrá una etiqueta 1 si
    el link del documento aparece en la respuesta aceptada, y 0 en caso contrario.
    """
    examples = []
    doc_collection = weaviate_wrapper.client.collections.get("Documentation")

    for q in train_data:
        question_text = q.get("question_content", "")
        answer_text = q.get("accepted_answer", "")

        # Vector usando el cliente explícitamente
        vector = embedding_client.generate_embedding(question_text)
        retrieved_docs = doc_collection.query.near_vector(near_vector=vector, limit=top_k)

        links_in_answer = re.findall(AZURE_DOCS_PATTERN, answer_text)
        links_in_answer = list(set([link.strip() for link in links_in_answer]))

        for doc in retrieved_docs.objects:
            doc_text = doc.properties.get("content", "")
            doc_link = doc.properties.get("link", "").strip()
            label = 1 if doc_link in links_in_answer else 0

            examples.append({
                "question": question_text,
                "document": doc_text,
                "link": doc_link,
                "label": label
            })

    return examples

def build_embedding_train_set(train_json_path: str, weaviate_wrapper, embedding_client, output_path: str, top_k: int = 10):
    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    labeled_data = []
    doc_collection = weaviate_wrapper.client.collections.get("Documentation")

    for q in train_data:
        question_text = q.get("question_content", "")
        answer_text = q.get("accepted_answer", "")
        vector = embedding_client.generate_embedding(question_text)
        if not vector:
            continue

        retrieved_docs = doc_collection.query.near_vector(near_vector=vector, limit=top_k)
        links_in_answer = re.findall(AZURE_DOCS_PATTERN, answer_text)
        links_in_answer = list(set([link.strip() for link in links_in_answer]))

        for doc in retrieved_docs.objects:
            doc_content = doc.properties.get("content", "")
            doc_link = doc.properties.get("link", "").strip()
            label = 1 if doc_link in links_in_answer else 0
            doc_vector = embedding_client.generate_embedding(doc_content)
            if doc_vector:
                labeled_data.append({
                    "embedding": doc_vector,
                    "label": label
                })

    with open(output_path, "w") as f:
        json.dump(labeled_data, f, indent=2)
    print(f"✅ Labeled embedding dataset saved to: {output_path}")

def generate_embedding_dataset(labeled_data: List[dict], embedding_client, output_path: str):
    result = []
    for item in labeled_data:
        doc_text = item["document"]
        label = item["label"]

        doc_vector = embedding_client.generate_embedding(doc_text)
        if doc_vector:
            result.append({
                "embedding": doc_vector,
                "label": label
            })
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Embedding dataset saved to {output_path} ({len(result)} samples)")
