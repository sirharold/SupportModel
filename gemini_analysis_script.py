
import json
import sys

# Cargar el archivo JSON
file_path = '/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/data/cumulative_results_20251010_131215.json'
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: El archivo no se encontró en la ruta {file_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: El archivo {file_path} no es un JSON válido.")
    sys.exit(1)

results = {}
model_names = data.get('results', {}).keys()

for model in model_names:
    model_data = data['results'][model]
    
    # --- Análisis "Antes" --- #
    all_before_metrics = model_data.get('all_before_metrics', [])
    top_1_similarities = []
    if all_before_metrics:
        for item in all_before_metrics:
            doc_scores = item.get('document_scores', [])
            if doc_scores:
                top_1_similarities.append(doc_scores[0].get('cosine_similarity', 0))
    
    avg_top_1_sim = sum(top_1_similarities) / len(top_1_similarities) if top_1_similarities else 0

    # --- Análisis "Después" --- #
    ragas_bert_evals = model_data.get('ragas_bert_evaluations', [])
    context_precision = []
    context_recall = []
    answer_correctness = []
    bert_f1 = []

    if ragas_bert_evals:
        for item in ragas_bert_evals:
            context_precision.append(item.get('context_precision', 0))
            context_recall.append(item.get('context_recall', 0))
            answer_correctness.append(item.get('answer_correctness', 0))
            bert_f1.append(item.get('bert_f1', 0))

    avg_context_precision = sum(context_precision) / len(context_precision) if context_precision else 0
    avg_context_recall = sum(context_recall) / len(context_recall) if context_recall else 0
    avg_answer_correctness = sum(answer_correctness) / len(answer_correctness) if answer_correctness else 0
    avg_bert_f1 = sum(bert_f1) / len(bert_f1) if bert_f1 else 0

    results[model] = {
        "avg_top_1_retrieval_similarity": avg_top_1_sim,
        "avg_context_precision": avg_context_precision,
        "avg_context_recall": avg_context_recall,
        "avg_answer_correctness": avg_answer_correctness,
        "avg_bert_f1": avg_bert_f1
    }

# Imprimir resultados en una tabla formateada
print(f"{'Modelo':<15} | {'Similitud Top-1 (Antes)':<25} | {'Context Precision (Después)':<28} | {'Context Recall (Después)':<27} | {'Answer Correctness (Después)':<30} | {'BERT F1 (Después)':<20}")
print("-" * 160)

for model, metrics in results.items():
    print(f"{model:<15} | {metrics['avg_top_1_retrieval_similarity']:<25.4f} | {metrics['avg_context_precision']:<28.4f} | {metrics['avg_context_recall']:<27.4f} | {metrics['avg_answer_correctness']:<30.4f} | {metrics['avg_bert_f1']:<20.4f}")
