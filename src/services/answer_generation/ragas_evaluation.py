"""
Real RAGAS and BERTScore evaluation for individual search responses.
"""

from typing import Dict, List, Tuple
import numpy as np
from openai import OpenAI
from bert_score import score as bert_scorer
from src.data.json_utils import robust_json_parse
from src.services.auth.auth import ensure_huggingface_login
from src.services.storage.weaviate_utils import WeaviateConfig


def evaluate_answer_with_ragas_and_bertscore(
    question: str,
    answer: str,
    source_docs: List[Dict],
    openai_client: OpenAI
) -> Dict:
    """
    Evalúa la respuesta usando RAGAS real y BERTScore.
    
    Args:
        question: Pregunta original
        answer: Respuesta generada
        source_docs: Documentos fuente usados
        openai_client: Cliente OpenAI para RAGAS
        
    Returns:
        Dict con métricas RAGAS y BERTScore
    """
    metrics = {}
    
    # 1. RAGAS Metrics using OpenAI
    try:
        ragas_metrics = _calculate_ragas_metrics(question, answer, source_docs, openai_client)
        metrics.update(ragas_metrics)
    except Exception as e:
        print(f"[DEBUG] Error calculating RAGAS metrics: {e}")
        metrics.update({
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "ragas_error": str(e)
        })
    
    # 2. BERTScore
    try:
        bert_metrics = _calculate_bert_score(answer, source_docs)
        metrics.update(bert_metrics)
    except Exception as e:
        print(f"[DEBUG] Error calculating BERTScore: {e}")
        metrics.update({
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
            "bert_error": str(e)
        })
    
    return metrics


def _calculate_ragas_metrics(
    question: str,
    answer: str,
    source_docs: List[Dict],
    openai_client: OpenAI
) -> Dict:
    """Calcula métricas RAGAS usando OpenAI."""
    
    # Preparar contexto
    context_parts = []
    for i, doc in enumerate(source_docs[:5]):  # Top 5 docs
        context_parts.append(f"[{i+1}] {doc.get('title', '')}: {doc.get('content', '')[:300]}")
    context = "\n".join(context_parts)
    
    # Prompt para evaluación RAGAS
    evaluation_prompt = f"""
Evalúa esta respuesta RAG usando las métricas RAGAS estándar:

PREGUNTA: {question}

RESPUESTA: {answer}

CONTEXTO PROPORCIONADO:
{context}

Evalúa las siguientes métricas (0.0 a 1.0):

1. FAITHFULNESS: ¿Todas las afirmaciones en la respuesta están respaldadas por el contexto?
   - 1.0 = Cada afirmación está completamente respaldada
   - 0.0 = Las afirmaciones no tienen respaldo en el contexto

2. ANSWER_RELEVANCY: ¿La respuesta aborda directamente la pregunta?
   - 1.0 = Responde completamente la pregunta
   - 0.0 = No responde la pregunta

3. CONTEXT_PRECISION: ¿Los documentos más relevantes están priorizados en el contexto?
   - 1.0 = Orden perfecto por relevancia
   - 0.0 = Orden aleatorio o incorrecto

4. CONTEXT_RECALL: ¿El contexto contiene toda la información necesaria para responder?
   - 1.0 = Contiene toda la información necesaria
   - 0.0 = Falta información crítica
"""

    tools = [{
        "type": "function",
        "function": {
            "name": "evaluate_ragas",
            "description": "Evalúa métricas RAGAS",
            "parameters": {
                "type": "object",
                "properties": {
                    "faithfulness": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Fidelidad de la respuesta al contexto"
                    },
                    "answer_relevancy": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Relevancia de la respuesta a la pregunta"
                    },
                    "context_precision": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Precisión del orden del contexto"
                    },
                    "context_recall": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Completitud del contexto"
                    }
                },
                "required": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            }
        }
    }]
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un evaluador experto de sistemas RAG. Evalúa objetivamente basándote en los criterios RAGAS."},
            {"role": "user", "content": evaluation_prompt}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "evaluate_ragas"}},
        temperature=0.1,
        seed=42
    )
    
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        raw_arguments = tool_call.function.arguments
        tool_args, success, error = robust_json_parse(raw_arguments, {})
        
        if success:
            return {
                "faithfulness": tool_args.get("faithfulness", 0.0),
                "answer_relevancy": tool_args.get("answer_relevancy", 0.0),
                "context_precision": tool_args.get("context_precision", 0.0),
                "context_recall": tool_args.get("context_recall", 0.0)
            }
    
    raise Exception("Failed to extract RAGAS metrics from OpenAI response")


def _calculate_bert_score(answer: str, source_docs: List[Dict]) -> Dict:
    """Calcula BERTScore comparando la respuesta con el contenido de los documentos."""
    
    # Ensure HuggingFace login
    config = WeaviateConfig.from_env()
    ensure_huggingface_login(token=config.huggingface_api_key)
    
    # Concatenar contenido de los documentos top como referencia
    reference_texts = []
    for doc in source_docs[:3]:  # Top 3 docs
        content = doc.get('content', '')
        if content:
            # Tomar párrafos relevantes del contenido
            reference_texts.append(content[:1000])  # Primeros 1000 chars
    
    if not reference_texts:
        return {
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
            "bert_error": "No reference content available"
        }
    
    # Calcular BERTScore
    # Usamos el contenido combinado como referencia
    reference = " ".join(reference_texts)
    
    P, R, F1 = bert_scorer(
        [answer],  # Candidate
        [reference],  # Reference
        lang="es",  # Spanish
        model_type="microsoft/deberta-xlarge-mnli",
        num_layers=18,
        device="cpu",
        batch_size=1
    )
    
    return {
        "bert_precision": float(P[0]),
        "bert_recall": float(R[0]),
        "bert_f1": float(F1[0])
    }