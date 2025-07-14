"""
Enhanced evaluation system that integrates advanced RAG metrics
with the existing comparison framework.
"""

import time
from typing import Dict, List, Tuple
from utils.advanced_rag_metrics import (
    calculate_hallucination_score,
    calculate_context_utilization,
    calculate_answer_completeness,
    calculate_user_satisfaction_proxy
)
from utils.qa_pipeline import answer_question_with_rag

def evaluate_rag_with_advanced_metrics(
    question: str,
    weaviate_wrapper,
    embedding_client,
    openai_client,
    gemini_client=None,
    local_tinyllama_client=None,
    local_mistral_client=None,
    generative_model_name: str = "tinyllama-1.1b",
    top_k: int = 10
) -> Dict:
    """
    Perform complete RAG evaluation with advanced metrics.
    
    Args:
        question: User question
        weaviate_wrapper: Weaviate client
        embedding_client: Embedding client
        openai_client: OpenAI client
        gemini_client: Gemini client (optional)
        local_tinyllama_client: Local Llama client (optional)
        local_mistral_client: Local Mistral client (optional)
        generative_model_name: Name of generative model
        top_k: Number of documents to retrieve
    
    Returns:
        Dict with comprehensive evaluation results
    """
    start_time = time.time()
    
    # 1. Execute RAG pipeline
    try:
        results, debug_info, generated_answer, rag_metrics = answer_question_with_rag(
            question=question,
            weaviate_wrapper=weaviate_wrapper,
            embedding_client=embedding_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            local_tinyllama_client=local_tinyllama_client,
            local_mistral_client=local_mistral_client,
            top_k=top_k,
            generative_model_name=generative_model_name,
            evaluate_quality=True
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "advanced_metrics": {}
        }
    
    # 2. Calculate advanced metrics if answer generation was successful
    advanced_metrics = {}
    
    # Calculate advanced metrics if we have a generated answer, regardless of RAG status
    # This allows evaluation of fallback answers when local models fail
    if generated_answer and generated_answer.strip():
        try:
            # Hallucination Detection
            hallucination_metrics = calculate_hallucination_score(
                answer=generated_answer,
                context_docs=results,
                question=question
            )
            
            # Context Utilization
            context_metrics = calculate_context_utilization(
                answer=generated_answer,
                context_docs=results,
                question=question
            )
            
            # Answer Completeness
            completeness_metrics = calculate_answer_completeness(
                answer=generated_answer,
                question=question,
                context_docs=results
            )
            
            # User Satisfaction Proxy
            satisfaction_metrics = calculate_user_satisfaction_proxy(
                answer=generated_answer,
                question=question,
                context_docs=results,
                response_time=response_time
            )
            
            advanced_metrics = {
                "hallucination": hallucination_metrics,
                "context_utilization": context_metrics,
                "completeness": completeness_metrics,
                "satisfaction": satisfaction_metrics
            }
            
        except Exception as e:
            advanced_metrics = {"error": f"Error calculating advanced metrics: {e}"}
    
    # 3. Combine all results
    evaluation_result = {
        "question": question,
        "model": generative_model_name,
        "response_time": response_time,
        "documents_retrieved": len(results),
        "generated_answer": generated_answer,
        "basic_rag_metrics": rag_metrics,
        "advanced_metrics": advanced_metrics,
        "debug_info": debug_info,
        "status": "success" if generated_answer else "failed"
    }
    
    return evaluation_result

def batch_evaluate_with_advanced_metrics(
    questions: List[str],
    weaviate_wrapper,
    embedding_client,
    openai_client,
    gemini_client=None,
    local_tinyllama_client=None,
    local_mistral_client=None,
    generative_model_name: str = "tinyllama-1.1b",
    top_k: int = 10
) -> List[Dict]:
    """
    Perform batch evaluation with advanced metrics.
    
    Args:
        questions: List of questions to evaluate
        ... (other args same as single evaluation)
    
    Returns:
        List of evaluation results
    """
    results = []
    
    for i, question in enumerate(questions):
        print(f"Evaluating question {i+1}/{len(questions)}: {question[:50]}...")
        
        evaluation = evaluate_rag_with_advanced_metrics(
            question=question,
            weaviate_wrapper=weaviate_wrapper,
            embedding_client=embedding_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            local_tinyllama_client=local_tinyllama_client,
            local_mistral_client=local_mistral_client,
            generative_model_name=generative_model_name,
            top_k=top_k
        )
        
        results.append(evaluation)
    
    return results

def create_advanced_metrics_summary(evaluations: List[Dict]) -> Dict:
    """
    Create summary statistics from multiple evaluations.
    
    Args:
        evaluations: List of evaluation results
    
    Returns:
        Dict with summary statistics
    """
    if not evaluations:
        return {}
    
    # Filter successful evaluations
    successful = [e for e in evaluations if e.get('status') == 'success' and 'advanced_metrics' in e]
    
    if not successful:
        return {"error": "No successful evaluations to summarize"}
    
    # Extract metrics
    hallucination_scores = []
    utilization_scores = []
    completeness_scores = []
    satisfaction_scores = []
    response_times = []
    
    for eval_result in successful:
        adv_metrics = eval_result.get('advanced_metrics', {})
        
        if 'hallucination' in adv_metrics:
            hallucination_scores.append(adv_metrics['hallucination']['hallucination_score'])
        
        if 'context_utilization' in adv_metrics:
            utilization_scores.append(adv_metrics['context_utilization']['utilization_score'])
        
        if 'completeness' in adv_metrics:
            completeness_scores.append(adv_metrics['completeness']['completeness_score'])
        
        if 'satisfaction' in adv_metrics:
            satisfaction_scores.append(adv_metrics['satisfaction']['satisfaction_score'])
        
        response_times.append(eval_result.get('response_time', 0))
    
    # Calculate summary statistics
    def safe_mean(values):
        return sum(values) / len(values) if values else 0
    
    def safe_std(values):
        if len(values) < 2:
            return 0
        mean_val = safe_mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    summary = {
        "total_evaluations": len(evaluations),
        "successful_evaluations": len(successful),
        "success_rate": len(successful) / len(evaluations),
        
        # Hallucination metrics (lower is better)
        "avg_hallucination_score": safe_mean(hallucination_scores),
        "std_hallucination_score": safe_std(hallucination_scores),
        
        # Context utilization (higher is better)
        "avg_utilization_score": safe_mean(utilization_scores),
        "std_utilization_score": safe_std(utilization_scores),
        
        # Completeness (higher is better)
        "avg_completeness_score": safe_mean(completeness_scores),
        "std_completeness_score": safe_std(completeness_scores),
        
        # User satisfaction (higher is better)
        "avg_satisfaction_score": safe_mean(satisfaction_scores),
        "std_satisfaction_score": safe_std(satisfaction_scores),
        
        # Performance
        "avg_response_time": safe_mean(response_times),
        "std_response_time": safe_std(response_times),
        
        # Quality grades
        "quality_grade": calculate_overall_quality_grade(
            safe_mean(hallucination_scores),
            safe_mean(utilization_scores),
            safe_mean(completeness_scores),
            safe_mean(satisfaction_scores)
        )
    }
    
    return summary

def calculate_overall_quality_grade(
    hallucination_score: float,
    utilization_score: float,
    completeness_score: float,
    satisfaction_score: float
) -> str:
    """
    Calculate an overall quality grade based on advanced metrics.
    
    Args:
        hallucination_score: 0-1 (lower is better)
        utilization_score: 0-1 (higher is better)
        completeness_score: 0-1 (higher is better)
        satisfaction_score: 0-1 (higher is better)
    
    Returns:
        Quality grade (A+, A, B+, B, C+, C, D, F)
    """
    # Invert hallucination score (so higher is better for all metrics)
    adjusted_hallucination = 1 - hallucination_score
    
    # Calculate weighted average
    weights = {
        'hallucination': 0.3,  # Very important
        'utilization': 0.2,
        'completeness': 0.25,
        'satisfaction': 0.25
    }
    
    overall_score = (
        adjusted_hallucination * weights['hallucination'] +
        utilization_score * weights['utilization'] +
        completeness_score * weights['completeness'] +
        satisfaction_score * weights['satisfaction']
    )
    
    # Convert to letter grade
    if overall_score >= 0.95:
        return "A+"
    elif overall_score >= 0.90:
        return "A"
    elif overall_score >= 0.85:
        return "B+"
    elif overall_score >= 0.80:
        return "B"
    elif overall_score >= 0.75:
        return "C+"
    elif overall_score >= 0.70:
        return "C"
    elif overall_score >= 0.60:
        return "D"
    else:
        return "F"

def get_advanced_metrics_interpretation() -> Dict:
    """
    Get interpretation guide for advanced metrics.
    
    Returns:
        Dict with metric explanations and thresholds
    """
    return {
        "metrics": {
            "hallucination_score": {
                "description": "Porcentaje de información no soportada por el contexto",
                "range": "0.0 - 1.0",
                "interpretation": "0.0 = Sin alucinaciones, 1.0 = Completamente alucinado",
                "thresholds": {
                    "excellent": "< 0.1",
                    "good": "< 0.2", 
                    "acceptable": "< 0.3",
                    "poor": ">= 0.3"
                }
            },
            "utilization_score": {
                "description": "Qué tan bien se utiliza el contexto recuperado",
                "range": "0.0 - 1.0",
                "interpretation": "0.0 = No usa contexto, 1.0 = Usa todo el contexto",
                "thresholds": {
                    "excellent": "> 0.8",
                    "good": "> 0.6",
                    "acceptable": "> 0.4",
                    "poor": "<= 0.4"
                }
            },
            "completeness_score": {
                "description": "Completitud de la respuesta basada en tipo de pregunta",
                "range": "0.0 - 1.0", 
                "interpretation": "0.0 = Incompleta, 1.0 = Completamente completa",
                "thresholds": {
                    "excellent": "> 0.9",
                    "good": "> 0.7",
                    "acceptable": "> 0.5",
                    "poor": "<= 0.5"
                }
            },
            "satisfaction_score": {
                "description": "Proxy de satisfacción del usuario (claridad + directness + actionabilidad)",
                "range": "0.0 - 1.0",
                "interpretation": "0.0 = Muy insatisfactorio, 1.0 = Muy satisfactorio",
                "thresholds": {
                    "excellent": "> 0.8",
                    "good": "> 0.6",
                    "acceptable": "> 0.4", 
                    "poor": "<= 0.4"
                }
            }
        },
        "overall_quality_grades": {
            "A+": "Excelencia excepcional (95%+)",
            "A": "Excelente calidad (90-95%)",
            "B+": "Muy buena calidad (85-90%)",
            "B": "Buena calidad (80-85%)",
            "C+": "Calidad aceptable (75-80%)",
            "C": "Calidad mínima (70-75%)",
            "D": "Calidad insuficiente (60-70%)",
            "F": "Calidad inaceptable (<60%)"
        }
    }