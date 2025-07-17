"""
OpenRouter answer generation using Mistral-7B-Instruct and other models.
Cost-effective alternative to OpenAI with support for multiple models.
"""

from typing import List, Dict, Tuple, Optional
import logging
import time
from utils.openrouter_client import get_cached_deepseek_openrouter_client

logger = logging.getLogger(__name__)

def generate_final_answer_openrouter(
    question: str,
    retrieved_docs: List[dict],
    model_name: str = "deepseek-v3-chat",
    max_length: int = 512
) -> Tuple[str, Dict]:
    """
    Generate final answer using OpenRouter models.
    
    Args:
        question: User's question
        retrieved_docs: List of relevant documents
        model_name: OpenRouter model to use ('mistral-7b-instruct', etc.)
        max_length: Maximum response length
        
    Returns:
        Tuple of (generated_answer, generation_info)
    """
    start_time = time.time()
    
    generation_info = {
        "status": "success",
        "model": model_name,
        "docs_used": len(retrieved_docs),
        "openrouter_model": True,
        "cost": 0.0  # OpenRouter Mistral is free
    }
    
    try:
        # Prepare context from retrieved documents
        context_parts = []
        max_docs = 5  # Good balance for Mistral 7B
        max_content_per_doc = 600  # Generous for Mistral
        
        for i, doc in enumerate(retrieved_docs[:max_docs]):
            title = doc.get('title', 'N/A')
            content = doc.get('content', '')
            link = doc.get('link', '')
            
            # Truncate content if too long
            if len(content) > max_content_per_doc:
                content = content[:max_content_per_doc] + "..."
            
            # Format document
            context_parts.append(f"Document {i+1}: {title}\nContent: {content}\nSource: {link}\n")
        
        context = "\n".join(context_parts)
        
        # Log context size for monitoring
        logger.info(f"OpenRouter context prepared: {len(context)} chars, {len(context_parts)} docs for {model_name}")
        
        # Generate answer using OpenRouter model
        logger.info(f"Generating answer using OpenRouter {model_name} with {len(retrieved_docs)} documents")
        
        if model_name == "deepseek-v3-chat":
            try:
                deepseek_client = get_cached_deepseek_openrouter_client()
                logger.info(f"OpenRouter DeepSeek V3 client obtained, generating answer...")
                answer = deepseek_client.generate_answer(question, context, max_length)
                logger.info(f"OpenRouter answer generated successfully: {len(answer) if answer else 0} characters")
            except Exception as e:
                logger.error(f"Error with OpenRouter DeepSeek client: {e}")
                error_str = str(e).lower()
                if "api_key" in error_str:
                    answer = "Error: OpenRouter API key inválida o no configurada. Verifica tu OPEN_ROUTER_KEY."
                elif "rate" in error_str or "limit" in error_str:
                    answer = "Error: Límite de rate excedido en OpenRouter. Espera unos segundos y reintenta."
                elif "503" in error_str:
                    answer = "Error: Modelo DeepSeek temporalmente no disponible en OpenRouter."
                else:
                    answer = f"Error: OpenRouter DeepSeek no disponible - {str(e)[:100]}"
        else:
            raise ValueError(f"Unknown OpenRouter model: {model_name}")
        
        # Validate answer
        if not answer or answer.startswith("Error"):
            error_msg = answer if answer else "Empty response"
            logger.error(f"OpenRouter model generation failed: {error_msg}")
            generation_info["status"] = "error"
            generation_info["error"] = error_msg
            answer = f"Error generando respuesta con OpenRouter {model_name}: {error_msg}"
        else:
            # Clean up the answer
            answer = answer.strip()
            logger.info(f"OpenRouter {model_name} generated answer: {len(answer)} characters")
            
            # Add timing info
            generation_info["generation_time"] = time.time() - start_time
            generation_info["answer_length"] = len(answer)
            
    except Exception as e:
        error_msg = f"OpenRouter generation error: {str(e)}"
        logger.error(error_msg)
        generation_info["status"] = "error"
        generation_info["error"] = error_msg
        answer = f"Error: No se pudo generar respuesta con OpenRouter {model_name}"
    
    return answer, generation_info

def refine_query_openrouter(
    query: str,
    model_name: str = "deepseek-v3-chat"
) -> Tuple[str, str]:
    """
    Refine query using OpenRouter models.
    
    Args:
        query: Original query
        model_name: OpenRouter model to use for refinement
        
    Returns:
        Tuple of (refined_query, refinement_log)
    """
    try:
        if model_name == "deepseek-v3-chat":
            deepseek_client = get_cached_deepseek_openrouter_client()
            refined_query = deepseek_client.refine_query(query)
            
            if refined_query and refined_query.strip() != query:
                log = f"🔹 Query refined using OpenRouter {model_name}: {query} -> {refined_query}"
                return refined_query, log
            else:
                log = f"🔹 OpenRouter {model_name} refinement skipped to save API calls"
                return query, log
        else:
            log = f"🔹 Unknown OpenRouter model for refinement: {model_name}"
            return query, log
            
    except Exception as e:
        logger.error(f"Error refining query with OpenRouter {model_name}: {e}")
        log = f"🔹 OpenRouter query refinement disabled to save API calls. Using original query."
        return query, log