"""
Local answer generation using Llama 3.1 8B and Mistral 7B models.
Cost-free alternative to OpenAI and Gemini APIs.
"""

from typing import List, Dict, Tuple, Optional
import logging
from src.services.local_models import get_tinyllama_client, get_mistral_client

logger = logging.getLogger(__name__)

def generate_final_answer_local(
    question: str,
    retrieved_docs: List[dict],
    model_name: str = "tinyllama-1.1b",
    max_length: int = 512
) -> Tuple[str, Dict]:
    """
    Generate final answer using local models.
    
    Args:
        question: User's question
        retrieved_docs: List of relevant documents
        model_name: Local model to use ('tinyllama-1.1b' or 'mistral-7b')
        max_length: Maximum response length
        
    Returns:
        Tuple of (generated_answer, generation_info)
    """
    import time
    start_time = time.time()
    
    generation_info = {
        "status": "success",
        "model": model_name,
        "docs_used": len(retrieved_docs),
        "local_model": True,
        "cost": 0.0
    }
    
    try:
        # Prepare context from retrieved documents
        # Build optimized context for faster generation
        context_parts = []
        max_docs = 3 if model_name == "tinyllama-1.1b" else 5  # Fewer docs for TinyLlama
        max_content_per_doc = 300 if model_name == "tinyllama-1.1b" else 500
        
        for i, doc in enumerate(retrieved_docs[:max_docs]):
            title = doc.get('title', 'N/A')
            content = doc.get('content', '')
            link = doc.get('link', '')
            
            # More aggressive truncation for TinyLlama speed
            if len(content) > max_content_per_doc:
                content = content[:max_content_per_doc] + "..."
            
            # Simplified format for faster processing
            context_parts.append(f"Doc {i+1}: {title}\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Log context size for monitoring
        logger.info(f"Context prepared: {len(context)} chars, {len(context_parts)} docs for {model_name}")
        
        # Generate answer using the specified local model
        logger.info(f"Generating answer using {model_name} with {len(retrieved_docs)} documents")
        
        if model_name == "tinyllama-1.1b":
            try:
                tinyllama_client = get_tinyllama_client()
                logger.info(f"TinyLlama client obtained, generating answer...")
                answer = tinyllama_client.generate_answer(question, context, max_length)
                logger.info(f"Answer generated successfully: {len(answer) if answer else 0} characters")
            except Exception as e:
                logger.error(f"Error with TinyLlama client: {e}")
                answer = f"Error: {str(e)}"
        elif model_name == "mistral-7b":
            try:
                mistral_client = get_mistral_client()
                logger.info(f"Mistral client obtained, generating answer...")
                answer = mistral_client.generate_answer(question, context, max_length)
                logger.info(f"Answer generated successfully: {len(answer) if answer else 0} characters")
            except Exception as e:
                logger.error(f"Error with Mistral client: {e}")
                if "gated repo" in str(e) or "Access to model" in str(e):
                    answer = "Error: Mistral requiere autorización. Usa TinyLlama en su lugar."
                elif "sentencepiece" in str(e):
                    answer = "Error: Dependencias faltantes para Mistral. Usa TinyLlama en su lugar."
                elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                    answer = "Error: Mistral es muy grande para descargar. Usa TinyLlama (más liviano)."
                else:
                    answer = f"Error: Mistral no disponible - {str(e)[:100]}. Usa TinyLlama."
        else:
            raise ValueError(f"Unknown local model: {model_name}")
        
        # Validate answer
        if not answer or answer.startswith("Error"):
            error_msg = answer if answer else "Empty response"
            logger.error(f"Local model generation failed: {error_msg}")
            generation_info["status"] = "error"
            generation_info["error"] = error_msg
            return f"Lo siento, no pude generar una respuesta con el modelo local. Error: {error_msg}", generation_info
        
        # Calculate generation metrics
        generation_time = time.time() - start_time
        generation_info.update({
            "generation_time": generation_time,
            "context_length": len(context),
            "answer_length": len(answer) if answer else 0
        })
        
        logger.info(f"Answer generated in {generation_time:.2f}s (context: {len(context)} chars, answer: {len(answer) if answer else 0} chars)")
        
        # Clean up the answer
        answer = answer.strip()
        
        # Add source information - ensure at least 3 Microsoft Learn links
        if retrieved_docs:
            source_links = []
            for doc in retrieved_docs[:6]:  # Check up to 6 documents
                link = doc.get('link', '')
                title = doc.get('title', 'N/A')
                if link and 'learn.microsoft.com' in link:
                    source_links.append(f"- **{title}**  \n  {link}")
            
            # Ensure we have at least 3 links if available
            if source_links:
                answer += "\n\n## Enlaces y Referencias\n\n"
                answer += "\n\n".join(source_links[:max(3, len(source_links))])
                answer += "\n\n*Consulta la documentación oficial de Microsoft Learn para información más detallada.*"
            elif retrieved_docs:
                # Fallback: show any available links even if not from learn.microsoft.com
                fallback_links = []
                for doc in retrieved_docs[:3]:
                    link = doc.get('link', '')
                    title = doc.get('title', 'N/A')
                    if link:
                        fallback_links.append(f"- **{title}**  \n  {link}")
                
                if fallback_links:
                    answer += "\n\n## Enlaces y Referencias\n\n"
                    answer += "\n\n".join(fallback_links)
        
        generation_info["answer_length"] = len(answer)
        return answer, generation_info
        
    except Exception as e:
        logger.error(f"Error generating answer with local model: {e}")
        generation_info["status"] = "error"
        generation_info["error"] = str(e)
        return f"Error generando respuesta: {e}", generation_info


def refine_query_local(query: str, model_name: str = "mistral-7b") -> Tuple[str, str]:
    """
    Refine query using local models.
    
    Args:
        query: Original user query
        model_name: Model to use for refinement
        
    Returns:
        Tuple of (refined_query, refinement_log)
    """
    try:
        if model_name == "mistral-7b":
            mistral_client = get_mistral_client()
            refined_query = mistral_client.refine_query(query)
        else:
            # Fallback to original query if model not supported
            refined_query = query
        
        if refined_query and refined_query.strip() != query.strip():
            refinement_log = f"🔹 Query refined locally with {model_name}: {refined_query}"
        else:
            refinement_log = f"🔹 Query refinement skipped or returned original query"
            refined_query = query
        
        return refined_query, refinement_log
        
    except Exception as e:
        logger.error(f"Error refining query with local model: {e}")
        refinement_log = f"🔹 Query refinement failed: {e}. Using original query."
        return query, refinement_log


def evaluate_answer_quality_local(
    question: str,
    answer: str,
    source_docs: List[dict],
    model_name: str = "tinyllama-1.1b"
) -> Dict:
    """
    Evaluate answer quality using local models.
    
    Args:
        question: Original question
        answer: Generated answer
        source_docs: Source documents used
        model_name: Local model to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluation_metrics = {
        "evaluation_model": model_name,
        "local_evaluation": True,
        "cost": 0.0
    }
    
    try:
        # Simple heuristic evaluation for now
        # In production, you could use a more sophisticated local evaluation model
        
        # Basic metrics
        evaluation_metrics.update({
            "answer_length": len(answer),
            "question_length": len(question),
            "docs_used": len(source_docs),
            "has_sources": "**Fuentes:**" in answer,
            "answer_quality": "good" if len(answer) > 50 else "short"
        })
        
        # Could add more sophisticated evaluation using local models
        # For example, using a local BERT model to check semantic similarity
        
        return evaluation_metrics
        
    except Exception as e:
        logger.error(f"Error evaluating answer quality: {e}")
        evaluation_metrics["evaluation_error"] = str(e)
        return evaluation_metrics


# Additional helper function for Mistral answer generation
def _generate_answer_mistral(question: str, context: str, max_length: int = 512) -> str:
    """Helper function to generate answers using Mistral 7B."""
    mistral_client = get_mistral_client()
    
    # Since Mistral is smaller, we'll use a simpler prompt
    prompt = f"""<s>[INST] Based on the following context, answer the user's question. Be concise and accurate.

Context: {context}

Question: {question}

Answer: [/INST]"""
    
    return mistral_client.model_manager.generate_text(
        mistral_client.model_name, prompt, max_length=max_length, temperature=0.1
    )


# Add method to MistralClient for answer generation
def _add_answer_generation_to_mistral():
    """Add answer generation capability to MistralClient."""
    from src.services.local_models import LocalMistralClient
    
    def generate_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Generate answer using Mistral 7B."""
        if not self.ensure_loaded():
            return "Error: Mistral model not available"
        
        prompt = f"""<s>[INST] Based on the following context, answer the user's question. Be concise and accurate.

Context: {context}

Question: {question}

Answer: [/INST]"""
        
        return self.model_manager.generate_text(
            self.model_name, prompt, max_length=max_length, temperature=0.1
        )
    
    LocalMistralClient.generate_answer = generate_answer

# Apply the enhancement
_add_answer_generation_to_mistral()