"""
Local answer generation using Llama 3.1 8B and Mistral 7B models.
Cost-free alternative to OpenAI and Gemini APIs.
"""

from typing import List, Dict, Tuple, Optional
import logging
from utils.local_models import get_llama_client, get_mistral_client

logger = logging.getLogger(__name__)

def generate_final_answer_local(
    question: str,
    retrieved_docs: List[dict],
    model_name: str = "llama-3.1-8b",
    max_length: int = 512
) -> Tuple[str, Dict]:
    """
    Generate final answer using local models.
    
    Args:
        question: User's question
        retrieved_docs: List of relevant documents
        model_name: Local model to use ('llama-3.1-8b' or 'mistral-7b')
        max_length: Maximum response length
        
    Returns:
        Tuple of (generated_answer, generation_info)
    """
    generation_info = {
        "status": "success",
        "model": model_name,
        "docs_used": len(retrieved_docs),
        "local_model": True,
        "cost": 0.0
    }
    
    try:
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Use top 5 documents
            title = doc.get('title', 'N/A')
            content = doc.get('content', '')
            link = doc.get('link', '')
            
            # Truncate content to avoid context overflow
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"Document {i+1}:\nTitle: {title}\nContent: {content}\nSource: {link}\n")
        
        context = "\n".join(context_parts)
        
        # Generate answer using the specified local model
        if model_name == "llama-3.1-8b":
            llama_client = get_llama_client()
            answer = llama_client.generate_answer(question, context, max_length)
        elif model_name == "mistral-7b":
            mistral_client = get_mistral_client()
            answer = mistral_client.generate_answer(question, context, max_length)
        else:
            raise ValueError(f"Unknown local model: {model_name}")
        
        # Validate answer
        if not answer or answer.startswith("Error"):
            generation_info["status"] = "error"
            generation_info["error"] = answer if answer else "Empty response"
            return "Lo siento, no pude generar una respuesta con el modelo local.", generation_info
        
        # Clean up the answer
        answer = answer.strip()
        
        # Add source information
        if retrieved_docs:
            source_links = []
            for doc in retrieved_docs[:3]:  # Show top 3 sources
                link = doc.get('link', '')
                title = doc.get('title', 'N/A')
                if link:
                    source_links.append(f"- [{title}]({link})")
            
            if source_links:
                answer += "\n\n**Fuentes:**\n" + "\n".join(source_links)
        
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
    model_name: str = "llama-3.1-8b"
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
    from utils.local_models import LocalMistralClient
    
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