"""
Local model management for cost-free inference.
Supports Llama 3.1 8B, Mistral 7B, and local SentenceTransformers.
"""

import os
import gc
import torch
import threading
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModelManager:
    """Manages local LLM models with memory optimization."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_memory_gb = 8  # Adjust based on your system
        
        logger.info(f"LocalModelManager initialized on device: {self.device}")
        
    def get_quantization_config(self) -> BitsAndBytesConfig:
        """Get quantization config for memory optimization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self, model_name: str, model_path: str) -> bool:
        """Load a model with memory optimization."""
        try:
            with self.model_lock:
                if model_name in self.models:
                    logger.info(f"Model {model_name} already loaded")
                    return True
                
                logger.info(f"Loading model: {model_name} from {model_path}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with quantization if GPU available
                if self.device == "cuda":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=self.get_quantization_config(),
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    model.to(self.device)
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                
                logger.info(f"Successfully loaded {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def generate_text(self, model_name: str, prompt: str, max_length: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text using the specified model."""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not loaded")
                return f"Error: Model {model_name} not available"
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                             max_length=2048).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text with {model_name}: {e}")
            return f"Error generating response: {e}"
    
    def unload_model(self, model_name: str):
        """Unload a specific model to free memory."""
        try:
            with self.model_lock:
                if model_name in self.models:
                    del self.models[model_name]
                    del self.tokenizers[model_name]
                    logger.info(f"Unloaded model: {model_name}")
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    def cleanup(self):
        """Cleanup all loaded models."""
        with self.model_lock:
            for model_name in list(self.models.keys()):
                self.unload_model(model_name)
            logger.info("All models cleaned up")


class LocalLlamaClient:
    """Client for Llama 3.1 8B model."""
    
    def __init__(self, model_manager: LocalModelManager):
        self.model_manager = model_manager
        self.model_name = "llama-3.1-8b"
        self.model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.loaded = False
    
    def ensure_loaded(self) -> bool:
        """Ensure the model is loaded."""
        if not self.loaded:
            self.loaded = self.model_manager.load_model(self.model_name, self.model_path)
        return self.loaded
    
    def generate_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Generate answer using Llama 3.1 8B."""
        if not self.ensure_loaded():
            return "Error: Llama model not available"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context. 
Be concise and accurate. If the context doesn't contain enough information, say so.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return self.model_manager.generate_text(
            self.model_name, prompt, max_length=max_length, temperature=0.1
        )


class LocalMistralClient:
    """Client for Mistral 7B model."""
    
    def __init__(self, model_manager: LocalModelManager):
        self.model_manager = model_manager
        self.model_name = "mistral-7b"
        self.model_path = "mistralai/Mistral-7B-Instruct-v0.3"
        self.loaded = False
    
    def ensure_loaded(self) -> bool:
        """Ensure the model is loaded."""
        if not self.loaded:
            self.loaded = self.model_manager.load_model(self.model_name, self.model_path)
        return self.loaded
    
    def refine_query(self, query: str) -> str:
        """Refine query using Mistral 7B."""
        if not self.ensure_loaded():
            return query  # Fallback to original query
        
        prompt = f"""<s>[INST] You are a query refinement expert. Your task is to clean and improve the following user query for better search results.

Remove greetings, pleasantries, and unnecessary words. Make it clear and concise.

Original query: {query}

Refined query: [/INST]"""
        
        response = self.model_manager.generate_text(
            self.model_name, prompt, max_length=100, temperature=0.1
        )
        
        # Extract the refined query from the response
        if response and len(response.strip()) > 0:
            return response.strip()
        return query


# Global model manager instance
_model_manager = None

def get_model_manager() -> LocalModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = LocalModelManager()
    return _model_manager

def get_llama_client() -> LocalLlamaClient:
    """Get Llama client instance."""
    return LocalLlamaClient(get_model_manager())

def get_mistral_client() -> LocalMistralClient:
    """Get Mistral client instance."""
    return LocalMistralClient(get_model_manager())

def cleanup_models():
    """Cleanup all loaded models."""
    global _model_manager
    if _model_manager:
        _model_manager.cleanup()
        _model_manager = None