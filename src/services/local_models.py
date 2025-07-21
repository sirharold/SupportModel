"""
Local model management for cost-free inference.
Supports TinyLlama 1.1B, Mistral 7B, and local SentenceTransformers.
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
                
                # Check if this is a large model and warn about requirements
                if "mistral" in model_path.lower():
                    import psutil
                    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
                    if available_memory < 6:
                        logger.error(f"Insufficient memory for Mistral: {available_memory:.1f}GB available, 6GB+ required")
                        return False
                    logger.warning(f"Loading Mistral with {available_memory:.1f}GB available memory")
                
                logger.info(f"Loading model: {model_name} from {model_path}")
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                
                # Special handling for large models like Mistral
                if "mistral" in model_path.lower():
                    logger.warning("Loading Mistral model - this may take several minutes and requires significant memory")
                    # Add timeout for large model downloads
                    import socket
                    original_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(60)  # 1 minute timeout
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
                    finally:
                        socket.setdefaulttimeout(original_timeout)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
                    
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info("Tokenizer loaded successfully")
                
                # Load model with quantization if GPU available
                logger.info(f"Loading model weights (this may take several minutes)...")
                if self.device == "cuda":
                    logger.info("Using GPU with 4-bit quantization")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=self.get_quantization_config(),
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        token=True
                    )
                else:
                    logger.info("Using CPU (this will be slower)")
                    # Special handling for Phi models
                    if "phi" in model_path.lower():
                        logger.info("Loading Phi model with optimized settings...")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            token=True,
                            attn_implementation="eager",  # Use eager attention for compatibility
                            _attn_implementation="eager"
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            token=True
                        )
                    logger.info("Moving model to CPU...")
                    model.to(self.device)
                
                logger.info("Model weights loaded, finalizing setup...")
                
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
            
            # Generate with fallback handling
            with torch.no_grad():
                # Optimized generation parameters for speed
                generation_kwargs = {
                    **inputs,
                    'max_new_tokens': min(max_length, 256),  # Limit for faster generation
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False,  # Disable cache to avoid DynamicCache issues
                }
                
                # Use faster generation settings
                if model_name == "tinyllama-1.1b":
                    # Optimized for TinyLlama speed
                    generation_kwargs.update({
                        'do_sample': False,  # Greedy decoding is faster
                        'num_beams': 1,      # No beam search for speed
                        'early_stopping': True,
                    })
                else:
                    # Standard settings for other models
                    generation_kwargs.update({
                        'temperature': temperature,
                        'top_p': top_p,
                        'do_sample': True,
                    })
                
                try:
                    outputs = model.generate(**generation_kwargs)
                except Exception as cache_error:
                    logger.warning(f"Generation error, trying minimal fallback: {cache_error}")
                    # Minimal fallback
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,  # Very short for speed
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,
                    )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Extract clean response based on model type
            if model_name == "tinyllama-1.1b":
                # Look for the assistant tag and extract what comes after
                assistant_marker = "<|assistant|>"
                if assistant_marker in response:
                    response = response.split(assistant_marker)[-1].strip()
                
                # Remove any remaining system or user tags
                response = response.replace("<|system|>", "").replace("<|user|>", "").replace("</s>", "").strip()
                
                # Remove common prefixes like "Answer:"
                if response.startswith("Answer:"):
                    response = response[7:].strip()
                elif response.startswith("Response:"):
                    response = response[9:].strip()
            
            elif model_name == "mistral-7b":
                # For Mistral, remove [INST] tags and extract response
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                
                # Remove common Mistral tags
                response = response.replace("<s>", "").replace("</s>", "").strip()
            
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


class LocalTinyLlamaClient:
    """Client for TinyLlama 1.1B model."""
    
    def __init__(self, model_manager: LocalModelManager):
        self.model_manager = model_manager
        self.model_name = "tinyllama-1.1b"
        self.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.loaded = False
    
    def ensure_loaded(self) -> bool:
        """Ensure the model is loaded."""
        if not self.loaded:
            logger.info(f"Loading TinyLlama model {self.model_name} from {self.model_path}")
            self.loaded = self.model_manager.load_model(self.model_name, self.model_path)
            if self.loaded:
                logger.info(f"Successfully loaded TinyLlama model {self.model_name}")
            else:
                logger.error(f"Failed to load TinyLlama model {self.model_name}")
        return self.loaded
    
    def generate_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Generate answer using TinyLlama 1.1B."""
        if not self.ensure_loaded():
            logger.error(f"TinyLlama model {self.model_name} could not be loaded")
            return "Error: TinyLlama model not available"
        
        prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the provided context. 
Be concise and accurate. If the context doesn't contain enough information, say so.
</s>
<|user|>
Context: {context}

Question: {question}
</s>
<|assistant|>
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
            logger.info(f"Loading Mistral model {self.model_name} from {self.model_path}")
            self.loaded = self.model_manager.load_model(self.model_name, self.model_path)
            if self.loaded:
                logger.info(f"Successfully loaded Mistral model {self.model_name}")
            else:
                logger.error(f"Failed to load Mistral model {self.model_name}")
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
    
    def generate_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Generate answer using Mistral 7B."""
        if not self.ensure_loaded():
            logger.error(f"Mistral model {self.model_name} could not be loaded")
            return "Error: Mistral model not available"
        
        prompt = f"""<s>[INST] You are a helpful assistant that answers questions based on the provided context.
Be concise and accurate. If the context doesn't contain enough information, say so.

Context: {context}

Question: {question} [/INST]"""
        
        return self.model_manager.generate_text(
            self.model_name, prompt, max_length=max_length, temperature=0.1
        )


# Global model manager instance
_model_manager = None

def get_model_manager() -> LocalModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = LocalModelManager()
    return _model_manager

def get_tinyllama_client() -> LocalTinyLlamaClient:
    """Get TinyLlama client instance."""
    return LocalTinyLlamaClient(get_model_manager())

# Preload function for better performance
def preload_tinyllama_model():
    """Preload TinyLlama model for faster subsequent use."""
    logger.info("Preloading TinyLlama model...")
    client = get_tinyllama_client()
    success = client.ensure_loaded()
    if success:
        logger.info("TinyLlama model preloaded successfully")
    else:
        logger.error("Failed to preload TinyLlama model")
    return success

def get_mistral_client() -> LocalMistralClient:
    """Get Mistral client instance."""
    return LocalMistralClient(get_model_manager())

def cleanup_models():
    """Cleanup all loaded models."""
    global _model_manager
    if _model_manager:
        _model_manager.cleanup()
        _model_manager = None