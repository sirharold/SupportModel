"""
Cliente OpenRouter para Llama-4-Scout y otros modelos
"""

import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
import streamlit as st

# Ensure environment variables are loaded
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Cliente para interactuar con modelos de OpenRouter"""
    
    def __init__(self, api_key: str = None):
        """
        Inicializa el cliente OpenRouter
        
        Args:
            api_key: API key de OpenRouter. Si no se proporciona, se obtiene del environment
        """
        self.api_key = api_key or os.getenv('OPEN_ROUTER_KEY')
        
        logger.info(f"OpenRouter API key found: {bool(self.api_key)}")
        if self.api_key:
            logger.info(f"API key starts with: {self.api_key[:10]}...")
        
        if not self.api_key:
            logger.error("OpenRouter API key not found in environment")
            raise ValueError("OpenRouter API key is required. Please set OPEN_ROUTER_KEY environment variable.")
        
        # Crear cliente OpenAI con base_url de OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        logger.info("OpenRouter client initialized successfully")
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con OpenRouter haciendo una petición simple.
        
        Returns:
            True si la conexión es exitosa, False en caso contrario
        """
        try:
            # Hacer una petición simple para probar la conexión
            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout:free",
                messages=[
                    {"role": "user", "content": "Hi"}
                ],
                max_tokens=3,
                temperature=0.1
            )
            return True
        except Exception as e:
            error_str = str(e).lower()
            # Don't consider 503 errors as connection failures - they indicate the API is working but model unavailable
            if "503" in error_str and ("no instances available" in error_str or "provider returned error" in error_str):
                logger.warning(f"Model temporarily unavailable but connection OK: {e}")
                return True  # Connection is working, model just temporarily unavailable
            logger.error(f"Connection test failed: {e}")
            return False
    
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        model: str = "meta-llama/llama-4-scout:free",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        """
        Genera respuesta usando modelo de OpenRouter
        
        Args:
            question: Pregunta del usuario
            context: Contexto de documentos relevantes
            model: Modelo a usar (default: llama-4-scout:free)
            max_tokens: Máximo número de tokens
            temperature: Temperatura para generación
            
        Returns:
            Respuesta generada por el modelo
        """
        try:
            # Construir prompt para respuesta
            prompt = f"""You are a Microsoft Azure expert assistant. Answer the question based on the provided context.

INSTRUCTIONS:
- Provide a clear, comprehensive answer in Spanish
- Structure your response with proper formatting
- Be accurate and only use information from the context
- If context is insufficient, clearly state what information is missing

Context: {context}

Question: {question}

Provide a detailed answer in Spanish:"""
            
            # Realizar request a OpenRouter
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://azure-qa-support.app",  # Para rankings
                    "X-Title": "Azure Q&A Support System",  # Para rankings
                },
                extra_body={},
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extraer respuesta
            response = completion.choices[0].message.content
            
            # Limpiar respuesta
            if response:
                response = response.strip()
                # Remover prefijo "Answer:" si está presente
                if response.startswith("Answer:"):
                    response = response[7:].strip()
                
                # Verificar que la respuesta no esté vacía después de la limpieza
                if not response:
                    logger.warning("Empty response after cleaning")
                    return "El modelo generó una respuesta vacía. Por favor, intenta reformular tu pregunta."
            else:
                logger.warning("No response content from OpenRouter")
                return "No se recibió respuesta del modelo. Por favor, intenta nuevamente."
            
            logger.info(f"OpenRouter response generated: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error generating OpenRouter response: {e}")
            error_str = str(e).lower()
            
            # Provide more specific error messages
            if "api_key" in error_str:
                return "Error: OpenRouter API key inválida o no configurada. Verifica tu OPEN_ROUTER_KEY."
            elif "rate" in error_str or "limit" in error_str:
                return "Error: Límite de rate excedido. Espera unos segundos y vuelve a intentar."
            elif "timeout" in error_str:
                return "Error: Timeout en la conexión. Verifica tu conexión a internet."
            elif "503" in error_str and "no instances available" in error_str:
                return "Error: El modelo llama-4-scout no está disponible temporalmente en OpenRouter. Intenta con otro modelo o espera unos minutos."
            elif "503" in error_str:
                return "Error: Servicio temporalmente no disponible en OpenRouter. Por favor, intenta nuevamente en unos minutos."
            elif "provider returned error" in error_str:
                return "Error: El proveedor del modelo está experimentando problemas. Intenta con otro modelo o espera unos minutos."
            else:
                return f"Error: No se pudo generar respuesta - {str(e)}"
    
    def refine_query(self, query: str, model: str = "meta-llama/llama-4-scout:free") -> str:
        """
        Refina una query usando modelo de OpenRouter
        
        Args:
            query: Query original
            model: Modelo a usar
            
        Returns:
            Query refinada
        """
        try:
            prompt = f"""You are a query refinement expert. Your task is to clean and improve the following user query for better search results.

Remove greetings, pleasantries, and unnecessary words. Make it clear and concise.

Original query: {query}

Refined query:"""
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://azure-qa-support.app",
                    "X-Title": "Azure Q&A Support System",
                },
                extra_body={},
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            response = completion.choices[0].message.content
            
            if response and response.strip():
                return response.strip()
            else:
                return query
                
        except Exception as e:
            logger.error(f"Error refining query with OpenRouter: {e}")
            # Provide more specific error messages for query refinement
            if "api_key" in str(e).lower():
                logger.warning("OpenRouter API key error during query refinement, using original query")
            elif "rate" in str(e).lower():
                logger.warning("Rate limit exceeded during query refinement, using original query")
            return query  # Fallback to original query

class OpenRouterLlama4Client:
    """Cliente específico para Llama-4-Scout"""
    
    def __init__(self, api_key: str = None):
        """Inicializa cliente para Llama-4-Scout"""
        self.client = OpenRouterClient(api_key)
        self.model_name = "meta-llama/llama-4-scout:free"
        
    def generate_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Genera respuesta usando Llama-4-Scout"""
        return self.client.generate_answer(
            question=question,
            context=context,
            model=self.model_name,
            max_tokens=max_length,
            temperature=0.1
        )
    
    def refine_query(self, query: str) -> str:
        """Refina query usando Llama-4-Scout"""
        return self.client.refine_query(query, self.model_name)

# Funciones de conveniencia
def get_openrouter_client(api_key: str = None) -> OpenRouterClient:
    """Obtiene cliente OpenRouter"""
    return OpenRouterClient(api_key)

def get_llama4_scout_client(api_key: str = None) -> OpenRouterLlama4Client:
    """Obtiene cliente específico para Llama-4-Scout"""
    return OpenRouterLlama4Client(api_key)

# Cache para Streamlit
@st.cache_resource
def get_cached_llama4_scout_client():
    """Obtiene cliente Llama-4-Scout con cache de Streamlit"""
    return get_llama4_scout_client()