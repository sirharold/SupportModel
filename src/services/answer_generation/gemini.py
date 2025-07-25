# utils/gemini_answer_generator.py
from typing import List, Dict, Tuple
import google.generativeai as genai
import json

def generate_final_answer_gemini(
    question: str,
    retrieved_docs: List[Dict],
    gemini_client: genai.GenerativeModel,
    max_context_length: int = 12000,
    include_citations: bool = True
) -> Tuple[str, Dict]:
    """
    Genera una respuesta final utilizando el modelo Gemini de Google.
    """
    if not retrieved_docs:
        return "No se encontraron documentos relevantes para responder tu pregunta.", {
            "status": "no_documents",
            "docs_used": 0,
            "context_length": 0
        }

    try:
        context, docs_used, context_info = _prepare_context(
            retrieved_docs, max_context_length, include_citations
        )

        prompt = _create_gemini_prompt(question, context, include_citations)
        
        tools = _get_gemini_tools()

        response = gemini_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500
            ),
            tools=tools
        )

        answer, generation_info = _extract_gemini_answer(response)

        debug_info = {
            "status": "success",
            "docs_used": docs_used,
            "context_length": len(context),
            "context_info": context_info,
            "generation_info": generation_info,
            "model_used": "gemini-1.5-flash"
        }

        return answer, debug_info

    except Exception as e:
        fallback_answer = _create_fallback_answer(question, retrieved_docs)
        return fallback_answer, {
            "status": "error",
            "error": str(e),
            "docs_used": len(retrieved_docs),
            "fallback_used": True
        }

def _prepare_context(docs: List[Dict], max_length: int, include_citations: bool) -> Tuple[str, int, Dict]:
    """Prepara el contexto para el prompt de Gemini."""
    # Esta función puede ser la misma que la de OpenAI, así que la podemos reutilizar.
    # Por simplicidad, la copio aquí, pero podría ser refactorizada a un módulo común.
    context_parts = []
    current_length = 0
    docs_used = 0
    
    sorted_docs = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, doc in enumerate(sorted_docs):
        title = doc.get('title', 'Sin título')
        content = doc.get('content', '')
        link = doc.get('link', '')
        score = doc.get('score', 0)
        
        max_doc_content = min(1500, max_length // max(len(sorted_docs), 3))
        if len(content) > max_doc_content:
            content = content[:max_doc_content] + "..."
        
        if include_citations:
            doc_section = f"""
[DOCUMENTO {i+1}] - Score: {score:.3f}
Título: {title}
URL: {link}
Contenido: {content}
---"""
        else:
            doc_section = f"""
Documento {i+1}:
{title}
{content}
---"""
        
        if current_length + len(doc_section) <= max_length:
            context_parts.append(doc_section)
            current_length += len(doc_section)
            docs_used += 1
        else:
            break
    
    context = "\n".join(context_parts)
    
    context_info = {
        "total_docs_available": len(docs),
        "docs_included": docs_used,
        "avg_score": sum(doc.get('score', 0) for doc in sorted_docs[:docs_used]) / docs_used if docs_used > 0 else 0,
    }
    
    return context, docs_used, context_info


def _create_gemini_prompt(question: str, context: str, include_citations: bool) -> str:
    """Crea el prompt para Gemini."""
    citation_instruction = ""
    if include_citations:
        citation_instruction = """
6. SIEMPRE incluye citas usando el formato [DOCUMENTO X] donde X es el número del documento.
7. Al final, incluye una sección "Referencias" con los enlaces de los documentos utilizados."""

    return f"""Eres un experto en Microsoft Azure que ayuda a desarrolladores y arquitectos de soluciones.

Tu tarea es proporcionar respuestas comprehensivas y precisas basándote ÚNICAMENTE en la documentación oficial proporcionada.

REGLAS IMPORTANTES:
1. Basa tu respuesta SOLO en la información proporcionada en los documentos.
2. Si la información no está en los documentos, di claramente que no tienes esa información.
3. Estructura tu respuesta de manera clara y organizada.
4. Incluye pasos específicos cuando sea apropiado.
5. Menciona consideraciones importantes y mejores prácticas.
{citation_instruction}

PREGUNTA DEL USUARIO:
{question}

DOCUMENTACIÓN DISPONIBLE:
{context}

Por favor, proporciona una respuesta comprehensiva basada en la documentación anterior.
"""

def _get_gemini_tools():
    """Define las herramientas para la generación estructurada de respuestas con Gemini."""
    return [
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="provide_comprehensive_answer",
                    description="Proporciona una respuesta comprehensiva y bien estructurada basada en documentación.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "answer": genai.protos.Schema(type=genai.protos.Type.STRING, description="La respuesta principal detallada y bien estructurada."),
                            "key_points": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(type=genai.protos.Type.STRING),
                                description="3-5 puntos clave más importantes de la respuesta."
                            ),
                            "confidence": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Confianza en la respuesta basada en la calidad de la documentación (0-1)."),
                            "completeness": genai.protos.Schema(type=genai.protos.Type.STRING, enum=["complete", "partial", "limited"], description="Si la documentación permite una respuesta completa, parcial o limitada."),
                            "references_used": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(type=genai.protos.Type.INTEGER),
                                description="Números de los documentos utilizados para generar la respuesta."
                            )
                        },
                        required=["answer", "key_points", "confidence", "completeness", "references_used"]
                    )
                )
            ]
        )
    ]

def _extract_gemini_answer(response) -> Tuple[str, Dict]:
    """Extrae la respuesta generada de la respuesta de Gemini."""
    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "provide_comprehensive_answer":
            args = {key: value for key, value in function_call.args.items()}
            
            answer = args.get("answer", "")
            key_points = args.get("key_points", [])
            
            if key_points:
                answer += "\n\n**Puntos Clave:**\n"
                for i, point in enumerate(key_points, 1):
                    answer += f"{i}. {point}\n"
            
            generation_info = {
                "confidence": args.get("confidence", 0.5),
                "completeness": args.get("completeness", "unknown"),
                "references_used": args.get("references_used", []),
                "key_points_count": len(key_points)
            }
            
            return answer, generation_info
        else:
            raise ValueError(f"Unexpected function call: {function_call.name}")
            
    except Exception as e:
        # Fallback a contenido de texto si falla la extracción de la función
        if response.text:
            return response.text, {"error": f"Error extracting function call: {e}", "fallback_to_text": True}
        return "Error procesando la respuesta generada.", {"error": str(e)}


def _create_fallback_answer(question: str, docs: List[Dict]) -> str:
    """Crea una respuesta de fallback cuando falla la generación."""
    # Esta función también es reutilizable.
    if not docs:
        return "No se encontraron documentos relevantes para tu pregunta."
    
    top_docs = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)[:3]
    
    fallback = f"No pude generar una respuesta completa, pero encontré documentación relevante sobre tu pregunta:\n\n"
    
    for i, doc in enumerate(top_docs, 1):
        title = doc.get('title', 'Sin título')
        link = doc.get('link', '#')
        score = doc.get('score', 0)
        
        fallback += f"{i}. **{title}** (Score: {score:.3f})\n"
        fallback += f"   🔗 {link}\n\n"
    
    fallback += "Te recomiendo revisar estos documentos para obtener información detallada."
    
    return fallback
