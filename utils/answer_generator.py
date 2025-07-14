from typing import List, Dict, Tuple
from openai import OpenAI
import json
import re
from utils.json_utils import robust_json_parse

def generate_final_answer(
    question: str, 
    retrieved_docs: List[Dict], 
    openai_client: OpenAI,
    max_context_length: int = 12000,
    include_citations: bool = True
) -> Tuple[str, Dict]:
    """
    Genera una respuesta final sintetizando información de múltiples documentos.
    
    Args:
        question: Pregunta del usuario
        retrieved_docs: Lista de documentos recuperados con scores
        openai_client: Cliente de OpenAI
        max_context_length: Longitud máxima del contexto para el LLM
        include_citations: Si incluir citas en la respuesta
    
    Returns:
        Tuple de (respuesta_generada, info_debug)
    """
    print(f"[DEBUG] Generating answer for question: {question}")
    print(f"[DEBUG] Using {len(retrieved_docs)} documents for generation")
    
    if not retrieved_docs:
        return "No se encontraron documentos relevantes para responder tu pregunta.", {
            "status": "no_documents",
            "docs_used": 0,
            "context_length": 0
        }
    
    try:
        # 1. Preparar contexto con los mejores documentos
        context, docs_used, context_info = _prepare_context(
            retrieved_docs, max_context_length, include_citations
        )
        
        print(f"[DEBUG] Context prepared with {docs_used} documents, length: {len(context)}")
        
        # 2. Crear prompt para generación
        system_prompt = _create_system_prompt(include_citations)
        user_prompt = _create_user_prompt(question, context)
        
        # 3. Generar respuesta con OpenAI
        tools = _get_answer_generation_tools()
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "provide_comprehensive_answer"}},
            temperature=0.1,  # Temperatura muy baja para JSON más consistente
            max_tokens=1500,
            seed=42  # Añadir seed para más consistencia
        )
        
        # 4. Extraer respuesta estructurada
        answer, generation_info = _extract_generated_answer(response, retrieved_docs)
        
        # 5. Compilar información de debug
        debug_info = {
            "status": "success",
            "docs_used": docs_used,
            "context_length": len(context),
            "context_info": context_info,
            "generation_info": generation_info,
            "model_used": "gpt-4"
        }
        
        print(f"[DEBUG] Answer generated successfully, length: {len(answer)}")
        return answer, debug_info
        
    except Exception as e:
        print(f"[DEBUG] Error in answer generation: {e}")
        fallback_answer = _create_fallback_answer(question, retrieved_docs)
        return fallback_answer, {
            "status": "error",
            "error": str(e),
            "docs_used": len(retrieved_docs),
            "fallback_used": True
        }

def _prepare_context(docs: List[Dict], max_length: int, include_citations: bool) -> Tuple[str, int, Dict]:
    """Prepara el contexto optimizando el uso de tokens."""
    context_parts = []
    current_length = 0
    docs_used = 0
    
    # Priorizar documentos por score (ya deberían estar ordenados)
    sorted_docs = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, doc in enumerate(sorted_docs):
        # Preparar información del documento
        title = doc.get('title', 'Sin título')
        content = doc.get('content', '')
        link = doc.get('link', '')
        score = doc.get('score', 0)
        
        # Truncar contenido si es muy largo
        max_doc_content = min(1500, max_length // max(len(sorted_docs), 3))
        if len(content) > max_doc_content:
            content = content[:max_doc_content] + "..."
        
        # Crear sección del documento
        if include_citations:
            doc_section = f"""
[DOCUMENTO {i+1}] - Score: {score:.3f}
Título: {title}
**ENLACE OFICIAL**: {link}
Contenido: {content}
---"""
        else:
            doc_section = f"""
Documento {i+1}:
{title}
{content}
---"""
        
        # Verificar si podemos agregar este documento
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
        "score_range": [
            min(doc.get('score', 0) for doc in sorted_docs[:docs_used]) if docs_used > 0 else 0,
            max(doc.get('score', 0) for doc in sorted_docs[:docs_used]) if docs_used > 0 else 0
        ]
    }
    
    return context, docs_used, context_info

def _create_system_prompt(include_citations: bool) -> str:
    """Crea el prompt del sistema para la generación."""
    base_prompt = """Eres un experto en Microsoft Azure que ayuda a desarrolladores y arquitectos de soluciones. 

Tu tarea es proporcionar respuestas comprehensivas y precisas basándote ÚNICAMENTE en la documentación oficial proporcionada.

REGLAS IMPORTANTES:
1. Basa tu respuesta SOLO en la información proporcionada en los documentos
2. Si la información no está en los documentos, di claramente que no tienes esa información
3. Estructura tu respuesta de manera clara y organizada
4. Incluye pasos específicos cuando sea apropiado
5. Menciona consideraciones importantes y mejores prácticas"""

    if include_citations:
        base_prompt += """
6. SIEMPRE incluye citas usando el formato [DOCUMENTO X] donde X es el número del documento
7. OBLIGATORIO: Al final, incluye una sección "Enlaces y Referencias" con AL MENOS 3 enlaces de los documentos utilizados
8. Formato de enlaces: Usa el URL completo proporcionado en cada documento
9. Los enlaces deben ser funcionales y corresponder a los documentos citados en la respuesta"""
    
    return base_prompt

def _create_user_prompt(question: str, context: str) -> str:
    """Crea el prompt del usuario con la pregunta y contexto."""
    return f"""
PREGUNTA DEL USUARIO:
{question}

DOCUMENTACIÓN DISPONIBLE:
{context}

Por favor, proporciona una respuesta comprehensiva basada en la documentación anterior."""

def _sanitize_json_string(json_string: str) -> str:
    """
    Sanitiza una cadena JSON eliminando caracteres de control inválidos.
    
    Args:
        json_string: Cadena JSON potencialmente con caracteres de control
        
    Returns:
        Cadena JSON sanitizada
    """
    import re
    
    # Método más robusto: usar regex para remover todos los caracteres de control
    # excepto los permitidos en JSON (\t, \n, \r)
    # ASCII control characters (0-31) except \t(9), \n(10), \r(13)
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_string)
    
    # También remover caracteres Unicode problemáticos
    sanitized = re.sub(r'[\u0080-\u009F]', '', sanitized)  # C1 control characters
    sanitized = re.sub(r'[\u2028\u2029]', '', sanitized)   # Line/Paragraph separators
    
    # Fix common JSON issues without breaking valid JSON
    # Remove any remaining non-printable characters
    sanitized = re.sub(r'[^\x20-\x7E\t\n\r]', '', sanitized)
    
    return sanitized

def _ensure_links_included(answer: str, retrieved_docs: List[Dict], references_used: List[int]) -> str:
    """
    Asegura que la respuesta incluya al menos 3 enlaces oficiales de Microsoft Learn.
    
    Args:
        answer: Respuesta generada
        retrieved_docs: Documentos recuperados
        references_used: Números de documentos utilizados según el modelo
        
    Returns:
        Respuesta con enlaces garantizados
    """
    # Verificar si ya hay una sección de enlaces en la respuesta
    has_links_section = any(keyword in answer.lower() for keyword in [
        "enlaces", "referencias", "links", "documentación oficial"
    ])
    
    # Extraer enlaces de los documentos utilizados
    available_links = []
    for i, doc in enumerate(retrieved_docs[:6]):  # Tomar máximo 6 documentos
        link = doc.get('link', '')
        title = doc.get('title', f'Documento {i+1}')
        if link and 'learn.microsoft.com' in link:
            available_links.append({
                'title': title,
                'link': link,
                'doc_num': i + 1
            })
    
    # Si no hay sección de enlaces o hay menos de 3 enlaces, añadir/completar
    links_in_answer = answer.count('learn.microsoft.com')
    
    if not has_links_section or links_in_answer < 3:
        # Seleccionar los mejores enlaces
        links_to_add = available_links[:max(3, len(available_links))]
        
        if links_to_add:
            if not has_links_section:
                answer += "\n\n## Enlaces y Referencias\n\n"
            else:
                # Si ya hay una sección pero faltan enlaces, añadir antes del final
                answer += "\n\n**Enlaces adicionales:**\n\n"
            
            for link_info in links_to_add:
                answer += f"- **{link_info['title']}**  \n"
                answer += f"  {link_info['link']}\n\n"
            
            answer += "*Consulta la documentación oficial de Microsoft Learn para información más detallada.*"
    
    return answer

def _get_answer_generation_tools():
    """Define las herramientas para la generación estructurada de respuestas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "provide_comprehensive_answer",
                "description": "Proporciona una respuesta comprehensiva y bien estructurada basada en documentación.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "La respuesta principal detallada y bien estructurada. DEBE incluir al final una sección 'Enlaces y Referencias' con al menos 3 enlaces oficiales de Microsoft Learn de los documentos utilizados."
                        },
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 puntos clave más importantes de la respuesta."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confianza en la respuesta basada en la calidad de la documentación (0-1)."
                        },
                        "completeness": {
                            "type": "string",
                            "enum": ["complete", "partial", "limited"],
                            "description": "Si la documentación permite una respuesta completa, parcial o limitada."
                        },
                        "references_used": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Números de los documentos utilizados para generar la respuesta."
                        }
                    },
                    "required": ["answer", "key_points", "confidence", "completeness", "references_used"]
                }
            }
        }
    ]

def _extract_generated_answer(response, retrieved_docs: List[Dict] = None) -> Tuple[str, Dict]:
    """Extrae la respuesta generada de la respuesta de OpenAI."""
    try:
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "provide_comprehensive_answer":
                # Use robust JSON parsing
                raw_arguments = tool_call.function.arguments
                tool_args, success, error = robust_json_parse(raw_arguments, {})
                
                if not success:
                    print(f"[DEBUG] All JSON parsing strategies failed: {error}")
                    print(f"[DEBUG] Original JSON: {raw_arguments[:200]}...")
                    
                    # Final fallback: try to extract content directly
                    fallback_content = message.content or ""
                    if not fallback_content and hasattr(message, 'text'):
                        fallback_content = message.text or ""
                    
                    if fallback_content:
                        return fallback_content, {
                            "error": f"JSON parsing failed: {error}",
                            "fallback": True,
                            "confidence": 0.3
                        }
                    else:
                        return "No se pudo procesar la respuesta del modelo. Por favor, intenta nuevamente.", {
                            "error": f"JSON parsing failed and no fallback content: {error}",
                            "fallback": True,
                            "confidence": 0.1
                        }
                
                answer = tool_args.get("answer", "")
                key_points = tool_args.get("key_points", [])
                confidence = tool_args.get("confidence", 0.5)
                completeness = tool_args.get("completeness", "unknown")
                references_used = tool_args.get("references_used", [])
                
                # Enriquecer la respuesta con puntos clave
                if key_points:
                    answer += "\n\n**Puntos Clave:**\n"
                    for i, point in enumerate(key_points, 1):
                        answer += f"{i}. {point}\n"
                
                # Asegurar que los enlaces estén incluidos
                if retrieved_docs:
                    answer = _ensure_links_included(answer, retrieved_docs, references_used)
                
                generation_info = {
                    "confidence": confidence,
                    "completeness": completeness,
                    "references_used": references_used,
                    "key_points_count": len(key_points)
                }
                
                return answer, generation_info
            else:
                raise ValueError(f"Unexpected tool call: {tool_call.function.name}")
        else:
            # Fallback a contenido directo si no hay tool calls
            answer = message.content or "No se pudo generar una respuesta."
            return answer, {"confidence": 0.3, "completeness": "unknown", "fallback": True}
            
    except Exception as e:
        print(f"[DEBUG] Error extracting generated answer: {e}")
        print(f"[DEBUG] Exception type: {type(e).__name__}")
        
        # Try to get any available content from the response
        try:
            message = response.choices[0].message
            fallback_content = message.content or ""
            if fallback_content:
                return fallback_content, {
                    "error": f"Extraction failed but content available: {str(e)}",
                    "fallback": True,
                    "confidence": 0.3
                }
        except:
            pass
        
        return "Error procesando la respuesta generada. Por favor, intenta nuevamente.", {
            "error": str(e),
            "error_type": type(e).__name__,
            "fallback": True
        }

def _create_fallback_answer(question: str, docs: List[Dict]) -> str:
    """Crea una respuesta de fallback cuando falla la generación."""
    if not docs:
        return "No se encontraron documentos relevantes para tu pregunta."
    
    # Crear una respuesta básica con los títulos de los documentos más relevantes
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

def evaluate_answer_quality(
    question: str,
    answer: str,
    source_docs: List[Dict],
    openai_client: OpenAI
) -> Dict[str, float]:
    """
    Evalúa la calidad de la respuesta generada usando métricas RAG.
    
    Returns:
        Dict con métricas: faithfulness, answer_relevancy, context_utilization
    """
    try:
        # Preparar contexto para evaluación
        context = "\n".join([
            f"Doc {i+1}: {doc.get('title', '')} - {doc.get('content', '')[:500]}"
            for i, doc in enumerate(source_docs[:5])
        ])
        
        evaluation_prompt = f"""
Evalúa la calidad de esta respuesta RAG en una escala de 0.0 a 1.0:

PREGUNTA: {question}

RESPUESTA GENERADA: {answer}

CONTEXTO UTILIZADO: {context}

Evalúa:
1. FAITHFULNESS (0-1): ¿La respuesta es fiel al contexto proporcionado?
2. ANSWER_RELEVANCY (0-1): ¿La respuesta responde directamente la pregunta?
3. CONTEXT_UTILIZATION (0-1): ¿Se utilizó bien la información del contexto?
"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_rag_quality",
                    "description": "Evalúa la calidad de una respuesta RAG",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "faithfulness": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Fidelidad al contexto (0-1)"
                            },
                            "answer_relevancy": {
                                "type": "number", 
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Relevancia de la respuesta a la pregunta (0-1)"
                            },
                            "context_utilization": {
                                "type": "number",
                                "minimum": 0, 
                                "maximum": 1,
                                "description": "Utilización del contexto (0-1)"
                            }
                        },
                        "required": ["faithfulness", "answer_relevancy", "context_utilization"]
                    }
                }
            }
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un evaluador experto de sistemas RAG."},
                {"role": "user", "content": evaluation_prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "evaluate_rag_quality"}},
            temperature=0.1
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            # Use robust JSON parsing
            raw_arguments = tool_call.function.arguments
            tool_args, success, error = robust_json_parse(raw_arguments, {})
            
            if success:
                return {
                    "faithfulness": tool_args.get("faithfulness", 0.5),
                    "answer_relevancy": tool_args.get("answer_relevancy", 0.5),
                    "context_utilization": tool_args.get("context_utilization", 0.5)
                }
            else:
                print(f"[DEBUG] JSON parsing failed in evaluation: {error}")
    
    except Exception as e:
        print(f"[DEBUG] Error in answer quality evaluation: {e}")
    
    return {
        "faithfulness": 0.5,
        "answer_relevancy": 0.5, 
        "context_utilization": 0.5
    }