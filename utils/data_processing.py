"""
Utilidades para procesamiento de datos.
"""

import re
import random
from typing import List, Dict
from utils.weaviate_utils_improved import WeaviateClientWrapper
from config import WEAVIATE_CLASS_CONFIG


def extract_ms_links(accepted_answer: str) -> List[str]:
    """
    Extrae links de Microsoft Learn de la respuesta aceptada.
    
    Args:
        accepted_answer: Texto de la respuesta aceptada
        
    Returns:
        Lista de links de Microsoft Learn encontrados
    """
    # Patrón para encontrar links de Microsoft Learn
    pattern = r'https://learn\.microsoft\.com[\w/\-\?=&%\.]+'
    links = re.findall(pattern, accepted_answer)
    return list(set(links))  # Eliminar duplicados


def filter_questions_with_links(questions_and_answers: List[Dict]) -> List[Dict]:
    """
    Filtra preguntas que tienen links en la respuesta aceptada.
    
    Args:
        questions_and_answers: Lista de preguntas y respuestas
        
    Returns:
        Lista filtrada de preguntas con links en respuesta aceptada
    """
    filtered_questions = []
    
    for qa in questions_and_answers:
        accepted_answer = qa.get('accepted_answer', '')
        
        # Extraer links de Microsoft Learn
        ms_links = extract_ms_links(accepted_answer)
        
        # Solo incluir si hay links
        if ms_links and len(ms_links) > 0:
            # Añadir los links extraídos al diccionario
            qa_copy = qa.copy()
            qa_copy['ms_links'] = ms_links
            qa_copy['question'] = qa.get('question_content', qa.get('title', ''))
            filtered_questions.append(qa_copy)
    
    return filtered_questions


def fetch_random_questions_from_weaviate(
    weaviate_wrapper: WeaviateClientWrapper,
    embedding_model_name: str,
    num_questions: int,
    sample_size: int = 500
) -> List[Dict]:
    """
    Extrae preguntas aleatorias desde Weaviate que tengan enlaces de Microsoft Learn.
    
    Args:
        weaviate_wrapper: Cliente de Weaviate
        embedding_model_name: Nombre del modelo de embedding para determinar la clase
        num_questions: Número de preguntas a devolver
        sample_size: Tamaño de muestra inicial para filtrar (debe ser > num_questions)
        
    Returns:
        Lista de preguntas filtradas con enlaces MS Learn
    """
    # Obtener la clase de preguntas según el modelo
    questions_class = WEAVIATE_CLASS_CONFIG.get(embedding_model_name, {}).get("questions", "Questions")
    
    # Si necesitamos más preguntas de las que esperamos filtrar, aumentar sample_size
    if sample_size < num_questions * 3:  # Factor de seguridad
        sample_size = max(num_questions * 5, 1000)
    
    try:
        print(f"Fetching {sample_size} questions from class: {questions_class}")
        
        # Obtener la colección de preguntas
        questions_collection = weaviate_wrapper.client.collections.get(questions_class)
        print(f"Successfully got collection: {questions_collection.name}")
        
        # Realizar consulta para obtener muestra de preguntas
        results = questions_collection.query.fetch_objects(
            limit=sample_size,
            return_properties=["title", "question_content", "accepted_answer", "tags"]
        )
        print(f"Query returned {len(results.objects)} objects")
        
        if not results.objects:
            print(f"Warning: No data found for class {questions_class}")
            return []
        
        # Convertir objetos de Weaviate a diccionarios
        questions = []
        for obj in results.objects:
            question_dict = {
                'title': obj.properties.get('title', ''),
                'question_content': obj.properties.get('question_content', ''),
                'accepted_answer': obj.properties.get('accepted_answer', ''),
                'tags': obj.properties.get('tags', [])
            }
            questions.append(question_dict)
        
        # Filtrar preguntas que tengan enlaces de Microsoft Learn
        print(f"Filtering {len(questions)} questions for MS Learn links...")
        filtered_questions = []
        for question in questions:
            accepted_answer = question.get('accepted_answer', '')
            if accepted_answer:
                ms_links = extract_ms_links(accepted_answer)
                if ms_links and len(ms_links) > 0:
                    # Agregar los links extraídos al diccionario
                    question_copy = question.copy()
                    question_copy['ms_links'] = ms_links
                    # Normalizar campo de pregunta
                    question_copy['question'] = question.get('question_content', question.get('title', ''))
                    filtered_questions.append(question_copy)
        
        print(f"Found {len(filtered_questions)} questions with MS Learn links")
        
        # Si tenemos suficientes preguntas, seleccionar aleatoriamente
        if len(filtered_questions) >= num_questions:
            selected = random.sample(filtered_questions, num_questions)
            print(f"Returning {len(selected)} randomly selected questions")
            return selected
        else:
            print(f"Warning: Only found {len(filtered_questions)} questions with MS Learn links out of {sample_size} sampled")
            return filtered_questions
            
    except Exception as e:
        import traceback
        print(f"Error fetching questions from Weaviate: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return []