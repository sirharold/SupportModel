"""
Utilidades para procesamiento de datos.
"""

import re
import random
from typing import List, Dict
from utils.chromadb_utils import ChromaDBClientWrapper
from config import CHROMADB_COLLECTION_CONFIG


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


def fetch_random_questions_from_chromadb(
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_model_name: str,
    num_questions: int,
    sample_size: int = None
) -> List[Dict]:
    """
    Extrae preguntas aleatorias desde ChromaDB que tengan enlaces de Microsoft Learn.
    MEJORADO: Ahora busca en TODA la base de datos, no solo una muestra limitada.
    
    Args:
        chromadb_wrapper: Cliente de ChromaDB
        embedding_model_name: Nombre del modelo de embedding para determinar la colección
        num_questions: Número de preguntas a devolver
        sample_size: Tamaño de muestra inicial (None = obtener TODAS las preguntas)
        
    Returns:
        Lista de preguntas filtradas con enlaces MS Learn
    """
    # Obtener la clase de preguntas según el modelo
    questions_class = CHROMADB_COLLECTION_CONFIG.get(embedding_model_name, {}).get("questions", "Questions")
    
    try:
        # FASE 1: Determinar el tamaño total de la colección
        print(f"🔍 Verificando tamaño total de la colección: {questions_class}")
        try:
            total_count = chromadb_wrapper._questions_collection.count()
            print(f"📊 Total de preguntas en la base de datos: {total_count:,}")
        except Exception as e:
            print(f"⚠️ No se pudo obtener el conteo total: {e}")
            total_count = 10000  # Estimación conservadora
        
        # FASE 2: Determinar cuántas preguntas necesitamos obtener inicialmente
        # Necesitamos obtener MUCHAS más porque solo ~20-25% tienen enlaces MS Learn
        if sample_size is None:
            # Estrategia adaptativa: obtener suficientes para garantizar que encontremos num_questions
            if total_count < 5000:
                # Para bases de datos pequeñas, obtener todo
                initial_fetch = total_count
                print(f"📦 Base de datos pequeña: obteniendo TODAS las {initial_fetch:,} preguntas")
            else:
                # Para bases de datos grandes, usar un múltiplo inteligente
                # Asumiendo ~20% tienen enlaces MS Learn, necesitamos 5x más
                initial_fetch = min(num_questions * 8, total_count, 15000)  # Límite de seguridad
                print(f"📦 Base de datos grande: obteniendo {initial_fetch:,} preguntas (estimado para encontrar {num_questions})")
        else:
            initial_fetch = min(sample_size, total_count)
            print(f"📦 Usando sample_size especificado: {initial_fetch:,} preguntas")
        
        # FASE 3: Obtener las preguntas en lotes para manejar memoria
        all_questions = []
        batch_size = 2000  # Procesar de a 2000 para no sobrecargar memoria
        
        for offset in range(0, initial_fetch, batch_size):
            current_batch_size = min(batch_size, initial_fetch - offset)
            print(f"📥 Obteniendo lote {offset//batch_size + 1}: preguntas {offset+1:,} a {offset+current_batch_size:,}")
            
            # Obtener lote actual
            batch_questions = chromadb_wrapper.get_sample_questions(
                limit=current_batch_size,
                random_sample=True  # Mantener aleatorización
            )
            
            if not batch_questions:
                print(f"⚠️ Lote vacío en offset {offset}")
                break
                
            print(f"✅ Obtenidas {len(batch_questions)} preguntas en este lote")
            all_questions.extend(batch_questions)
        
        print(f"📊 Total de preguntas obtenidas: {len(all_questions):,}")
        
        if not all_questions:
            print(f"❌ No se encontraron preguntas en la colección {questions_class}")
            return []
        
        # FASE 4: Normalizar formato de preguntas
        print(f"🔧 Normalizando formato de {len(all_questions):,} preguntas...")
        formatted_questions = []
        for question in all_questions:
            question_dict = {
                'title': question.get('title', ''),
                'question_content': question.get('question_content', ''),
                'accepted_answer': question.get('accepted_answer', ''),
                'tags': question.get('tags', [])
            }
            formatted_questions.append(question_dict)
        
        # FASE 5: Filtrar preguntas que tengan enlaces de Microsoft Learn
        print(f"🔍 Filtrando {len(formatted_questions):,} preguntas buscando enlaces MS Learn...")
        filtered_questions = []
        
        for i, question in enumerate(formatted_questions):
            if i % 1000 == 0:  # Progreso cada 1000 preguntas
                print(f"   Procesadas {i:,}/{len(formatted_questions):,} preguntas...")
                
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
        
        print(f"✅ Encontradas {len(filtered_questions):,} preguntas con enlaces MS Learn")
        print(f"📊 Tasa de éxito: {len(filtered_questions)/len(formatted_questions)*100:.1f}% de preguntas tienen enlaces MS Learn")
        
        # FASE 6: Selección final aleatoria
        if len(filtered_questions) >= num_questions:
            selected = random.sample(filtered_questions, num_questions)
            print(f"🎯 Seleccionadas {len(selected)} preguntas aleatoriamente del conjunto filtrado")
            return selected
        else:
            print(f"⚠️ Solo se encontraron {len(filtered_questions)} preguntas con enlaces MS Learn")
            print(f"💡 Considera aumentar el tamaño de muestra o reducir el número de preguntas solicitadas")
            return filtered_questions
            
    except Exception as e:
        import traceback
        print(f"❌ Error obteniendo preguntas de ChromaDB: {e}")
        print(f"📝 Traceback completo: {traceback.format_exc()}")
        return []