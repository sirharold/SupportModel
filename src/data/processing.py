"""
Utilidades para procesamiento de datos.
"""

import re
import random
from typing import List, Dict
import time
from src.services.storage.chromadb_utils import ChromaDBClientWrapper
from src.config.config import CHROMADB_COLLECTION_CONFIG


def extract_ms_links(accepted_answer: str) -> List[str]:
    """
    Extrae links de Microsoft Learn de la respuesta aceptada y los normaliza.
    
    Args:
        accepted_answer: Texto de la respuesta aceptada
        
    Returns:
        Lista de links de Microsoft Learn normalizados (sin parámetros ni anchors)
    """
    # Importar función de normalización
    from src.data.extract_links import normalize_url
    
    # Patrón para encontrar links de Microsoft Learn (incluye parámetros, anchors, etc.)
    pattern = r'https://learn\.microsoft\.com[^\s\)\]\"\'\<\>]+'
    raw_links = re.findall(pattern, accepted_answer)
    
    # Normalizar cada link (eliminar parámetros y anchors)
    normalized_links = []
    for link in raw_links:
        normalized = normalize_url(link)
        if normalized:  # Solo agregar si la normalización fue exitosa
            normalized_links.append(normalized)
    
    return list(set(normalized_links))  # Eliminar duplicados


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
            qa_copy['accepted_answer_links'] = ms_links  # FIXED: Add field expected by Colab evaluation
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
                    question_copy['accepted_answer_links'] = ms_links  # FIXED: Add field expected by Colab evaluation
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


def get_single_random_question_with_valid_links(
    chromadb_wrapper,
    embedding_model_name: str = 'ada',
    max_attempts: int = 10,
    candidate_batch_size: int = 20
) -> Dict:
    """
    Obtiene UNA pregunta aleatoria donde los links del accepted_answer 
    EXISTEN como documentos en la collection de documentos.
    
    FUNCIÓN REUTILIZABLE para cualquier parte del sistema que necesite
    preguntas con links validados.
    
    Args:
        chromadb_wrapper: Cliente de ChromaDB
        embedding_model_name: Modelo para determinar colección
        max_attempts: Máximo número de intentos para encontrar pregunta válida
        candidate_batch_size: Tamaño del lote de candidatas por intento
        
    Returns:
        Dict con pregunta validada o None si no se encuentra
        
    Example:
        question = get_single_random_question_with_valid_links(chromadb_wrapper)
        if question:
            title = question['title']
            content = question['question_content']
            valid_links = question['validated_links']
    """
    from src.services.storage.chromadb_utils import ChromaDBClientWrapper
    
    print(f"🔍 Buscando pregunta aleatoria con links válidos...")
    print(f"📊 Parámetros: modelo={embedding_model_name}, intentos_max={max_attempts}")
    
    total_candidates_checked = 0
    total_questions_with_links = 0
    total_questions_with_valid_links = 0
    
    for attempt in range(max_attempts):
        print(f"🎲 Intento {attempt + 1}/{max_attempts}")
        
        try:
            # 1. Obtener lote de preguntas candidatas con links MS Learn
            candidates = fetch_random_questions_from_chromadb(
                chromadb_wrapper=chromadb_wrapper,
                embedding_model_name=embedding_model_name,
                num_questions=candidate_batch_size
            )
            
            if not candidates:
                print(f"⚠️ No se encontraron preguntas candidatas en intento {attempt + 1}")
                continue
                
            total_candidates_checked += len(candidates)
            print(f"📥 Obtenidas {len(candidates)} preguntas candidatas")
            
            # 2. Para cada candidata, validar que links existan como documentos
            for i, question in enumerate(candidates):
                ms_links = question.get('ms_links', [])
                if not ms_links:
                    continue
                    
                total_questions_with_links += 1
                print(f"   🔗 Pregunta {i+1}: {len(ms_links)} links encontrados")
                
                # 3. VALIDACIÓN CRUZADA: ¿Existen los links como documentos?
                try:
                    existing_docs = chromadb_wrapper.lookup_docs_by_links_batch(ms_links)
                    
                    if existing_docs:  # Al menos 1 link existe como documento
                        total_questions_with_valid_links += 1
                        validated_links = [doc.get('link') for doc in existing_docs if doc.get('link')]
                        
                        # Enriquecer pregunta con información de validación
                        validated_question = question.copy()
                        validated_question['validated_links'] = validated_links
                        validated_question['total_links'] = len(ms_links)
                        validated_question['valid_links'] = len(validated_links)
                        validated_question['validation_success_rate'] = len(validated_links) / len(ms_links)
                        validated_question['validation_timestamp'] = time.time()
                        
                        print(f"✅ PREGUNTA VÁLIDA ENCONTRADA!")
                        print(f"   📋 Título: {validated_question.get('title', 'Sin título')[:50]}...")
                        print(f"   🔗 Links válidos: {len(validated_links)}/{len(ms_links)}")
                        print(f"   📊 Estadísticas: candidatas={total_candidates_checked}, "
                              f"con_links={total_questions_with_links}, válidas={total_questions_with_valid_links}")
                        
                        return validated_question
                        
                    else:
                        print(f"   ❌ Ningún link existe como documento")
                        
                except Exception as e:
                    print(f"   ⚠️ Error validando links: {e}")
                    continue
            
            print(f"   🔍 Intento {attempt + 1} completado - No se encontró pregunta válida")
            
        except Exception as e:
            print(f"❌ Error en intento {attempt + 1}: {e}")
            continue
    
    # No se encontró pregunta válida después de todos los intentos
    print(f"⚠️ NO SE ENCONTRÓ PREGUNTA VÁLIDA después de {max_attempts} intentos")
    print(f"📊 Estadísticas finales:")
    print(f"   - Candidatas revisadas: {total_candidates_checked}")
    print(f"   - Con links MS Learn: {total_questions_with_links}")
    print(f"   - Con links válidos: {total_questions_with_valid_links}")
    
    if total_candidates_checked > 0:
        success_rate = (total_questions_with_valid_links / total_candidates_checked) * 100
        print(f"   - Tasa de éxito: {success_rate:.1f}%")
    
    return None


def get_random_question_simple(
    chromadb_wrapper,
    embedding_model_name: str = 'ada'
) -> Dict:
    """
    Versión simplificada y rápida para obtener una pregunta aleatoria.
    Solo valida que tenga links MS Learn, no que existan como documentos.
    
    FUNCIÓN REUTILIZABLE para casos donde no se necesita validación completa.
    
    Returns:
        Dict con pregunta o None si no se encuentra
    """
    try:
        questions = fetch_random_questions_from_chromadb(
            chromadb_wrapper=chromadb_wrapper,
            embedding_model_name=embedding_model_name,
            num_questions=1
        )
        
        if questions:
            return questions[0]
        return None
        
    except Exception as e:
        print(f"❌ Error obteniendo pregunta simple: {e}")
        return None