"""
Funciones optimizadas para trabajar con la colección questions_withlinks.

Esta colección ya contiene solo preguntas con links validados, eliminando
la necesidad de hacer validación cruzada en tiempo real.
"""

import random
import time
from typing import List, Dict, Optional
from src.services.storage.chromadb_utils import ChromaDBClientWrapper
from src.config.config import CHROMADB_COLLECTION_CONFIG


def get_optimized_random_question(
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_model_name: str = 'ada',
    max_attempts: int = 3
) -> Optional[Dict]:
    """
    Obtiene una pregunta aleatoria de la colección optimizada questions_withlinks.
    
    Esta función es mucho más rápida que get_single_random_question_with_valid_links
    porque todas las preguntas en esta colección ya están pre-validadas.
    
    Args:
        chromadb_wrapper: Cliente de ChromaDB
        embedding_model_name: Modelo para logging (ya no se usa para selección)
        max_attempts: Máximo número de intentos
        
    Returns:
        Dict con pregunta validada o None si no se encuentra
    """
    print(f"🚀 Obteniendo pregunta optimizada de questions_withlinks...")
    
    try:
        # Conectar a la colección optimizada
        client = chromadb_wrapper.client
        collection = client.get_collection(name="questions_withlinks")
        
        # Obtener el conteo total
        total_count = collection.count()
        print(f"📊 Colección optimizada tiene {total_count:,} preguntas pre-validadas")
        
        if total_count == 0:
            print("❌ Colección optimizada está vacía")
            return None
        
        for attempt in range(max_attempts):
            try:
                # Obtener una muestra aleatoria pequeña
                sample_size = min(10, total_count)
                results = collection.get(
                    limit=sample_size,
                    include=['metadatas']
                )
                
                if not results['metadatas']:
                    continue
                
                # Seleccionar una pregunta aleatoria de la muestra
                random_question = random.choice(results['metadatas'])
                
                # Convertir strings separados por | de vuelta a listas
                if 'ms_links' in random_question and isinstance(random_question['ms_links'], str):
                    random_question['ms_links'] = random_question['ms_links'].split('|') if random_question['ms_links'] else []
                
                if 'validated_links' in random_question and isinstance(random_question['validated_links'], str):
                    random_question['validated_links'] = random_question['validated_links'].split('|') if random_question['validated_links'] else []
                
                if 'accepted_answer_links' in random_question and isinstance(random_question['accepted_answer_links'], str):
                    random_question['accepted_answer_links'] = random_question['accepted_answer_links'].split('|') if random_question['accepted_answer_links'] else []
                
                print(f"✅ Pregunta optimizada obtenida en intento {attempt + 1}")
                print(f"   📋 Título: {random_question.get('title', 'Sin título')[:50]}...")
                print(f"   🔗 Links válidos: {random_question.get('valid_links', 0)}/{random_question.get('total_links', 0)}")
                
                return random_question
                
            except Exception as e:
                print(f"⚠️ Error en intento {attempt + 1}: {e}")
                continue
        
        print(f"❌ No se pudo obtener pregunta después de {max_attempts} intentos")
        return None
        
    except Exception as e:
        print(f"❌ Error crítico accediendo a questions_withlinks: {e}")
        print("💡 Asegúrate de que la colección questions_withlinks esté creada")
        return None


def get_optimized_questions_batch(
    chromadb_wrapper: ChromaDBClientWrapper,
    num_questions: int,
    embedding_model_name: str = 'ada'
) -> List[Dict]:
    """
    Obtiene un lote de preguntas de la colección questions_withlinks.
    Esta colección contiene 2,067 preguntas con links ya validados.

    Args:
        chromadb_wrapper: Cliente de ChromaDB
        num_questions: Número de preguntas a obtener
        embedding_model_name: Modelo para logging

    Returns:
        Lista de preguntas pre-validadas
    """
    print(f"📥 Cargando {num_questions} preguntas desde questions_withlinks...")

    try:
        # Conectar a la colección questions_withlinks (2,067 preguntas validadas)
        client = chromadb_wrapper.client
        collection = client.get_collection(name="questions_withlinks")

        # Obtener el conteo total
        total_count = collection.count()
        print(f"📊 Colección questions_withlinks: {total_count:,} preguntas validadas disponibles")

        if total_count == 0:
            print("❌ Colección questions_withlinks está vacía")
            return []
        
        # Obtener preguntas (limitado por lo disponible)
        actual_limit = min(num_questions, total_count)
        
        if actual_limit < total_count:
            # Obtener muestra aleatoria más grande y luego seleccionar
            sample_size = min(actual_limit * 3, total_count)
            results = collection.get(
                limit=sample_size,
                include=['metadatas']
            )
            
            # Seleccionar aleatoriamente del conjunto más grande
            all_questions = results['metadatas']
            random.shuffle(all_questions)
            selected_questions = all_questions[:actual_limit]
        else:
            # Obtener todas las disponibles
            results = collection.get(
                limit=actual_limit,
                include=['metadatas']
            )
            selected_questions = results['metadatas']
        
        # Procesar preguntas para convertir strings a listas
        processed_questions = []
        for question in selected_questions:
            # Convertir strings separados por | de vuelta a listas
            if 'ms_links' in question and isinstance(question['ms_links'], str):
                question['ms_links'] = question['ms_links'].split('|') if question['ms_links'] else []
            
            if 'validated_links' in question and isinstance(question['validated_links'], str):
                question['validated_links'] = question['validated_links'].split('|') if question['validated_links'] else []
            
            if 'accepted_answer_links' in question and isinstance(question['accepted_answer_links'], str):
                question['accepted_answer_links'] = question['accepted_answer_links'].split('|') if question['accepted_answer_links'] else []
            
            processed_questions.append(question)
        
        print(f"✅ Cargadas {len(processed_questions)} preguntas validadas")
        avg_valid = sum(q.get('valid_links', 0) for q in processed_questions) / len(processed_questions)
        print(f"📊 Promedio de links válidos: {avg_valid:.1f} por pregunta")

        return processed_questions

    except Exception as e:
        print(f"❌ Error cargando preguntas: {e}")
        return []


def get_collection_stats() -> Dict:
    """
    Obtiene estadísticas de la colección questions_withlinks.
    
    Returns:
        Dict con estadísticas de la colección
    """
    try:
        from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client
        
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        collection = client.get_collection(name="questions_withlinks")
        
        # Obtener muestra para estadísticas
        sample_results = collection.get(
            limit=100,
            include=['metadatas']
        )
        
        total_count = collection.count()
        
        if sample_results['metadatas']:
            # Calcular estadísticas de la muestra
            sample_questions = sample_results['metadatas']
            total_links_sample = sum(q.get('total_links', 0) for q in sample_questions)
            valid_links_sample = sum(q.get('valid_links', 0) for q in sample_questions)
            
            avg_total_links = total_links_sample / len(sample_questions)
            avg_valid_links = valid_links_sample / len(sample_questions)
            avg_success_rate = sum(q.get('validation_success_rate', 0) for q in sample_questions) / len(sample_questions)
            
            return {
                'total_questions': total_count,
                'sample_size': len(sample_questions),
                'avg_total_links_per_question': round(avg_total_links, 2),
                'avg_valid_links_per_question': round(avg_valid_links, 2),
                'avg_validation_success_rate': round(avg_success_rate * 100, 1),
                'collection_name': 'questions_withlinks',
                'status': 'ready'
            }
        else:
            return {
                'total_questions': total_count,
                'status': 'empty' if total_count == 0 else 'error'
            }
            
    except Exception as e:
        return {
            'error': str(e),
            'status': 'not_available'
        }


# Función de compatibilidad para el código existente
def get_single_random_question_with_valid_links_optimized(
    chromadb_wrapper: ChromaDBClientWrapper,
    embedding_model_name: str = 'ada',
    max_attempts: int = 3,
    candidate_batch_size: int = 5  # Ya no se usa, pero mantenido para compatibilidad
) -> Optional[Dict]:
    """
    Función de compatibilidad que usa la versión optimizada.
    
    Esta función reemplaza get_single_random_question_with_valid_links
    pero es mucho más rápida.
    """
    return get_optimized_random_question(
        chromadb_wrapper=chromadb_wrapper,
        embedding_model_name=embedding_model_name,
        max_attempts=max_attempts
    )