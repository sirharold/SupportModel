#!/usr/bin/env python3
"""
Script para crear la colección questions_withlinks en ChromaDB.

Esta colección contendrá solo las preguntas que tienen en su respuesta aceptada
al menos un link que al normalizarlo existe en la colección de documentos.

Esto optimiza las búsquedas evitando hacer validación cruzada en tiempo real.
"""

import os
import sys
import time
from typing import List, Dict, Set
import chromadb
from chromadb.config import Settings

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client, ChromaDBClientWrapper
from src.data.processing import extract_ms_links
from src.data.extract_links import normalize_url


def get_all_normalized_doc_links(chromadb_wrapper: ChromaDBClientWrapper) -> Set[str]:
    """
    Obtiene todos los links normalizados de la colección de documentos.
    
    Returns:
        Set de links normalizados de todos los documentos
    """
    print("📥 Obteniendo todos los links de documentos...")
    
    try:
        # Obtener todos los documentos
        results = chromadb_wrapper._docs_collection.get(
            include=['metadatas']
        )
        
        normalized_links = set()
        for metadata in results['metadatas']:
            doc_link = metadata.get('link', '')
            if doc_link:
                normalized_link = normalize_url(doc_link)
                if normalized_link:
                    normalized_links.add(normalized_link)
        
        print(f"✅ Obtenidos {len(normalized_links)} links únicos normalizados de documentos")
        return normalized_links
        
    except Exception as e:
        print(f"❌ Error obteniendo links de documentos: {e}")
        return set()


def get_all_questions(chromadb_wrapper: ChromaDBClientWrapper) -> List[Dict]:
    """
    Obtiene todas las preguntas de la colección questions_ada.
    
    Returns:
        Lista de todas las preguntas con sus metadatos
    """
    print("📥 Obteniendo todas las preguntas...")
    
    try:
        # Obtener todas las preguntas
        results = chromadb_wrapper._questions_collection.get(
            include=['metadatas', 'documents', 'embeddings']
        )
        
        questions = []
        for i in range(len(results['metadatas'])):
            question_data = {
                'metadata': results['metadatas'][i],
                'document': results['documents'][i] if i < len(results['documents']) else '',
                'embedding': results['embeddings'][i] if i < len(results['embeddings']) else None
            }
            questions.append(question_data)
        
        print(f"✅ Obtenidas {len(questions)} preguntas totales")
        return questions
        
    except Exception as e:
        print(f"❌ Error obteniendo preguntas: {e}")
        return []


def filter_questions_with_valid_links(
    questions: List[Dict], 
    valid_doc_links: Set[str]
) -> List[Dict]:
    """
    Filtra preguntas que tienen links válidos en su respuesta aceptada.
    
    Args:
        questions: Lista de preguntas
        valid_doc_links: Set de links válidos de documentos (normalizados)
        
    Returns:
        Lista de preguntas filtradas con información de validación
    """
    print("🔍 Filtrando preguntas con links válidos...")
    
    filtered_questions = []
    total_processed = 0
    
    for question in questions:
        total_processed += 1
        
        if total_processed % 1000 == 0:
            print(f"   Procesadas {total_processed:,}/{len(questions):,} preguntas...")
        
        metadata = question['metadata']
        accepted_answer = metadata.get('accepted_answer', '')
        
        if not accepted_answer:
            continue
        
        # Extraer y normalizar links de la respuesta aceptada
        ms_links = extract_ms_links(accepted_answer)
        if not ms_links:
            continue
        
        # Normalizar links y verificar si existen en documentos
        valid_links = []
        for link in ms_links:
            normalized_link = normalize_url(link)
            if normalized_link and normalized_link in valid_doc_links:
                valid_links.append(normalized_link)
        
        # Solo incluir si tiene al menos 1 link válido
        if valid_links:
            # Enriquecer metadata con información de validación
            enriched_metadata = metadata.copy()
            # Convertir listas a strings para ChromaDB
            enriched_metadata['ms_links'] = '|'.join(ms_links)  # Separados por |
            enriched_metadata['accepted_answer_links'] = '|'.join(ms_links)  # Para compatibilidad
            enriched_metadata['validated_links'] = '|'.join(valid_links)
            enriched_metadata['total_links'] = len(ms_links)
            enriched_metadata['valid_links'] = len(valid_links)
            enriched_metadata['validation_success_rate'] = len(valid_links) / len(ms_links)
            enriched_metadata['validation_timestamp'] = time.time()
            
            # Normalizar campo de pregunta
            enriched_metadata['question'] = metadata.get('question_content', metadata.get('title', ''))
            
            filtered_question = {
                'metadata': enriched_metadata,
                'document': question['document'],
                'embedding': question['embedding']
            }
            
            filtered_questions.append(filtered_question)
    
    print(f"✅ Filtradas {len(filtered_questions):,} preguntas con links válidos")
    print(f"📊 Tasa de éxito: {len(filtered_questions)/len(questions)*100:.1f}% de preguntas tienen links válidos")
    
    return filtered_questions


def create_questions_withlinks_collection(client, filtered_questions: List[Dict]):
    """
    Crea la nueva colección questions_withlinks con las preguntas filtradas.
    
    Args:
        client: Cliente de ChromaDB
        filtered_questions: Lista de preguntas filtradas
    """
    collection_name = "questions_withlinks"
    
    print(f"🔧 Creando colección '{collection_name}'...")
    
    try:
        # Eliminar colección si existe
        try:
            client.delete_collection(name=collection_name)
            print(f"🗑️ Colección existente '{collection_name}' eliminada")
        except:
            pass  # No existe, está bien
        
        # Crear nueva colección
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Preguntas con links validados que existen en documentos"}
        )
        
        print(f"✅ Colección '{collection_name}' creada")
        
        # Preparar datos para inserción en lotes
        batch_size = 100
        total_batches = (len(filtered_questions) + batch_size - 1) // batch_size
        
        print(f"📤 Insertando {len(filtered_questions):,} preguntas en {total_batches} lotes...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(filtered_questions))
            batch = filtered_questions[start_idx:end_idx]
            
            # Preparar datos del lote (sin embeddings para evitar problemas)
            ids = []
            metadatas = []
            documents = []
            
            for i, question in enumerate(batch):
                # Generar ID único
                question_id = f"q_{start_idx + i}_{int(time.time())}"
                ids.append(question_id)
                
                # Metadata - asegurar que todos los valores sean tipos compatibles con ChromaDB
                clean_metadata = {}
                for key, value in question['metadata'].items():
                    if isinstance(value, list):
                        # Convertir listas a strings separadas por |
                        clean_metadata[key] = '|'.join(str(v) for v in value) if value else ''
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        clean_metadata[key] = value
                    else:
                        # Convertir otros tipos a string
                        clean_metadata[key] = str(value)
                
                metadatas.append(clean_metadata)
                
                # Document content - usar question_content o title como documento
                doc_content = question['metadata'].get('question_content', '')
                if not doc_content:
                    doc_content = question['metadata'].get('title', '')
                documents.append(doc_content or f"Question {question_id}")
            
            # Insertar lote (ChromaDB generará embeddings automáticamente)
            try:
                collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=documents
                )
            except Exception as e:
                print(f"   ❌ Error en lote {batch_idx + 1}: {e}")
                continue
            
            print(f"   ✅ Lote {batch_idx + 1}/{total_batches} insertado ({len(batch)} preguntas)")
        
        # Verificar conteo final
        final_count = collection.count()
        print(f"🎉 Colección '{collection_name}' creada exitosamente con {final_count:,} preguntas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando colección: {e}")
        return False


def main():
    """Función principal para crear la colección questions_withlinks."""
    
    print("🚀 CREACIÓN DE COLECCIÓN questions_withlinks")
    print("=" * 60)
    
    try:
        # 1. Conectar a ChromaDB
        print("🔌 Conectando a ChromaDB...")
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        # 2. Inicializar wrapper para colección ada (preguntas y documentos)
        print("🔧 Inicializando cliente ChromaDB...")
        chromadb_wrapper = ChromaDBClientWrapper(
            client,
            documents_class="docs_ada",
            questions_class="questions_ada",
            retry_attempts=3
        )
        
        # 3. Obtener todos los links válidos de documentos
        valid_doc_links = get_all_normalized_doc_links(chromadb_wrapper)
        if not valid_doc_links:
            print("❌ No se pudieron obtener links de documentos. Abortando.")
            return False
        
        # 4. Obtener todas las preguntas
        all_questions = get_all_questions(chromadb_wrapper)
        if not all_questions:
            print("❌ No se pudieron obtener preguntas. Abortando.")
            return False
        
        # 5. Filtrar preguntas con links válidos
        filtered_questions = filter_questions_with_valid_links(all_questions, valid_doc_links)
        if not filtered_questions:
            print("❌ No se encontraron preguntas con links válidos. Abortando.")
            return False
        
        # 6. Crear nueva colección
        success = create_questions_withlinks_collection(client, filtered_questions)
        
        if success:
            print("\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
            print("=" * 60)
            print(f"📊 Estadísticas finales:")
            print(f"   • Preguntas totales procesadas: {len(all_questions):,}")
            print(f"   • Preguntas con links válidos: {len(filtered_questions):,}")
            print(f"   • Tasa de filtrado: {len(filtered_questions)/len(all_questions)*100:.1f}%")
            print(f"   • Links únicos en documentos: {len(valid_doc_links):,}")
            print(f"\n✅ Colección 'questions_withlinks' lista para usar!")
            print(f"💡 Ahora puedes actualizar el código para usar esta colección optimizada.")
            return True
        else:
            print("\n❌ Error durante la creación de la colección.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)