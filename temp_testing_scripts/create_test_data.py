#!/usr/bin/env python3
"""
Script para crear datos de prueba en ChromaDB para testing
"""

import random
from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client

def create_test_questions(collection, count=100):
    """Crea preguntas de prueba"""
    documents = []
    metadatas = []
    embeddings = []
    ids = []
    
    for i in range(count):
        # Crear pregunta de prueba
        question_id = f"test_question_{i}"
        ids.append(question_id)
        
        question_text = f"¿Cómo configurar Azure Service {i} para la aplicación empresarial?"
        documents.append(question_text)
        
        metadata = {
            'title': f'Configuración de Azure Service {i}',
            'question_content': question_text,
            'accepted_answer': f'Para configurar Azure Service {i}, debes seguir estos pasos... Ver más en https://learn.microsoft.com/azure/service-{i}',
            'tags': ['azure', 'configuration', f'service-{i}']
        }
        metadatas.append(metadata)
        
        # Vector embedding dummy (768 dimensiones)
        embedding = [random.uniform(-1, 1) for _ in range(768)]
        embeddings.append(embedding)
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Created {count} test questions")

def create_test_docs(collection, count=200):
    """Crea documentos de prueba"""
    documents = []
    metadatas = []
    embeddings = []
    ids = []
    
    for i in range(count):
        doc_id = f"test_doc_{i}"
        ids.append(doc_id)
        
        doc_text = f"Este es un documento de Azure sobre el servicio {i}. Contiene información importante sobre configuración y mejores prácticas."
        documents.append(doc_text)
        
        metadata = {
            'title': f'Documentación de Azure Service {i}',
            'content': doc_text,
            'link': f'https://learn.microsoft.com/azure/service-{i}',
            'text': doc_text
        }
        metadatas.append(metadata)
        
        # Vector embedding dummy (768 dimensiones)
        embedding = [random.uniform(-1, 1) for _ in range(768)]
        embeddings.append(embedding)
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Created {count} test documents")

def main():
    print("=== Creating Test Data for ChromaDB ===\n")
    
    try:
        # Conectar a ChromaDB
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        print("✅ Connected to ChromaDB")
        
        # Crear datos de prueba para cada colección
        collections_config = {
            'questions_mpnet': ('questions', 50),
            'questions_minilm': ('questions', 50), 
            'questions_ada': ('questions', 50),
            'docs_mpnet': ('docs', 100),
            'docs_minilm': ('docs', 100),
            'docs_ada': ('docs', 100)
        }
        
        for collection_name, (type_name, count) in collections_config.items():
            try:
                collection = client.get_collection(collection_name)
                current_count = collection.count()
                
                if current_count > 0:
                    print(f"⚠️  {collection_name} already has {current_count} items, skipping...")
                    continue
                
                print(f"Creating test data for {collection_name}...")
                
                if type_name == 'questions':
                    create_test_questions(collection, count)
                else:
                    create_test_docs(collection, count)
                    
            except Exception as e:
                print(f"❌ Error creating data for {collection_name}: {e}")
        
        print("\n=== Final Status ===")
        # Verificar el estado final
        from utils.chromadb_utils import list_chromadb_collections
        collections = list_chromadb_collections()
        for collection_name in collections:
            collection = client.get_collection(collection_name)
            count = collection.count()
            print(f"{collection_name}: {count} items")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()