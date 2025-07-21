#!/usr/bin/env python3
"""
Script para verificar el estado de las colecciones de ChromaDB
"""

from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client, list_chromadb_collections

def main():
    print("=== ChromaDB Collections Status ===\n")
    
    try:
        # Listar todas las colecciones
        collections = list_chromadb_collections()
        print(f"Available collections: {collections}")
        
        if not collections:
            print("❌ No collections found!")
            return
        
        # Obtener cliente
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        # Verificar cada colección
        for collection_name in collections:
            try:
                collection = client.get_collection(collection_name)
                count = collection.count()
                print(f"✅ {collection_name}: {count} items")
                
                # Obtener una muestra pequeña para verificar la estructura
                if count > 0:
                    sample = collection.get(limit=1, include=['metadatas', 'documents'])
                    if sample['metadatas']:
                        print(f"   Sample metadata keys: {list(sample['metadatas'][0].keys())}")
                    else:
                        print("   No metadata found")
                else:
                    print("   Collection is empty!")
                    
            except Exception as e:
                print(f"❌ Error accessing {collection_name}: {e}")
        
        print("\n=== Testing ChromaDBClientWrapper ===")
        from utils.chromadb_utils import ChromaDBClientWrapper
        
        # Probar con diferentes configuraciones
        test_configs = [
            ("DocumentsMpnet", "QuestionsMlpnet"),  # multi-qa-mpnet-base-dot-v1
            ("DocumentsMiniLM", "QuestionsMiniLM"),  # all-MiniLM-L6-v2  
            ("Documentation", "Questions")  # ada
        ]
        
        for docs_class, questions_class in test_configs:
            try:
                wrapper = ChromaDBClientWrapper(client, docs_class, questions_class)
                stats = wrapper.get_collection_stats()
                print(f"✅ {docs_class}/{questions_class}: {stats}")
                
                # Intentar obtener preguntas de muestra
                sample_questions = wrapper.get_sample_questions(limit=2, random_sample=True)
                print(f"   Sample questions found: {len(sample_questions)}")
                
            except Exception as e:
                print(f"❌ Error with {docs_class}/{questions_class}: {e}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()