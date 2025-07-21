#!/usr/bin/env python3
"""
Script para migrar datos de Weaviate a ChromaDB
"""

import os
from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client
from config import CHROMADB_COLLECTION_CONFIG

def migrate_collection(weaviate_client, chromadb_client, weaviate_class_name, chromadb_collection_name):
    """Migra una colección específica de Weaviate a ChromaDB"""
    try:
        print(f"\n=== Migrating {weaviate_class_name} to {chromadb_collection_name} ===")
        
        # Obtener datos de Weaviate
        weaviate_collection = weaviate_client.collections.get(weaviate_class_name)
        
        # Contar elementos en Weaviate
        count_result = weaviate_collection.aggregate.over_all(total_count=True)
        total_count = count_result.total_count if count_result else 0
        print(f"Found {total_count} items in Weaviate collection {weaviate_class_name}")
        
        if total_count == 0:
            print(f"⚠️  Collection {weaviate_class_name} is empty in Weaviate")
            return
        
        # Obtener colección de ChromaDB
        chromadb_collection = chromadb_client.get_collection(chromadb_collection_name)
        
        # Migrar en lotes
        batch_size = 100
        migrated_count = 0
        
        for offset in range(0, total_count, batch_size):
            print(f"Processing batch {offset}-{min(offset + batch_size, total_count)}")
            
            # Obtener lote de Weaviate
            results = weaviate_collection.query.fetch_objects(
                limit=batch_size,
                offset=offset,
                include_vector=True
            )
            
            if not results.objects:
                break
            
            # Preparar datos para ChromaDB
            documents = []
            metadatas = []
            embeddings = []
            ids = []
            
            for i, obj in enumerate(results.objects):
                # ID único
                doc_id = f"{chromadb_collection_name}_{offset + i}"
                ids.append(doc_id)
                
                # Metadata
                metadata = obj.properties.copy()
                metadatas.append(metadata)
                
                # Documento (usar el campo 'text' o 'content' si existe)
                document_text = metadata.get('text', '') or metadata.get('content', '') or metadata.get('question_content', '') or ''
                documents.append(document_text)
                
                # Vector embedding
                if obj.vector:
                    embeddings.append(obj.vector)
                else:
                    # Si no hay vector, crear uno dummy (esto no debería pasar)
                    embeddings.append([0.0] * 768)  # Asume 768 dimensiones
            
            # Insertar en ChromaDB
            try:
                chromadb_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                migrated_count += len(documents)
                print(f"✅ Migrated {len(documents)} items (total: {migrated_count})")
                
            except Exception as e:
                print(f"❌ Error inserting batch: {e}")
                continue
        
        print(f"✅ Migration complete: {migrated_count} items migrated to {chromadb_collection_name}")
        
    except Exception as e:
        print(f"❌ Error migrating {weaviate_class_name}: {e}")

def main():
    print("=== Weaviate to ChromaDB Migration ===\n")
    
    try:
        # Verificar si tenemos credenciales de Weaviate
        weaviate_config = None
        try:
            weaviate_config = WeaviateConfig.from_env()
            weaviate_client = get_weaviate_client(weaviate_config)
            print("✅ Connected to Weaviate")
        except Exception as e:
            print(f"❌ Could not connect to Weaviate: {e}")
            print("Make sure your .env file has WCS_URL and WCS_API_KEY")
            return
        
        # Conectar a ChromaDB
        chromadb_config = ChromaDBConfig.from_env()
        chromadb_client = get_chromadb_client(chromadb_config)
        print("✅ Connected to ChromaDB")
        
        # Mapeo de colecciones
        migrations = [
            # (weaviate_class, chromadb_collection)
            ("DocumentsMpnet", "docs_mpnet"),
            ("QuestionsMlpnet", "questions_mpnet"),
            ("DocumentsMiniLM", "docs_minilm"), 
            ("QuestionsMiniLM", "questions_minilm"),
            ("Documentation", "docs_ada"),
            ("Questions", "questions_ada")
        ]
        
        for weaviate_class, chromadb_collection in migrations:
            migrate_collection(weaviate_client, chromadb_client, weaviate_class, chromadb_collection)
        
        print("\n=== Migration Summary ===")
        # Verificar el estado final
        from utils.chromadb_utils import list_chromadb_collections
        collections = list_chromadb_collections()
        for collection_name in collections:
            collection = chromadb_client.get_collection(collection_name)
            count = collection.count()
            print(f"{collection_name}: {count} items")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    main()