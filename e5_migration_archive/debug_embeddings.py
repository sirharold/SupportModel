#!/usr/bin/env python3
"""
Script para debuggear la estructura de embeddings en ChromaDB
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb

def debug_collection_embeddings(collection_name: str):
    """Debuggear embeddings de colección"""
    try:
        client = chromadb.PersistentClient(path="/Users/haroldgomez/chromadb2/")
        collection = client.get_collection(collection_name)
        
        print(f"📥 Debugging collection '{collection_name}'...")
        count = collection.count()
        print(f"📊 Total items: {count:,}")
        
        # Obtener una muestra pequeña
        print("🔄 Getting sample data...")
        sample_data = collection.get(limit=5, include=['metadatas', 'documents', 'embeddings'])
        
        print(f"Sample documents: {len(sample_data['documents'])}")
        print(f"Sample metadatas: {len(sample_data['metadatas'])}")
        print(f"Sample embeddings: {len(sample_data['embeddings'])}")
        
        print("\n--- EMBEDDING ANALYSIS ---")
        embeddings_data = sample_data.get('embeddings', [])
        if embeddings_data is not None and len(embeddings_data) > 0:
            print(f"Embeddings container type: {type(embeddings_data)}")
            print(f"Number of embeddings: {len(embeddings_data)}")
            
            for i, emb in enumerate(embeddings_data[:3]):
                print(f"Embedding {i}:")
                if emb is None:
                    print("  None")
                else:
                    print(f"  Type: {type(emb)}")
                    print(f"  Length: {len(emb) if hasattr(emb, '__len__') else 'N/A'}")
                    if hasattr(emb, '__len__') and len(emb) > 0:
                        print(f"  First 5 values: {emb[:5]}")
                    print()
        else:
            print("No embeddings found or empty!")
        
        print("\n--- METADATA ANALYSIS ---")
        for i, meta in enumerate(sample_data['metadatas'][:2]):
            print(f"Metadata {i}: {meta}")
        
        print("\n--- DOCUMENT ANALYSIS ---")
        for i, doc in enumerate(sample_data['documents'][:2]):
            print(f"Document {i} (first 100 chars): {doc[:100]}...")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "docs_ada"
    debug_collection_embeddings(collection_name)

if __name__ == "__main__":
    main()