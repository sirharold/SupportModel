#!/usr/bin/env python3
"""
Script para verificar que la migración E5-Large se completó correctamente
"""

import json
from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client
from utils.clients import initialize_clients

def verify_migration():
    """Verificar migración completa"""
    print("🔍 E5-Large Migration Verification")
    print("===================================")
    
    try:
        client = get_chromadb_client(ChromaDBConfig.from_env())
        
        # Verificar colecciones
        print("\n📊 COLLECTION VERIFICATION")
        
        # Colección fuente
        try:
            source_collection = client.get_collection("docs_ada")
            source_count = source_collection.count()
            print(f"✅ Source (docs_ada): {source_count:,} items")
        except Exception as e:
            print(f"❌ Source collection error: {e}")
            return False
        
        # Colección destino
        try:
            target_collection = client.get_collection("docs_e5large")
            target_count = target_collection.count()
            print(f"✅ Target (docs_e5large): {target_count:,} items")
            
            if target_count == 0:
                print("❌ Target collection is empty!")
                return False
                
        except Exception as e:
            print(f"❌ Target collection error: {e}")
            return False
        
        # Comparar conteos
        coverage = (target_count / source_count * 100) if source_count > 0 else 0
        print(f"📈 Coverage: {coverage:.1f}%")
        
        if coverage < 95:
            print("⚠️  WARNING: Low coverage, migration may be incomplete")
        
        # Verificar estructura de datos
        print("\n🔍 DATA STRUCTURE VERIFICATION")
        
        sample = target_collection.get(limit=3, include=['metadatas', 'documents', 'embeddings'])
        
        if not sample['embeddings']:
            print("❌ No embeddings found!")
            return False
        
        # Verificar dimensiones
        embedding_dims = len(sample['embeddings'][0]) if sample['embeddings'] else 0
        print(f"📏 Embedding dimensions: {embedding_dims}")
        
        if embedding_dims != 1024:
            print(f"❌ Wrong dimensions! Expected 1024, got {embedding_dims}")
            return False
        
        # Verificar metadatos
        if sample['metadatas']:
            sample_keys = list(sample['metadatas'][0].keys())
            print(f"🗂️  Metadata keys: {sample_keys}")
            
            expected_keys = ['title', 'content', 'link']
            missing_keys = [key for key in expected_keys if key not in sample_keys]
            if missing_keys:
                print(f"⚠️  Missing metadata keys: {missing_keys}")
        
        # Verificar documentos
        if sample['documents']:
            avg_doc_length = sum(len(doc or "") for doc in sample['documents']) / len(sample['documents'])
            print(f"📄 Average document length: {avg_doc_length:.0f} chars")
        
        # Verificar checkpoint final
        print("\n📋 CHECKPOINT VERIFICATION")
        try:
            with open("checkpoint_docs_e5large.json", "r") as f:
                checkpoint = json.load(f)
            
            processed = checkpoint.get("processed_count", 0)
            spent = checkpoint.get("spent", 0.0)
            method = checkpoint.get("current_method", "unknown")
            stats = checkpoint.get("stats", {})
            
            print(f"✅ Final checkpoint found")
            print(f"   Processed: {processed:,} items")
            print(f"   Final method: {method}")
            print(f"   Total cost: ${spent:.3f}")
            print(f"   Success rate: {stats.get('successful_batches', 0)}/{stats.get('successful_batches', 0) + stats.get('failed_batches', 0)} batches")
            
        except FileNotFoundError:
            print("⚠️  No checkpoint file found")
        except Exception as e:
            print(f"❌ Checkpoint error: {e}")
        
        # Test de integración con la aplicación
        print("\n🧪 APPLICATION INTEGRATION TEST")
        try:
            chromadb_wrapper, embedding_client, *_ = initialize_clients("e5-large-v2")
            
            # Verificar stats
            stats = chromadb_wrapper.get_collection_stats()
            print(f"✅ Application can access E5 collections")
            print(f"   docs_e5large: {stats.get('docs_e5large_count', 0):,} items")
            
            # Test de búsqueda
            sample_results = chromadb_wrapper.search_docs_by_vector(
                vector=[0.1] * 1024,  # Vector dummy
                top_k=3,
                include_distance=True
            )
            
            if sample_results:
                print(f"✅ Search test successful: {len(sample_results)} results")
                avg_distance = sum(r.get('distance', 0) for r in sample_results) / len(sample_results)
                print(f"   Average distance: {avg_distance:.3f}")
            else:
                print("⚠️  Search test returned no results")
            
        except Exception as e:
            print(f"❌ Application integration failed: {e}")
            return False
        
        # Resumen final
        print("\n" + "="*50)
        print("🎉 VERIFICATION SUMMARY")
        print("="*50)
        
        if coverage >= 95 and embedding_dims == 1024 and target_count > 0:
            print("✅ Migration completed successfully!")
            print(f"   ✅ {target_count:,} documents migrated")
            print(f"   ✅ {coverage:.1f}% coverage")
            print(f"   ✅ Correct embedding dimensions (1024)")
            print(f"   ✅ Application integration working")
            print("\n🚀 E5-Large-v2 is ready to use!")
            print("   You can now select 'e5-large-v2' in your application")
            return True
        else:
            print("❌ Migration verification failed!")
            print("   Please check the issues above")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def quick_check():
    """Verificación rápida"""
    try:
        client = get_chromadb_client(ChromaDBConfig.from_env())
        target_collection = client.get_collection("docs_e5large")
        count = target_collection.count()
        
        print(f"docs_e5large: {count:,} items")
        
        if count > 0:
            sample = target_collection.get(limit=1, include=['embeddings'])
            dims = len(sample['embeddings'][0]) if sample['embeddings'] else 0
            print(f"Embedding dimensions: {dims}")
            
            if dims == 1024:
                print("✅ Migration appears successful")
            else:
                print("❌ Wrong embedding dimensions")
        else:
            print("❌ Collection is empty")
            
    except Exception as e:
        print(f"❌ Quick check failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_check()
    else:
        verify_migration()