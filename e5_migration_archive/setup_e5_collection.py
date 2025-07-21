#!/usr/bin/env python3
"""
Script para preparar la colección docs_e5large antes de la migración
"""

from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client

def setup_e5_collection():
    """Crear colección para E5-Large si no existe"""
    try:
        client = get_chromadb_client(ChromaDBConfig.from_env())
        
        # Verificar colección fuente
        try:
            source_collection = client.get_collection("docs_ada")
            source_count = source_collection.count()
            print(f"✅ Source collection 'docs_ada': {source_count:,} items")
            
            if source_count == 0:
                print("❌ WARNING: Source collection is empty!")
                return False
                
        except Exception as e:
            print(f"❌ Source collection 'docs_ada' not found: {e}")
            return False
        
        # Crear/verificar colección destino
        try:
            target_collection = client.get_collection("docs_e5large")
            target_count = target_collection.count()
            print(f"ℹ️  Target collection 'docs_e5large' already exists: {target_count:,} items")
            
            if target_count > 0:
                response = input("Collection already has data. Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Migration cancelled by user")
                    return False
                    
        except Exception:
            print("📝 Creating target collection 'docs_e5large'...")
            target_collection = client.create_collection(
                name="docs_e5large",
                metadata={
                    "description": "Documents with E5-Large-v2 embeddings",
                    "model": "intfloat/e5-large-v2",
                    "dimensions": 1024,
                    "created_at": str(time.time())
                }
            )
            print("✅ Target collection created successfully")
        
        print(f"\n🎯 Ready for migration:")
        print(f"   Source: docs_ada ({source_count:,} items)")
        print(f"   Target: docs_e5large (ready)")
        print(f"\nRun migration with: python migrate_docs_to_e5.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    import time
    
    print("🚀 E5-Large Collection Setup")
    print("============================")
    
    success = setup_e5_collection()
    
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")