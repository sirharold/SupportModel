#!/usr/bin/env python3
"""
Script to create the missing questions_e5large collection in ChromaDB
This collection is needed for the E5-Large-v2 model comparison functionality.
"""

import sys
import os
import time
sys.path.append('.')

from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client
from src.config.config import CHROMADB_COLLECTION_CONFIG

def create_questions_e5large_collection():
    """Create the questions_e5large collection based on questions_ada collection structure"""
    
    print("🚀 Creating questions_e5large collection")
    print("=" * 50)
    
    try:
        # Initialize ChromaDB client
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        # Check current collections
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        print(f"📋 Current collections: {collection_names}")
        
        # Check if questions_e5large already exists
        if "questions_e5large" in collection_names:
            questions_e5large = client.get_collection("questions_e5large")
            count = questions_e5large.count()
            print(f"ℹ️  questions_e5large already exists with {count} items")
            
            response = input("\n🤔 Collection already exists. Recreate it? (y/N): ")
            if response.lower() != 'y':
                print("❌ Operation cancelled by user")
                return False
                
            # Delete existing collection
            print("🗑️  Deleting existing collection...")
            client.delete_collection("questions_e5large")
        
        # Check if source collection exists
        if "questions_ada" not in collection_names:
            print("❌ Source collection 'questions_ada' not found!")
            print(f"Available collections: {collection_names}")
            return False
        
        # Get source collection info
        questions_ada = client.get_collection("questions_ada")
        source_count = questions_ada.count()
        print(f"📊 Source collection 'questions_ada' has {source_count:,} items")
        
        if source_count == 0:
            print("❌ Source collection is empty!")
            return False
        
        # Create target collection
        print("📝 Creating questions_e5large collection...")
        questions_e5large = client.create_collection(
            name="questions_e5large",
            metadata={
                "description": "Questions with E5-Large-v2 embeddings (1024 dimensions)",
                "model": "intfloat/e5-large-v2",
                "dimensions": 1024,
                "source": "questions_ada",
                "created_at": str(time.time())
            }
        )
        print("✅ Collection created successfully!")
        
        # Verify collection was created
        updated_collections = client.list_collections()
        updated_names = [col.name for col in updated_collections]
        print(f"📋 Updated collections: {updated_names}")
        
        if "questions_e5large" in updated_names:
            print("✅ questions_e5large collection verified in ChromaDB")
            
            # Verify configuration
            e5_config = CHROMADB_COLLECTION_CONFIG.get("e5-large-v2")
            if e5_config:
                expected_questions_collection = e5_config["questions"]
                if expected_questions_collection == "questions_e5large":
                    print("✅ Configuration matches: questions_e5large is correctly configured")
                else:
                    print(f"⚠️  Configuration mismatch: expected {expected_questions_collection}, created questions_e5large")
            else:
                print("⚠️  E5-Large-v2 configuration not found in CHROMADB_COLLECTION_CONFIG")
            
            return True
        
        else:
            print("❌ Failed to create collection")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_all_collections():
    """Verify that all required collections exist for all models"""
    
    print("\n🔍 Verifying all required collections")
    print("=" * 40)
    
    try:
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        collections = client.list_collections()
        existing_names = [col.name for col in collections]
        
        print(f"📋 Existing collections: {existing_names}")
        print()
        
        # Check all configured collections
        missing_collections = []
        
        for model_key, model_config in CHROMADB_COLLECTION_CONFIG.items():
            docs_collection = model_config["documents"]
            questions_collection = model_config["questions"]
            
            print(f"🔧 {model_key}:")
            
            # Check documents collection
            if docs_collection in existing_names:
                docs_col = client.get_collection(docs_collection)
                docs_count = docs_col.count()
                print(f"   ✅ {docs_collection}: {docs_count:,} items")
            else:
                print(f"   ❌ {docs_collection}: NOT FOUND")
                missing_collections.append(docs_collection)
            
            # Check questions collection
            if questions_collection in existing_names:
                questions_col = client.get_collection(questions_collection)
                questions_count = questions_col.count()
                print(f"   ✅ {questions_collection}: {questions_count:,} items")
            else:
                print(f"   ❌ {questions_collection}: NOT FOUND")
                missing_collections.append(questions_collection)
        
        if missing_collections:
            print(f"\n⚠️  Missing collections: {missing_collections}")
            return False
        else:
            print("\n✅ All collections are present!")
            return True
            
    except Exception as e:
        print(f"❌ Error verifying collections: {e}")
        return False

def main():
    print("🔧 ChromaDB Questions E5-Large Collection Creator")
    print("=" * 55)
    print()
    
    # First verify current state
    print("Step 1: Verify current state")
    verify_all_collections()
    
    print("\nStep 2: Create missing questions_e5large collection")
    success = create_questions_e5large_collection()
    
    print("\nStep 3: Final verification")
    all_good = verify_all_collections()
    
    print("\n" + "=" * 55)
    if success and all_good:
        print("✅ SUCCESS: questions_e5large collection created and verified!")
        print()
        print("ℹ️  Note: This collection is now empty and ready for data.")
        print("   You'll need to populate it with E5-Large embeddings of your questions.")
        print("   The collection is configured for 1024-dimensional embeddings.")
    else:
        print("❌ FAILED: Could not create or verify the collection")
        print("   Please check the errors above and try again.")

if __name__ == "__main__":
    main()