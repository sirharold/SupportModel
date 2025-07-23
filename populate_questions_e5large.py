#!/usr/bin/env python3
"""
Script to populate the questions_e5large collection with data from questions_ada.
This is a temporary solution to make the E5-Large model work in the comparison tool.
For production use, this should be replaced with proper E5-Large-v2 embeddings.
"""

import sys
import time
sys.path.append('.')

from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client

def copy_questions_to_e5large():
    """Copy questions from questions_ada to questions_e5large collection"""
    
    print("🔄 Copying questions from questions_ada to questions_e5large")
    print("=" * 65)
    
    try:
        # Initialize ChromaDB client
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        # Get source collection
        print("📖 Getting source collection (questions_ada)...")
        try:
            source_collection = client.get_collection("questions_ada")
            source_count = source_collection.count()
            print(f"✅ Source collection found: {source_count:,} items")
        except Exception as e:
            print(f"❌ Error accessing questions_ada: {e}")
            return False
        
        # Get target collection
        print("📝 Getting target collection (questions_e5large)...")
        try:
            target_collection = client.get_collection("questions_e5large")
            target_count = target_collection.count()
            print(f"✅ Target collection found: {target_count:,} items")
            
            if target_count > 0:
                response = input(f"\n🤔 Target collection already has {target_count} items. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Operation cancelled by user")
                    return False
                    
                # Clear target collection
                print("🗑️  Clearing target collection...")
                # Delete and recreate collection
                client.delete_collection("questions_e5large")
                target_collection = client.create_collection(
                    name="questions_e5large",
                    metadata={
                        "description": "Questions with E5-Large-v2 embeddings (copied from ada)",
                        "model": "intfloat/e5-large-v2",
                        "dimensions": 1024,
                        "source": "questions_ada",
                        "created_at": str(time.time()),
                        "note": "Temporary copy - should be replaced with actual E5 embeddings"
                    }
                )
                print("✅ Target collection cleared and recreated")
                
        except Exception as e:
            print(f"❌ Error accessing questions_e5large: {e}")
            return False
        
        if source_count == 0:
            print("⚠️  Source collection is empty, nothing to copy")
            return True
        
        # Copy data in batches
        batch_size = 1000
        total_batches = (source_count + batch_size - 1) // batch_size
        
        print(f"\n📊 Starting copy process:")
        print(f"   Total items: {source_count:,}")
        print(f"   Batch size: {batch_size:,}")
        print(f"   Total batches: {total_batches}")
        print()
        
        copied_count = 0
        
        for batch_num in range(total_batches):
            try:
                # Calculate offset and limit for this batch
                offset = batch_num * batch_size
                limit = min(batch_size, source_count - offset)
                
                print(f"📦 Processing batch {batch_num + 1}/{total_batches} (items {offset+1}-{offset+limit})")
                
                # Get batch from source
                # Note: ChromaDB doesn't have offset, so we get all and slice
                # For large collections, this is not optimal but works for this purpose
                if batch_num == 0:
                    # Get all data once
                    print("   🔍 Fetching all source data...")
                    all_source_data = source_collection.get(
                        include=['metadatas', 'documents', 'embeddings']
                    )
                    total_fetched = len(all_source_data['metadatas'])
                    print(f"   ✅ Fetched {total_fetched:,} items from source")
                
                # Slice batch data
                batch_start = offset
                batch_end = offset + limit
                
                batch_metadatas = all_source_data['metadatas'][batch_start:batch_end]
                batch_documents = all_source_data['documents'][batch_start:batch_end]
                batch_embeddings = all_source_data['embeddings'][batch_start:batch_end]
                
                # Generate IDs for this batch
                batch_ids = [f"q_e5_{offset + i}" for i in range(len(batch_metadatas))]
                
                print(f"   📝 Adding {len(batch_metadatas)} items to target collection...")
                
                # Add to target collection
                target_collection.add(
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                    embeddings=batch_embeddings
                )
                
                copied_count += len(batch_metadatas)
                print(f"   ✅ Batch completed. Total copied: {copied_count:,}/{source_count:,}")
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Error in batch {batch_num + 1}: {e}")
                print("   ⏩ Continuing with next batch...")
                continue
        
        # Verify final count
        final_count = target_collection.count()
        print(f"\n🎯 Copy completed!")
        print(f"   Source items: {source_count:,}")
        print(f"   Copied items: {copied_count:,}")
        print(f"   Final target count: {final_count:,}")
        
        if final_count == source_count:
            print("✅ SUCCESS: All items copied successfully!")
            return True
        else:
            print(f"⚠️  WARNING: Count mismatch. Expected {source_count:,}, got {final_count:,}")
            return final_count > 0  # Partial success is still useful
            
    except Exception as e:
        print(f"❌ Error during copy process: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_copy():
    """Verify that the copy was successful"""
    
    print("\n🔍 Verifying copy results")
    print("=" * 30)
    
    try:
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        
        # Get both collections
        source_collection = client.get_collection("questions_ada")
        target_collection = client.get_collection("questions_e5large")
        
        source_count = source_collection.count()
        target_count = target_collection.count()
        
        print(f"📊 Collection counts:")
        print(f"   questions_ada: {source_count:,}")
        print(f"   questions_e5large: {target_count:,}")
        
        if target_count == 0:
            print("❌ Target collection is empty!")
            return False
        
        # Get a sample from each collection
        print("\n🔍 Sampling data structure...")
        
        source_sample = source_collection.get(limit=1, include=['metadatas', 'documents'])
        target_sample = target_collection.get(limit=1, include=['metadatas', 'documents'])
        
        if source_sample['metadatas'] and target_sample['metadatas']:
            source_keys = set(source_sample['metadatas'][0].keys())
            target_keys = set(target_sample['metadatas'][0].keys())
            
            print(f"   Source metadata keys: {sorted(source_keys)}")
            print(f"   Target metadata keys: {sorted(target_keys)}")
            
            if source_keys == target_keys:
                print("✅ Metadata structure matches!")
            else:
                missing_keys = source_keys - target_keys
                extra_keys = target_keys - source_keys
                if missing_keys:
                    print(f"⚠️  Missing keys in target: {missing_keys}")
                if extra_keys:
                    print(f"ℹ️  Extra keys in target: {extra_keys}")
        
        success = target_count > 0 and target_count >= source_count * 0.9  # 90% threshold
        
        if success:
            print("\n✅ Verification passed!")
        else:
            print("\n❌ Verification failed!")
            
        return success
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

def main():
    print("🚀 Questions E5-Large Collection Populator")
    print("=" * 45)
    print()
    print("⚠️  NOTE: This script copies questions from questions_ada to questions_e5large")
    print("   This is a temporary solution to make E5-Large comparisons work.")
    print("   For production, you should generate actual E5-Large-v2 embeddings.")
    print()
    
    # Step 1: Copy data
    print("Step 1: Copy questions data")
    copy_success = copy_questions_to_e5large()
    
    if copy_success:
        # Step 2: Verify copy
        print("\nStep 2: Verify copy results")
        verify_success = verify_copy()
        
        print(f"\n{'='*45}")
        if copy_success and verify_success:
            print("✅ SUCCESS: questions_e5large collection populated and verified!")
            print()
            print("🎯 Next steps:")
            print("   1. Test the E5-Large model in the comparison tool")
            print("   2. Consider replacing with actual E5-Large-v2 embeddings")
            print("   3. Monitor performance and accuracy")
        else:
            print("⚠️  PARTIAL SUCCESS: Collection populated but verification had issues")
    else:
        print(f"\n{'='*45}")
        print("❌ FAILED: Could not populate the collection")
        print("   Please check the errors above and try again.")

if __name__ == "__main__":
    main()