#!/usr/bin/env python3
"""
Test script to verify the safe embedding client works without segmentation faults.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.embedding_safe import SafeEmbeddingClient
from dotenv import load_dotenv

def test_safe_embedding():
    print("🧪 Testing SafeEmbeddingClient...")
    
    # Load environment
    load_dotenv()
    
    try:
        # Initialize client
        print("1️⃣ Initializing SafeEmbeddingClient...")
        client = SafeEmbeddingClient(
            model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY")
        )
        print("✅ Client initialized successfully")
        
        # Test query embedding
        print("2️⃣ Testing query embedding...")
        test_query = "How to configure Azure Functions with Key Vault?"
        query_embedding = client.generate_query_embedding(test_query)
        print(f"✅ Query embedding generated: {len(query_embedding)} dimensions")
        
        # Test document embedding
        print("3️⃣ Testing document embedding...")
        test_doc = "Azure Functions can be configured to use Key Vault for secure secrets management."
        doc_embedding = client.generate_document_embedding(test_doc)
        print(f"✅ Document embedding generated: {len(doc_embedding)} dimensions")
        
        # Test cleanup
        print("4️⃣ Testing cleanup...")
        client.cleanup()
        print("✅ Cleanup completed successfully")
        
        print("🎉 All tests passed! SafeEmbeddingClient is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_safe_embedding()
    sys.exit(0 if success else 1)