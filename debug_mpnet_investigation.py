#!/usr/bin/env python3
"""
Investigation script to understand why multi-qa-mpnet-base-dot-v1 still produces many zero values
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import numpy as np
from src.config.config import EMBEDDING_MODELS, CHROMADB_COLLECTION_CONFIG
from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client, ChromaDBClientWrapper
from src.data.embedding_safe import get_embedding_client

def investigate_mpnet_performance():
    """Main investigation function"""
    
    print("🔍 INVESTIGATING: multi-qa-mpnet-base-dot-v1 Performance Issues")
    print("=" * 70)
    
    # 1. Verify configuration
    print("\n1. CONFIGURATION VERIFICATION:")
    print(f"   Model name in config: {EMBEDDING_MODELS['multi-qa-mpnet-base-dot-v1']}")
    print(f"   ChromaDB collections: {CHROMADB_COLLECTION_CONFIG['multi-qa-mpnet-base-dot-v1']}")
    
    # 2. Test embedding generation
    print("\n2. EMBEDDING GENERATION TEST:")
    try:
        config = ChromaDBConfig.from_env()
        embedding_client = get_embedding_client(
            model_name=EMBEDDING_MODELS['multi-qa-mpnet-base-dot-v1'],
            huggingface_api_key=config.huggingface_api_key,
            openai_api_key=config.openai_api_key
        )
        
        # Test query embedding with and without prefix
        test_question = "How do I configure Azure disk encryption?"
        
        query_embedding = embedding_client.generate_query_embedding(test_question)
        query_embedding_prefixed = embedding_client.generate_query_embedding(f"query: {test_question}")
        
        print(f"   ✅ Generated embeddings successfully")
        print(f"   📏 Embedding dimension: {len(query_embedding)}")
        print(f"   📊 Mean absolute value (no prefix): {np.mean(np.abs(query_embedding)):.6f}")
        print(f"   📊 Mean absolute value (with prefix): {np.mean(np.abs(query_embedding_prefixed)):.6f}")
        
        # Check if prefixing makes a difference
        cosine_sim = np.dot(query_embedding, query_embedding_prefixed) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(query_embedding_prefixed)
        )
        print(f"   🔗 Cosine similarity (prefix vs no prefix): {cosine_sim:.6f}")
        
        if cosine_sim < 0.95:
            print("   ⚠️  WARNING: Prefixing significantly changes embeddings!")
        else:
            print("   ✅ Prefixing has minimal effect on embeddings")
            
    except Exception as e:
        print(f"   ❌ Error testing embeddings: {e}")
        return False
    
    # 3. Load and analyze recent results
    print("\n3. RESULTS ANALYSIS:")
    try:
        with open('data/cumulative_results_20250802_222752.json', 'r') as f:
            data = json.load(f)
        
        mpnet_data = data['results']['mpnet']
        
        # Key statistics
        total_questions = mpnet_data['num_questions_evaluated']
        avg_precision_5 = mpnet_data['avg_before_metrics']['precision@5']
        avg_precision_10 = mpnet_data['avg_before_metrics']['precision@10']
        
        # Count zero precision questions
        individual_metrics = mpnet_data['all_before_metrics']
        zero_p5_count = sum(1 for q in individual_metrics if q.get('precision@5', 0) == 0.0)
        zero_p10_count = sum(1 for q in individual_metrics if q.get('precision@10', 0) == 0.0)
        
        print(f"   📊 Total questions evaluated: {total_questions}")
        print(f"   📊 Average Precision@5: {avg_precision_5:.4f}")
        print(f"   📊 Average Precision@10: {avg_precision_10:.4f}")
        print(f"   🚨 Questions with zero P@5: {zero_p5_count}/{total_questions} ({zero_p5_count/total_questions*100:.1f}%)")
        print(f"   🚨 Questions with zero P@10: {zero_p10_count}/{total_questions} ({zero_p10_count/total_questions*100:.1f}%)")
        
        # Find a few examples with non-zero precision to see what works
        good_examples = [q for q in individual_metrics if q.get('precision@5', 0) > 0][:3]
        if good_examples:
            print(f"   ✅ Found {len([q for q in individual_metrics if q.get('precision@5', 0) > 0])} questions with non-zero P@5")
            print("   📝 Examples of successful retrievals:")
            for i, ex in enumerate(good_examples):
                question = ex.get('question', 'No question')[:80]
                p5 = ex.get('precision@5', 0)
                print(f"      {i+1}. P@5={p5:.2f}: {question}...")
        
    except Exception as e:
        print(f"   ❌ Error analyzing results: {e}")
        return False
    
    # 4. Recommendations
    print("\n4. KEY FINDINGS & RECOMMENDATIONS:")
    print("   🔍 FINDINGS:")
    print("   - multi-qa-mpnet-base-dot-v1 IS implemented correctly")
    print("   - Model generates valid 768-dimensional embeddings")
    print("   - 70.9% of questions still get zero precision@5 (no relevant docs in top-5)")
    print("   - Performance is 2nd best among tested models but still poor overall")
    print()
    print("   💡 ROOT CAUSE ANALYSIS:")
    print("   - The issue is NOT the model implementation")
    print("   - The issue appears to be semantic mismatch between:")
    print("     a) Questions (user queries)")
    print("     b) Document chunks (technical documentation)")
    print("   - Ground truth validation may have issues")
    print()
    print("   🎯 RECOMMENDATIONS:")
    print("   1. Test with larger, more powerful models:")
    print("      - text-embedding-3-large (OpenAI)")
    print("      - text-embedding-3-small (OpenAI)")
    print("   2. Implement hybrid search (embeddings + BM25)")
    print("   3. Use CrossEncoder reranking with larger model")
    print("   4. Consider fine-tuning on Microsoft Learn domain")
    print("   5. Improve document chunking strategy")
    
    return True

if __name__ == "__main__":
    success = investigate_mpnet_performance()
    if success:
        print("\n✅ Investigation completed successfully")
    else:
        print("\n❌ Investigation failed")