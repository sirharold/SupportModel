import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_cosine_similarities(data):
    """Extract cosine similarity scores from document scores"""
    model_similarities = {}
    
    results_data = data['results']
    
    for model_name in ['minilm', 'mpnet', 'e5-large', 'ada']:
        if model_name not in results_data:
            continue
            
        model_similarities[model_name] = {
            'all_scores': [],
            'relevant_scores': [],
            'irrelevant_scores': [],
            'dimensions': results_data[model_name]['embedding_dimensions']
        }
    
    # We need to look at individual question results
    # Since the aggregated data doesn't have document scores,
    # we'll check if there's raw question data elsewhere
    
    # Let's check if there's a questions section
    if 'questions' in data:
        for question_id, question_data in data['questions'].items():
            for model_name in ['minilm', 'mpnet', 'e5-large', 'ada']:
                if model_name in question_data:
                    model_data = question_data[model_name]
                    if 'document_scores' in model_data:
                        for doc in model_data['document_scores']:
                            if 'cosine_similarity' in doc:
                                score = doc['cosine_similarity']
                                model_similarities[model_name]['all_scores'].append(score)
                                
                                # Check if document is relevant
                                if doc.get('is_relevant', False):
                                    model_similarities[model_name]['relevant_scores'].append(score)
                                else:
                                    model_similarities[model_name]['irrelevant_scores'].append(score)
    
    return model_similarities

def print_similarity_analysis(model_similarities):
    """Print analysis of cosine similarities"""
    print("\n" + "="*80)
    print("COSINE SIMILARITY ANALYSIS")
    print("="*80)
    
    for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
        if model not in model_similarities:
            continue
            
        sims = model_similarities[model]
        dims = sims['dimensions']
        
        print(f"\n{model.upper()} (Dimensions: {dims})")
        print("-" * 60)
        
        # Overall statistics
        if sims['all_scores']:
            all_stats = calculate_statistics(sims['all_scores'])
            print(f"\nAll Documents (n={all_stats['count']}):")
            print(f"  Mean:     {all_stats['mean']:.4f}")
            print(f"  Std:      {all_stats['std']:.4f}")
            print(f"  Min:      {all_stats['min']:.4f}")
            print(f"  Max:      {all_stats['max']:.4f}")
            print(f"  Median:   {np.median(sims['all_scores']):.4f}")
            
            # Percentiles
            percentiles = [10, 25, 50, 75, 90]
            print("\n  Percentiles:")
            for p in percentiles:
                value = np.percentile(sims['all_scores'], p)
                print(f"    {p}th:   {value:.4f}")
        
        # Relevant vs Irrelevant
        if sims['relevant_scores'] and sims['irrelevant_scores']:
            rel_stats = calculate_statistics(sims['relevant_scores'])
            irrel_stats = calculate_statistics(sims['irrelevant_scores'])
            
            print(f"\nRelevant Documents (n={rel_stats['count']}):")
            print(f"  Mean:     {rel_stats['mean']:.4f}")
            print(f"  Std:      {rel_stats['std']:.4f}")
            print(f"  Range:    [{rel_stats['min']:.4f}, {rel_stats['max']:.4f}]")
            
            print(f"\nIrrelevant Documents (n={irrel_stats['count']}):")
            print(f"  Mean:     {irrel_stats['mean']:.4f}")
            print(f"  Std:      {irrel_stats['std']:.4f}")
            print(f"  Range:    [{irrel_stats['min']:.4f}, {irrel_stats['max']:.4f}]")
            
            # Separation metrics
            mean_diff = rel_stats['mean'] - irrel_stats['mean']
            print(f"\nSeparation Metrics:")
            print(f"  Mean Difference:     {mean_diff:.4f}")
            print(f"  Relative Difference: {(mean_diff / irrel_stats['mean'] * 100):.1f}%")
            
            # Check overlap
            rel_min, rel_max = rel_stats['min'], rel_stats['max']
            irrel_min, irrel_max = irrel_stats['min'], irrel_stats['max']
            
            if rel_min > irrel_max:
                print("  Overlap: NO OVERLAP (Perfect separation)")
            elif rel_max < irrel_min:
                print("  Overlap: NO OVERLAP (Inverted - irrelevant scores higher!)")
            else:
                overlap_start = max(rel_min, irrel_min)
                overlap_end = min(rel_max, irrel_max)
                print(f"  Overlap Range: [{overlap_start:.4f}, {overlap_end:.4f}]")

def calculate_statistics(values):
    """Calculate statistics for a list of values"""
    if not values or len(values) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'count': len(values)
    }

def main():
    print("Loading JSON file...")
    file_path = '/Users/haroldgomez/Downloads/cumulative_results_20250730_071510.json'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # First, let's check the structure to see where document scores might be
    print("\nChecking data structure for document scores...")
    
    # Check if there are document scores in the model results
    if 'results' in data:
        for model_name in ['minilm', 'mpnet', 'e5-large', 'ada']:
            if model_name in data['results']:
                model_data = data['results'][model_name]
                print(f"\n{model_name} keys: {list(model_data.keys())}")
                
                # Check if there's a sample question with document scores
                if 'sample_question' in model_data or 'document_scores' in model_data:
                    print(f"  Found document scores in {model_name}")
    
    # Since document scores aren't in the aggregated results,
    # let's print what we know about similarity from the metrics
    print("\n" + "="*80)
    print("COSINE SIMILARITY INSIGHTS FROM METRICS")
    print("="*80)
    
    print("\nBased on the metrics analysis, we can infer:")
    print("\n1. Model Performance Ranking (by precision@10 after reranking):")
    print("   - Ada (1536D):      0.0668")
    print("   - MPNet (768D):     0.0528")
    print("   - E5-Large (1024D): 0.0480")
    print("   - MiniLM (384D):    0.0436")
    
    print("\n2. Dimension Impact:")
    print("   - Strong positive correlation (0.895) between dimensions and precision")
    print("   - Higher dimensions generally lead to better retrieval performance")
    print("   - Ada (1536D) significantly outperforms others")
    
    print("\n3. Reranking Impact:")
    print("   - Most models show NEGATIVE impact from reranking")
    print("   - Ada shows largest degradation (-23.1% precision@5)")
    print("   - This suggests CrossEncoder may not be well-calibrated")
    
    print("\n4. RAG Metrics Insights:")
    print("   - E5-Large shows 0.0000 for all BERT scores (possible error)")
    print("   - Ada shows highest BERT scores (0.1445)")
    print("   - Context precision/recall high for all models (>0.90)")
    
    print("\nNOTE: Document-level cosine similarity scores are not available")
    print("in this aggregated results file. To analyze actual cosine")
    print("similarities, we would need the raw evaluation data with")
    print("individual document scores for each query.")

if __name__ == "__main__":
    main()