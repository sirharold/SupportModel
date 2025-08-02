import json
import numpy as np
from collections import defaultdict

def extract_metrics(data):
    """Extract and analyze metrics for each model"""
    models = ['ada', 'e5-large', 'mpnet', 'minilm']
    results = {}
    
    for model in models:
        if model not in data:
            print(f"Model {model} not found in data")
            continue
            
        model_data = data[model]
        results[model] = {
            'dimensions': get_model_dimensions(model),
            'before_reranking': {},
            'after_reranking': {},
            'cosine_similarities': [],
            'ragas_scores': {},
            'bert_scores': {}
        }
        
        # Extract metrics for each question
        for question_id, question_data in model_data.items():
            if 'cumulative_results' not in question_data:
                continue
                
            cum_results = question_data['cumulative_results']
            
            # Before reranking metrics
            if 'before_reranking' in cum_results:
                for metric, value in cum_results['before_reranking'].items():
                    if metric not in results[model]['before_reranking']:
                        results[model]['before_reranking'][metric] = []
                    results[model]['before_reranking'][metric].append(value)
            
            # After reranking metrics
            if 'after_reranking' in cum_results:
                for metric, value in cum_results['after_reranking'].items():
                    if metric not in results[model]['after_reranking']:
                        results[model]['after_reranking'][metric] = []
                    results[model]['after_reranking'][metric].append(value)
            
            # Document scores (cosine similarities)
            if 'document_scores' in question_data:
                for doc in question_data['document_scores']:
                    if 'cosine_similarity' in doc:
                        results[model]['cosine_similarities'].append(doc['cosine_similarity'])
            
            # RAGAS scores
            if 'ragas_scores' in cum_results:
                for metric, value in cum_results['ragas_scores'].items():
                    if metric not in results[model]['ragas_scores']:
                        results[model]['ragas_scores'][metric] = []
                    if value is not None:
                        results[model]['ragas_scores'][metric].append(value)
            
            # BERT scores
            if 'bert_scores' in cum_results:
                for metric, value in cum_results['bert_scores'].items():
                    if metric not in results[model]['bert_scores']:
                        results[model]['bert_scores'][metric] = []
                    if value is not None:
                        results[model]['bert_scores'][metric].append(value)
    
    return results

def get_model_dimensions(model):
    """Get embedding dimensions for each model"""
    dimensions = {
        'minilm': 384,
        'mpnet': 768,
        'e5-large': 1024,
        'ada': 1536
    }
    return dimensions.get(model, 'unknown')

def calculate_statistics(values):
    """Calculate mean, std, min, max for a list of values"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }

def main():
    print("Loading JSON file...")
    with open('/Users/haroldgomez/Downloads/cumulative_results_20250730_071510.json', 'r') as f:
        data = json.load(f)
    
    print("Extracting metrics...")
    # Get the actual results data
    if 'results' in data:
        results_data = data['results']
    else:
        results_data = data
    
    results = extract_metrics(results_data)
    
    # Print summary for each model
    for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
        if model not in results:
            continue
            
        print(f"\n{'='*80}")
        print(f"MODEL: {model.upper()} (Dimensions: {results[model]['dimensions']})")
        print(f"{'='*80}")
        
        # Traditional metrics comparison
        print("\n1. BEST TRADITIONAL METRICS:")
        print("-" * 50)
        
        metrics_to_show = ['precision@5', 'precision@10', 'recall@5', 'recall@10', 
                          'f1@5', 'f1@10', 'ndcg@5', 'ndcg@10', 'mrr']
        
        print("\nBEFORE RERANKING:")
        for metric in metrics_to_show:
            if metric in results[model]['before_reranking']:
                stats = calculate_statistics(results[model]['before_reranking'][metric])
                print(f"  {metric:<15}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
        
        print("\nAFTER RERANKING:")
        for metric in metrics_to_show:
            if metric in results[model]['after_reranking']:
                stats = calculate_statistics(results[model]['after_reranking'][metric])
                print(f"  {metric:<15}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
        
        # Improvement analysis
        print("\nIMPROVEMENT (After - Before):")
        for metric in metrics_to_show:
            if (metric in results[model]['before_reranking'] and 
                metric in results[model]['after_reranking']):
                before = results[model]['before_reranking'][metric]
                after = results[model]['after_reranking'][metric]
                if before and after:
                    improvement = np.mean(after) - np.mean(before)
                    print(f"  {metric:<15}: {improvement:+.4f}")
        
        # Cosine similarities
        print("\n2. COSINE SIMILARITY DISTRIBUTION:")
        print("-" * 50)
        if results[model]['cosine_similarities']:
            stats = calculate_statistics(results[model]['cosine_similarities'])
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
        
        # RAGAS scores
        print("\n3. RAGAS METRICS:")
        print("-" * 50)
        for metric, values in results[model]['ragas_scores'].items():
            if values:
                stats = calculate_statistics(values)
                print(f"  {metric:<30}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
        
        # BERT scores
        print("\n4. BERT SCORES:")
        print("-" * 50)
        for metric, values in results[model]['bert_scores'].items():
            if values:
                stats = calculate_statistics(values)
                print(f"  {metric:<30}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
    
    # Summary comparison across dimensions
    print(f"\n{'='*80}")
    print("DIMENSION IMPACT SUMMARY")
    print(f"{'='*80}")
    
    print("\nAverage Performance by Dimension:")
    print("-" * 50)
    
    key_metrics = ['precision@10', 'recall@10', 'f1@10', 'ndcg@10', 'mrr']
    
    for stage in ['before_reranking', 'after_reranking']:
        print(f"\n{stage.upper().replace('_', ' ')}:")
        print(f"{'Model':<10} {'Dims':<6} " + " ".join([f"{m:<12}" for m in key_metrics]))
        print("-" * 80)
        
        for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
            if model not in results:
                continue
            
            row = f"{model:<10} {results[model]['dimensions']:<6}"
            for metric in key_metrics:
                if metric in results[model][stage]:
                    mean_val = np.mean(results[model][stage][metric])
                    row += f" {mean_val:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
            print(row)

if __name__ == "__main__":
    main()