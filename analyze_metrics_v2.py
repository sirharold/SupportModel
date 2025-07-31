import json
import numpy as np
from collections import defaultdict

def analyze_model_metrics(model_data):
    """Analyze metrics for a single model"""
    results = {
        'model_name': model_data['model_name'],
        'dimensions': model_data['embedding_dimensions'],
        'num_questions': model_data['num_questions_evaluated'],
        'avg_before_metrics': model_data.get('avg_before_metrics', {}),
        'avg_after_metrics': model_data.get('avg_after_metrics', {}),
        'all_before_metrics': model_data.get('all_before_metrics', {}),
        'all_after_metrics': model_data.get('all_after_metrics', {}),
        'rag_metrics': model_data.get('rag_metrics', {}),
        'individual_rag_metrics': model_data.get('individual_rag_metrics', {})
    }
    
    return results

def calculate_statistics(values):
    """Calculate statistics for a list of values"""
    if not values or len(values) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    
    # Filter out None values
    values = [v for v in values if v is not None]
    if not values:
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
    with open('/Users/haroldgomez/Downloads/cumulative_results_20250730_071510.json', 'r') as f:
        data = json.load(f)
    
    results_data = data['results']
    model_results = {}
    
    # Process each model
    for model_name in ['minilm', 'mpnet', 'e5-large', 'ada']:
        if model_name in results_data:
            model_results[model_name] = analyze_model_metrics(results_data[model_name])
    
    # Print detailed analysis for each model
    for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
        if model not in model_results:
            continue
        
        results = model_results[model]
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model.upper()} (Dimensions: {results['dimensions']})")
        print(f"Questions Evaluated: {results['num_questions']}")
        print(f"{'='*80}")
        
        # 1. Average Traditional Metrics
        print("\n1. AVERAGE TRADITIONAL METRICS:")
        print("-" * 50)
        
        metrics_to_show = ['precision@5', 'precision@10', 'recall@5', 'recall@10', 
                          'f1@5', 'f1@10', 'ndcg@5', 'ndcg@10', 'mrr']
        
        print("\nBEFORE RERANKING:")
        for metric in metrics_to_show:
            if metric in results['avg_before_metrics']:
                value = results['avg_before_metrics'][metric]
                print(f"  {metric:<15}: {value:.4f}")
        
        print("\nAFTER RERANKING:")
        for metric in metrics_to_show:
            if metric in results['avg_after_metrics']:
                value = results['avg_after_metrics'][metric]
                print(f"  {metric:<15}: {value:.4f}")
        
        # Calculate improvements
        print("\nIMPROVEMENT (After - Before):")
        for metric in metrics_to_show:
            if (metric in results['avg_before_metrics'] and 
                metric in results['avg_after_metrics']):
                before = results['avg_before_metrics'][metric]
                after = results['avg_after_metrics'][metric]
                improvement = after - before
                improvement_pct = (improvement / before * 100) if before > 0 else 0
                print(f"  {metric:<15}: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # 2. Distribution Analysis (using all_before/after_metrics)
        print("\n2. METRICS DISTRIBUTION (from all questions):")
        print("-" * 50)
        
        # Show distribution for key metrics
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'mrr']
        
        for stage, all_metrics in [('Before Reranking', results['all_before_metrics']), 
                                   ('After Reranking', results['all_after_metrics'])]:
            print(f"\n{stage}:")
            for metric in key_metrics:
                if metric in all_metrics:
                    stats = calculate_statistics(all_metrics[metric])
                    if stats['count'] > 0:
                        print(f"  {metric:<15}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                              f"min={stats['min']:.4f}, max={stats['max']:.4f}")
        
        # 3. RAG Metrics
        print("\n3. RAG METRICS (RAGAS + BERT):")
        print("-" * 50)
        
        if results['rag_metrics']:
            ragas_metrics = results['rag_metrics'].get('ragas', {})
            bert_metrics = results['rag_metrics'].get('bert', {})
            
            print("\nRAGAS Scores:")
            for metric, value in ragas_metrics.items():
                if value is not None:
                    print(f"  {metric:<30}: {value:.4f}")
            
            print("\nBERT Scores:")
            for metric, value in bert_metrics.items():
                if value is not None:
                    print(f"  {metric:<30}: {value:.4f}")
        
        # 4. Individual RAG metrics distribution
        if results['individual_rag_metrics'] and isinstance(results['individual_rag_metrics'], list):
            print("\n4. RAG METRICS DISTRIBUTION:")
            print("-" * 50)
            
            # Extract metrics from list of dictionaries
            ragas_metrics = defaultdict(list)
            bert_metrics = defaultdict(list)
            
            for item in results['individual_rag_metrics']:
                # RAGAS metrics
                for metric in ['faithfulness', 'answer_relevancy', 'answer_correctness', 
                              'context_precision', 'context_recall', 'semantic_similarity']:
                    if metric in item and item[metric] is not None:
                        ragas_metrics[metric].append(item[metric])
                
                # BERT metrics
                for metric in ['bert_precision', 'bert_recall', 'bert_f1']:
                    if metric in item and item[metric] is not None:
                        bert_metrics[metric].append(item[metric])
            
            if ragas_metrics:
                print("\nRAGAS Distribution:")
                for metric, values in ragas_metrics.items():
                    stats = calculate_statistics(values)
                    if stats['count'] > 0:
                        print(f"  {metric:<30}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
            
            if bert_metrics:
                print("\nBERT Distribution:")
                for metric, values in bert_metrics.items():
                    stats = calculate_statistics(values)
                    if stats['count'] > 0:
                        print(f"  {metric:<30}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    
    # Summary comparison across dimensions
    print(f"\n{'='*80}")
    print("DIMENSION IMPACT SUMMARY")
    print(f"{'='*80}")
    
    print("\n1. Performance by Embedding Dimensions:")
    print("-" * 60)
    
    # Create comparison table
    key_metrics = ['precision@10', 'recall@10', 'f1@10', 'ndcg@10', 'mrr']
    
    for stage_name, stage_key in [('Before Reranking', 'avg_before_metrics'), 
                                  ('After Reranking', 'avg_after_metrics')]:
        print(f"\n{stage_name}:")
        print(f"{'Model':<10} {'Dims':<6} " + " ".join([f"{m:<12}" for m in key_metrics]))
        print("-" * 80)
        
        for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
            if model not in model_results:
                continue
            
            results = model_results[model]
            row = f"{model:<10} {results['dimensions']:<6}"
            
            for metric in key_metrics:
                if metric in results[stage_key]:
                    value = results[stage_key][metric]
                    row += f" {value:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
            print(row)
    
    # 2. Analyze dimension correlation
    print("\n\n2. Dimension-Performance Correlation Analysis:")
    print("-" * 60)
    
    dimensions = []
    metric_values = defaultdict(list)
    
    for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
        if model in model_results:
            dims = model_results[model]['dimensions']
            dimensions.append(dims)
            
            # Collect after-reranking metrics
            for metric in key_metrics:
                if metric in model_results[model]['avg_after_metrics']:
                    metric_values[metric].append(model_results[model]['avg_after_metrics'][metric])
    
    print("\nCorrelation between dimensions and metrics (after reranking):")
    for metric, values in metric_values.items():
        if len(values) == len(dimensions):
            correlation = np.corrcoef(dimensions, values)[0, 1]
            print(f"  {metric:<15}: {correlation:.3f}")
    
    # 3. Best performing model analysis
    print("\n\n3. Best Performing Model by Metric:")
    print("-" * 60)
    
    for metric in key_metrics:
        best_model = None
        best_value = -1
        
        for model in ['minilm', 'mpnet', 'e5-large', 'ada']:
            if model in model_results:
                value = model_results[model]['avg_after_metrics'].get(metric, -1)
                if value > best_value:
                    best_value = value
                    best_model = model
        
        if best_model:
            dims = model_results[best_model]['dimensions']
            print(f"  {metric:<15}: {best_model} ({dims}D) = {best_value:.4f}")

if __name__ == "__main__":
    main()