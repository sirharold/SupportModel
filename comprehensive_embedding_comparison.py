#!/usr/bin/env python3
"""
Comprehensive Embedding Models Comparison Script
Evaluates all embedding models and generates detailed comparison report
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import argparse
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing utilities
from src.config.config import EMBEDDING_MODELS, GENERATIVE_MODELS, CHROMADB_COLLECTION_CONFIG
from src.evaluation.cumulative import run_cumulative_metrics_for_models
from src.evaluation.comparison import compare_models_with_advanced_metrics
from src.evaluation.enhanced_evaluation import evaluate_rag_with_advanced_metrics
from src.services.storage.chromadb_utils import ChromaDBConfig
from src.utils.memory_utils import get_memory_usage, cleanup_memory


def comprehensive_embedding_comparison(
    num_questions: int = 200,
    generative_model: str = "llama-3.3-70b",
    top_k: int = 10,
    use_llm_reranker: bool = True,
    batch_size: int = 50,
    output_dir: str = "comparison_results",
    include_visualizations: bool = True
):
    """
    Run comprehensive comparison across all embedding models
    
    Args:
        num_questions: Number of questions to evaluate
        generative_model: Model for LLM reranking
        top_k: Number of documents to retrieve
        use_llm_reranker: Whether to use LLM reranking
        batch_size: Batch size for processing
        output_dir: Directory to save results
        include_visualizations: Whether to generate charts
    """
    
    print("🚀 Starting Comprehensive Embedding Comparison")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   Questions: {num_questions}")
    print(f"   Models: {list(EMBEDDING_MODELS.keys())}")
    print(f"   Generative Model: {generative_model}")
    print(f"   Top-K: {top_k}")
    print(f"   LLM Reranking: {use_llm_reranker}")
    print(f"   Batch Size: {batch_size}")
    print("=" * 60)
    
    # Initialize results storage
    all_results = {}
    execution_metrics = {
        'start_time': datetime.now(),
        'model_times': {},
        'total_questions_processed': 0,
        'memory_usage': []
    }
    
    # Record initial memory
    initial_memory = get_memory_usage()
    execution_metrics['memory_usage'].append(('initial', initial_memory))
    
    try:
        # Phase 1: Run cumulative evaluation for all models
        print("\n📈 Phase 1: Cumulative Metrics Evaluation")
        print("-" * 40)
        
        phase1_start = time.time()
        cumulative_results = run_cumulative_metrics_for_models(
            num_questions=num_questions,
            model_names=list(EMBEDDING_MODELS.keys()),
            generative_model_name=generative_model,
            top_k=top_k,
            use_llm_reranker=use_llm_reranker,
            batch_size=batch_size
        )
        phase1_time = time.time() - phase1_start
        execution_metrics['phase1_time'] = phase1_time
        
        print(f"✅ Phase 1 completed in {phase1_time:.2f} seconds")
        
        # Record memory after phase 1
        phase1_memory = get_memory_usage()
        execution_metrics['memory_usage'].append(('after_phase1', phase1_memory))
        
        # Phase 2: Advanced metrics comparison
        print("\n🔬 Phase 2: Advanced Metrics Comparison")
        print("-" * 40)
        
        phase2_start = time.time()
        advanced_results = {}
        
        # Run pairwise comparisons between models
        models = list(EMBEDDING_MODELS.keys())
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                print(f"   Comparing {model1} vs {model2}...")
                
                try:
                    comparison_result = compare_models_with_advanced_metrics(
                        model1_name=model1,
                        model2_name=model2,
                        num_questions=min(num_questions, 100),  # Use subset for pairwise
                        top_k=top_k,
                        use_llm_reranker=use_llm_reranker
                    )
                    advanced_results[f"{model1}_vs_{model2}"] = comparison_result
                    
                except Exception as e:
                    print(f"     ⚠️ Error comparing {model1} vs {model2}: {e}")
                    advanced_results[f"{model1}_vs_{model2}"] = {"error": str(e)}
        
        phase2_time = time.time() - phase2_start
        execution_metrics['phase2_time'] = phase2_time
        
        print(f"✅ Phase 2 completed in {phase2_time:.2f} seconds")
        
        # Record memory after phase 2
        phase2_memory = get_memory_usage()
        execution_metrics['memory_usage'].append(('after_phase2', phase2_memory))
        
        # Phase 3: Individual model deep analysis
        print("\n🎯 Phase 3: Individual Model Analysis")
        print("-" * 40)
        
        phase3_start = time.time()
        individual_results = {}
        
        for model_name in EMBEDDING_MODELS.keys():
            print(f"   Analyzing {model_name}...")
            
            try:
                # Run enhanced evaluation for each model
                enhanced_result = evaluate_rag_with_advanced_metrics(
                    question="Sample evaluation question for model analysis",
                    model_name=model_name,
                    generative_model_name=generative_model,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker
                )
                individual_results[model_name] = enhanced_result
                
            except Exception as e:
                print(f"     ⚠️ Error analyzing {model_name}: {e}")
                individual_results[model_name] = {"error": str(e)}
        
        phase3_time = time.time() - phase3_start
        execution_metrics['phase3_time'] = phase3_time
        
        print(f"✅ Phase 3 completed in {phase3_time:.2f} seconds")
        
        # Compile comprehensive results
        all_results = {
            'cumulative_metrics': cumulative_results,
            'advanced_comparisons': advanced_results,
            'individual_analysis': individual_results,
            'execution_metrics': execution_metrics,
            'configuration': {
                'num_questions': num_questions,
                'generative_model': generative_model,
                'top_k': top_k,
                'use_llm_reranker': use_llm_reranker,
                'batch_size': batch_size,
                'models_evaluated': list(EMBEDDING_MODELS.keys())
            }
        }
        
        # Phase 4: Generate summary report
        print("\n📊 Phase 4: Generating Summary Report")
        print("-" * 40)
        
        summary_report = generate_summary_report(all_results)
        all_results['summary_report'] = summary_report
        
        # Phase 5: Save results and generate visualizations
        print("\n💾 Phase 5: Saving Results")
        print("-" * 40)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comprehensive results
        results_file = f"{output_dir}/comprehensive_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Results saved to: {results_file}")
        
        # Generate CSV summary for easy analysis
        csv_file = f"{output_dir}/comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        generate_csv_summary(cumulative_results, csv_file)
        print(f"✅ CSV summary saved to: {csv_file}")
        
        # Generate visualizations
        if include_visualizations:
            viz_file = f"{output_dir}/comparison_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            generate_comparison_visualizations(cumulative_results, viz_file)
            print(f"✅ Visualizations saved to: {viz_file}")
        
        # Final memory cleanup
        cleanup_memory()
        final_memory = get_memory_usage()
        execution_metrics['memory_usage'].append(('final', final_memory))
        
        # Print final summary
        total_time = time.time() - time.mktime(execution_metrics['start_time'].timetuple())
        print("\n🎉 COMPREHENSIVE COMPARISON COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
        print(f"📊 Questions Processed: {num_questions}")
        print(f"🔍 Models Evaluated: {len(EMBEDDING_MODELS)}")
        print(f"💾 Memory Usage: {initial_memory:.1f} MB → {final_memory:.1f} MB")
        print(f"📁 Results Directory: {output_dir}")
        print("=" * 60)
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ Error during comprehensive comparison: {e}")
        raise
    finally:
        cleanup_memory()


def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary of comparison results"""
    
    cumulative_results = results.get('cumulative_metrics', {})
    
    if not cumulative_results:
        return {"error": "No cumulative results to summarize"}
    
    # Extract key metrics for each model
    model_summary = {}
    
    for model_name, model_data in cumulative_results.items():
        if isinstance(model_data, dict) and 'avg_metrics' in model_data:
            avg_metrics = model_data['avg_metrics']
            
            model_summary[model_name] = {
                'precision': avg_metrics.get('avg_precision', 0),
                'recall': avg_metrics.get('avg_recall', 0),
                'f1_score': avg_metrics.get('avg_f1', 0),
                'map_score': avg_metrics.get('avg_map', 0),
                'mrr_score': avg_metrics.get('avg_mrr', 0),
                'ndcg_score': avg_metrics.get('avg_ndcg', 0),
                'total_questions': model_data.get('total_questions', 0),
                'avg_execution_time': model_data.get('avg_execution_time', 0)
            }
    
    # Determine best performing model for each metric
    best_models = {}
    metrics = ['precision', 'recall', 'f1_score', 'map_score', 'mrr_score', 'ndcg_score']
    
    for metric in metrics:
        if model_summary:
            best_model = max(model_summary.keys(), 
                           key=lambda k: model_summary[k].get(metric, 0))
            best_score = model_summary[best_model].get(metric, 0)
            best_models[metric] = {'model': best_model, 'score': best_score}
    
    # Overall ranking
    model_rankings = []
    for model_name, metrics_data in model_summary.items():
        # Calculate weighted average score
        weighted_score = (
            metrics_data.get('f1_score', 0) * 0.3 +
            metrics_data.get('map_score', 0) * 0.25 +
            metrics_data.get('mrr_score', 0) * 0.25 +
            metrics_data.get('ndcg_score', 0) * 0.2
        )
        model_rankings.append({
            'model': model_name,
            'weighted_score': weighted_score,
            'metrics': metrics_data
        })
    
    model_rankings.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    return {
        'model_summary': model_summary,
        'best_models_by_metric': best_models,
        'overall_ranking': model_rankings,
        'top_model': model_rankings[0] if model_rankings else None,
        'models_evaluated': len(model_summary),
        'summary_generated': datetime.now().isoformat()
    }


def generate_csv_summary(cumulative_results: Dict[str, Any], output_file: str):
    """Generate CSV summary of results"""
    
    summary_data = []
    
    for model_name, model_data in cumulative_results.items():
        if isinstance(model_data, dict) and 'avg_metrics' in model_data:
            avg_metrics = model_data['avg_metrics']
            
            summary_data.append({
                'Model': model_name,
                'Precision': avg_metrics.get('avg_precision', 0),
                'Recall': avg_metrics.get('avg_recall', 0),
                'F1_Score': avg_metrics.get('avg_f1', 0),
                'MAP': avg_metrics.get('avg_map', 0),
                'MRR': avg_metrics.get('avg_mrr', 0),
                'NDCG': avg_metrics.get('avg_ndcg', 0),
                'Total_Questions': model_data.get('total_questions', 0),
                'Avg_Execution_Time': model_data.get('avg_execution_time', 0)
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)


def generate_comparison_visualizations(cumulative_results: Dict[str, Any], output_file: str):
    """Generate comparison visualizations"""
    
    try:
        # Extract data for visualization
        models = []
        metrics_data = {
            'Precision': [],
            'Recall': [],
            'F1_Score': [],
            'MAP': [],
            'MRR': [],
            'NDCG': []
        }
        
        for model_name, model_data in cumulative_results.items():
            if isinstance(model_data, dict) and 'avg_metrics' in model_data:
                models.append(model_name)
                avg_metrics = model_data['avg_metrics']
                
                metrics_data['Precision'].append(avg_metrics.get('avg_precision', 0))
                metrics_data['Recall'].append(avg_metrics.get('avg_recall', 0))
                metrics_data['F1_Score'].append(avg_metrics.get('avg_f1', 0))
                metrics_data['MAP'].append(avg_metrics.get('avg_map', 0))
                metrics_data['MRR'].append(avg_metrics.get('avg_mrr', 0))
                metrics_data['NDCG'].append(avg_metrics.get('avg_ndcg', 0))
        
        if not models:
            print("⚠️ No data available for visualizations")
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Embedding Models Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = list(metrics_data.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics_list):
            row = i // 3
            col = i % 3
            
            ax = axes[row, col]
            bars = ax.bar(models, metrics_data[metric], color=colors)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Error generating visualizations: {e}")


def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Comprehensive Embedding Models Comparison')
    parser.add_argument('--questions', type=int, default=200, help='Number of questions to evaluate')
    parser.add_argument('--generative-model', default='llama-3.3-70b', help='Generative model for reranking')
    parser.add_argument('--top-k', type=int, default=10, help='Top-K documents to retrieve')
    parser.add_argument('--no-llm-reranker', action='store_true', help='Disable LLM reranking')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Run comprehensive comparison
    results = comprehensive_embedding_comparison(
        num_questions=args.questions,
        generative_model=args.generative_model,
        top_k=args.top_k,
        use_llm_reranker=not args.no_llm_reranker,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        include_visualizations=not args.no_visualizations
    )
    
    return results


if __name__ == "__main__":
    main()