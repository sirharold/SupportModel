#!/usr/bin/env python3
"""
Results Processing Library - Handle evaluation results, statistics, and file operations
"""

import json
import os
import time
import gc
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pytz
from pathlib import Path

# Chile timezone
CHILE_TZ = pytz.timezone('America/Santiago')

class ResultsProcessor:
    """Handles processing and analysis of evaluation results"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def process_single_model_results(self, individual_results: List[Dict[str, Any]], 
                                   model_name: str, use_llm_reranking: bool = True) -> Dict[str, Any]:
        """
        Process results for a single model
        
        Args:
            individual_results: List of individual question results
            model_name: Name of the embedding model
            use_llm_reranking: Whether LLM reranking was used
            
        Returns:
            Processed results dictionary
        """
        if not individual_results:
            return {'error': 'No results to process', 'model_name': model_name}
        
        if self.debug:
            print(f"📊 Processing results for {model_name}: {len(individual_results)} questions")
        
        # Separate retrieval metrics from RAG metrics
        retrieval_metrics_before = []
        retrieval_metrics_after = []
        rag_metrics_before = []
        rag_metrics_after = []
        
        # Extract metrics from individual results
        for result in individual_results:
            # Retrieval metrics
            if 'retrieval_metrics_before' in result:
                retrieval_metrics_before.append(result['retrieval_metrics_before'])
            
            if 'retrieval_metrics_after' in result:
                retrieval_metrics_after.append(result['retrieval_metrics_after'])
            
            # RAG metrics
            if 'rag_metrics_before' in result:
                rag_metrics_before.append(result['rag_metrics_before'])
            
            if 'rag_metrics_after' in result:
                rag_metrics_after.append(result['rag_metrics_after'])
        
        # Calculate averages
        processed_results = {
            'model_name': model_name,
            'num_questions_evaluated': len(individual_results),
            'use_llm_reranking': use_llm_reranking,
            'timestamp': datetime.now(CHILE_TZ).isoformat()
        }
        
        # Process retrieval metrics
        if retrieval_metrics_before:
            processed_results['avg_before_metrics'] = self._calculate_average_metrics(retrieval_metrics_before)
        
        if retrieval_metrics_after:
            processed_results['avg_after_metrics'] = self._calculate_average_metrics(retrieval_metrics_after)
        elif retrieval_metrics_before:
            # If no after metrics, use before metrics
            processed_results['avg_after_metrics'] = processed_results['avg_before_metrics'].copy()
        
        # Process RAG metrics
        if rag_metrics_before:
            processed_results['avg_rag_before'] = self._calculate_average_metrics(rag_metrics_before)
        
        if rag_metrics_after:
            processed_results['avg_rag_after'] = self._calculate_average_metrics(rag_metrics_after)
        elif rag_metrics_before:
            processed_results['avg_rag_after'] = processed_results['avg_rag_before'].copy()
        
        # Calculate improvements
        if 'avg_before_metrics' in processed_results and 'avg_after_metrics' in processed_results:
            processed_results['improvements'] = self._calculate_improvements(
                processed_results['avg_before_metrics'],
                processed_results['avg_after_metrics']
            )
        
        # Calculate RAG improvements
        if 'avg_rag_before' in processed_results and 'avg_rag_after' in processed_results:
            processed_results['rag_improvements'] = self._calculate_improvements(
                processed_results['avg_rag_before'],
                processed_results['avg_rag_after']
            )
        
        # Add individual question details
        processed_results['individual_question_results'] = individual_results
        
        # Calculate additional statistics
        processed_results['statistics'] = self._calculate_result_statistics(individual_results)
        
        if self.debug:
            print(f"✅ Processed results for {model_name}")
            if 'avg_before_metrics' in processed_results:
                print(f"📈 Average metrics calculated: {len(processed_results['avg_before_metrics'])} metrics")
        
        return processed_results
    
    def process_multiple_models_results(self, models_results: Dict[str, List[Dict[str, Any]]], 
                                      use_llm_reranking: bool = True) -> Dict[str, Any]:
        """
        Process results for multiple models
        
        Args:
            models_results: Dictionary mapping model names to their individual results
            use_llm_reranking: Whether LLM reranking was used
            
        Returns:
            Comprehensive comparison results
        """
        if self.debug:
            print(f"📊 Processing results for {len(models_results)} models")
        
        # Process each model individually
        processed_models = {}
        for model_name, individual_results in models_results.items():
            processed_models[model_name] = self.process_single_model_results(
                individual_results, model_name, use_llm_reranking
            )
        
        # Create comparison summary
        comparison_results = {
            'timestamp': datetime.now(CHILE_TZ).isoformat(),
            'num_models': len(processed_models),
            'use_llm_reranking': use_llm_reranking,
            'models_results': processed_models,
            'comparison_summary': self._create_comparison_summary(processed_models),
            'ranking': self._create_model_ranking(processed_models, use_llm_reranking)
        }
        
        if self.debug:
            print(f"✅ Processed comparison for {len(processed_models)} models")
        
        return comparison_results
    
    def _calculate_average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics from a list of metric dictionaries"""
        if not metrics_list:
            return {}
        
        # Get all unique metric names
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Calculate averages
        averages = {}
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics and metrics[key] is not None:
                    try:
                        values.append(float(metrics[key]))
                    except (ValueError, TypeError):
                        continue
            
            averages[key] = float(np.mean(values)) if values else 0.0
        
        return averages
    
    def _calculate_improvements(self, before_metrics: Dict[str, float], 
                              after_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate improvements between before and after metrics"""
        improvements = {}
        
        for metric_name in before_metrics:
            if metric_name in after_metrics:
                before_val = before_metrics[metric_name]
                after_val = after_metrics[metric_name]
                
                # Calculate absolute and percentage improvements
                absolute_improvement = after_val - before_val
                percentage_improvement = ((after_val - before_val) / before_val * 100) if before_val > 0 else 0.0
                
                improvements[metric_name] = {
                    'before': before_val,
                    'after': after_val,
                    'absolute': absolute_improvement,
                    'percentage': percentage_improvement,
                    'improved': absolute_improvement > 0
                }
        
        return improvements
    
    def _calculate_result_statistics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate additional statistics from individual results"""
        if not individual_results:
            return {}
        
        stats = {
            'total_questions': len(individual_results),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'avg_retrieved_docs': 0,
            'avg_reranked_docs': 0
        }
        
        retrieved_docs_counts = []
        reranked_docs_counts = []
        
        for result in individual_results:
            # Count successful vs failed evaluations
            if result.get('retrieval_metrics_before') or result.get('rag_metrics_before'):
                stats['successful_evaluations'] += 1
            else:
                stats['failed_evaluations'] += 1
            
            # Collect document counts
            if 'num_retrieved_docs' in result:
                retrieved_docs_counts.append(result['num_retrieved_docs'])
            
            if 'num_reranked_docs' in result:
                reranked_docs_counts.append(result['num_reranked_docs'])
        
        # Calculate averages
        if retrieved_docs_counts:
            stats['avg_retrieved_docs'] = float(np.mean(retrieved_docs_counts))
            stats['std_retrieved_docs'] = float(np.std(retrieved_docs_counts))
        
        if reranked_docs_counts:
            stats['avg_reranked_docs'] = float(np.mean(reranked_docs_counts))
            stats['std_reranked_docs'] = float(np.std(reranked_docs_counts))
        
        return stats
    
    def _create_comparison_summary(self, processed_models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary comparing all models"""
        if not processed_models:
            return {}
        
        summary = {
            'best_model_overall': None,
            'best_model_by_metric': {},
            'metric_ranges': {},
            'model_strengths': {}
        }
        
        # Key retrieval metrics for comparison
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5', 'map@5']
        
        # Find best model for each metric
        for metric in key_metrics:
            best_model = None
            best_value = -1
            values = []
            
            for model_name, results in processed_models.items():
                metrics = results.get('avg_after_metrics', results.get('avg_before_metrics', {}))
                if metric in metrics:
                    value = metrics[metric]
                    values.append(value)
                    if value > best_value:
                        best_value = value
                        best_model = model_name
            
            if best_model:
                summary['best_model_by_metric'][metric] = {
                    'model': best_model,
                    'value': best_value
                }
            
            if values:
                summary['metric_ranges'][metric] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        # Calculate overall best model (simple average of key metrics)
        model_scores = {}
        for model_name, results in processed_models.items():
            metrics = results.get('avg_after_metrics', results.get('avg_before_metrics', {}))
            score = np.mean([metrics.get(m, 0) for m in key_metrics if m in metrics])
            model_scores[model_name] = score
        
        if model_scores:
            best_overall = max(model_scores.items(), key=lambda x: x[1])
            summary['best_model_overall'] = {
                'model': best_overall[0],
                'score': best_overall[1]
            }
        
        return summary
    
    def _create_model_ranking(self, processed_models: Dict[str, Dict[str, Any]], 
                            use_llm_reranking: bool = True) -> List[Dict[str, Any]]:
        """Create ranking of models based on performance"""
        if not processed_models:
            return []
        
        # Key metrics for ranking (weighted)
        ranking_weights = {
            'precision@5': 0.25,
            'recall@5': 0.25,
            'f1@5': 0.25,
            'mrr': 0.15,
            'ndcg@5': 0.10
        }
        
        model_rankings = []
        
        for model_name, results in processed_models.items():
            metrics = results.get('avg_after_metrics' if use_llm_reranking else 'avg_before_metrics', {})
            
            # Calculate weighted score
            weighted_score = 0
            valid_metrics = 0
            
            for metric, weight in ranking_weights.items():
                if metric in metrics:
                    weighted_score += metrics[metric] * weight
                    valid_metrics += weight
            
            # Normalize score
            final_score = weighted_score / valid_metrics if valid_metrics > 0 else 0
            
            model_rankings.append({
                'model': model_name,
                'score': final_score,
                'num_questions': results.get('num_questions_evaluated', 0),
                'key_metrics': {k: metrics.get(k, 0) for k in ranking_weights.keys()}
            })
        
        # Sort by score (descending)
        model_rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for i, ranking in enumerate(model_rankings):
            ranking['rank'] = i + 1
        
        return model_rankings


class ResultsSaver:
    """Handles saving results to various formats"""
    
    def __init__(self, output_path: str, debug: bool = False):
        self.output_path = Path(output_path)
        self.debug = debug
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def save_results_json(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save results to JSON file
        
        Args:
            results: Results dictionary to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"cumulative_results_{timestamp}.json"
        
        filepath = self.output_path / filename
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            if self.debug:
                print(f"💾 Results saved to: {filepath}")
                print(f"📊 File size: {filepath.stat().st_size / 1024:.2f} KB")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error saving results to JSON: {e}")
            raise
    
    def save_results_excel(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save results to Excel file with multiple sheets
        
        Args:
            results: Results dictionary to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"cumulative_results_{timestamp}.xlsx"
        
        filepath = self.output_path / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                if 'comparison_summary' in results:
                    summary_df = self._create_summary_dataframe(results)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Model rankings sheet
                if 'ranking' in results:
                    ranking_df = pd.DataFrame(results['ranking'])
                    ranking_df.to_excel(writer, sheet_name='Model_Rankings', index=False)
                
                # Individual model sheets
                if 'models_results' in results:
                    for model_name, model_results in results['models_results'].items():
                        model_df = self._create_model_dataframe(model_results)
                        sheet_name = f"Model_{model_name}"[:31]  # Excel sheet name limit
                        model_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            if self.debug:
                print(f"📊 Results saved to Excel: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error saving results to Excel: {e}")
            raise
    
    def save_summary_csv(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save summary results to CSV file
        
        Args:
            results: Results dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"results_summary_{timestamp}.csv"
        
        filepath = self.output_path / filename
        
        try:
            summary_df = self._create_summary_dataframe(results)
            summary_df.to_csv(filepath, index=False, encoding='utf-8')
            
            if self.debug:
                print(f"📄 Summary saved to CSV: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error saving summary to CSV: {e}")
            raise
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _create_summary_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame from results"""
        summary_data = []
        
        if 'models_results' in results:
            for model_name, model_results in results['models_results'].items():
                row = {'Model': model_name}
                
                # Add average metrics
                avg_metrics = model_results.get('avg_after_metrics', model_results.get('avg_before_metrics', {}))
                for metric, value in avg_metrics.items():
                    row[metric] = value
                
                # Add other key information
                row['Num_Questions'] = model_results.get('num_questions_evaluated', 0)
                row['LLM_Reranking'] = model_results.get('use_llm_reranking', False)
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _create_model_dataframe(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Create detailed DataFrame for a single model"""
        data = []
        
        # Basic information
        info_row = {
            'Metric': 'Model_Info',
            'Value': model_results.get('model_name', 'Unknown'),
            'Description': f"Questions: {model_results.get('num_questions_evaluated', 0)}"
        }
        data.append(info_row)
        
        # Average metrics before
        if 'avg_before_metrics' in model_results:
            for metric, value in model_results['avg_before_metrics'].items():
                data.append({
                    'Metric': f"Before_{metric}",
                    'Value': value,
                    'Description': f"Average {metric} before reranking"
                })
        
        # Average metrics after
        if 'avg_after_metrics' in model_results:
            for metric, value in model_results['avg_after_metrics'].items():
                data.append({
                    'Metric': f"After_{metric}",
                    'Value': value,
                    'Description': f"Average {metric} after reranking"
                })
        
        # Improvements
        if 'improvements' in model_results:
            for metric, improvement in model_results['improvements'].items():
                data.append({
                    'Metric': f"Improvement_{metric}",
                    'Value': improvement.get('percentage', 0),
                    'Description': f"Percentage improvement in {metric}"
                })
        
        return pd.DataFrame(data)


class ResultsAnalyzer:
    """Provides analysis and insights from evaluation results"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def analyze_model_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance patterns and provide insights
        
        Args:
            results: Processed evaluation results
            
        Returns:
            Analysis insights
        """
        analysis = {
            'timestamp': datetime.now(CHILE_TZ).isoformat(),
            'performance_insights': {},
            'recommendations': [],
            'anomalies': []
        }
        
        if 'models_results' not in results:
            return analysis
        
        models_results = results['models_results']
        
        # Analyze each model
        for model_name, model_results in models_results.items():
            model_analysis = self._analyze_single_model(model_name, model_results)
            analysis['performance_insights'][model_name] = model_analysis
        
        # Generate comparative insights
        analysis['comparative_insights'] = self._generate_comparative_insights(models_results)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(models_results, results.get('ranking', []))
        
        # Detect anomalies
        analysis['anomalies'] = self._detect_anomalies(models_results)
        
        if self.debug:
            print(f"🔍 Analysis completed for {len(models_results)} models")
        
        return analysis
    
    def _analyze_single_model(self, model_name: str, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance of a single model"""
        analysis = {
            'model_name': model_name,
            'strengths': [],
            'weaknesses': [],
            'consistency': None,
            'improvement_effectiveness': None
        }
        
        # Get metrics
        before_metrics = model_results.get('avg_before_metrics', {})
        after_metrics = model_results.get('avg_after_metrics', {})
        improvements = model_results.get('improvements', {})
        
        # Analyze strengths and weaknesses
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5']
        
        for metric in key_metrics:
            if metric in after_metrics:
                value = after_metrics[metric]
                if value >= 0.7:
                    analysis['strengths'].append(f"High {metric}: {value:.3f}")
                elif value < 0.4:
                    analysis['weaknesses'].append(f"Low {metric}: {value:.3f}")
        
        # Analyze improvement effectiveness
        if improvements:
            positive_improvements = sum(1 for imp in improvements.values() if imp.get('improved', False))
            total_improvements = len(improvements)
            effectiveness = positive_improvements / total_improvements if total_improvements > 0 else 0
            
            analysis['improvement_effectiveness'] = {
                'score': effectiveness,
                'positive_improvements': positive_improvements,
                'total_metrics': total_improvements,
                'description': self._get_effectiveness_description(effectiveness)
            }
        
        return analysis
    
    def _generate_comparative_insights(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights comparing all models"""
        insights = {
            'performance_gaps': {},
            'metric_leaders': {},
            'consistency_ranking': []
        }
        
        # Find metric leaders
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5']
        
        for metric in key_metrics:
            values = []
            for model_name, results in models_results.items():
                avg_metrics = results.get('avg_after_metrics', results.get('avg_before_metrics', {}))
                if metric in avg_metrics:
                    values.append((model_name, avg_metrics[metric]))
            
            if values:
                values.sort(key=lambda x: x[1], reverse=True)
                insights['metric_leaders'][metric] = values[0]
                
                # Calculate performance gap
                if len(values) > 1:
                    gap = values[0][1] - values[-1][1]
                    insights['performance_gaps'][metric] = {
                        'gap': gap,
                        'leader': values[0],
                        'lowest': values[-1]
                    }
        
        return insights
    
    def _generate_recommendations(self, models_results: Dict[str, Dict[str, Any]], 
                                ranking: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not ranking:
            return recommendations
        
        # Best overall model recommendation
        best_model = ranking[0]
        recommendations.append({
            'type': 'best_choice',
            'title': 'Recommended Model',
            'description': f"Use {best_model['model']} for best overall performance (score: {best_model['score']:.3f})",
            'action': f"Deploy {best_model['model']} for production use cases requiring balanced performance"
        })
        
        # Specialized use case recommendations
        if len(ranking) > 1:
            for i, model_ranking in enumerate(ranking[1:], 1):
                model_name = model_ranking['model']
                model_results = models_results.get(model_name, {})
                
                # Find this model's strongest metric
                avg_metrics = model_results.get('avg_after_metrics', model_results.get('avg_before_metrics', {}))
                if avg_metrics:
                    best_metric = max(avg_metrics.items(), key=lambda x: x[1])
                    if best_metric[1] > 0.6:  # Only recommend if reasonably good
                        recommendations.append({
                            'type': 'specialized',
                            'title': f'Specialized Use Case - {model_name}',
                            'description': f"Consider {model_name} for scenarios prioritizing {best_metric[0]} ({best_metric[1]:.3f})",
                            'action': f"Use {model_name} when {best_metric[0]} is the primary concern"
                        })
        
        return recommendations
    
    def _detect_anomalies(self, models_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect unusual patterns or potential issues"""
        anomalies = []
        
        for model_name, results in models_results.items():
            improvements = results.get('improvements', {})
            
            # Check for negative improvements in multiple metrics
            negative_improvements = [metric for metric, imp in improvements.items() 
                                   if imp.get('percentage', 0) < -5]  # More than 5% degradation
            
            if len(negative_improvements) >= 3:
                anomalies.append({
                    'type': 'performance_degradation',
                    'model': model_name,
                    'description': f"Multiple metrics degraded after reranking: {negative_improvements}",
                    'severity': 'high'
                })
            
            # Check for extremely low performance
            avg_metrics = results.get('avg_after_metrics', results.get('avg_before_metrics', {}))
            low_metrics = [metric for metric, value in avg_metrics.items() if value < 0.1]
            
            if low_metrics:
                anomalies.append({
                    'type': 'low_performance',
                    'model': model_name,
                    'description': f"Extremely low performance in: {low_metrics}",
                    'severity': 'medium'
                })
        
        return anomalies
    
    def _get_effectiveness_description(self, effectiveness: float) -> str:
        """Get description for improvement effectiveness score"""
        if effectiveness >= 0.8:
            return "Highly effective reranking"
        elif effectiveness >= 0.6:
            return "Moderately effective reranking"
        elif effectiveness >= 0.4:
            return "Somewhat effective reranking"
        else:
            return "Limited reranking effectiveness"


# Factory functions
def create_results_processor(debug: bool = False) -> ResultsProcessor:
    """Create ResultsProcessor instance"""
    return ResultsProcessor(debug)

def create_results_saver(output_path: str, debug: bool = False) -> ResultsSaver:
    """Create ResultsSaver instance"""
    return ResultsSaver(output_path, debug)

def create_results_analyzer(debug: bool = False) -> ResultsAnalyzer:
    """Create ResultsAnalyzer instance"""
    return ResultsAnalyzer(debug)


# Complete pipeline function
def process_and_save_results(individual_results: Dict[str, List[Dict[str, Any]]], 
                           output_path: str, use_llm_reranking: bool = True,
                           debug: bool = False) -> Dict[str, str]:
    """
    Complete pipeline to process and save results
    
    Args:
        individual_results: Dictionary mapping model names to their individual results
        output_path: Path to save results
        use_llm_reranking: Whether LLM reranking was used
        debug: Enable debug output
        
    Returns:
        Dictionary with paths to saved files
    """
    if debug:
        print("🚀 Starting complete results processing pipeline...")
    
    # Process results
    processor = create_results_processor(debug)
    processed_results = processor.process_multiple_models_results(individual_results, use_llm_reranking)
    
    # Analyze results
    analyzer = create_results_analyzer(debug)
    analysis = analyzer.analyze_model_performance(processed_results)
    
    # Combine results with analysis
    final_results = {**processed_results, 'analysis': analysis}
    
    # Save results
    saver = create_results_saver(output_path, debug)
    
    saved_files = {}
    saved_files['json'] = saver.save_results_json(final_results)
    saved_files['excel'] = saver.save_results_excel(final_results)
    saved_files['csv'] = saver.save_summary_csv(final_results)
    
    if debug:
        print("✅ Complete results processing pipeline finished!")
        print(f"📁 Saved files: {list(saved_files.keys())}")
    
    return saved_files


if __name__ == "__main__":
    # Test the results processing components
    print("🧪 Testing Results Processing Library...")
    
    # Create sample data for testing
    sample_individual_results = {
        'test_model': [
            {
                'retrieval_metrics_before': {'precision@5': 0.6, 'recall@5': 0.5, 'f1@5': 0.55},
                'retrieval_metrics_after': {'precision@5': 0.65, 'recall@5': 0.55, 'f1@5': 0.6},
                'rag_metrics_before': {'faithfulness': 0.7, 'answer_relevancy': 0.8},
                'rag_metrics_after': {'faithfulness': 0.75, 'answer_relevancy': 0.82}
            }
        ]
    }
    
    # Test results processor
    print("📊 Testing Results Processor...")
    processor = create_results_processor(debug=True)
    processed = processor.process_multiple_models_results(sample_individual_results)
    print(f"✅ Processed results for {processed['num_models']} models")
    
    # Test results analyzer
    print("🔍 Testing Results Analyzer...")
    analyzer = create_results_analyzer(debug=True)
    analysis = analyzer.analyze_model_performance(processed)
    print(f"✅ Generated {len(analysis['recommendations'])} recommendations")
    
    print("🎉 Results Processing Library test completed!")