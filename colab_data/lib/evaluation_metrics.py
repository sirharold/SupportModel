#!/usr/bin/env python3
"""
Evaluation Metrics Library - All retrieval metrics calculations
"""

import numpy as np
from typing import List, Dict

class RetrievalMetricsCalculator:
    """Comprehensive retrieval metrics calculator"""
    
    def __init__(self):
        self.supported_metrics = [
            'precision', 'recall', 'f1', 'ndcg', 'map', 'mrr'
        ]
        self.default_k_values = [1, 3, 5, 10]
    
    def calculate_ndcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        
        Args:
            relevance_scores: List of relevance scores (1.0 for relevant, 0.0 for non-relevant)
            k: Number of top documents to consider
            
        Returns:
            NDCG@k score between 0 and 1
        """
        if k <= 0 or not relevance_scores:
            return 0.0
        
        # Calculate DCG@k
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]) if rel > 0)
        
        # Calculate IDCG@k (ideal DCG)
        ideal_relevance = sorted(relevance_scores[:k], reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance) if rel > 0)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_map_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Mean Average Precision at K
        
        Args:
            relevance_scores: List of relevance scores (1.0 for relevant, 0.0 for non-relevant)
            k: Number of top documents to consider
            
        Returns:
            MAP@k score between 0 and 1
        """
        if k <= 0 or not relevance_scores:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, rel in enumerate(relevance_scores[:k]):
            if rel > 0:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / relevant_count if relevant_count > 0 else 0.0
    
    def calculate_mrr_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Mean Reciprocal Rank at K
        
        Args:
            relevance_scores: List of relevance scores (1.0 for relevant, 0.0 for non-relevant)
            k: Number of top documents to consider
            
        Returns:
            MRR@k score between 0 and 1
        """
        if k <= 0 or not relevance_scores:
            return 0.0
        
        # Only consider top k documents
        top_k_scores = relevance_scores[:k]
        
        # Find first relevant document within top k
        for rank, relevance in enumerate(top_k_scores, 1):
            if relevance > 0:  # Found first relevant document
                return 1.0 / rank
        
        # No relevant document found in top k
        return 0.0
    
    def normalize_link(self, link: str) -> str:
        """
        Normalize URL for comparison
        
        Args:
            link: URL string
            
        Returns:
            Normalized URL string
        """
        if not link:
            return ""
        return link.split('#')[0].split('?')[0].rstrip('/')
    
    def calculate_retrieval_metrics(self, retrieved_docs: List[Dict], ground_truth_links: List[str], 
                                  top_k_values: List[int] = None) -> Dict:
        """
        Calculate comprehensive retrieval metrics
        
        Args:
            retrieved_docs: List of retrieved documents with 'link' field
            ground_truth_links: List of ground truth document links
            top_k_values: List of k values to calculate metrics for
            
        Returns:
            Dictionary of calculated metrics
        """
        if top_k_values is None:
            top_k_values = self.default_k_values
        
        # Normalize ground truth links
        gt_normalized = set(self.normalize_link(link) for link in ground_truth_links)
        
        # Calculate relevance scores for retrieved documents
        relevance_scores = []
        retrieved_links_normalized = []
        
        for doc in retrieved_docs:
            link = self.normalize_link(doc.get('link', ''))
            retrieved_links_normalized.append(link)
            relevance_scores.append(1.0 if link in gt_normalized else 0.0)
        
        # Calculate metrics for each k value
        metrics = {}
        
        for k in top_k_values:
            top_k_relevance = relevance_scores[:k]
            top_k_links = retrieved_links_normalized[:k]
            
            # Get relevant retrieved documents
            retrieved_links = set(link for link in top_k_links if link)
            relevant_retrieved = retrieved_links.intersection(gt_normalized)
            
            # Calculate basic metrics
            precision_k = len(relevant_retrieved) / k if k > 0 else 0.0
            recall_k = len(relevant_retrieved) / len(gt_normalized) if gt_normalized else 0.0
            f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
            
            # Store metrics
            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k
            metrics[f'f1@{k}'] = f1_k
            metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(top_k_relevance, k)
            metrics[f'map@{k}'] = self.calculate_map_at_k(top_k_relevance, k)
            metrics[f'mrr@{k}'] = self.calculate_mrr_at_k(relevance_scores, k)
        
        # Overall MRR (considering all retrieved documents)
        overall_mrr = self.calculate_mrr_at_k(relevance_scores, len(relevance_scores))
        metrics['mrr'] = overall_mrr
        
        return metrics
    
    def calculate_averages(self, metrics_list: List[Dict]) -> Dict:
        """
        Calculate average metrics from a list of individual metric dictionaries
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary of averaged metrics
        """
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        metric_keys = []
        
        # Collect all possible metric keys
        for k in self.default_k_values:
            metric_keys.extend([
                f'precision@{k}', f'recall@{k}', f'f1@{k}', 
                f'ndcg@{k}', f'map@{k}', f'mrr@{k}'
            ])
        metric_keys.append('mrr')  # Overall MRR
        
        # Calculate averages
        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def format_metrics_for_display(self, metrics: Dict, precision: int = 3) -> Dict:
        """
        Format metrics for display with specified precision
        
        Args:
            metrics: Dictionary of metrics
            precision: Number of decimal places
            
        Returns:
            Dictionary of formatted metrics
        """
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted[key] = round(float(value), precision)
            else:
                formatted[key] = value
        return formatted
    
    def get_metric_summary(self, metrics: Dict) -> Dict:
        """
        Get a summary of key metrics for quick overview
        
        Args:
            metrics: Dictionary of all calculated metrics
            
        Returns:
            Dictionary of key metrics
        """
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']
        
        summary = {}
        for metric in key_metrics:
            if metric in metrics:
                summary[metric] = metrics[metric]
        
        return summary
    
    def compare_metrics(self, before_metrics: Dict, after_metrics: Dict) -> Dict:
        """
        Compare two sets of metrics and calculate improvements
        
        Args:
            before_metrics: Metrics before intervention
            after_metrics: Metrics after intervention
            
        Returns:
            Dictionary of improvements and comparisons
        """
        if not before_metrics or not after_metrics:
            return {}
        
        comparison = {}
        
        for metric_name in before_metrics:
            if metric_name in after_metrics:
                before_val = before_metrics[metric_name]
                after_val = after_metrics[metric_name]
                
                # Calculate absolute and percentage improvements
                absolute_improvement = after_val - before_val
                percentage_improvement = ((after_val - before_val) / before_val * 100) if before_val > 0 else 0
                
                comparison[metric_name] = {
                    'before': before_val,
                    'after': after_val,
                    'absolute_improvement': absolute_improvement,
                    'percentage_improvement': percentage_improvement,
                    'improved': absolute_improvement > 0
                }
        
        return comparison

# Factory function for easy instantiation
def create_metrics_calculator() -> RetrievalMetricsCalculator:
    """Create and return a RetrievalMetricsCalculator instance"""
    return RetrievalMetricsCalculator()

# Convenience functions for direct use
def calculate_metrics(retrieved_docs: List[Dict], ground_truth_links: List[str], 
                     top_k_values: List[int] = None) -> Dict:
    """
    Convenience function to calculate retrieval metrics
    
    Args:
        retrieved_docs: List of retrieved documents
        ground_truth_links: List of ground truth links
        top_k_values: List of k values to evaluate
        
    Returns:
        Dictionary of calculated metrics
    """
    calculator = RetrievalMetricsCalculator()
    return calculator.calculate_retrieval_metrics(retrieved_docs, ground_truth_links, top_k_values)

def calculate_average_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Convenience function to calculate average metrics
    
    Args:
        metrics_list: List of individual metric dictionaries
        
    Returns:
        Dictionary of averaged metrics
    """
    calculator = RetrievalMetricsCalculator()
    return calculator.calculate_averages(metrics_list)

if __name__ == "__main__":
    # Test the metrics calculator
    print("🧪 Testing RetrievalMetricsCalculator...")
    
    # Sample data for testing
    retrieved_docs = [
        {'link': 'https://example.com/doc1'},
        {'link': 'https://example.com/doc2'},
        {'link': 'https://different.com/doc3'},
        {'link': 'https://example.com/doc4'},
        {'link': 'https://other.com/doc5'}
    ]
    
    ground_truth_links = [
        'https://example.com/doc1',
        'https://example.com/doc4',
        'https://example.com/doc6'  # This one is not retrieved
    ]
    
    # Calculate metrics
    metrics = calculate_metrics(retrieved_docs, ground_truth_links)
    
    # Display results
    calculator = RetrievalMetricsCalculator()
    formatted_metrics = calculator.format_metrics_for_display(metrics)
    summary = calculator.get_metric_summary(formatted_metrics)
    
    print("✅ Test completed!")
    print(f"📊 Key metrics: {summary}")
    print(f"🔍 All metrics calculated: {len(formatted_metrics)} metrics")