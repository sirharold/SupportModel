"""
Advanced metrics for RAG system evaluation.
Includes metrics specific to retrieval quality and semantic coherence.
"""

import numpy as np
from typing import List, Dict, Set
import re
from collections import Counter

def calculate_retrieval_metrics(results: List[Dict], ground_truth_links: Set[str]) -> Dict:
    """
    Calculate advanced retrieval metrics.
    
    Args:
        results: List of retrieved documents
        ground_truth_links: Set of expected/relevant document links
    
    Returns:
        Dictionary with advanced metrics
    """
    if not results:
        return {
            "precision_at_k": 0,
            "recall_at_k": 0,
            "f1_at_k": 0,
            "mrr": 0,
            "ndcg_at_k": 0,
            "diversity_score": 0,
            "coverage_score": 0
        }
    
    retrieved_links = set(doc.get('link', '') for doc in results if doc.get('link'))
    
    # Precision@K and Recall@K
    if ground_truth_links:
        relevant_retrieved = retrieved_links.intersection(ground_truth_links)
        precision_at_k = len(relevant_retrieved) / len(retrieved_links) if retrieved_links else 0
        recall_at_k = len(relevant_retrieved) / len(ground_truth_links) if ground_truth_links else 0
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0
    else:
        precision_at_k = recall_at_k = f1_at_k = 0
    
    # Mean Reciprocal Rank (MRR)
    mrr = 0
    if ground_truth_links:
        for i, doc in enumerate(results):
            if doc.get('link', '') in ground_truth_links:
                mrr = 1 / (i + 1)
                break
    
    # Simplified NDCG@K (based on scores and relevance)
    dcg = 0
    idcg = 0
    
    if ground_truth_links and results:
        # DCG calculation
        for i, doc in enumerate(results):
            relevance = 1 if doc.get('link', '') in ground_truth_links else 0
            dcg += relevance / np.log2(i + 2)
        
        # IDCG calculation (ideal case where all relevant docs are at top)
        relevant_count = min(len(ground_truth_links), len(results))
        for i in range(relevant_count):
            idcg += 1 / np.log2(i + 2)
        
        ndcg_at_k = dcg / idcg if idcg > 0 else 0
    else:
        ndcg_at_k = 0
    
    # Diversity Score (based on score variance)
    scores = [doc.get('score', 0) for doc in results]
    diversity_score = 1 - (np.std(scores) / np.mean(scores)) if scores and np.mean(scores) > 0 else 0
    diversity_score = max(0, min(1, diversity_score))  # Clamp between 0 and 1
    
    # Coverage Score (unique services/technologies mentioned)
    coverage_score = calculate_coverage_score(results)
    
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "f1_at_k": f1_at_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg_at_k,
        "diversity_score": diversity_score,
        "coverage_score": coverage_score
    }

def calculate_coverage_score(results: List[Dict]) -> float:
    """
    Calculate how well the results cover different Azure services/topics.
    
    Args:
        results: List of retrieved documents
    
    Returns:
        Coverage score between 0 and 1
    """
    if not results:
        return 0
    
    # Azure services and technologies to look for
    azure_services = {
        'storage', 'blob', 'sql', 'cosmos', 'functions', 'app service',
        'virtual machine', 'vm', 'kubernetes', 'aks', 'container', 'devops',
        'active directory', 'azure ad', 'logic apps', 'service bus',
        'event hub', 'cognitive services', 'machine learning', 'databricks'
    }
    
    technologies = {
        'python', 'javascript', 'c#', 'java', 'docker', 'api', 'rest',
        'json', 'authentication', 'oauth', 'powershell', 'cli', 'terraform'
    }
    
    all_keywords = azure_services.union(technologies)
    
    found_keywords = set()
    
    for doc in results:
        title = doc.get('title', '').lower()
        content = doc.get('content', '').lower()
        text = f"{title} {content}"
        
        for keyword in all_keywords:
            if keyword in text:
                found_keywords.add(keyword)
    
    # Coverage is the ratio of found keywords to total possible keywords
    coverage_score = len(found_keywords) / len(all_keywords) if all_keywords else 0
    return min(1.0, coverage_score * 3)  # Scale up since finding all is unlikely

def calculate_semantic_coherence(results: List[Dict]) -> float:
    """
    Calculate semantic coherence of retrieved documents.
    
    Args:
        results: List of retrieved documents
    
    Returns:
        Coherence score between 0 and 1
    """
    if len(results) < 2:
        return 1.0  # Single or no document is perfectly coherent
    
    # Simple word overlap-based coherence
    all_texts = []
    for doc in results:
        title = doc.get('title', '')
        content = doc.get('content', '')[:500]  # First 500 chars
        text = f"{title} {content}".lower()
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text)
        all_texts.append(set(words))
    
    # Calculate pairwise overlaps
    overlaps = []
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            if all_texts[i] and all_texts[j]:
                overlap = len(all_texts[i].intersection(all_texts[j]))
                union = len(all_texts[i].union(all_texts[j]))
                jaccard = overlap / union if union > 0 else 0
                overlaps.append(jaccard)
    
    return np.mean(overlaps) if overlaps else 0

def calculate_freshness_score(results: List[Dict]) -> float:
    """
    Calculate how "fresh" or up-to-date the retrieved documents appear.
    
    Args:
        results: List of retrieved documents
    
    Returns:
        Freshness score between 0 and 1
    """
    if not results:
        return 0
    
    # Look for indicators of freshness in titles and content
    fresh_indicators = {
        '2024', '2023', 'latest', 'new', 'updated', 'current', 'preview',
        'general availability', 'ga', 'public preview', 'v2', 'version 2'
    }
    
    stale_indicators = {
        '2020', '2019', '2018', 'deprecated', 'legacy', 'old', 'classic',
        'v1', 'version 1'
    }
    
    fresh_count = 0
    stale_count = 0
    
    for doc in results:
        title = doc.get('title', '').lower()
        content = doc.get('content', '')[:200].lower()
        text = f"{title} {content}"
        
        for indicator in fresh_indicators:
            if indicator in text:
                fresh_count += 1
                break
        
        for indicator in stale_indicators:
            if indicator in text:
                stale_count += 1
                break
    
    total_scored = fresh_count + stale_count
    if total_scored == 0:
        return 0.5  # Neutral score if no indicators found
    
    return fresh_count / total_scored

def calculate_query_document_alignment(query: str, results: List[Dict]) -> float:
    """
    Calculate how well the retrieved documents align with the original query.
    
    Args:
        query: Original user query
        results: List of retrieved documents
    
    Returns:
        Alignment score between 0 and 1
    """
    if not results or not query:
        return 0
    
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    query_words = {word for word in query_words if len(word) > 2}  # Filter short words
    
    if not query_words:
        return 0
    
    alignment_scores = []
    
    for doc in results:
        title = doc.get('title', '').lower()
        content = doc.get('content', '')[:300].lower()  # First 300 chars
        doc_text = f"{title} {content}"
        doc_words = set(re.findall(r'\b\w+\b', doc_text))
        
        if doc_words:
            overlap = len(query_words.intersection(doc_words))
            alignment = overlap / len(query_words)
            alignment_scores.append(alignment)
    
    return np.mean(alignment_scores) if alignment_scores else 0