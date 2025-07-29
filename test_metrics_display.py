#!/usr/bin/env python3
"""Test script to debug metrics display issue"""

import json
import sys
sys.path.append('.')

from src.apps.cumulative_metrics_results import display_results_visualizations

# Load the results file
with open('/Users/haroldgomez/Downloads/cumulative_results_1753578255.json', 'r') as f:
    results_data = json.load(f)

# Extract the processed results (same as Streamlit page does)
processed_results = results_data['results']

print("📊 DEBUGGING METRICS DISPLAY")
print("=" * 50)

# Check what's being passed to display functions
for model_name, model_data in processed_results.items():
    print(f"\n📊 Model: {model_name}")
    
    # Check if avg_before_metrics and avg_after_metrics exist
    if 'avg_before_metrics' in model_data:
        before = model_data['avg_before_metrics']
        print(f"  ✅ avg_before_metrics found with {len(before)} keys")
        
        # Check specific metrics
        test_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'ndcg@5', 'mrr']
        print("  📈 Key metrics:")
        for metric in test_metrics:
            value = before.get(metric, 'NOT FOUND')
            print(f"    {metric}: {value}")
    else:
        print("  ❌ avg_before_metrics NOT FOUND")
    
    if 'avg_after_metrics' in model_data:
        after = model_data['avg_after_metrics']
        print(f"  ✅ avg_after_metrics found with {len(after)} keys")
    else:
        print("  ❌ avg_after_metrics NOT FOUND")

# Test the adaptation logic from cumulative_metrics_results.py
print("\n🔍 TESTING ADAPTATION LOGIC:")
model_results = processed_results['ada']

if 'avg_before_metrics' in model_results and 'avg_after_metrics' in model_results:
    adapted_results = {
        'num_questions_evaluated': model_results.get('num_questions_evaluated', results_data['config']['num_questions']),
        'avg_before_metrics': model_results['avg_before_metrics'],
        'avg_after_metrics': model_results['avg_after_metrics'],
        'individual_before_metrics': model_results.get('all_before_metrics', []),
        'individual_after_metrics': model_results.get('all_after_metrics', []),
        'rag_metrics': model_results.get('rag_metrics', {}),
        'individual_rag_metrics': model_results.get('individual_rag_metrics', [])
    }
    
    print("✅ Adapted results structure created")
    
    # Check what's in the adapted avg_before_metrics
    print("\n📊 Adapted avg_before_metrics sample:")
    for i, (key, value) in enumerate(adapted_results['avg_before_metrics'].items()):
        if i < 10:  # Show first 10
            print(f"  {key}: {value}")
    
    # Specific check for the metrics you mentioned
    print("\n🎯 Specific metrics check in adapted structure:")
    test_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5', 'ndcg@5', 'mrr']
    for metric in test_metrics:
        before_val = adapted_results['avg_before_metrics'].get(metric, 'NOT FOUND')
        after_val = adapted_results['avg_after_metrics'].get(metric, 'NOT FOUND')
        print(f"  {metric}: before={before_val}, after={after_val}")

print("\n✅ Debug complete")