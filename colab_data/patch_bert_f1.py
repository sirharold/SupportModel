#!/usr/bin/env python3
"""
Script to patch missing bert_f1 values in cumulative results.

If bert_f1 is None but bert_precision and bert_recall exist,
calculates bert_f1 using the harmonic mean formula:
  bert_f1 = 2 * (P * R) / (P + R)

If bert_f1 already has a value, it is preserved.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

def calculate_bert_f1(precision, recall):
    """Calculate BERTScore F1 from precision and recall using harmonic mean."""
    if precision is None or recall is None:
        return None

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def patch_results_file(file_path):
    """Patch the cumulative results file with calculated bert_f1 values."""

    file_path = Path(file_path)

    # Create backup
    backup_path = file_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    print(f"📦 Creating backup: {backup_path.name}")
    shutil.copy2(file_path, backup_path)

    # Load data
    print(f"📂 Loading: {file_path.name}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Statistics
    stats = {
        'total_processed': 0,
        'bert_f1_calculated': 0,
        'bert_f1_preserved': 0,
        'bert_f1_missing_data': 0,
        'by_model': {}
    }

    # Process each model
    for model in ['ada', 'e5-large', 'mpnet', 'minilm']:
        print(f"\n🔧 Processing model: {model.upper()}")

        model_stats = {
            'total': 0,
            'calculated': 0,
            'preserved': 0,
            'missing_data': 0
        }

        individual_metrics = data['results'][model]['individual_rag_metrics']

        for i, metric in enumerate(individual_metrics):
            model_stats['total'] += 1

            bert_f1 = metric.get('bert_f1')
            bert_precision = metric.get('bert_precision')
            bert_recall = metric.get('bert_recall')

            if bert_f1 is not None:
                # Value already exists, preserve it
                model_stats['preserved'] += 1
            elif bert_precision is not None and bert_recall is not None:
                # Calculate F1
                calculated_f1 = calculate_bert_f1(bert_precision, bert_recall)
                metric['bert_f1'] = calculated_f1
                model_stats['calculated'] += 1
            else:
                # Cannot calculate (missing precision or recall)
                model_stats['missing_data'] += 1

        # Recalculate average bert_f1 for this model
        all_f1_values = [
            m['bert_f1'] for m in individual_metrics
            if m.get('bert_f1') is not None
        ]

        if all_f1_values:
            new_avg_f1 = sum(all_f1_values) / len(all_f1_values)
            old_avg_f1 = data['results'][model]['rag_metrics'].get('avg_bert_f1')

            data['results'][model]['rag_metrics']['avg_bert_f1'] = new_avg_f1

            print(f"  ✅ Calculated: {model_stats['calculated']}")
            print(f"  ✓  Preserved: {model_stats['preserved']}")
            print(f"  ⚠️  Missing data: {model_stats['missing_data']}")
            print(f"  📊 Average bert_f1: {old_avg_f1:.6f} → {new_avg_f1:.6f}")
        else:
            print(f"  ⚠️  No valid bert_f1 values found for {model}")

        stats['by_model'][model] = model_stats
        stats['total_processed'] += model_stats['total']
        stats['bert_f1_calculated'] += model_stats['calculated']
        stats['bert_f1_preserved'] += model_stats['preserved']
        stats['bert_f1_missing_data'] += model_stats['missing_data']

    # Save patched file (without indentation for faster write)
    print(f"\n💾 Saving patched file: {file_path.name}")
    with open(file_path, 'w') as f:
        json.dump(data, f)

    # Print summary
    print("\n" + "="*60)
    print("📊 PATCH SUMMARY")
    print("="*60)
    print(f"Total entries processed: {stats['total_processed']}")
    print(f"bert_f1 calculated: {stats['bert_f1_calculated']}")
    print(f"bert_f1 preserved: {stats['bert_f1_preserved']}")
    print(f"bert_f1 missing data: {stats['bert_f1_missing_data']}")
    print("="*60)

    print(f"\n✅ Patch completed successfully!")
    print(f"📦 Backup saved at: {backup_path}")
    print(f"📄 Patched file: {file_path}")

    return stats

if __name__ == "__main__":
    # Path to the cumulative results file
    results_file = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/data/cumulative_results_20251010_131215.json"

    print("="*60)
    print("🔧 BERT F1 PATCH UTILITY")
    print("="*60)
    print("This script will:")
    print("  1. Create a backup of the original file")
    print("  2. Calculate missing bert_f1 values from bert_precision and bert_recall")
    print("  3. Preserve existing bert_f1 values")
    print("  4. Recalculate average bert_f1 for each model")
    print("="*60)

    patch_results_file(results_file)
