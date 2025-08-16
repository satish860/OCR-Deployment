"""
Analyze failures in the batch benchmark results to identify patterns.
"""
import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_batch_results(results_file: str):
    """
    Analyze the batch benchmark results to identify failure patterns.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    
    print("BATCH BENCHMARK FAILURE ANALYSIS")
    print("=" * 50)
    
    # Overall stats
    total = len(results)
    perfect = sum(1 for r in results if r['accuracy'] == 100.0)
    good = sum(1 for r in results if 80.0 <= r['accuracy'] < 100.0)
    poor = sum(1 for r in results if r['accuracy'] < 80.0)
    
    print(f"Total Examples: {total}")
    print(f"Perfect (100%): {perfect} ({perfect/total*100:.1f}%)")
    print(f"Good (80-99%): {good} ({good/total*100:.1f}%)")
    print(f"Poor (<80%): {poor} ({poor/total*100:.1f}%)")
    
    # Document type analysis
    print(f"\nDOCUMENT TYPE ANALYSIS:")
    print("-" * 30)
    
    doc_types = {}
    for result in results:
        doc_type_str = result.get('document_type', '{}')
        try:
            doc_meta = json.loads(doc_type_str)
            doc_format = doc_meta.get('format', 'UNKNOWN')
        except:
            doc_format = 'UNKNOWN'
        
        if doc_format not in doc_types:
            doc_types[doc_format] = {'total': 0, 'perfect': 0, 'avg_accuracy': []}
        
        doc_types[doc_format]['total'] += 1
        doc_types[doc_format]['avg_accuracy'].append(result['accuracy'])
        if result['accuracy'] == 100.0:
            doc_types[doc_format]['perfect'] += 1
    
    for doc_type, stats in doc_types.items():
        avg_acc = sum(stats['avg_accuracy']) / len(stats['avg_accuracy'])
        perfect_rate = stats['perfect'] / stats['total'] * 100
        print(f"{doc_type}: {stats['total']} examples, {avg_acc:.1f}% avg accuracy, {perfect_rate:.1f}% perfect")
    
    # Look at specific failures
    print(f"\nFAILURE PATTERN ANALYSIS:")
    print("-" * 30)
    
    # Check poor performing examples
    poor_examples = [r for r in results if r['accuracy'] < 50.0]
    print(f"\nVery Poor Examples (<50% accuracy): {len(poor_examples)}")
    
    for i, example in enumerate(poor_examples[:5]):  # Show first 5
        doc_type_str = example.get('document_type', '{}')
        try:
            doc_meta = json.loads(doc_type_str)
            doc_format = doc_meta.get('format', 'UNKNOWN')
        except:
            doc_format = 'UNKNOWN'
        
        ocr_length = example.get('ocr_length', 0)
        fields = example.get('total_fields', 0)
        matched = example.get('matched_fields', 0)
        
        print(f"  Example {example['example_id']}: {doc_format}, {example['accuracy']:.1f}% accuracy")
        print(f"    Fields: {matched}/{fields}, OCR length: {ocr_length}")
        
        # Show a snippet of the OCR result
        modal_result = example.get('modal_result', {})
        ocr_text = modal_result.get('result', '')[:200]
        print(f"    OCR preview: {ocr_text}...")
        print()
    
    # Check if there are patterns in OCR quality
    print(f"\nOCR LENGTH vs ACCURACY ANALYSIS:")
    print("-" * 30)
    
    # Group by OCR length ranges
    length_ranges = {
        'Very Short (0-500)': [],
        'Short (500-1000)': [],
        'Medium (1000-2000)': [],
        'Long (2000-5000)': [],
        'Very Long (5000+)': []
    }
    
    for result in results:
        ocr_len = result.get('ocr_length', 0)
        accuracy = result['accuracy']
        
        if ocr_len <= 500:
            length_ranges['Very Short (0-500)'].append(accuracy)
        elif ocr_len <= 1000:
            length_ranges['Short (500-1000)'].append(accuracy)
        elif ocr_len <= 2000:
            length_ranges['Medium (1000-2000)'].append(accuracy)
        elif ocr_len <= 5000:
            length_ranges['Long (2000-5000)'].append(accuracy)
        else:
            length_ranges['Very Long (5000+)'].append(accuracy)
    
    for range_name, accuracies in length_ranges.items():
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            perfect_count = sum(1 for acc in accuracies if acc == 100.0)
            perfect_rate = perfect_count / len(accuracies) * 100
            print(f"{range_name}: {len(accuracies)} examples, {avg_acc:.1f}% avg accuracy, {perfect_rate:.1f}% perfect")
    
    # Check specific problematic examples
    print(f"\nSPECIFIC ISSUE EXAMPLES:")
    print("-" * 30)
    
    # Find examples with very few fields
    few_fields = [r for r in results if r.get('total_fields', 0) <= 5 and r['accuracy'] < 80]
    print(f"\nLow field count + low accuracy: {len(few_fields)} examples")
    for example in few_fields[:3]:
        print(f"  Example {example['example_id']}: {example['total_fields']} fields, {example['accuracy']:.1f}% accuracy")
    
    # Find examples where extraction took very long
    slow_extractions = [r for r in results if r.get('extraction_time', 0) > 10]
    print(f"\nSlow extractions (>10s): {len(slow_extractions)} examples")
    for example in slow_extractions[:3]:
        print(f"  Example {example['example_id']}: {example.get('extraction_time', 0):.1f}s, {example['accuracy']:.1f}% accuracy")


def compare_single_vs_batch():
    """
    Compare our known good single example with batch results.
    """
    print(f"\nSINGLE vs BATCH COMPARISON:")
    print("-" * 30)
    
    # We know example 000 should be perfect
    print("Example 000 (medal rankings table):")
    print("  Single test: 100% accuracy (49/49 fields)")
    print("  Batch test: Check results...")


if __name__ == "__main__":
    # Use the most recent batch results
    results_file = "benchmark/results/batch_benchmark_20250816_141941.json"
    
    if Path(results_file).exists():
        analyze_batch_results(results_file)
        compare_single_vs_batch()
    else:
        print(f"Results file not found: {results_file}")
        print("Please run the batch benchmark first.")