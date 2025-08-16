"""
Compare results between standard prompting and smart prompting benchmarks.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_latest_results(pattern: str = "batch_benchmark_"):
    """Load the most recent benchmark results."""
    results_dir = Path("benchmark/results")
    files = list(results_dir.glob(f"{pattern}*.json"))
    
    if not files:
        return None
        
    # Sort by modification time
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded results from: {latest_file.name}")
    return data


def compare_prompting_methods():
    """Compare standard vs smart prompting results."""
    print("PROMPTING METHODS COMPARISON")
    print("=" * 50)
    
    # Load the two most recent results
    results_dir = Path("benchmark/results")
    files = list(results_dir.glob("batch_benchmark_*.json"))
    
    if len(files) < 2:
        print("Need at least 2 benchmark result files to compare")
        return
    
    # Sort by modification time
    sorted_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Load the two most recent
    latest_file = sorted_files[0]
    previous_file = sorted_files[1]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        latest_data = json.load(f)
    
    with open(previous_file, 'r', encoding='utf-8') as f:
        previous_data = json.load(f)
    
    print(f"Latest results: {latest_file.name}")
    print(f"Previous results: {previous_file.name}")
    
    # Determine which is smart vs standard
    latest_smart = latest_data.get("configuration", {}).get("smart_prompting", False)
    previous_smart = previous_data.get("configuration", {}).get("smart_prompting", False)
    
    if latest_smart and not previous_smart:
        smart_data = latest_data
        standard_data = previous_data
        print("\nLatest = Smart Prompting, Previous = Standard Prompting")
    elif previous_smart and not latest_smart:
        smart_data = previous_data
        standard_data = latest_data
        print("\nPrevious = Smart Prompting, Latest = Standard Prompting")
    else:
        print(f"\nBoth files have same prompting type (smart: {latest_smart})")
        return
    
    # Compare overall metrics
    print(f"\n--- OVERALL COMPARISON ---")
    
    standard_summary = standard_data["summary"]
    smart_summary = smart_data["summary"]
    
    print(f"Standard Prompting:")
    print(f"  Total Examples: {standard_summary['total_examples']}")
    print(f"  Successful Tests: {standard_summary['successful_tests']}")
    print(f"  Overall Accuracy: {standard_summary['overall_accuracy']:.1f}%")
    print(f"  Perfect Extractions: {standard_summary['perfect_extractions']}")
    
    print(f"\nSmart Prompting:")
    print(f"  Total Examples: {smart_summary['total_examples']}")
    print(f"  Successful Tests: {smart_summary['successful_tests']}")
    print(f"  Overall Accuracy: {smart_summary['overall_accuracy']:.1f}%")
    print(f"  Perfect Extractions: {smart_summary['perfect_extractions']}")
    
    # Calculate improvements
    accuracy_improvement = smart_summary['overall_accuracy'] - standard_summary['overall_accuracy']
    perfect_improvement = smart_summary['perfect_extractions'] - standard_summary['perfect_extractions']
    
    print(f"\n--- IMPROVEMENTS ---")
    print(f"Accuracy improvement: {accuracy_improvement:+.1f} percentage points")
    print(f"Perfect extractions improvement: {perfect_improvement:+d}")
    
    # Analyze by document type
    print(f"\n--- DOCUMENT TYPE ANALYSIS ---")
    
    def analyze_by_doc_type(data, label):
        doc_type_stats = defaultdict(lambda: {'count': 0, 'total_accuracy': 0, 'perfect_count': 0})
        
        for result in data["detailed_results"]:
            try:
                doc_type_str = result.get('document_type', '{}')
                doc_meta = json.loads(doc_type_str)
                doc_format = doc_meta.get('format', 'UNKNOWN')
                
                stats = doc_type_stats[doc_format]
                stats['count'] += 1
                stats['total_accuracy'] += result.get('accuracy', 0)
                if result.get('perfect_extraction', False):
                    stats['perfect_count'] += 1
                    
            except Exception as e:
                continue
        
        print(f"\n{label}:")
        for doc_type, stats in sorted(doc_type_stats.items()):
            avg_acc = stats['total_accuracy'] / stats['count'] if stats['count'] > 0 else 0
            perfect_rate = stats['perfect_count'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f"  {doc_type}: {stats['count']} examples, {avg_acc:.1f}% avg accuracy, {perfect_rate:.1f}% perfect")
        
        return doc_type_stats
    
    standard_doc_stats = analyze_by_doc_type(standard_data, "Standard Prompting")
    smart_doc_stats = analyze_by_doc_type(smart_data, "Smart Prompting")
    
    # Compare document type improvements
    print(f"\n--- DOCUMENT TYPE IMPROVEMENTS ---")
    for doc_type in set(standard_doc_stats.keys()) | set(smart_doc_stats.keys()):
        if doc_type in standard_doc_stats and doc_type in smart_doc_stats:
            standard_avg = standard_doc_stats[doc_type]['total_accuracy'] / standard_doc_stats[doc_type]['count']
            smart_avg = smart_doc_stats[doc_type]['total_accuracy'] / smart_doc_stats[doc_type]['count']
            improvement = smart_avg - standard_avg
            print(f"  {doc_type}: {standard_avg:.1f}% → {smart_avg:.1f}% ({improvement:+.1f}%)")
    
    # Performance comparison
    print(f"\n--- PERFORMANCE COMPARISON ---")
    
    standard_timing = standard_data["timing"]
    smart_timing = smart_data["timing"]
    
    print(f"Standard Modal Time: {standard_timing['modal_batch_time']:.1f}s")
    print(f"Smart Modal Time: {smart_timing['modal_batch_time']:.1f}s")
    print(f"Time difference: {smart_timing['modal_batch_time'] - standard_timing['modal_batch_time']:+.1f}s")
    
    return {
        "accuracy_improvement": accuracy_improvement,
        "perfect_improvement": perfect_improvement,
        "standard_data": standard_data,
        "smart_data": smart_data
    }


if __name__ == "__main__":
    compare_prompting_methods()