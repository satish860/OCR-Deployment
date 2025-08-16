"""
Analyze the poor-performing examples (<70% accuracy) from the filtered benchmark results.
"""
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_poor_examples(results_file: str):
    """
    Extract and analyze examples with <70% accuracy.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    
    # Find poor performing examples (<70% accuracy)
    poor_examples = [r for r in results if r['accuracy'] < 70.0]
    
    print("POOR PERFORMING EXAMPLES ANALYSIS (<70% accuracy)")
    print("=" * 60)
    print(f"Found {len(poor_examples)} poor examples out of {len(results)} total")
    print()
    
    for i, example in enumerate(poor_examples, 1):
        print(f"--- POOR EXAMPLE #{i} ---")
        print(f"Example ID: {example['example_id']}")
        print(f"Accuracy: {example['accuracy']:.1f}%")
        print(f"Fields Matched: {example['matched_fields']}/{example['total_fields']}")
        
        # Parse document type
        try:
            doc_type_str = example.get('document_type', '{}')
            doc_meta = json.loads(doc_type_str)
            doc_format = doc_meta.get('format', 'UNKNOWN')
            doc_quality = doc_meta.get('documentQuality', 'UNKNOWN')
            print(f"Document Type: {doc_format}")
            print(f"Document Quality: {doc_quality}")
        except:
            print(f"Document Type: PARSE ERROR")
        
        print(f"OCR Length: {example.get('ocr_length', 0)} characters")
        print(f"Extraction Time: {example.get('extraction_time', 0):.2f} seconds")
        
        # Show Modal OCR result preview
        modal_result = example.get('modal_result', {})
        ocr_text = modal_result.get('result', '')
        print(f"OCR Preview (first 300 chars):")
        print(f"'{ocr_text[:300]}...'")
        
        # Show what was expected vs extracted
        print(f"Processing Status: {example['success']}")
        if not example['success']:
            print(f"Error: {example.get('error', 'Unknown error')}")
        
        print()
        print("-" * 60)
        print()


def compare_good_vs_poor(results_file: str):
    """
    Compare characteristics of good vs poor examples.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    
    poor_examples = [r for r in results if r['accuracy'] < 70.0]
    good_examples = [r for r in results if r['accuracy'] >= 90.0]
    
    print("GOOD vs POOR EXAMPLES COMPARISON")
    print("=" * 50)
    
    def analyze_group(examples, group_name):
        print(f"\n{group_name} Examples ({len(examples)}):")
        
        # OCR length distribution
        ocr_lengths = [e.get('ocr_length', 0) for e in examples]
        avg_ocr_length = sum(ocr_lengths) / len(ocr_lengths) if ocr_lengths else 0
        
        # Field count distribution
        field_counts = [e.get('total_fields', 0) for e in examples]
        avg_field_count = sum(field_counts) / len(field_counts) if field_counts else 0
        
        # Extraction time
        extraction_times = [e.get('extraction_time', 0) for e in examples]
        avg_extraction_time = sum(extraction_times) / len(extraction_times) if extraction_times else 0
        
        # Document types
        doc_types = {}
        for example in examples:
            try:
                doc_type_str = example.get('document_type', '{}')
                doc_meta = json.loads(doc_type_str)
                doc_format = doc_meta.get('format', 'UNKNOWN')
                doc_types[doc_format] = doc_types.get(doc_format, 0) + 1
            except:
                doc_types['PARSE_ERROR'] = doc_types.get('PARSE_ERROR', 0) + 1
        
        print(f"  Average OCR Length: {avg_ocr_length:.0f} characters")
        print(f"  Average Field Count: {avg_field_count:.1f}")
        print(f"  Average Extraction Time: {avg_extraction_time:.2f} seconds")
        print(f"  Document Types: {dict(sorted(doc_types.items()))}")
    
    analyze_group(poor_examples, "POOR (<70%)")
    analyze_group(good_examples, "GOOD (≥90%)")


def find_specific_failure_patterns(results_file: str):
    """
    Look for specific patterns in failures.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    poor_examples = [r for r in results if r['accuracy'] < 70.0]
    
    print("SPECIFIC FAILURE PATTERNS")
    print("=" * 40)
    
    # Check for very short OCR results
    short_ocr = [e for e in poor_examples if e.get('ocr_length', 0) < 100]
    print(f"Very short OCR (<100 chars): {len(short_ocr)} examples")
    
    # Check for very long extraction times
    slow_extraction = [e for e in poor_examples if e.get('extraction_time', 0) > 10]
    print(f"Slow extraction (>10s): {len(slow_extraction)} examples")
    
    # Check for very few fields
    few_fields = [e for e in poor_examples if e.get('total_fields', 0) <= 3]
    print(f"Very few fields (≤3): {len(few_fields)} examples")
    
    # Check for zero matched fields
    zero_matches = [e for e in poor_examples if e.get('matched_fields', 0) == 0]
    print(f"Zero matched fields: {len(zero_matches)} examples")
    
    if zero_matches:
        print("\nZERO MATCH EXAMPLES:")
        for example in zero_matches:
            try:
                doc_type_str = example.get('document_type', '{}')
                doc_meta = json.loads(doc_type_str)
                doc_format = doc_meta.get('format', 'UNKNOWN')
            except:
                doc_format = 'PARSE_ERROR'
            print(f"  Example {example['example_id']}: {doc_format}, OCR: {example.get('ocr_length', 0)} chars")


if __name__ == "__main__":
    # Use the most recent filtered results
    results_file = "benchmark/results/batch_benchmark_20250816_151218.json"
    
    if Path(results_file).exists():
        analyze_poor_examples(results_file)
        compare_good_vs_poor(results_file)
        find_specific_failure_patterns(results_file)
    else:
        print(f"Results file not found: {results_file}")
        print("Please run the filtered benchmark first.")