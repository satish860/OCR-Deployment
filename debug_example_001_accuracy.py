"""
Debug Example 001 accuracy - analyze chart processing output vs ground truth
to understand why accuracy is only 54% instead of higher.
"""
import json
import sys
from pathlib import Path

def debug_example_001_accuracy():
    """Debug why Example 001 only achieved 54% accuracy after chart processing"""
    
    print("DEBUGGING EXAMPLE 001 ACCURACY")
    print("=" * 60)
    
    # Load the results with chart processing
    results_file = "benchmark/results/chart_benchmark_20250816_223453_with_charts.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Find Example 001
    example_001 = None
    for example in results['detailed_results']:
        if example['example_id'] == '001':
            example_001 = example
            break
    
    if not example_001:
        print("Example 001 not found")
        return
    
    print("EXAMPLE 001 DETAILS:")
    print(f"  Format: {json.loads(example_001['document_type'])['format']}")
    print(f"  Accuracy: {example_001['accuracy']:.1f}%")
    print(f"  Total fields: {example_001['total_fields']}")
    print(f"  Matched fields: {example_001['matched_fields']}")
    print(f"  Charts processed: {example_001['charts_processed']}")
    print(f"  Chart processing time: {example_001['chart_processing_time']:.1f}s")
    print()
    
    # Get the processed OCR result (with chart descriptions)
    processed_ocr = example_001['modal_result']['result']
    print("PROCESSED OCR RESULT:")
    print(f"Length: {len(processed_ocr)} characters")
    print()
    
    # Parse and analyze the OCR data
    try:
        # Clean the OCR result
        cleaned_ocr = processed_ocr.strip()
        if not cleaned_ocr.startswith('['):
            json_start = cleaned_ocr.find('[')
            if json_start != -1:
                cleaned_ocr = cleaned_ocr[json_start:]
        
        ocr_data = json.loads(cleaned_ocr)
        
        print("OCR ELEMENTS ANALYSIS:")
        for i, element in enumerate(ocr_data):
            category = element.get('category', 'Unknown')
            bbox = element.get('bbox', [])
            text = element.get('text', 'N/A')
            
            print(f"Element {i}: {category}")
            if bbox and len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                print(f"  BBox: {bbox} (Size: {width}x{height})")
            
            if category == 'Picture':
                print("  PICTURE ELEMENT - Should have chart description:")
                if 'chart_analysis' in element:
                    chart_info = element['chart_analysis']
                    print(f"    Chart processed: {chart_info.get('processed', False)}")
                    if 'tokens_used' in chart_info:
                        print(f"    Tokens used: {chart_info['tokens_used']}")
                    if 'error' in chart_info:
                        print(f"    Error: {chart_info['error']}")
                
                if text != 'N/A' and len(text) > 0:
                    print(f"    Chart description length: {len(text)} chars")
                    print(f"    First 200 chars: {text[:200]}...")
                else:
                    print("    NO CHART DESCRIPTION FOUND!")
            else:
                if text != 'N/A' and len(text) > 100:
                    print(f"  Text (first 100 chars): {text[:100]}...")
                elif text != 'N/A':
                    print(f"  Text: {text}")
            print()
    
    except Exception as e:
        print(f"Error parsing OCR data: {e}")
        return
    
    print("\nNOW LET'S LOAD THE GROUND TRUTH TO SEE WHAT WAS EXPECTED:")
    print("=" * 60)
    
    # Load the original 50-example results to get ground truth for Example 001
    original_results_file = "benchmark/results/chart_benchmark_20250816_222015_with_charts.json"
    
    try:
        with open(original_results_file, 'r', encoding='utf-8') as f:
            original_results = json.load(f)
        
        # Find the original Example 001 with ground truth
        original_001 = None
        for example in original_results['detailed_results']:
            if example['example_id'] == '001':
                original_001 = example
                break
        
        if original_001:
            print("GROUND TRUTH ANALYSIS:")
            print(f"Expected format: {json.loads(original_001['document_type'])['format']}")
            print(f"Total fields expected: {original_001['total_fields']}")
            print(f"Fields matched originally: {original_001['matched_fields']}")
            print()
            
            # Try to understand what the extraction was trying to match
            print("This suggests the ground truth expects extraction of:")
            print(f"  {original_001['total_fields']} specific data fields")
            print(f"  Original OCR got {original_001['matched_fields']} correct")
            print(f"  Chart processing got {example_001['matched_fields']} correct")
            print()
            
            improvement = example_001['matched_fields'] - original_001['matched_fields']
            print(f"IMPROVEMENT: {improvement} additional fields matched")
            
    except Exception as e:
        print(f"Could not load ground truth: {e}")
    
    print("\nDIAGNOSIS AND RECOMMENDATIONS:")
    print("=" * 60)
    print("To improve chart processing accuracy:")
    print("1. Examine what specific data fields the ground truth expects")
    print("2. Check if chart descriptions contain the expected information")
    print("3. Consider if chart prompt needs to be more specific")
    print("4. Verify if LLM extraction prompt needs adjustment for chart data")

if __name__ == "__main__":
    debug_example_001_accuracy()