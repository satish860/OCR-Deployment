"""
Debug chart processing with Example 001 from the benchmark results.
This will help us understand why charts aren't being detected and processed.
"""
import json
import sys
from pathlib import Path
import base64

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

from test_single_page import (
    is_likely_chart,
    extract_image_region,
    analyze_chart_with_llm,
    process_charts_in_ocr_result,
    load_openai_client
)

def debug_example_001():
    """Debug Example 001 - a CHART document with 0% accuracy"""
    
    print("DEBUGGING CHART PROCESSING - Example 001")
    print("=" * 60)
    
    # Load the benchmark results to get Example 001
    results_file = "benchmark/results/chart_benchmark_20250816_222015_with_charts.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Find Example 001
    example_001 = None
    for example in results['detailed_results']:
        if example['example_id'] == '001':
            example_001 = example
            break
    
    if not example_001:
        print("Example 001 not found in results")
        return
    
    print(f"Example 001 Details:")
    print(f"   Format: {json.loads(example_001['document_type'])['format']}")
    print(f"   Quality: {json.loads(example_001['document_type'])['documentQuality']}")
    print(f"   Accuracy: {example_001['accuracy']}%")
    print(f"   Total fields: {example_001['total_fields']}")
    print(f"   Matched fields: {example_001['matched_fields']}")
    print()
    
    # Get the OCR result
    ocr_result = example_001['modal_result']['result']
    print(f"OCR Result length: {len(ocr_result)} characters")
    print()
    
    # Parse the OCR JSON to examine Picture elements
    try:
        # Clean the OCR result - remove leading/trailing whitespace and find JSON array
        cleaned_ocr = ocr_result.strip()
        if not cleaned_ocr.startswith('['):
            # Find the start of the JSON array
            json_start = cleaned_ocr.find('[')
            if json_start != -1:
                cleaned_ocr = cleaned_ocr[json_start:]
            else:
                raise ValueError("No JSON array found in OCR result")
        
        ocr_data = json.loads(cleaned_ocr)
        print(f"OCR Data Analysis:")
        print(f"   Total elements: {len(ocr_data)}")
        
        picture_elements = []
        for i, element in enumerate(ocr_data):
            category = element.get('category', 'Unknown')
            bbox = element.get('bbox', [])
            text = element.get('text', 'N/A')
            
            print(f"   Element {i}: {category}")
            if bbox:
                width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
                height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
                print(f"     BBox: {bbox} (Size: {width}x{height})")
            
            if category == 'Picture':
                picture_elements.append((i, element))
                print(f"     PICTURE ELEMENT FOUND!")
            
            if text != 'N/A':
                print(f"     Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print()
        
        print(f"Picture Elements Found: {len(picture_elements)}")
        print()
        
        # Test chart detection logic on each Picture element
        for i, element in picture_elements:
            print(f"Testing Chart Detection on Picture Element {i}:")
            bbox = element.get('bbox', [])
            
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = max(width / height, height / width) if height > 0 else float('inf')
                
                print(f"   BBox: {bbox}")
                print(f"   Size: {width}x{height}")
                print(f"   Aspect ratio: {aspect_ratio:.2f}")
                print(f"   Category: {element.get('category')}")
                
                # Test our detection logic step by step
                print("   Detection Tests:")
                print(f"     Category == 'Picture': {element.get('category') == 'Picture'}")
                print(f"     Width >= 100: {width >= 100} ({width})")
                print(f"     Height >= 100: {height >= 100} ({height})")
                print(f"     Aspect ratio <= 5: {aspect_ratio <= 5} ({aspect_ratio:.2f})")
                
                is_chart = is_likely_chart(element)
                print(f"   IS LIKELY CHART: {is_chart}")
                
                if is_chart:
                    print("   This should be processed as a chart!")
                else:
                    print("   This was NOT detected as a chart")
            else:
                print(f"   Invalid bbox: {bbox}")
            print()
        
    except json.JSONDecodeError as e:
        print(f"Error parsing OCR JSON: {e}")
        print("First 500 chars of OCR result:")
        print(ocr_result[:500])
        return
    
    # Test the full chart processing pipeline
    print("Testing Full Chart Processing Pipeline:")
    print()
    
    # We need the base64 image to test extraction
    print("Note: We can't test image extraction without the original base64 image")
    print("   But we can test the detection and processing logic structure")
    
    # Test if charts would be processed
    openai_client = load_openai_client()
    if openai_client:
        print("OpenAI client available for chart analysis")
    else:
        print("OpenAI client not available")
    
    # Simulate processing
    simulated_charts_processed = 0
    for i, element in picture_elements:
        if is_likely_chart(element):
            simulated_charts_processed += 1
            print(f"   Would process Picture Element {i} as chart")
    
    print(f"\nSummary:")
    print(f"   Picture elements found: {len(picture_elements)}")
    print(f"   Charts that would be processed: {simulated_charts_processed}")
    print(f"   Expected improvement: {'YES' if simulated_charts_processed > 0 else 'NO'}")
    
    if simulated_charts_processed == 0:
        print("\nDEBUG CONCLUSION:")
        print("   No charts were detected for processing!")
        print("   This explains why chart processing didn't improve accuracy.")
        print("\nNEXT STEPS:")
        print("   1. Review chart detection criteria")
        print("   2. Check if Picture elements are correctly formatted")
        print("   3. Verify chart processing pipeline integration")

if __name__ == "__main__":
    debug_example_001()