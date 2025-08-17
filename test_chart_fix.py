"""
Test the chart processing fix with Example 001 data
"""
import json
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

from test_single_page import (
    process_charts_in_ocr_result,
    load_openai_client
)

def test_chart_processing_fix():
    """Test if the chart processing fix works with Example 001 OCR data"""
    
    print("Testing Chart Processing Fix")
    print("=" * 40)
    
    # Load Example 001 OCR result
    results_file = "benchmark/results/chart_benchmark_20250816_222015_with_charts.json"
    
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
    
    # Get OCR result
    ocr_result = example_001['modal_result']['result']
    print(f"Original OCR result length: {len(ocr_result)}")
    print(f"First 100 chars: {repr(ocr_result[:100])}")
    print()
    
    # Load OpenAI client
    openai_client = load_openai_client()
    if not openai_client:
        print("OpenAI client not available - using mock processing")
        openai_client = "mock"  # For testing parsing logic
    
    # Test the parsing logic first
    print("Testing JSON parsing...")
    
    try:
        # Test our parsing logic
        cleaned_ocr = ocr_result.strip()
        print(f"After strip: {repr(cleaned_ocr[:50])}")
        
        if not cleaned_ocr.startswith('['):
            json_start = cleaned_ocr.find('[')
            print(f"JSON start position: {json_start}")
            if json_start != -1:
                cleaned_ocr = cleaned_ocr[json_start:]
                print(f"After cleaning: {repr(cleaned_ocr[:50])}")
        
        ocr_data = json.loads(cleaned_ocr)
        print(f"Successfully parsed JSON with {len(ocr_data)} elements")
        
        # Count Picture elements manually
        picture_count = 0
        chart_count = 0
        
        from test_single_page import is_likely_chart
        
        for i, element in enumerate(ocr_data):
            if element.get('category') == 'Picture':
                picture_count += 1
                if is_likely_chart(element):
                    chart_count += 1
                    bbox = element.get('bbox', [])
                    print(f"  Chart {chart_count}: Element {i}, bbox {bbox}")
        
        print(f"Picture elements: {picture_count}")
        print(f"Chart elements (detected): {chart_count}")
        
        # Now test the actual function
        print("\nTesting chart processing function...")
        processed_ocr, charts_processed = process_charts_in_ocr_result(
            ocr_result, "dummy_base64", openai_client
        )
        
        print(f"Function result - Charts processed: {charts_processed}")
        
        if chart_count > 0 and charts_processed == 0:
            print("ISSUE: Charts detected manually but not by function")
        elif charts_processed > 0:
            print("SUCCESS: Chart processing working!")
        else:
            print("No charts detected by either method")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chart_processing_fix()