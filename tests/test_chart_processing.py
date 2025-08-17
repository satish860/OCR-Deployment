"""
Test script specifically for chart processing functionality.
Uses the sample chart image to validate the LLM-based chart extraction.
"""
import time
import base64
import json
from test_single_page import (
    load_openai_client,
    analyze_chart_with_llm,
    is_likely_chart,
    extract_image_region
)

def test_chart_with_sample_image():
    """Test chart processing with the sample chart image"""
    chart_path = "dots.ocr/assets/chart.png"
    
    print("Chart Processing Test")
    print("=" * 50)
    print(f"Testing with: {chart_path}")
    
    # Load OpenAI client
    print("\nInitializing OpenAI client...")
    openai_client = load_openai_client()
    
    if not openai_client:
        print("ERROR: Could not initialize OpenAI client")
        print("Make sure OPENAI_API_KEY environment variable is set")
        return
    
    print("OpenAI client ready")
    
    # Load and encode the chart image
    print(f"\nLoading chart image: {chart_path}")
    try:
        with open(chart_path, "rb") as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode()
        chart_data_url = f"data:image/png;base64,{image_base64}"
        
        print(f"Chart image loaded: {len(image_base64):,} characters")
        
    except Exception as e:
        print(f"ERROR: Failed to load chart image: {e}")
        return
    
    # Analyze the chart with LLM
    print(f"\nAnalyzing chart with OpenAI Vision API...")
    start_time = time.time()
    
    analysis = analyze_chart_with_llm(chart_data_url, openai_client)
    
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.2f}s")
    
    if analysis["success"]:
        print(f"Tokens used: {analysis.get('tokens_used', 'unknown')}")
        print("\nChart Analysis Result:")
        print("-" * 80)
        print(analysis["description"])
        print("-" * 80)
        
        # Save result to file
        with open("chart_analysis_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Chart Analysis Result\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis time: {analysis_time:.2f}s\n")
            f.write(f"Tokens used: {analysis.get('tokens_used', 'unknown')}\n")
            f.write(f"\n{'-'*80}\n")
            f.write(analysis["description"])
        
        print(f"\nResult saved to: chart_analysis_result.txt")
        print(f"SUCCESS: Chart analysis completed!")
        
    else:
        print(f"ERROR: Chart analysis failed: {analysis['error']}")

def test_chart_detection_logic():
    """Test the chart detection logic with sample elements"""
    print("\nChart Detection Logic Test")
    print("-" * 30)
    
    # Test cases for chart detection
    test_cases = [
        {
            "name": "Large Picture Element (likely chart)",
            "element": {"category": "Picture", "bbox": [100, 200, 600, 500]},
            "expected": True
        },
        {
            "name": "Small Picture Element (likely icon)",
            "element": {"category": "Picture", "bbox": [100, 200, 150, 250]},
            "expected": False
        },
        {
            "name": "Very Wide Picture (likely banner)",
            "element": {"category": "Picture", "bbox": [0, 100, 800, 150]},
            "expected": False
        },
        {
            "name": "Text Element",
            "element": {"category": "Text", "bbox": [100, 200, 600, 500]},
            "expected": False
        },
        {
            "name": "Square Chart",
            "element": {"category": "Picture", "bbox": [100, 200, 400, 500]},
            "expected": True
        }
    ]
    
    for test_case in test_cases:
        result = is_likely_chart(test_case["element"])
        status = "PASS" if result == test_case["expected"] else "FAIL"
        print(f"[{status}] {test_case['name']}: {result} (expected: {test_case['expected']})")

def simulate_ocr_with_chart():
    """Simulate an OCR result that contains a chart"""
    print("\nSimulated OCR + Chart Processing Test")
    print("-" * 40)
    
    # Create a mock OCR result with a Picture element
    mock_ocr_result = [
        {
            "bbox": [50, 100, 300, 200],
            "category": "Text",
            "text": "End-to-end evaluation results of different models"
        },
        {
            "bbox": [50, 220, 700, 600],  # Large area that would be detected as chart
            "category": "Picture"
            # Note: no "text" field - this is what we want to fill in
        },
        {
            "bbox": [50, 620, 400, 680],
            "category": "Text", 
            "text": "Figure 1: Performance comparison across models"
        }
    ]
    
    print("Mock OCR result structure:")
    for i, element in enumerate(mock_ocr_result):
        chart_likely = is_likely_chart(element)
        print(f"  Element {i}: {element['category']} - Chart likely: {chart_likely}")
    
    return mock_ocr_result

if __name__ == "__main__":
    print("Chart Processing Test Suite")
    print("=" * 60)
    
    # Test 1: Chart detection logic
    test_chart_detection_logic()
    
    # Test 2: Simulate OCR structure
    simulate_ocr_with_chart()
    
    # Test 3: Actual chart analysis (requires OpenAI API key)
    test_chart_with_sample_image()