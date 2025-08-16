"""
Test OCR benchmark examples with LLM-based extraction for structured comparison.
This provides a more accurate evaluation by using the JSON schema.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.test_single_example import test_modal_endpoint, load_ground_truth
from benchmark.extraction_utils import (
    load_openai_client, 
    extract_and_compare, 
    generate_comparison_report
)


def test_example_with_extraction(example_id="000", modal_url="https://marker--dotsocr-v2.modal.run", 
                                prompt_mode="prompt_ocr"):
    """
    Test a single example with LLM extraction for accurate comparison.
    
    Args:
        example_id: ID of the example to test (e.g., "000")
        modal_url: Modal endpoint URL
        prompt_mode: OCR prompt mode to use
    """
    print("OCR Benchmark Test with LLM Extraction")
    print("=" * 50)
    print(f"Example ID: {example_id}")
    print(f"Modal URL: {modal_url}")
    print(f"Prompt Mode: {prompt_mode}")
    
    # File paths
    image_path = f"benchmark/data/example_{example_id}.png"
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"Error: Example files not found for ID {example_id}")
        print(f"Please run 'python benchmark/load_example.py' first")
        return None
    
    # Load ground truth and schema
    print("\nStep 1: Loading ground truth data...")
    try:
        truth_json, truth_md, metadata = load_ground_truth("benchmark/data", example_id)
        
        # Parse JSON schema from metadata
        json_schema_str = metadata.get("json_schema", "{}")
        if isinstance(json_schema_str, str):
            json_schema = json.loads(json_schema_str)
        else:
            json_schema = json_schema_str
            
        print(f"Ground truth loaded with {len(str(truth_json))} characters")
        print(f"Schema loaded: {list(json_schema.get('properties', {}).keys())}")
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None
    
    # Test Modal OCR
    print(f"\nStep 2: Getting OCR result from Modal...")
    modal_result = test_modal_endpoint(modal_url, image_path, prompt_mode)
    
    if not modal_result:
        print("Error: Could not get OCR result from Modal")
        return None
    
    print(f"Modal OCR completed: {len(modal_result)} characters")
    
    # Load OpenAI client
    print("\nStep 3: Initializing OpenAI client...")
    openai_client = load_openai_client()
    
    if not openai_client:
        print("Error: Could not initialize OpenAI client")
        print("Make sure OPENAI_API_KEY environment variable is set")
        return None
    
    print("OpenAI client ready")
    
    # Extract and compare
    print("\nStep 4: Extracting structured data with LLM...")
    results = extract_and_compare(
        ocr_text=modal_result,
        ground_truth=truth_json,
        json_schema=json.dumps(json_schema),
        openai_client=openai_client
    )
    
    if not results["extraction_successful"]:
        print(f"Extraction failed: {results['extraction_error']}")
        return None
    
    print("LLM extraction completed")
    
    # Generate and save report
    print("\nStep 5: Generating comparison report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark/results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_file = results_dir / f"extraction_test_{example_id}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "example_id": example_id,
            "modal_url": modal_url,
            "prompt_mode": prompt_mode,
            "metadata": metadata,
            "ground_truth": truth_json,
            "modal_ocr_result": modal_result,
            "extraction_results": results
        }, f, indent=2, ensure_ascii=False)
    
    # Generate human-readable report
    report_file = results_dir / f"extraction_report_{example_id}_{timestamp}.txt"
    report = generate_comparison_report(results["comparison_results"], str(report_file))
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    comparison = results["comparison_results"]
    accuracy = comparison["accuracy_percentage"]
    
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"Extraction Success: {'Yes' if results['overall_success'] else 'No'}")
    print(f"Fields Matched: {comparison['matches']}/{comparison['total_fields']}")
    
    if comparison['mismatches'] > 0:
        print(f"Mismatched Fields: {comparison['mismatches']}")
    if comparison['missing_fields'] > 0:
        print(f"Missing Fields: {comparison['missing_fields']}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Human-readable report: {report_file}")
    
    return results


def test_multiple_prompt_modes(example_id="000", modal_url="https://marker--dotsocr-v2.modal.run"):
    """
    Test the same example with different prompt modes to compare performance.
    """
    prompt_modes = ["prompt_ocr", "prompt_layout_all_en"]
    results = {}
    
    print("Testing Multiple Prompt Modes")
    print("=" * 50)
    
    for prompt_mode in prompt_modes:
        print(f"\nTesting with {prompt_mode}...")
        result = test_example_with_extraction(example_id, modal_url, prompt_mode)
        
        if result:
            accuracy = result["comparison_results"]["accuracy_percentage"]
            results[prompt_mode] = {
                "accuracy": accuracy,
                "success": result["overall_success"],
                "extraction_successful": result["extraction_successful"]
            }
            print(f"{prompt_mode}: {accuracy:.1f}% accuracy")
        else:
            results[prompt_mode] = {"error": "Test failed"}
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("PROMPT MODE COMPARISON")
    print("=" * 50)
    
    for mode, result in results.items():
        if "error" not in result:
            status = "SUCCESS" if result["success"] else "PARTIAL"
            print(f"{mode}: {result['accuracy']:.1f}% ({status})")
        else:
            print(f"{mode}: FAILED")
    
    # Recommend best prompt mode
    if results:
        best_mode = max([k for k, v in results.items() if "error" not in v], 
                       key=lambda k: results[k]["accuracy"])
        best_accuracy = results[best_mode]["accuracy"]
        print(f"\nRecommended prompt mode: {best_mode} ({best_accuracy:.1f}% accuracy)")
    
    return results


def main():
    """Main function for testing single example."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file or environment variables")
        print("Please add OPENAI_API_KEY=your_api_key to your .env file")
        return
    
    # Test single example with prompt_ocr (most reliable)
    print("Starting single example OCR benchmark test with LLM extraction...")
    print("Using prompt_ocr mode for clean text extraction")
    
    result = test_example_with_extraction(
        example_id="000",
        prompt_mode="prompt_ocr"
    )
    
    if result:
        if result["overall_success"]:
            print("\nSUCCESS: Perfect extraction achieved!")
        else:
            accuracy = result["comparison_results"]["accuracy_percentage"]
            print(f"\nPARTIAL SUCCESS: {accuracy:.1f}% accuracy achieved")
            print("Check the detailed report for specific issues")
    else:
        print("\nFAILED: Test could not complete")
        print("Check Modal endpoint and OpenAI API key")


if __name__ == "__main__":
    main()