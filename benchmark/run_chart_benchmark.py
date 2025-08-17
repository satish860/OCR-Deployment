"""
Enhanced OCR benchmark with chart processing capabilities.
Removes document type filtering and adds LLM-based chart analysis.
"""
import json
import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
import requests

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.extraction_utils import (
    load_openai_client, 
    extract_and_compare
)

# Import chart processing functions from our test script
sys.path.insert(0, str(project_root / "tests"))
from test_single_page import (
    extract_image_region,
    is_likely_chart,
    analyze_chart_with_llm,
    process_charts_in_ocr_result
)


def image_to_base64(image) -> str:
    """Convert PIL image to base64 string."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def download_and_prepare_batch(max_examples: int = None) -> List[Dict[str, Any]]:
    """
    Download all examples and prepare them for batch processing.
    
    Args:
        max_examples: Maximum number of examples to download (None for all)
        
    Returns:
        List of prepared examples with base64 images
    """
    print("Loading OCR benchmark dataset...")
    
    try:
        # Use streaming to handle large datasets
        dataset = load_dataset("getomni-ai/ocr-benchmark", split="test", streaming=True)
        
        if max_examples:
            print(f"Downloading first {max_examples} examples...")
            examples = list(dataset.take(max_examples))
        else:
            print("Downloading all examples...")
            examples = list(dataset)
        
        print(f"Successfully loaded {len(examples)} examples")
        
        # Prepare examples with base64 images
        prepared_examples = []
        print("Converting images to base64...")
        
        for i, example in enumerate(tqdm(examples, desc="Preparing images")):
            try:
                # Convert image to base64
                image_b64 = image_to_base64(example["image"])
                
                # Parse ground truth and metadata
                truth_json = example["true_json_output"]
                if isinstance(truth_json, str):
                    # Handle double-encoded JSON
                    if truth_json.startswith('"') and truth_json.endswith('"'):
                        json_string = json.loads(truth_json)
                        truth_json = json.loads(json_string)
                    else:
                        truth_json = json.loads(truth_json)
                
                prepared_example = {
                    "example_id": f"{i:03d}",
                    "original_id": example["id"],
                    "image_b64": image_b64,
                    "ground_truth": truth_json,
                    "ground_truth_md": example["true_markdown_output"],
                    "metadata": example["metadata"],
                    "json_schema": example["json_schema"]
                }
                
                prepared_examples.append(prepared_example)
                
            except Exception as e:
                print(f"Error preparing example {i}: {e}")
                continue
        
        print(f"Successfully prepared {len(prepared_examples)} examples for batch processing")
        return prepared_examples
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def send_batch_to_modal(examples: List[Dict[str, Any]], modal_url: str, 
                       prompt_mode: str = "prompt_layout_all_en") -> Dict[str, Any]:
    """
    Send all images to Modal in a single batch request.
    
    Args:
        examples: List of prepared examples with base64 images
        modal_url: Modal endpoint URL
        prompt_mode: OCR prompt mode to use
        
    Returns:
        Modal batch response
    """
    print(f"\nSending batch of {len(examples)} images to Modal...")
    print(f"Modal URL: {modal_url}")
    print(f"Prompt Mode: {prompt_mode}")
    
    images_b64 = [example["image_b64"] for example in examples]
    
    request_data = {
        "images": images_b64,
        "prompt_mode": prompt_mode,
        "temperature": 0.1,
        "top_p": 0.9
    }

    try:
        print("Sending batch request to Modal...")
        start_time = time.time()
        
        response = requests.post(
            modal_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=1800  # 30 minutes timeout for batch processing
        )
        
        processing_time = time.time() - start_time
        print(f"Modal batch processing completed in {processing_time:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                print(f"Batch OCR successful: {len(result.get('results', []))} results received")
                return {
                    "success": True,
                    "results": result["results"],
                    "total_pages": result.get("total_pages", len(examples)),
                    "processing_mode": result.get("processing_mode", "batch"),
                    "processing_time": processing_time
                }
            else:
                print(f"Modal batch failed: {result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }
        else:
            print(f"HTTP error {response.status_code}: {response.text}")
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        print(f"Request failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def process_batch_extractions_with_charts(examples: List[Dict[str, Any]], modal_results: List[Dict[str, Any]], 
                                        openai_client, process_charts: bool = True) -> List[Dict[str, Any]]:
    """
    Process LLM extractions for all examples using the batch OCR results with optional chart processing.
    
    Args:
        examples: Original examples with ground truth
        modal_results: Results from Modal batch OCR
        openai_client: OpenAI client instance
        process_charts: Whether to process charts with LLM analysis
        
    Returns:
        List of extraction results
    """
    print(f"\nProcessing LLM extractions for {len(examples)} examples...")
    if process_charts:
        print("Chart processing ENABLED - will analyze Picture elements")
    else:
        print("Chart processing DISABLED")
    
    extraction_results = []
    total_charts_processed = 0
    total_chart_time = 0
    
    with tqdm(total=len(examples), desc="Processing extractions", unit="example") as pbar:
        for i, (example, modal_result) in enumerate(zip(examples, modal_results)):
            pbar.set_description(f"Extracting example {example['example_id']}")
            
            try:
                # Check if Modal OCR was successful for this example
                if not modal_result.get("success"):
                    extraction_results.append({
                        "example_id": example["example_id"],
                        "success": False,
                        "error": f"Modal OCR failed: {modal_result.get('error', 'Unknown error')}",
                        "accuracy": 0.0,
                        "charts_processed": 0,
                        "chart_processing_time": 0
                    })
                    pbar.update(1)
                    continue
                
                # Get OCR result
                ocr_text = modal_result.get("result", "")
                if not ocr_text:
                    extraction_results.append({
                        "example_id": example["example_id"],
                        "success": False,
                        "error": "Empty OCR result from Modal",
                        "accuracy": 0.0,
                        "charts_processed": 0,
                        "chart_processing_time": 0
                    })
                    pbar.update(1)
                    continue
                
                # Process charts if enabled
                charts_processed = 0
                chart_time = 0
                
                if process_charts and openai_client:
                    chart_start = time.time()
                    ocr_text, charts_processed = process_charts_in_ocr_result(
                        ocr_text, example["image_b64"], openai_client
                    )
                    chart_time = time.time() - chart_start
                    total_charts_processed += charts_processed
                    total_chart_time += chart_time
                
                # Parse JSON schema
                json_schema_str = example.get("json_schema", "{}")
                if isinstance(json_schema_str, str):
                    json_schema = json.loads(json_schema_str)
                else:
                    json_schema = json_schema_str
                
                # Extract and compare with LLM
                extraction_start = time.time()
                results = extract_and_compare(
                    ocr_text=ocr_text,
                    ground_truth=example["ground_truth"],
                    json_schema=json.dumps(json_schema),
                    openai_client=openai_client
                )
                extraction_time = time.time() - extraction_start
                
                if not results["extraction_successful"]:
                    extraction_results.append({
                        "example_id": example["example_id"],
                        "success": False,
                        "error": f"LLM extraction failed: {results.get('extraction_error', 'Unknown error')}",
                        "accuracy": 0.0,
                        "extraction_time": extraction_time,
                        "charts_processed": charts_processed,
                        "chart_processing_time": chart_time
                    })
                    pbar.update(1)
                    continue
                
                # Success case
                accuracy = results["comparison_results"]["accuracy_percentage"]
                extraction_results.append({
                    "example_id": example["example_id"],
                    "original_id": example["original_id"],
                    "success": True,
                    "accuracy": accuracy,
                    "total_fields": results["comparison_results"]["total_fields"],
                    "matched_fields": results["comparison_results"]["matches"],
                    "perfect_extraction": results["overall_success"],
                    "document_type": example.get("metadata", ""),
                    "extraction_time": extraction_time,
                    "ocr_length": len(ocr_text),
                    "charts_processed": charts_processed,
                    "chart_processing_time": chart_time,
                    "modal_result": modal_result
                })
                
            except Exception as e:
                extraction_results.append({
                    "example_id": example["example_id"],
                    "success": False,
                    "error": f"Processing error: {str(e)}",
                    "accuracy": 0.0,
                    "charts_processed": 0,
                    "chart_processing_time": 0
                })
            
            pbar.update(1)
    
    print(f"\nChart Processing Summary:")
    print(f"Total charts processed: {total_charts_processed}")
    print(f"Total chart processing time: {total_chart_time:.1f}s")
    if total_charts_processed > 0:
        print(f"Average time per chart: {total_chart_time/total_charts_processed:.1f}s")
    
    return extraction_results


def run_chart_benchmark(max_examples: int = None, modal_url: str = "https://marker--dotsocr-v2.modal.run", 
                       prompt_mode: str = "prompt_layout_all_en", process_charts: bool = True) -> Dict[str, Any]:
    """
    Run the complete batch benchmark with chart processing (NO FILTERING).
    
    Args:
        max_examples: Maximum number of examples to test (None for all)
        modal_url: Modal endpoint URL
        prompt_mode: OCR prompt mode to use
        process_charts: Whether to process charts with LLM analysis
        
    Returns:
        Comprehensive results dictionary
    """
    print("Starting Enhanced OCR Benchmark with Chart Processing")
    print("=" * 60)
    print(f"Modal URL: {modal_url}")
    print(f"Prompt Mode: {prompt_mode}")
    print(f"Max Examples: {max_examples or 'All (1000)'}")
    print(f"Chart Processing: {'ENABLED' if process_charts else 'DISABLED'}")
    print(f"Document Type Filtering: DISABLED (testing all document types)")
    
    # Load environment and OpenAI client
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        return {"error": "Missing OpenAI API key"}
    
    openai_client = load_openai_client()
    if not openai_client:
        print("Error: Could not initialize OpenAI client")
        return {"error": "OpenAI client initialization failed"}
    
    start_time = time.time()
    
    # Step 1: Download and prepare examples
    examples = download_and_prepare_batch(max_examples)
    if not examples:
        return {"error": "No examples loaded"}
    
    # Step 2: Send batch to Modal
    modal_batch_result = send_batch_to_modal(examples, modal_url, prompt_mode)
    if not modal_batch_result["success"]:
        return {"error": f"Modal batch failed: {modal_batch_result['error']}"}
    
    modal_results = modal_batch_result["results"]
    modal_processing_time = modal_batch_result["processing_time"]
    
    # Step 3: Process LLM extractions with chart processing
    extraction_results = process_batch_extractions_with_charts(
        examples, modal_results, openai_client, process_charts
    )
    
    total_time = time.time() - start_time
    
    # Calculate statistics (NO FILTERING APPLIED)
    successful_tests = [r for r in extraction_results if r["success"]]
    failed_tests = [r for r in extraction_results if not r["success"]]
    
    if successful_tests:
        overall_accuracy = sum(r["accuracy"] for r in successful_tests) / len(successful_tests)
        perfect_extractions = sum(1 for r in successful_tests if r.get("perfect_extraction", False))
        avg_extraction_time = sum(r.get("extraction_time", 0) for r in successful_tests) / len(successful_tests)
        total_charts_processed = sum(r.get("charts_processed", 0) for r in successful_tests)
        total_chart_time = sum(r.get("chart_processing_time", 0) for r in successful_tests)
    else:
        overall_accuracy = 0.0
        perfect_extractions = 0
        avg_extraction_time = 0.0
        total_charts_processed = 0
        total_chart_time = 0
    
    # Compile final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "modal_url": modal_url,
            "prompt_mode": prompt_mode,
            "max_examples": max_examples,
            "total_examples_processed": len(examples),
            "chart_processing_enabled": process_charts,
            "document_filtering": False  # Always False - no filtering
        },
        "timing": {
            "total_time": total_time,
            "modal_batch_time": modal_processing_time,
            "extraction_processing_time": total_time - modal_processing_time,
            "avg_extraction_time_per_example": avg_extraction_time,
            "modal_throughput_images_per_second": len(examples) / modal_processing_time if modal_processing_time > 0 else 0,
            "total_chart_processing_time": total_chart_time,
            "avg_chart_processing_time": total_chart_time / total_charts_processed if total_charts_processed > 0 else 0
        },
        "summary": {
            "total_examples": len(examples),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "overall_accuracy": overall_accuracy,
            "perfect_extractions": perfect_extractions,
            "perfect_extraction_rate": (perfect_extractions / len(successful_tests) * 100) if successful_tests else 0,
            "modal_batch_success": True,
            "batch_efficiency": f"{len(examples)} images processed in {modal_processing_time:.1f}s",
            "charts_processed": total_charts_processed,
            "chart_processing_enabled": process_charts
        },
        "detailed_results": extraction_results,
        "modal_batch_info": {
            "processing_mode": modal_batch_result.get("processing_mode", "batch"),
            "total_pages": modal_batch_result.get("total_pages", len(examples))
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    chart_suffix = "_with_charts" if process_charts else "_no_charts"
    results_file = results_dir / f"chart_benchmark_{timestamp}{chart_suffix}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    return final_results


def print_chart_benchmark_summary(results: Dict[str, Any]):
    """
    Print a comprehensive summary of chart benchmark results.
    """
    if "error" in results:
        print(f"Chart benchmark failed: {results['error']}")
        return
    
    summary = results["summary"]
    timing = results["timing"]
    config = results["configuration"]
    
    print("\n" + "=" * 70)
    print("ENHANCED OCR BENCHMARK RESULTS WITH CHART PROCESSING")
    print("=" * 70)
    
    print(f"Total Examples Processed: {summary['total_examples']} (NO FILTERING)")
    print(f"Successful Extractions: {summary['successful_tests']}")
    print(f"Failed Extractions: {summary['failed_tests']}")
    print(f"Chart Processing: {'ENABLED' if config['chart_processing_enabled'] else 'DISABLED'}")
    
    print(f"\nACCURACY METRICS:")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1f}%")
    print(f"Perfect Extractions: {summary['perfect_extractions']}")
    print(f"Perfect Extraction Rate: {summary['perfect_extraction_rate']:.1f}%")
    
    print(f"\nCHART PROCESSING METRICS:")
    print(f"Total Charts Processed: {summary['charts_processed']}")
    if summary['charts_processed'] > 0:
        print(f"Total Chart Processing Time: {timing['total_chart_processing_time']:.1f}s")
        print(f"Average Time per Chart: {timing['avg_chart_processing_time']:.1f}s")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Total Processing Time: {timing['total_time']:.1f} seconds")
    print(f"Modal Batch Time: {timing['modal_batch_time']:.1f} seconds")
    print(f"Extraction Processing Time: {timing['extraction_processing_time']:.1f} seconds")
    print(f"Modal Throughput: {timing['modal_throughput_images_per_second']:.1f} images/second")
    print(f"Average Extraction Time: {timing['avg_extraction_time_per_example']:.2f} seconds/example")
    
    print(f"\nBATCH EFFICIENCY:")
    print(f"Batch Processing: SUCCESS")
    print(f"{summary['batch_efficiency']}")
    
    # Show accuracy distribution
    successful_results = [r for r in results["detailed_results"] if r["success"]]
    if successful_results:
        accuracies = [r["accuracy"] for r in successful_results]
        
        perfect_count = sum(1 for acc in accuracies if acc == 100.0)
        excellent_count = sum(1 for acc in accuracies if 90.0 <= acc < 100.0)
        good_count = sum(1 for acc in accuracies if 70.0 <= acc < 90.0)
        poor_count = sum(1 for acc in accuracies if acc < 70.0)
        
        print(f"\nACCURACY DISTRIBUTION:")
        print(f"Perfect (100%): {perfect_count}")
        print(f"Excellent (90-99%): {excellent_count}")
        print(f"Good (70-89%): {good_count}")
        print(f"Needs Improvement (<70%): {poor_count}")


def main():
    """Main function to run the chart benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced OCR benchmark with chart processing")
    parser.add_argument("--max-examples", type=int, default=50, 
                       help="Maximum number of examples to test (default: 50)")
    parser.add_argument("--prompt-mode", default="prompt_layout_all_en",
                       choices=["prompt_ocr", "prompt_layout_all_en", "prompt_layout_only_en"],
                       help="OCR prompt mode to use")
    parser.add_argument("--modal-url", default="https://marker--dotsocr-v2.modal.run",
                       help="Modal endpoint URL")
    parser.add_argument("--no-charts", action="store_true",
                       help="Disable chart processing (for comparison)")
    
    args = parser.parse_args()
    
    # Run chart benchmark
    results = run_chart_benchmark(
        max_examples=args.max_examples,
        modal_url=args.modal_url,
        prompt_mode=args.prompt_mode,
        process_charts=not args.no_charts
    )
    
    # Print summary
    print_chart_benchmark_summary(results)


if __name__ == "__main__":
    main()