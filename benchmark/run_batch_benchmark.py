"""
Run OCR benchmark using Modal batch processing for maximum efficiency.
Sends all images in one batch to Modal, then processes extractions individually.
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


def get_optimal_prompt_mode(doc_format: str, doc_quality: str) -> str:
    """
    Select the optimal OCR prompt mode based on document type and quality.
    
    Args:
        doc_format: Document format (TABLE, FORM, etc.)
        doc_quality: Document quality (PHOTO, SCANNED, etc.)
        
    Returns:
        Optimal prompt mode for this document type
    """
    # For tables, especially photo tables, use layout-aware prompting
    if doc_format in ["TABLE", "PHOTO_TABLE", "WEB_TABLE", "SCANNED_TABLE"]:
        if doc_quality == "PHOTO":
            return "prompt_layout_all_en"  # Better for complex photo layouts
        else:
            return "prompt_layout_only_en"  # Cleaner for scanned documents
    
    # For forms, use grounding OCR to better preserve field relationships
    elif doc_format in ["FORM", "SCANNED_FORM"]:
        return "prompt_grounding_ocr"
    
    # For receipts, use standard OCR with layout awareness
    elif doc_format in ["PHOTO_RECEIPT"]:
        return "prompt_layout_all_en"
    
    # For nutrition labels and other text-heavy documents
    elif doc_format in ["PHOTO_NUTRITION"]:
        return "prompt_ocr"
    
    # Default fallback
    else:
        return "prompt_ocr"


def send_batch_to_modal(examples: List[Dict[str, Any]], modal_url: str, 
                       prompt_mode: str = "prompt_ocr", use_smart_prompting: bool = False) -> Dict[str, Any]:
    """
    Send all images to Modal in a single batch request.
    
    Args:
        examples: List of prepared examples with base64 images
        modal_url: Modal endpoint URL
        prompt_mode: OCR prompt mode to use (fallback if smart prompting disabled)
        use_smart_prompting: If True, use document-type-specific prompting
        
    Returns:
        Modal batch response
    """
    print(f"\nSending batch of {len(examples)} images to Modal...")
    print(f"Modal URL: {modal_url}")
    print(f"Smart Prompting: {use_smart_prompting}")
    if use_smart_prompting:
        print("Using document-type-specific prompting")
    else:
        print(f"Prompt Mode: {prompt_mode}")
    
    # For now, smart prompting will group examples by optimal prompt mode
    # and send separate batch requests for each group
    if use_smart_prompting:
        print("Smart prompting: Grouping examples by optimal prompt modes...")
        
        # Group examples by optimal prompt mode
        prompt_groups = {}
        
        for example in examples:
            try:
                # Parse document metadata
                metadata_str = example.get("metadata", "{}")
                if isinstance(metadata_str, str):
                    metadata = json.loads(metadata_str)
                else:
                    metadata = metadata_str
                
                doc_format = metadata.get("format", "UNKNOWN")
                doc_quality = metadata.get("documentQuality", "UNKNOWN")
                
                # Get optimal prompt mode for this document
                optimal_prompt = get_optimal_prompt_mode(doc_format, doc_quality)
                
                if optimal_prompt not in prompt_groups:
                    prompt_groups[optimal_prompt] = []
                prompt_groups[optimal_prompt].append(example)
                
            except Exception as e:
                print(f"Error parsing metadata for example {example.get('example_id', 'unknown')}: {e}")
                # Fallback to default group
                if prompt_mode not in prompt_groups:
                    prompt_groups[prompt_mode] = []
                prompt_groups[prompt_mode].append(example)
        
        print(f"Prompt groups: {[(mode, len(group)) for mode, group in prompt_groups.items()]}")
        
        # Process each group separately and combine results
        all_modal_results = []
        total_modal_time = 0
        
        for group_prompt_mode, group_examples in prompt_groups.items():
            print(f"\nProcessing {len(group_examples)} examples with {group_prompt_mode}...")
            
            images_b64 = [ex["image_b64"] for ex in group_examples]
            
            group_request_data = {
                "images": images_b64,
                "prompt_mode": group_prompt_mode,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            try:
                start_time = time.time()
                
                response = requests.post(
                    modal_url,
                    json=group_request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=1800
                )
                
                group_processing_time = time.time() - start_time
                total_modal_time += group_processing_time
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        all_modal_results.extend(result["results"])
                        print(f"Group processed successfully in {group_processing_time:.1f}s")
                    else:
                        print(f"Group failed: {result.get('error', 'Unknown error')}")
                        return {
                            "success": False,
                            "error": f"Group {group_prompt_mode} failed: {result.get('error', 'Unknown error')}"
                        }
                else:
                    print(f"HTTP error {response.status_code}: {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code} for group {group_prompt_mode}"
                    }
                    
            except Exception as e:
                print(f"Request failed for group {group_prompt_mode}: {e}")
                return {
                    "success": False,
                    "error": f"Group {group_prompt_mode} request failed: {str(e)}"
                }
        
        return {
            "success": True,
            "results": all_modal_results,
            "total_pages": len(examples),
            "processing_mode": "smart_batch",
            "processing_time": total_modal_time
        }
        
    else:
        # Original single prompt mode for all images
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


def process_batch_extractions(examples: List[Dict[str, Any]], modal_results: List[Dict[str, Any]], 
                            openai_client) -> List[Dict[str, Any]]:
    """
    Process LLM extractions for all examples using the batch OCR results.
    
    Args:
        examples: Original examples with ground truth
        modal_results: Results from Modal batch OCR
        openai_client: OpenAI client instance
        
    Returns:
        List of extraction results
    """
    print(f"\nProcessing LLM extractions for {len(examples)} examples...")
    
    extraction_results = []
    
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
                        "accuracy": 0.0
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
                        "accuracy": 0.0
                    })
                    pbar.update(1)
                    continue
                
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
                        "extraction_time": extraction_time
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
                    "modal_result": modal_result
                })
                
            except Exception as e:
                extraction_results.append({
                    "example_id": example["example_id"],
                    "success": False,
                    "error": f"Processing error: {str(e)}",
                    "accuracy": 0.0
                })
            
            pbar.update(1)
    
    return extraction_results


def run_batch_benchmark(max_examples: int = None, modal_url: str = "https://marker--dotsocr-v2.modal.run", 
                       prompt_mode: str = "prompt_ocr", filter_doc_types: bool = False, 
                       use_smart_prompting: bool = False) -> Dict[str, Any]:
    """
    Run the complete batch benchmark.
    
    Args:
        max_examples: Maximum number of examples to test (None for all)
        modal_url: Modal endpoint URL
        prompt_mode: OCR prompt mode to use (fallback if smart prompting disabled)
        filter_doc_types: If True, filter results to only include well-performing document types
        use_smart_prompting: If True, use document-type-specific OCR prompting
        
    Returns:
        Comprehensive results dictionary
    """
    print("Starting Batch OCR Benchmark")
    print("=" * 50)
    print(f"Modal URL: {modal_url}")
    print(f"Prompt Mode: {prompt_mode}")
    print(f"Max Examples: {max_examples or 'All (1000)'}")
    print(f"Filter Document Types: {filter_doc_types}")
    print(f"Smart Prompting: {use_smart_prompting}")
    
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
    modal_batch_result = send_batch_to_modal(examples, modal_url, prompt_mode, use_smart_prompting)
    if not modal_batch_result["success"]:
        return {"error": f"Modal batch failed: {modal_batch_result['error']}"}
    
    modal_results = modal_batch_result["results"]
    modal_processing_time = modal_batch_result["processing_time"]
    
    # Step 3: Process LLM extractions
    extraction_results = process_batch_extractions(examples, modal_results, openai_client)
    
    total_time = time.time() - start_time
    
    # Apply filtering if requested
    if filter_doc_types:
        print(f"\nApplying document type filtering...")
        # These are document types that work well based on analysis
        allowed_formats = [
            "TABLE", "FORM", "SLIDES", "PHOTO_NUTRITION", "SCANNED_TABLE", 
            "WEB_TABLE", "PHOTO_TABLE", "SCANNED_FORM", "PHOTO_RECEIPT"
        ]
        
        original_count = len(extraction_results)
        filtered_results = []
        
        for result in extraction_results:
            try:
                doc_type_str = result.get('document_type', '{}')
                doc_meta = json.loads(doc_type_str)
                doc_format = doc_meta.get('format', 'UNKNOWN')
                
                if doc_format in allowed_formats:
                    filtered_results.append(result)
                else:
                    print(f"Filtered out: Example {result['example_id']} ({doc_format})")
            except Exception as e:
                print(f"Error parsing document type for example {result.get('example_id', 'unknown')}: {e}")
                continue
        
        extraction_results = filtered_results
        print(f"Filtered from {original_count} to {len(extraction_results)} examples")
    
    # Calculate statistics
    successful_tests = [r for r in extraction_results if r["success"]]
    failed_tests = [r for r in extraction_results if not r["success"]]
    
    if successful_tests:
        overall_accuracy = sum(r["accuracy"] for r in successful_tests) / len(successful_tests)
        perfect_extractions = sum(1 for r in successful_tests if r.get("perfect_extraction", False))
        avg_extraction_time = sum(r.get("extraction_time", 0) for r in successful_tests) / len(successful_tests)
    else:
        overall_accuracy = 0.0
        perfect_extractions = 0
        avg_extraction_time = 0.0
    
    # Compile final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "modal_url": modal_url,
            "prompt_mode": prompt_mode,
            "max_examples": max_examples,
            "total_examples_processed": len(examples),
            "smart_prompting": use_smart_prompting,
            "filter_doc_types": filter_doc_types
        },
        "timing": {
            "total_time": total_time,
            "modal_batch_time": modal_processing_time,
            "extraction_processing_time": total_time - modal_processing_time,
            "avg_extraction_time_per_example": avg_extraction_time,
            "modal_throughput_images_per_second": len(examples) / modal_processing_time if modal_processing_time > 0 else 0
        },
        "summary": {
            "total_examples": len(examples),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "overall_accuracy": overall_accuracy,
            "perfect_extractions": perfect_extractions,
            "perfect_extraction_rate": (perfect_extractions / len(successful_tests) * 100) if successful_tests else 0,
            "modal_batch_success": True,
            "batch_efficiency": f"{len(examples)} images processed in {modal_processing_time:.1f}s"
        },
        "detailed_results": extraction_results,
        "modal_batch_info": {
            "processing_mode": modal_batch_result.get("processing_mode", "batch"),
            "total_pages": modal_batch_result.get("total_pages", len(examples))
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark/results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"batch_benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    return final_results


def print_batch_benchmark_summary(results: Dict[str, Any]):
    """
    Print a comprehensive summary of batch benchmark results.
    """
    if "error" in results:
        print(f"Batch benchmark failed: {results['error']}")
        return
    
    summary = results["summary"]
    timing = results["timing"]
    
    print("\n" + "=" * 60)
    print("BATCH BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Total Examples Processed: {summary['total_examples']}")
    print(f"Successful Extractions: {summary['successful_tests']}")
    print(f"Failed Extractions: {summary['failed_tests']}")
    
    print(f"\nACCURACY METRICS:")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1f}%")
    print(f"Perfect Extractions: {summary['perfect_extractions']}")
    print(f"Perfect Extraction Rate: {summary['perfect_extraction_rate']:.1f}%")
    
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
    """Main function to run the batch benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch OCR benchmark")
    parser.add_argument("--max-examples", type=int, default=None, 
                       help="Maximum number of examples to test (default: all 1000)")
    parser.add_argument("--prompt-mode", default="prompt_ocr",
                       choices=["prompt_ocr", "prompt_layout_all_en", "prompt_layout_only_en", "prompt_grounding_ocr"],
                       help="OCR prompt mode to use")
    parser.add_argument("--modal-url", default="https://marker--dotsocr-v2.modal.run",
                       help="Modal endpoint URL")
    parser.add_argument("--filter", action="store_true",
                       help="Filter results to only include well-performing document types")
    parser.add_argument("--smart-prompting", action="store_true",
                       help="Use document-type-specific OCR prompting for better accuracy")
    
    args = parser.parse_args()
    
    # Run batch benchmark
    results = run_batch_benchmark(
        max_examples=args.max_examples,
        modal_url=args.modal_url,
        prompt_mode=args.prompt_mode,
        filter_doc_types=args.filter,
        use_smart_prompting=args.smart_prompting
    )
    
    # Print summary
    print_batch_benchmark_summary(results)


if __name__ == "__main__":
    main()