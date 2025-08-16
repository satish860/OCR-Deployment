"""
Run OCR benchmark on all examples with progress tracking and accuracy reporting.
Tests the complete dataset and provides comprehensive accuracy metrics.
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.extraction_utils import (
    load_openai_client, 
    extract_and_compare
)
from benchmark.test_single_example import test_modal_endpoint


def download_all_examples(max_examples: int = None) -> List[Dict[str, Any]]:
    """
    Download all examples from the OCR benchmark dataset.
    
    Args:
        max_examples: Maximum number of examples to download (None for all)
        
    Returns:
        List of example dictionaries
    """
    print("Loading OCR benchmark dataset...")
    
    try:
        # Use streaming to handle large datasets
        dataset = load_dataset("getomni-ai/ocr-benchmark", split="test", streaming=True)
        
        examples = []
        if max_examples:
            print(f"Downloading first {max_examples} examples...")
            examples = list(dataset.take(max_examples))
        else:
            print("Downloading all examples...")
            examples = list(dataset)
        
        print(f"Successfully loaded {len(examples)} examples")
        return examples
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def save_example_locally(example: Dict[str, Any], example_id: str, data_dir: str = "benchmark/data") -> bool:
    """
    Save a single example to local files.
    
    Args:
        example: Example data from dataset
        example_id: Unique identifier for the example
        data_dir: Directory to save files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        save_path = Path(data_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        image = example["image"]
        image_path = save_path / f"example_{example_id}.png"
        image.save(image_path)
        
        # Save the ground truth JSON
        json_truth_path = save_path / f"example_{example_id}_truth.json"
        with open(json_truth_path, "w", encoding="utf-8") as f:
            json.dump(example["true_json_output"], f, indent=2, ensure_ascii=False)
        
        # Save the ground truth markdown
        md_truth_path = save_path / f"example_{example_id}_truth.md"
        with open(md_truth_path, "w", encoding="utf-8") as f:
            f.write(example["true_markdown_output"])
        
        # Save the metadata and schema
        metadata_path = save_path / f"example_{example_id}_metadata.json"
        metadata = {
            "id": example["id"],
            "metadata": example["metadata"],
            "json_schema": example["json_schema"]
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving example {example_id}: {e}")
        return False


def load_ground_truth_for_example(example_id: str, data_dir: str = "benchmark/data") -> tuple:
    """
    Load ground truth data for a specific example.
    
    Returns:
        (truth_json, truth_md, metadata) or (None, None, None) if error
    """
    try:
        data_path = Path(data_dir)
        
        # Load ground truth JSON
        json_path = data_path / f"example_{example_id}_truth.json"
        with open(json_path, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()
            # Handle double-encoded JSON
            if raw_content.startswith('"') and raw_content.endswith('"'):
                json_string = json.loads(raw_content)
                truth_json = json.loads(json_string)
            else:
                truth_json = json.loads(raw_content)
        
        # Load ground truth markdown
        md_path = data_path / f"example_{example_id}_truth.md"
        with open(md_path, "r", encoding="utf-8") as f:
            truth_md = f.read()
        
        # Load metadata
        metadata_path = data_path / f"example_{example_id}_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return truth_json, truth_md, metadata
        
    except Exception as e:
        print(f"Error loading ground truth for example {example_id}: {e}")
        return None, None, None


def test_single_example_full_pipeline(example_id: str, modal_url: str, prompt_mode: str, 
                                    openai_client, progress_bar=None) -> Dict[str, Any]:
    """
    Test a single example through the complete OCR + extraction pipeline.
    
    Returns:
        Dictionary with test results
    """
    if progress_bar:
        progress_bar.set_description(f"Testing example {example_id}")
    
    # File paths
    image_path = f"benchmark/data/example_{example_id}.png"
    
    # Check if files exist
    if not Path(image_path).exists():
        return {
            "example_id": example_id,
            "success": False,
            "error": "Image file not found",
            "accuracy": 0.0
        }
    
    try:
        # Load ground truth
        truth_json, truth_md, metadata = load_ground_truth_for_example(example_id)
        if truth_json is None:
            return {
                "example_id": example_id,
                "success": False,
                "error": "Could not load ground truth",
                "accuracy": 0.0
            }
        
        # Get OCR result from Modal
        modal_result = test_modal_endpoint(modal_url, image_path, prompt_mode)
        if not modal_result:
            return {
                "example_id": example_id,
                "success": False,
                "error": "Modal OCR failed",
                "accuracy": 0.0
            }
        
        # Parse JSON schema
        json_schema_str = metadata.get("json_schema", "{}")
        if isinstance(json_schema_str, str):
            json_schema = json.loads(json_schema_str)
        else:
            json_schema = json_schema_str
        
        # Extract and compare with LLM
        results = extract_and_compare(
            ocr_text=modal_result,
            ground_truth=truth_json,
            json_schema=json.dumps(json_schema),
            openai_client=openai_client
        )
        
        if not results["extraction_successful"]:
            return {
                "example_id": example_id,
                "success": False,
                "error": f"LLM extraction failed: {results.get('extraction_error', 'Unknown error')}",
                "accuracy": 0.0
            }
        
        # Return success results
        accuracy = results["comparison_results"]["accuracy_percentage"]
        return {
            "example_id": example_id,
            "success": True,
            "accuracy": accuracy,
            "total_fields": results["comparison_results"]["total_fields"],
            "matched_fields": results["comparison_results"]["matches"],
            "perfect_extraction": results["overall_success"],
            "document_type": metadata.get("metadata", ""),
            "processing_time": time.time()  # Will be calculated by caller
        }
        
    except Exception as e:
        return {
            "example_id": example_id,
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        }


def run_full_benchmark(max_examples: int = None, modal_url: str = "https://marker--dotsocr-v2.modal.run", 
                      prompt_mode: str = "prompt_ocr") -> Dict[str, Any]:
    """
    Run the complete benchmark on all examples.
    
    Args:
        max_examples: Maximum number of examples to test (None for all)
        modal_url: Modal endpoint URL
        prompt_mode: OCR prompt mode to use
        
    Returns:
        Comprehensive results dictionary
    """
    print("Starting Full OCR Benchmark")
    print("=" * 50)
    print(f"Modal URL: {modal_url}")
    print(f"Prompt Mode: {prompt_mode}")
    print(f"Max Examples: {max_examples or 'All'}")
    
    # Load environment and OpenAI client
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        return {"error": "Missing OpenAI API key"}
    
    openai_client = load_openai_client()
    if not openai_client:
        print("Error: Could not initialize OpenAI client")
        return {"error": "OpenAI client initialization failed"}
    
    # Download examples
    examples = download_all_examples(max_examples)
    if not examples:
        print("Error: No examples loaded")
        return {"error": "No examples loaded"}
    
    print(f"\nProcessing {len(examples)} examples...")
    
    # Save examples locally and test each one
    results = []
    failed_downloads = 0
    start_time = time.time()
    
    # Progress bar for overall process
    with tqdm(total=len(examples), desc="Processing examples", unit="example") as pbar:
        for i, example in enumerate(examples):
            example_id = f"{i:03d}"
            
            # Save example locally
            pbar.set_description(f"Saving example {example_id}")
            if not save_example_locally(example, example_id):
                failed_downloads += 1
                pbar.update(1)
                continue
            
            # Test the example
            test_start = time.time()
            result = test_single_example_full_pipeline(
                example_id, modal_url, prompt_mode, openai_client, pbar
            )
            result["processing_time"] = time.time() - test_start
            
            results.append(result)
            pbar.update(1)
    
    total_time = time.time() - start_time
    
    # Calculate overall statistics
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    if successful_tests:
        overall_accuracy = sum(r["accuracy"] for r in successful_tests) / len(successful_tests)
        perfect_extractions = sum(1 for r in successful_tests if r.get("perfect_extraction", False))
        avg_processing_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
    else:
        overall_accuracy = 0.0
        perfect_extractions = 0
        avg_processing_time = 0.0
    
    # Compile final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "modal_url": modal_url,
            "prompt_mode": prompt_mode,
            "max_examples": max_examples,
            "total_examples_attempted": len(examples)
        },
        "summary": {
            "total_examples": len(examples),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "failed_downloads": failed_downloads,
            "overall_accuracy": overall_accuracy,
            "perfect_extractions": perfect_extractions,
            "perfect_extraction_rate": (perfect_extractions / len(successful_tests) * 100) if successful_tests else 0,
            "total_processing_time": total_time,
            "avg_processing_time_per_example": avg_processing_time
        },
        "detailed_results": results
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark/results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"full_benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    return final_results


def print_benchmark_summary(results: Dict[str, Any]):
    """
    Print a comprehensive summary of benchmark results.
    """
    if "error" in results:
        print(f"Benchmark failed: {results['error']}")
        return
    
    summary = results["summary"]
    
    print("\n" + "=" * 60)
    print("FULL BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Total Examples Tested: {summary['total_examples']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    print(f"Failed Downloads: {summary['failed_downloads']}")
    
    print(f"\nACCURACY METRICS:")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1f}%")
    print(f"Perfect Extractions: {summary['perfect_extractions']}")
    print(f"Perfect Extraction Rate: {summary['perfect_extraction_rate']:.1f}%")
    
    print(f"\nPERFORMACE METRICS:")
    print(f"Total Processing Time: {summary['total_processing_time']:.1f} seconds")
    print(f"Average Time per Example: {summary['avg_processing_time_per_example']:.2f} seconds")
    
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
    
    # Show failed examples
    failed_results = [r for r in results["detailed_results"] if not r["success"]]
    if failed_results:
        print(f"\nFAILED EXAMPLES:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"  Example {result['example_id']}: {result['error']}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")


def main():
    """Main function to run the full benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full OCR benchmark")
    parser.add_argument("--max-examples", type=int, default=None, 
                       help="Maximum number of examples to test (default: all)")
    parser.add_argument("--prompt-mode", default="prompt_ocr",
                       choices=["prompt_ocr", "prompt_layout_all_en", "prompt_layout_only_en", "prompt_grounding_ocr"],
                       help="OCR prompt mode to use")
    parser.add_argument("--modal-url", default="https://marker--dotsocr-v2.modal.run",
                       help="Modal endpoint URL")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_full_benchmark(
        max_examples=args.max_examples,
        modal_url=args.modal_url,
        prompt_mode=args.prompt_mode
    )
    
    # Print summary
    print_benchmark_summary(results)


if __name__ == "__main__":
    main()