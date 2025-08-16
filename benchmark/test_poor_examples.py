"""
Test smart prompting specifically on the poor-performing examples we identified.
"""
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.run_batch_benchmark import download_and_prepare_batch, run_batch_benchmark


def test_poor_examples():
    """Test the specific poor-performing examples with smart prompting."""
    load_dotenv()
    
    # These are the example IDs that performed poorly in our previous filtered test
    # Converting to original dataset indices based on our previous mapping
    poor_example_indices = [4, 13, 36, 39, 46, 47, 56, 62]
    
    print("TESTING POOR-PERFORMING EXAMPLES WITH SMART PROMPTING")
    print("=" * 60)
    print(f"Testing examples: {poor_example_indices}")
    
    # Test each example individually to compare
    results_comparison = []
    
    for example_idx in poor_example_indices[:3]:  # Start with first 3
        print(f"\n--- Testing Example {example_idx:03d} ---")
        
        try:
            # Test with standard prompting
            print("Standard prompting...")
            standard_results = run_batch_benchmark(
                max_examples=example_idx + 1,
                filter_doc_types=False,
                use_smart_prompting=False
            )
            
            if "error" in standard_results:
                print(f"Standard test failed: {standard_results['error']}")
                continue
                
            # Get the specific example result
            standard_example = None
            for result in standard_results["detailed_results"]:
                if result["original_id"] == example_idx:
                    standard_example = result
                    break
            
            if not standard_example:
                print(f"Could not find example {example_idx} in standard results")
                continue
            
            # Test with smart prompting
            print("Smart prompting...")
            smart_results = run_batch_benchmark(
                max_examples=example_idx + 1,
                filter_doc_types=False,
                use_smart_prompting=True
            )
            
            if "error" in smart_results:
                print(f"Smart test failed: {smart_results['error']}")
                continue
                
            # Get the specific example result
            smart_example = None
            for result in smart_results["detailed_results"]:
                if result["original_id"] == example_idx:
                    smart_example = result
                    break
            
            if not smart_example:
                print(f"Could not find example {example_idx} in smart results")
                continue
            
            # Compare results
            standard_acc = standard_example.get("accuracy", 0)
            smart_acc = smart_example.get("accuracy", 0)
            improvement = smart_acc - standard_acc
            
            result_comparison = {
                "example_id": example_idx,
                "standard_accuracy": standard_acc,
                "smart_accuracy": smart_acc,
                "improvement": improvement,
                "standard_ocr_length": standard_example.get("ocr_length", 0),
                "smart_ocr_length": smart_example.get("ocr_length", 0),
                "document_type": standard_example.get("document_type", ""),
                "standard_prompt": standard_example.get("modal_result", {}).get("prompt_mode", "prompt_ocr"),
                "smart_prompt": smart_example.get("modal_result", {}).get("prompt_mode", "unknown")
            }
            
            results_comparison.append(result_comparison)
            
            print(f"Results: {standard_acc:.1f}% → {smart_acc:.1f}% ({improvement:+.1f}%)")
            print(f"OCR length: {standard_example.get('ocr_length', 0)} → {smart_example.get('ocr_length', 0)} chars")
            print(f"Prompt modes: {result_comparison['standard_prompt']} → {result_comparison['smart_prompt']}")
            
            time.sleep(5)  # Pause between tests
            
        except Exception as e:
            print(f"Error testing example {example_idx}: {e}")
            continue
    
    # Summary
    print(f"\n" + "=" * 60)
    print("POOR EXAMPLES SMART PROMPTING SUMMARY")
    print("=" * 60)
    
    if results_comparison:
        avg_standard = sum(r["standard_accuracy"] for r in results_comparison) / len(results_comparison)
        avg_smart = sum(r["smart_accuracy"] for r in results_comparison) / len(results_comparison)
        avg_improvement = avg_smart - avg_standard
        
        print(f"Examples tested: {len(results_comparison)}")
        print(f"Average standard accuracy: {avg_standard:.1f}%")
        print(f"Average smart accuracy: {avg_smart:.1f}%")
        print(f"Average improvement: {avg_improvement:+.1f} percentage points")
        
        print(f"\nDetailed results:")
        for r in results_comparison:
            doc_type = json.loads(r["document_type"]).get("format", "UNKNOWN")
            print(f"  Example {r['example_id']:03d} ({doc_type}): {r['standard_accuracy']:.1f}% → {r['smart_accuracy']:.1f}% ({r['improvement']:+.1f}%)")
    
    return results_comparison


if __name__ == "__main__":
    test_poor_examples()