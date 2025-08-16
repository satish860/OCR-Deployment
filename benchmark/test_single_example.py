"""
Test a single OCR benchmark example against the Modal endpoint.
"""
import json
import base64
import requests
from pathlib import Path
from datetime import datetime


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_modal_endpoint(modal_url, image_path, prompt_mode="prompt_layout_all_en"):
    """Test the Modal OCR endpoint with the benchmark image"""
    print(f"Testing Modal endpoint: {modal_url}")
    print(f"Image: {image_path}")
    print(f"Prompt mode: {prompt_mode}")
    
    # Convert image to base64
    image_b64 = image_to_base64(image_path)
    
    # Prepare request
    request_data = {
        "image": image_b64,
        "prompt_mode": prompt_mode,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    try:
        print("Sending request to Modal...")
        response = requests.post(
            modal_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result["result"]
            else:
                print(f"Modal returned error: {result.get('error', 'Unknown error')}")
                return None
        else:
            print(f"HTTP error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def load_ground_truth(data_dir="benchmark/data", example_id="000"):
    """Load ground truth data for comparison"""
    data_path = Path(data_dir)
    
    # Load ground truth JSON (it's double-encoded)
    json_path = data_path / f"example_{example_id}_truth.json"
    with open(json_path, "r", encoding="utf-8") as f:
        raw_content = f.read().strip()
        # First parse to get the JSON string, then parse that JSON
        json_string = json.loads(raw_content)
        truth_json = json.loads(json_string)
    
    # Load ground truth markdown
    md_path = data_path / f"example_{example_id}_truth.md"
    with open(md_path, "r", encoding="utf-8") as f:
        truth_md = f.read()
    
    # Load metadata
    metadata_path = data_path / f"example_{example_id}_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return truth_json, truth_md, metadata


def save_results(modal_result, ground_truth, metadata, results_dir="benchmark/results"):
    """Save test results for analysis"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_path / f"test_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "metadata": metadata,
        "ground_truth": ground_truth,
        "modal_result": modal_result,
        "modal_result_parsed": None
    }
    
    # Try to parse Modal result as JSON
    if modal_result:
        try:
            modal_json = json.loads(modal_result)
            results["modal_result_parsed"] = modal_json
        except json.JSONDecodeError:
            print("Modal result is not valid JSON")
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {result_file}")
    return str(result_file)


def main():
    """Main test function"""
    print("OCR Benchmark - Single Example Test")
    print("=" * 40)
    
    # Configuration
    modal_url = "https://marker--dotsocr-v2.modal.run"  # Update this with your actual URL
    image_path = "benchmark/data/example_000.png"
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        print("Please run 'python benchmark/load_example.py' first to download the example.")
        return
    
    # Load ground truth
    print("Loading ground truth data...")
    try:
        truth_json, truth_md, metadata = load_ground_truth()
        print(f"Ground truth loaded: {len(truth_json['metal_rankings'])} medal rankings")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return
    
    # Test with simple OCR extraction
    print("\n--- Testing with prompt_ocr ---")
    modal_result = test_modal_endpoint(modal_url, image_path, "prompt_ocr")
    
    if modal_result:
        print(f"Modal result length: {len(modal_result)} characters")
        print("First 500 characters:")
        print(modal_result[:500])
        
        # Save results
        result_file = save_results(modal_result, truth_json, metadata)
        
        print(f"\nTest completed! Check results in: {result_file}")
        
        # Basic comparison
        try:
            modal_json = json.loads(modal_result)
            if "metal_rankings" in modal_json:
                modal_count = len(modal_json["metal_rankings"])
                truth_count = len(truth_json["metal_rankings"])
                print(f"\nQuick comparison:")
                print(f"Ground truth entries: {truth_count}")
                print(f"Modal result entries: {modal_count}")
            else:
                print("Modal result doesn't contain 'metal_rankings' field")
        except json.JSONDecodeError:
            print("Modal result is not valid JSON - may need different prompt mode")
    else:
        print("Test failed - no result from Modal endpoint")


if __name__ == "__main__":
    main()