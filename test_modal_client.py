import argparse
import base64
import io
import requests
import json
from PIL import Image
from pathlib import Path

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_modal_ocr(modal_url, image_path, prompt_mode="prompt_layout_all_en"):
    """Test the Modal OCR endpoint"""
    
    # Prepare the request data
    image_b64 = image_to_base64(image_path)
    
    request_data = {
        "image": image_b64,
        "prompt_mode": prompt_mode,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    # Send request to Modal endpoint
    try:
        print(f"Sending OCR request to: {modal_url}")
        print(f"Image: {image_path}")
        print(f"Prompt mode: {prompt_mode}")
        
        response = requests.post(
            modal_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                print("\n=== OCR Result ===")
                try:
                    print(result["result"])
                except UnicodeEncodeError:
                    # Save to file instead due to Windows console encoding issues
                    with open("ocr_result.txt", "w", encoding="utf-8") as f:
                        f.write(result["result"])
                    print("OCR result saved to ocr_result.txt (contains Unicode characters)")
                print("==================")
                return result["result"]
            else:
                print(f"OCR failed: {result.get('error')}")
                return None
        else:
            print(f"HTTP Error {response.status_code}: {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

def test_openai_compatible_endpoint(modal_url, image_path, prompt_mode="prompt_layout_all_en"):
    """Test using OpenAI client (this endpoint is no longer available in the new deployment)"""
    print("OpenAI-compatible endpoint is not available in the new deployment.")
    print("The new deployment uses vLLM Python API directly instead of running an OpenAI server.")
    print("Use the custom OCR endpoint instead.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Test Modal DotsOCR deployment")
    parser.add_argument("--modal_url", type=str, required=True, 
                       help="Modal endpoint URL (e.g., https://your-app-name--dotsocr-ocr.modal.run/)")
    parser.add_argument("--image_path", type=str, 
                       default="dots.ocr/demo/demo_image1.jpg",
                       help="Path to test image")
    parser.add_argument("--prompt_mode", type=str, 
                       default="prompt_layout_all_en",
                       help="Prompt mode to use")
    parser.add_argument("--test_type", type=str, 
                       choices=["custom", "openai", "both"], 
                       default="both",
                       help="Type of test to run")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    print(f"Testing Modal DotsOCR deployment")
    print(f"Modal URL: {args.modal_url}")
    print(f"Image: {args.image_path}")
    print("-" * 50)
    
    if args.test_type in ["custom", "both"]:
        print("\n1. Testing custom OCR endpoint...")
        result1 = test_modal_ocr(args.modal_url, args.image_path, args.prompt_mode)
    
    if args.test_type in ["openai", "both"]:
        print("\n2. Testing OpenAI-compatible endpoint...")
        # For OpenAI format, we need the base URL without the specific endpoint
        openai_url = args.modal_url.replace("/dotsocr-ocr", "/v1")
        result2 = test_openai_compatible_endpoint(openai_url, args.image_path, args.prompt_mode)

if __name__ == "__main__":
    main()