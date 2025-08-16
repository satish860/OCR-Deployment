import requests
import base64
import time
import json
from pathlib import Path

# Modal endpoints
HEALTH_URL = "https://marker--health-v2.modal.run"
GENERATE_URL = "https://marker--dotsocr-v2.modal.run"

def load_test_image():
    """Load a test image and convert to base64"""
    import fitz  # PyMuPDF
    
    # Use the specific PDF file and extract first page as image
    pdf_path = Path("input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf")
    
    if pdf_path.exists():
        print(f"Using test PDF: {pdf_path}")
        
        # Open PDF and get first page
        doc = fitz.open(pdf_path)
        page = doc[0]  # First page
        
        # Render page as image (PNG)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        
        # Convert to base64
        image_b64 = base64.b64encode(img_data).decode('utf-8')
        
        doc.close()
        print(f"Extracted first page as PNG, size: {len(img_data)} bytes")
        return image_b64
    
    # Fallback to minimal PNG if PDF not found
    minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    print("PDF not found, using minimal test PNG (1x1 white pixel)")
    return minimal_png

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    print("(This may take several minutes to warm up the container...)")
    
    try:
        start_time = time.time()
        response = requests.get(HEALTH_URL, timeout=600)  # 10 minutes for container startup
        response_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f}s")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Health check PASSED")
            print(f"Container is now warmed up after {response_time:.2f}s")
            return True
        else:
            print("Health check FAILED")
            return False
            
    except Exception as e:
        print(f"Health check ERROR: {e}")
        print("Note: Health check timeout suggests container may be cold starting")
        return False

def test_single_page_ocr(image_b64, test_name="Single Page OCR", prompt_mode="prompt_layout_all_en", bbox=None):
    """Test single page OCR with different prompt modes"""
    print(f"\n=== Testing {test_name} ===")
    
    payload = {
        "image": image_b64,
        "prompt_mode": prompt_mode,
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    # Add bbox for grounding OCR
    if bbox:
        payload["bbox"] = bbox
    
    try:
        start_time = time.time()
        response = requests.post(
            GENERATE_URL, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for potential cold start
        )
        response_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f}s")
        print(f"Prompt Mode: {prompt_mode}")
        if bbox:
            print(f"Bounding Box: {bbox}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', 'N/A')}")
            if result.get('success'):
                ocr_result = result.get('result', '')
                print(f"OCR Result Length: {len(ocr_result)}")
                print(f"OCR Result Preview: {ocr_result[:100]}...")
                print(f"{test_name} PASSED")
                return True, response_time
            else:
                print(f"OCR failed: {result.get('error', 'Unknown error')}")
                return False, response_time
        else:
            print(f"HTTP Error: {response.text}")
            return False, response_time
            
    except Exception as e:
        print(f"{test_name} ERROR: {e}")
        return False, 0

def test_invalid_prompt_mode(image_b64):
    """Test invalid prompt mode handling"""
    print(f"\n=== Testing Invalid Prompt Mode ===")
    
    payload = {
        "image": image_b64,
        "prompt_mode": "invalid_mode",
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            GENERATE_URL, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if not result.get('success') and 'Invalid prompt_mode' in result.get('error', ''):
                print("Invalid prompt mode validation PASSED")
                return True
            else:
                print(f"Unexpected success: {result}")
                return False
        else:
            print(f"HTTP Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Test ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting Modal Endpoint Tests")
    print(f"Health URL: {HEALTH_URL}")
    print(f"Generate URL: {GENERATE_URL}")
    
    # Test 1: Health Check
    health_ok = test_health_endpoint()
    
    if not health_ok:
        print("\nHealth check failed, skipping OCR tests")
        return
    
    # Load test image
    print("\n=== Loading Test Image ===")
    image_b64 = load_test_image()
    print(f"Image base64 length: {len(image_b64)}")
    
    # Test 2: Test all prompt modes
    prompt_modes = [
        ("prompt_layout_all_en", "Full Layout + Text", None),
        ("prompt_layout_only_en", "Layout Only", None),
        ("prompt_ocr", "Simple OCR", None),
        ("prompt_grounding_ocr", "Grounding OCR", [100, 100, 500, 400])  # Example bbox
    ]
    
    print("\n=== Testing Different Prompt Modes ===")
    test_times = []
    
    for prompt_mode, description, bbox in prompt_modes:
        success, response_time = test_single_page_ocr(
            image_b64, 
            f"{description} ({prompt_mode})", 
            prompt_mode, 
            bbox
        )
        test_times.append((description, response_time, success))
        if not success:
            print(f"FAILED: {description}")
        time.sleep(1)  # Brief pause between tests
    
    # Test 3: Test invalid prompt mode
    test_invalid_prompt_mode(image_b64)
    
    # Test 4: Since health endpoint warmed up the container, test warm vs warm
    print("\n=== Performance Testing (Warm Container) ===")
    
    success_1, time_1 = test_single_page_ocr(image_b64, "First Warm Request")
    
    if success_1:
        success_2, time_2 = test_single_page_ocr(image_b64, "Second Warm Request")
        
        if success_2:
            print(f"\nWarm Container Performance:")
            print(f"First Request: {time_1:.2f}s")
            print(f"Second Request: {time_2:.2f}s")
            print(f"Consistency: {abs(time_1-time_2):.2f}s difference")
    
    # Summary of all test times
    print(f"\n=== Test Summary ===")
    print("Prompt Mode Performance:")
    for description, response_time, success in test_times:
        status = "PASS" if success else "FAIL"
        print(f"  {description:<20}: {response_time:>6.2f}s [{status}]")
        
    print(f"\nAll prompt mode tests completed!")

if __name__ == "__main__":
    main()