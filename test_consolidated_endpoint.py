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
    pdf_path = Path("input/2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf")
    
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
    
    try:
        start_time = time.time()
        response = requests.get(HEALTH_URL, timeout=30)
        response_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f}s")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Health check PASSED")
            return True
        else:
            print("Health check FAILED")
            return False
            
    except Exception as e:
        print(f"Health check ERROR: {e}")
        return False

def test_single_page_ocr(image_b64, test_name="Single Page OCR"):
    """Test single page OCR"""
    print(f"\n=== Testing {test_name} ===")
    
    payload = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
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
    
    # Test 2: Cold Start OCR (first request)
    print("\nWaiting 2 minutes for cold start simulation...")
    time.sleep(120)  # Wait to ensure cold start
    
    success_cold, time_cold = test_single_page_ocr(image_b64, "Cold Start OCR")
    
    if success_cold:
        # Test 3: Warm Start OCR (immediate follow-up request)
        print("\nRunning warm start test immediately...")
        success_warm, time_warm = test_single_page_ocr(image_b64, "Warm Start OCR")
        
        if success_warm:
            print(f"\nPerformance Comparison:")
            print(f"Cold Start Time: {time_cold:.2f}s")
            print(f"Warm Start Time: {time_warm:.2f}s")
            print(f"Speedup: {time_cold/time_warm:.1f}x faster")
            
            print(f"\nAll tests PASSED!")
        else:
            print(f"\nWarm start test failed")
    else:
        print(f"\nCold start test failed, skipping warm start")

if __name__ == "__main__":
    main()