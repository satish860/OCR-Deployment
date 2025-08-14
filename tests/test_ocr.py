import requests
import base64
import json

def test_health_check():
    """Test the health check endpoint"""
    # Replace with your actual Modal endpoint URL after deployment
    health_url = "https://your-username--dots-ocr-health.modal.run"
    
    try:
        response = requests.get(health_url, timeout=30)
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_ocr_with_sample_text():
    """Test OCR with a simple text request"""
    # Replace with your actual Modal endpoint URL after deployment
    ocr_url = "https://your-username--dots-ocr-dots-ocr-api.modal.run"
    
    # Sample request (you'll need to add a real base64 image)
    test_request = {
        "model": "dots.ocr",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,your-base64-image-here"
                    }
                }
            ]
        }]
    }
    
    try:
        response = requests.post(ocr_url, json=test_request, timeout=60)
        print(f"OCR test status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"OCR test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing OCR deployment...")
    
    print("\n1. Testing health check...")
    health_ok = test_health_check()
    
    print("\n2. Testing OCR endpoint...")
    ocr_ok = test_ocr_with_sample_text()
    
    if health_ok and ocr_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the deployment.")