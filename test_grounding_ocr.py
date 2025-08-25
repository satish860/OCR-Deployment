import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr_deployment.client import OCRClient


def test_grounding_ocr():
    """Test prompt_grounding_ocr with a specific bounding box"""
    
    print("Testing Grounding OCR Functionality")
    print("=" * 50)
    
    # Configuration
    process_url = "https://marker--process.modal.run"
    health_url = "https://marker--health.modal.run"
    pdf_path = "input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf"
    
    client = OCRClient(process_url, health_url)
    
    # Check health first
    print("1. Checking service health...")
    health = client.check_health()
    if not health["healthy"]:
        print(f"   Service unhealthy: {health['error']}")
        return
    print("   Service is healthy")
    
    # Convert first page to base64
    print("\n2. Converting first page to base64...")
    page_b64, page_size = client.pdf_page_to_base64(pdf_path, 0, dpi=100)
    print(f"   Page converted - Size: {page_size[0]}x{page_size[1]}")
    
    # Test grounding OCR with a sample bounding box
    # This bounding box should target a specific region of the page
    # Format: [x1, y1, x2, y2] - coordinates of the region to OCR
    test_bbox = [200, 300, 600, 500]  # Example coordinates
    
    print(f"\n3. Testing basic OCR functionality...")
    start_time = time.time()
    
    # Test with prompt_ocr mode first to verify service is working
    result = client.process_image(
        image_b64=page_b64,
        prompt_mode="prompt_ocr",
        max_tokens=1000,
        temperature=0.0
    )
    
    processing_time = time.time() - start_time
    
    if result.get("success"):
        extracted_text = result.get("result", "")
        print("   ✅ SUCCESS!")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Extracted text length: {len(extracted_text)} characters")
        print(f"\n   EXTRACTED TEXT:")
        print(f"   {'-' * 40}")
        print(f"   {extracted_text[:500]}...")  # Show first 500 chars
        print(f"   {'-' * 40}")
        
        print("\n4. Basic OCR is working! Simple text extraction successful.")
            
    else:
        print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    try:
        test_grounding_ocr()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()