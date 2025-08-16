import requests
import time
import base64
from pathlib import Path

# Modal endpoint
GENERATE_URL = "https://marker--dotsocr-v2.modal.run"

def load_test_image():
    """Load a simple test image"""
    import fitz  # PyMuPDF
    
    pdf_path = Path("input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf")
    
    if pdf_path.exists():
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        image_b64 = base64.b64encode(img_data).decode('utf-8')
        doc.close()
        return image_b64
    
    # Fallback
    minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    return minimal_png

def test_sequential_requests(num_requests=5):
    """Test sequential requests to measure startup vs steady-state performance"""
    
    print(f"Testing {num_requests} sequential requests to measure startup overhead...")
    
    image_b64 = load_test_image()
    
    payload = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    timings = []
    
    for i in range(num_requests):
        print(f"\nRequest {i+1}:")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                GENERATE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            end_time = time.time()
            duration = end_time - start_time
            timings.append(duration)
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"  Success: {len(result.get('result', ''))} chars")
                else:
                    print(f"  Error: {result.get('error', 'Unknown')}")
            else:
                print(f"  HTTP Error: {response.text[:100]}")
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            timings.append(duration)
            print(f"  Exception after {duration:.2f}s: {e}")
        
        # Short pause between requests
        if i < num_requests - 1:
            print("  Waiting 2 seconds...")
            time.sleep(2)
    
    # Analysis
    if timings:
        print(f"\n{'='*50}")
        print("TIMING ANALYSIS")
        print(f"{'='*50}")
        
        for i, timing in enumerate(timings):
            print(f"Request {i+1}: {timing:.2f}s")
        
        print(f"\nFirst request:    {timings[0]:.2f}s")
        if len(timings) > 1:
            avg_subsequent = sum(timings[1:]) / len(timings[1:])
            print(f"Avg subsequent:   {avg_subsequent:.2f}s")
            print(f"Startup overhead: {timings[0] - avg_subsequent:.2f}s")
            
            if timings[0] > avg_subsequent + 2:
                print(f"🔥 HIGH STARTUP OVERHEAD DETECTED!")
                print(f"   First request is {timings[0] - avg_subsequent:.1f}s slower")
            else:
                print(f"✅ Startup overhead is minimal")
        
        min_time = min(timings)
        max_time = max(timings)
        avg_time = sum(timings) / len(timings)
        
        print(f"\nSummary:")
        print(f"  Fastest: {min_time:.2f}s")
        print(f"  Slowest: {max_time:.2f}s") 
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Range:   {max_time - min_time:.2f}s")

def test_rapid_fire_requests(num_requests=3):
    """Test rapid-fire requests with minimal delay"""
    
    print(f"\nTesting {num_requests} rapid-fire requests (no delay)...")
    
    image_b64 = load_test_image()
    
    payload = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    timings = []
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.post(
                GENERATE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            end_time = time.time()
            duration = end_time - start_time
            timings.append(duration)
            
            print(f"Rapid request {i+1}: {duration:.2f}s - Status {response.status_code}")
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            timings.append(duration)
            print(f"Rapid request {i+1}: {duration:.2f}s - Exception: {e}")
    
    if timings:
        print(f"\nRapid-fire timings: {[f'{t:.2f}s' for t in timings]}")
        print(f"All similar? {max(timings) - min(timings) < 2.0}")

if __name__ == "__main__":
    print("MODAL STARTUP TIMING ANALYSIS")
    print("=" * 60)
    
    test_sequential_requests(5)
    test_rapid_fire_requests(3)
    
    print(f"\n🔍 DIAGNOSIS:")
    print(f"If first request is much slower: Container cold start issue")
    print(f"If all requests slow: Modal/network latency issue") 
    print(f"If rapid-fire faster: Container warm-up between requests")