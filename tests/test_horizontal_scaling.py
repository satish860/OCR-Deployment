import time
import requests
import concurrent.futures
import threading
from datetime import datetime

def test_single_request(endpoint, request_id):
    """Send a single request and track timing"""
    thread_name = threading.current_thread().name
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Simple test image data (small payload for quick testing)
    test_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    request_data = {
        "image": test_image_data,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 100,  # Small for testing
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    print(f"[{timestamp}] [{thread_name}] Request {request_id}: Starting...")
    
    try:
        response = requests.post(endpoint, json=request_data, timeout=60)
        elapsed = time.time() - start_time
        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if response.status_code == 200:
            print(f"[{end_timestamp}] [{thread_name}] Request {request_id}: ✅ SUCCESS in {elapsed:.2f}s")
            return {
                "request_id": request_id,
                "success": True,
                "elapsed": elapsed,
                "thread_name": thread_name,
                "start_time": timestamp,
                "end_time": end_timestamp,
                "status_code": response.status_code
            }
        else:
            print(f"[{end_timestamp}] [{thread_name}] Request {request_id}: ❌ FAILED - HTTP {response.status_code}")
            return {
                "request_id": request_id,
                "success": False,
                "elapsed": elapsed,
                "thread_name": thread_name,
                "error": f"HTTP {response.status_code}",
                "start_time": timestamp,
                "end_time": end_timestamp
            }
            
    except Exception as e:
        elapsed = time.time() - start_time
        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{end_timestamp}] [{thread_name}] Request {request_id}: ❌ EXCEPTION - {e}")
        return {
            "request_id": request_id,
            "success": False,
            "elapsed": elapsed,
            "thread_name": thread_name,
            "error": str(e),
            "start_time": timestamp,
            "end_time": end_timestamp
        }

def test_horizontal_scaling():
    """Test that multiple containers are being used for parallel requests"""
    print("=" * 80)
    print("HORIZONTAL SCALING VERIFICATION TEST")
    print("=" * 80)
    
    # Test both old and new endpoints to compare
    endpoints_to_test = [
        {
            "name": "V2 (Shared Model - Should Queue)",
            "url": "https://marker--dotsocr-v2.modal.run",
            "expected": "Sequential processing, long total time"
        },
        {
            "name": "V3 (Horizontal Scaling - Should Parallelize)", 
            "url": "https://marker--dotsocr-v3.modal.run",
            "expected": "Parallel processing, short total time"
        }
    ]
    
    num_requests = 5  # Test with 5 concurrent requests
    
    for endpoint_config in endpoints_to_test:
        endpoint_name = endpoint_config["name"]
        endpoint_url = endpoint_config["url"]
        expected_behavior = endpoint_config["expected"]
        
        print(f"\n🧪 Testing: {endpoint_name}")
        print(f"📍 URL: {endpoint_url}")
        print(f"🎯 Expected: {expected_behavior}")
        print("-" * 60)
        
        # Record overall start time
        overall_start = time.time()
        start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        print(f"[{start_timestamp}] Launching {num_requests} concurrent requests...")
        
        # Launch concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = []
            for i in range(num_requests):
                future = executor.submit(test_single_request, endpoint_url, i+1)
                futures.append(future)
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        overall_elapsed = time.time() - overall_start
        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Analyze results
        results.sort(key=lambda x: x["request_id"])
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"\n📊 RESULTS for {endpoint_name}:")
        print(f"⏱️ Total time: {overall_elapsed:.2f}s")
        print(f"✅ Successful: {len(successful)}/{num_requests}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            individual_times = [r["elapsed"] for r in successful]
            max_time = max(individual_times)
            min_time = min(individual_times)
            avg_time = sum(individual_times) / len(individual_times)
            
            print(f"📈 Individual request times:")
            print(f"   Max: {max_time:.2f}s")
            print(f"   Min: {min_time:.2f}s") 
            print(f"   Avg: {avg_time:.2f}s")
            
            # Key metric: If horizontal scaling works, total time should be close to max individual time
            # If queuing, total time will be much larger than max individual time
            efficiency = max_time / overall_elapsed if overall_elapsed > 0 else 0
            
            print(f"\n🔍 Scaling Analysis:")
            print(f"   Efficiency ratio: {efficiency:.2f}")
            if efficiency > 0.8:
                print(f"   ✅ EXCELLENT - Requests processed in parallel!")
            elif efficiency > 0.5:
                print(f"   🟡 GOOD - Some parallelism detected")
            else:
                print(f"   ❌ POOR - Requests appear to be queuing sequentially")
            
            # Show request timeline
            print(f"\n⏰ Request Timeline:")
            for result in results:
                status = "✅" if result["success"] else "❌"
                print(f"   Request {result['request_id']}: {result['start_time']} → {result['end_time']} ({result['elapsed']:.2f}s) {status}")
        
        print("=" * 60)

def test_health_endpoint():
    """Quick test of health endpoint"""
    print("\n🏥 Testing Health Endpoint...")
    
    health_url = "https://marker--health-v3.modal.run"
    
    try:
        response = requests.get(health_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check exception: {e}")

if __name__ == "__main__":
    print("Testing Modal Horizontal Scaling Configuration")
    test_health_endpoint()
    test_horizontal_scaling()
    
    print("\n" + "=" * 80)
    print("🎯 WHAT TO LOOK FOR:")
    print("• V2 should show sequential processing (low efficiency ratio)")
    print("• V3 should show parallel processing (high efficiency ratio)")
    print("• V3 total time should be close to individual request time")
    print("• Multiple containers should be visible in Modal dashboard")
    print("=" * 80)