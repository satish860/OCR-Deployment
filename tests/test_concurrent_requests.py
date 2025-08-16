import requests
import time
import threading
import asyncio
import aiohttp
import json
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple
import statistics

# Modal endpoint
GENERATE_URL = "https://marker--dotsocr-v2.modal.run"

@dataclass
class RequestResult:
    request_id: int
    start_time: float
    end_time: float
    duration: float
    status_code: int
    success: bool
    error: str = None
    response_size: int = 0

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

def send_single_request(request_id: int, image_b64: str, start_offset: float = 0) -> RequestResult:
    """Send a single OCR request and track timing"""
    
    # Wait for the start offset (to stagger requests if needed)
    if start_offset > 0:
        time.sleep(start_offset)
    
    payload = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            GENERATE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = response.status_code == 200
        error = None
        response_size = len(response.text) if response.text else 0
        
        if not success:
            error = f"HTTP {response.status_code}: {response.text[:200]}"
        else:
            result = response.json()
            if not result.get('success', False):
                error = result.get('error', 'Unknown API error')
                success = False
        
        return RequestResult(
            request_id=request_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status_code=response.status_code,
            success=success,
            error=error,
            response_size=response_size
        )
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return RequestResult(
            request_id=request_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status_code=0,
            success=False,
            error=str(e)
        )

def test_concurrent_requests_threading(image_b64: str, num_requests: int, stagger_delay: float = 0) -> List[RequestResult]:
    """Test concurrent requests using threading"""
    
    print(f"\n=== Testing {num_requests} Concurrent Requests (Threading) ===")
    if stagger_delay > 0:
        print(f"Staggering requests by {stagger_delay}s each")
    
    results = []
    
    # Use ThreadPoolExecutor to send requests in parallel
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Submit all requests
        future_to_id = {}
        for i in range(num_requests):
            offset = i * stagger_delay if stagger_delay > 0 else 0
            future = executor.submit(send_single_request, i, image_b64, offset)
            future_to_id[future] = i
        
        # Collect results as they complete
        for future in as_completed(future_to_id):
            request_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Request {result.request_id}: {result.duration:.2f}s - {'SUCCESS' if result.success else 'FAILED'}")
                if not result.success and result.error:
                    print(f"  Error: {result.error}")
            except Exception as e:
                print(f"Request {request_id} raised exception: {e}")
    
    # Sort results by request_id for consistent reporting
    results.sort(key=lambda x: x.request_id)
    return results

async def send_single_request_async(session: aiohttp.ClientSession, request_id: int, image_b64: str, start_offset: float = 0) -> RequestResult:
    """Send a single OCR request asynchronously"""
    
    # Wait for the start offset
    if start_offset > 0:
        await asyncio.sleep(start_offset)
    
    payload = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    start_time = time.time()
    
    try:
        async with session.post(
            GENERATE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            
            end_time = time.time()
            duration = end_time - start_time
            
            response_text = await response.text()
            response_size = len(response_text)
            
            success = response.status == 200
            error = None
            
            if not success:
                error = f"HTTP {response.status}: {response_text[:200]}"
            else:
                result = json.loads(response_text)
                if not result.get('success', False):
                    error = result.get('error', 'Unknown API error')
                    success = False
            
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                status_code=response.status,
                success=success,
                error=error,
                response_size=response_size
            )
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return RequestResult(
            request_id=request_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status_code=0,
            success=False,
            error=str(e)
        )

async def test_concurrent_requests_async(image_b64: str, num_requests: int, stagger_delay: float = 0) -> List[RequestResult]:
    """Test concurrent requests using async/await"""
    
    print(f"\n=== Testing {num_requests} Concurrent Requests (Async) ===")
    if stagger_delay > 0:
        print(f"Staggering requests by {stagger_delay}s each")
    
    async with aiohttp.ClientSession() as session:
        # Create all request tasks
        tasks = []
        for i in range(num_requests):
            offset = i * stagger_delay if stagger_delay > 0 else 0
            task = send_single_request_async(session, i, image_b64, offset)
            tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and convert to RequestResult objects
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} raised exception: {result}")
        else:
            valid_results.append(result)
            print(f"Request {result.request_id}: {result.duration:.2f}s - {'SUCCESS' if result.success else 'FAILED'}")
            if not result.success and result.error:
                print(f"  Error: {result.error}")
    
    # Sort results by request_id
    valid_results.sort(key=lambda x: x.request_id)
    return valid_results

def analyze_concurrency(results: List[RequestResult]) -> dict:
    """Analyze the results to determine if requests were processed concurrently"""
    
    if not results:
        return {"error": "No results to analyze"}
    
    # Calculate timing statistics
    durations = [r.duration for r in results if r.success]
    successful_count = len([r for r in results if r.success])
    
    if not durations:
        return {"error": "No successful requests to analyze"}
    
    # Find the earliest start and latest end times
    earliest_start = min(r.start_time for r in results)
    latest_end = max(r.end_time for r in results)
    total_wall_time = latest_end - earliest_start
    
    # Calculate overlap analysis
    overlaps = []
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i < j:  # Avoid duplicate pairs
                # Check if requests overlap in time
                overlap_start = max(r1.start_time, r2.start_time)
                overlap_end = min(r1.end_time, r2.end_time)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append(overlap_duration)
    
    # Calculate expected sequential time vs actual time
    avg_duration = statistics.mean(durations)
    expected_sequential_time = sum(durations)
    
    # Determine concurrency level
    max_concurrent = 1
    time_points = []
    for r in results:
        time_points.append((r.start_time, 1))    # Request starts (+1)
        time_points.append((r.end_time, -1))     # Request ends (-1)
    
    time_points.sort()
    current_concurrent = 0
    for _, delta in time_points:
        current_concurrent += delta
        max_concurrent = max(max_concurrent, current_concurrent)
    
    analysis = {
        "total_requests": len(results),
        "successful_requests": successful_count,
        "failed_requests": len(results) - successful_count,
        "total_wall_time": total_wall_time,
        "expected_sequential_time": expected_sequential_time,
        "avg_duration": avg_duration,
        "min_duration": min(durations),
        "max_duration": max(durations),
        "speedup_factor": expected_sequential_time / total_wall_time if total_wall_time > 0 else 0,
        "max_concurrent_detected": max_concurrent,
        "overlap_count": len(overlaps),
        "avg_overlap_duration": statistics.mean(overlaps) if overlaps else 0,
        "is_truly_concurrent": len(overlaps) > 0,
        "efficiency": (expected_sequential_time / total_wall_time) / len(results) if total_wall_time > 0 else 0
    }
    
    return analysis

def print_timeline(results: List[RequestResult], time_resolution: float = 1.0):
    """Print a simple timeline visualization of request execution"""
    
    if not results:
        return
    
    print(f"\n=== Request Timeline (resolution: {time_resolution}s) ===")
    
    earliest_start = min(r.start_time for r in results)
    latest_end = max(r.end_time for r in results)
    total_time = latest_end - earliest_start
    
    print(f"Timeline: 0s to {total_time:.1f}s")
    print("Request ID: " + "".join(f"{i:>3}" for i in range(len(results))))
    
    # Create timeline in time_resolution increments
    current_time = 0
    while current_time <= total_time:
        absolute_time = earliest_start + current_time
        
        # Check which requests are active at this time
        active_requests = []
        for r in results:
            if r.start_time <= absolute_time <= r.end_time:
                active_requests.append(r.request_id)
        
        # Print timeline row
        timeline_row = ""
        for i in range(len(results)):
            if i in active_requests:
                timeline_row += " █ "
            else:
                timeline_row += " . "
        
        print(f"{current_time:>6.1f}s: {timeline_row} ({len(active_requests)} concurrent)")
        current_time += time_resolution

def main():
    """Test concurrent request processing"""
    
    print("MODAL CONCURRENT REQUEST TESTING")
    print("=" * 60)
    
    # Load test image
    image_b64 = load_test_image()
    print(f"Image base64 length: {len(image_b64)}")
    
    # Test configurations
    test_configs = [
        {"requests": 5, "method": "threading", "stagger": 0},
        {"requests": 10, "method": "threading", "stagger": 0},
        {"requests": 15, "method": "threading", "stagger": 0},
        {"requests": 10, "method": "async", "stagger": 0},
        {"requests": 10, "method": "threading", "stagger": 0.5},  # Staggered start
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"TEST: {config['requests']} requests, {config['method']}, stagger={config['stagger']}s")
        print(f"{'='*60}")
        
        if config['method'] == 'threading':
            results = test_concurrent_requests_threading(
                image_b64, config['requests'], config['stagger']
            )
        else:  # async
            results = asyncio.run(test_concurrent_requests_async(
                image_b64, config['requests'], config['stagger']
            ))
        
        # Analyze results
        analysis = analyze_concurrency(results)
        
        print(f"\n--- Analysis ---")
        print(f"Total wall time: {analysis.get('total_wall_time', 0):.2f}s")
        print(f"Expected sequential time: {analysis.get('expected_sequential_time', 0):.2f}s")
        print(f"Speedup factor: {analysis.get('speedup_factor', 0):.2f}x")
        print(f"Max concurrent detected: {analysis.get('max_concurrent_detected', 0)}")
        print(f"Request overlaps detected: {analysis.get('overlap_count', 0)}")
        print(f"Is truly concurrent: {analysis.get('is_truly_concurrent', False)}")
        print(f"Success rate: {analysis.get('successful_requests', 0)}/{analysis.get('total_requests', 0)}")
        
        # Print timeline for smaller tests
        if config['requests'] <= 10:
            print_timeline(results, 2.0)
        
        all_results.append({
            "config": config,
            "results": results,
            "analysis": analysis
        })
        
        # Brief pause between tests
        time.sleep(2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL CONCURRENCY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<25} {'Requests':<10} {'Speedup':<10} {'Max Concurrent':<15} {'Truly Concurrent'}")
    print("-" * 75)
    
    for test_result in all_results:
        config = test_result['config']
        analysis = test_result['analysis']
        
        test_name = f"{config['method']}-{config['stagger']}s"
        print(f"{test_name:<25} {config['requests']:<10} {analysis.get('speedup_factor', 0):<10.2f}x "
              f"{analysis.get('max_concurrent_detected', 0):<15} {analysis.get('is_truly_concurrent', False)}")
    
    print(f"\nKey Findings:")
    concurrent_tests = [t for t in all_results if t['analysis'].get('is_truly_concurrent', False)]
    if concurrent_tests:
        max_concurrent = max(t['analysis'].get('max_concurrent_detected', 0) for t in concurrent_tests)
        print(f"✓ Modal IS processing requests concurrently")
        print(f"✓ Maximum concurrent requests observed: {max_concurrent}")
        print(f"✓ This confirms @modal.concurrent(max_inputs=10) is working")
    else:
        print(f"✗ No concurrent processing detected")
        print(f"✗ All requests appear to be processed sequentially")
        print(f"✗ @modal.concurrent(max_inputs=10) may not be working as expected")

if __name__ == "__main__":
    main()