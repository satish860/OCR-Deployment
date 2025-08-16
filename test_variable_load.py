import requests
import time
import threading
import asyncio
import aiohttp
import json
import base64
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple
import statistics

# Modal endpoint
GENERATE_URL = "https://marker--dotsocr-v2.modal.run"

@dataclass
class VariableLoadResult:
    request_id: int
    batch_size: int  # Number of pages in this request
    start_time: float
    end_time: float
    duration: float
    status_code: int
    success: bool
    pages_per_second: float
    error: str = None
    response_size: int = 0

def load_diverse_images(max_images: int = 1000):
    """Load diverse images from multiple PDFs"""
    import fitz  # PyMuPDF
    
    input_dir = Path("input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found, using minimal test image")
        minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        return [minimal_png] * min(max_images, 100)
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    images_b64 = []
    
    for pdf_path in pdf_files:
        if len(images_b64) >= max_images:
            break
            
        try:
            print(f"\nProcessing {pdf_path.name}...")
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            print(f"  Pages in PDF: {total_pages}")
            
            # Extract pages (limit to avoid memory issues)
            pages_to_extract = min(total_pages, max_images - len(images_b64))
            
            for page_num in range(pages_to_extract):
                if len(images_b64) >= max_images:
                    break
                    
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image_b64 = base64.b64encode(img_data).decode('utf-8')
                    images_b64.append(image_b64)
                    
                except Exception as e:
                    print(f"    Error extracting page {page_num}: {e}")
                    continue
            
            doc.close()
            print(f"  Extracted {pages_to_extract} pages")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    print(f"\nTotal images extracted: {len(images_b64)}")
    return images_b64

def generate_batch_sizes(num_requests: int, max_pages_per_request: int = 1000) -> List[int]:
    """Generate variable batch sizes for requests"""
    
    # Different batch size patterns
    patterns = [
        # Small batches (1-10 pages)
        lambda: random.randint(1, 10),
        # Medium batches (10-50 pages) 
        lambda: random.randint(10, 50),
        # Large batches (50-200 pages)
        lambda: random.randint(50, 200),
        # Very large batches (200-1000 pages)
        lambda: random.randint(200, min(1000, max_pages_per_request)),
    ]
    
    batch_sizes = []
    
    for i in range(num_requests):
        # Choose pattern based on request number to ensure variety
        if i < num_requests * 0.4:  # 40% small batches
            size = patterns[0]()
        elif i < num_requests * 0.7:  # 30% medium batches
            size = patterns[1]()
        elif i < num_requests * 0.9:  # 20% large batches
            size = patterns[2]()
        else:  # 10% very large batches
            size = patterns[3]()
        
        batch_sizes.append(size)
    
    # Shuffle to randomize order
    random.shuffle(batch_sizes)
    return batch_sizes

def send_variable_batch_request(request_id: int, batch_size: int, available_images: List[str]) -> VariableLoadResult:
    """Send a batch request with variable number of pages"""
    
    # Select random images for this batch
    if batch_size > len(available_images):
        # Repeat images if we need more than available
        images = (available_images * ((batch_size // len(available_images)) + 1))[:batch_size]
    else:
        images = random.sample(available_images, batch_size)
    
    payload = {
        "images": images,
        "prompt_mode": "prompt_layout_all_en", 
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    start_time = time.time()
    
    try:
        print(f"Request {request_id}: Starting batch of {batch_size} pages...")
        
        response = requests.post(
            GENERATE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600  # 10 minutes for large batches
        )
        
        end_time = time.time()
        duration = end_time - start_time
        pages_per_second = batch_size / duration if duration > 0 else 0
        
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
            else:
                # Verify we got results for all pages
                results = result.get('results', [])
                if len(results) != batch_size:
                    error = f"Expected {batch_size} results, got {len(results)}"
                    success = False
        
        status_text = 'SUCCESS' if success else 'FAILED'
        if not success and error:
            print(f"Request {request_id}: {duration:.1f}s for {batch_size} pages ({pages_per_second:.2f} pages/s) - {status_text}")
            print(f"  ERROR: {error}")
        else:
            print(f"Request {request_id}: {duration:.1f}s for {batch_size} pages ({pages_per_second:.2f} pages/s) - {status_text}")
        
        return VariableLoadResult(
            request_id=request_id,
            batch_size=batch_size,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status_code=response.status_code,
            success=success,
            pages_per_second=pages_per_second,
            error=error,
            response_size=response_size
        )
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        pages_per_second = 0
        
        print(f"Request {request_id}: EXCEPTION after {duration:.1f}s - {str(e)[:100]}")
        
        return VariableLoadResult(
            request_id=request_id,
            batch_size=batch_size,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status_code=0,
            success=False,
            pages_per_second=pages_per_second,
            error=str(e)
        )

def test_variable_concurrent_load(available_images: List[str], num_concurrent_requests: int, max_pages_per_request: int = 1000) -> List[VariableLoadResult]:
    """Test concurrent requests with variable batch sizes"""
    
    print(f"\n=== Testing {num_concurrent_requests} Concurrent Requests with Variable Batch Sizes ===")
    print(f"Max pages per request: {max_pages_per_request}")
    
    # Generate variable batch sizes
    batch_sizes = generate_batch_sizes(num_concurrent_requests, max_pages_per_request)
    total_pages = sum(batch_sizes)
    
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total pages across all requests: {total_pages}")
    print(f"Average batch size: {total_pages/num_concurrent_requests:.1f} pages")
    print(f"Largest batch: {max(batch_sizes)} pages")
    print(f"Smallest batch: {min(batch_sizes)} pages")
    
    results = []
    
    # Use ThreadPoolExecutor to send requests in parallel
    with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        # Submit all requests
        future_to_info = {}
        for i, batch_size in enumerate(batch_sizes):
            future = executor.submit(send_variable_batch_request, i, batch_size, available_images)
            future_to_info[future] = (i, batch_size)
        
        # Collect results as they complete
        for future in as_completed(future_to_info):
            request_id, batch_size = future_to_info[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Request {request_id} (batch size {batch_size}) raised exception: {e}")
    
    # Sort results by request_id for consistent reporting
    results.sort(key=lambda x: x.request_id)
    return results

def analyze_variable_load_results(results: List[VariableLoadResult]) -> dict:
    """Analyze variable load test results"""
    
    if not results:
        return {"error": "No results to analyze"}
    
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    if not successful_results:
        return {
            "error": "No successful requests to analyze",
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(failed_results),
            "success_rate": 0,
            "failure_details": [{"request_id": r.request_id, "batch_size": r.batch_size, "error": r.error} for r in failed_results]
        }
    
    # Calculate timing and throughput statistics
    durations = [r.duration for r in successful_results]
    batch_sizes = [r.batch_size for r in successful_results]
    pages_per_second = [r.pages_per_second for r in successful_results]
    
    # Overall metrics
    earliest_start = min(r.start_time for r in results)
    latest_end = max(r.end_time for r in results)
    total_wall_time = latest_end - earliest_start
    
    total_pages_processed = sum(r.batch_size for r in successful_results)
    overall_throughput = total_pages_processed / total_wall_time if total_wall_time > 0 else 0
    
    # Concurrency analysis
    max_concurrent = 1
    time_points = []
    for r in results:
        time_points.append((r.start_time, 1))    # Request starts (+1)
        time_points.append((r.end_time, -1))     # Request ends (-1)
    
    time_points.sort()
    current_concurrent = 0
    peak_concurrent_pages = 0
    current_pages = 0
    
    for timestamp, delta in time_points:
        if delta == 1:  # Request starting
            current_concurrent += 1
            # Find which request is starting
            starting_request = next(r for r in results if abs(r.start_time - timestamp) < 0.001)
            current_pages += starting_request.batch_size
        else:  # Request ending
            current_concurrent -= 1
            # Find which request is ending
            ending_request = next(r for r in results if abs(r.end_time - timestamp) < 0.001)
            current_pages -= ending_request.batch_size
        
        max_concurrent = max(max_concurrent, current_concurrent)
        peak_concurrent_pages = max(peak_concurrent_pages, current_pages)
    
    analysis = {
        "total_requests": len(results),
        "successful_requests": len(successful_results),
        "failed_requests": len(failed_results),
        "success_rate": len(successful_results) / len(results) * 100,
        
        # Timing metrics
        "total_wall_time": total_wall_time,
        "avg_duration": statistics.mean(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "median_duration": statistics.median(durations),
        
        # Batch size metrics
        "total_pages_processed": total_pages_processed,
        "avg_batch_size": statistics.mean(batch_sizes),
        "min_batch_size": min(batch_sizes),
        "max_batch_size": max(batch_sizes),
        "median_batch_size": statistics.median(batch_sizes),
        
        # Throughput metrics
        "overall_throughput_pages_per_sec": overall_throughput,
        "avg_individual_throughput": statistics.mean(pages_per_second),
        "min_individual_throughput": min(pages_per_second),
        "max_individual_throughput": max(pages_per_second),
        
        # Concurrency metrics
        "max_concurrent_requests": max_concurrent,
        "peak_concurrent_pages": peak_concurrent_pages,
        
        # Failure analysis
        "failure_details": [{"request_id": r.request_id, "batch_size": r.batch_size, "error": r.error} for r in failed_results]
    }
    
    return analysis

def print_variable_load_summary(results: List[VariableLoadResult], analysis: dict):
    """Print detailed summary of variable load test"""
    
    print(f"\n=== Variable Load Test Results ===")
    print(f"Requests: {analysis.get('successful_requests', 0)}/{analysis.get('total_requests', 0)} successful ({analysis.get('success_rate', 0):.1f}%)")
    
    if analysis.get('error'):
        print(f"ERROR: {analysis['error']}")
        if 'failure_details' in analysis:
            print(f"\n--- Failure Details ---")
            for failure in analysis['failure_details']:
                print(f"Request {failure['request_id']} ({failure['batch_size']} pages): {failure['error']}")
        return
    
    print(f"Total pages processed: {analysis.get('total_pages_processed', 0)}")
    print(f"Total wall time: {analysis.get('total_wall_time', 0):.2f}s")
    print(f"Overall throughput: {analysis.get('overall_throughput_pages_per_sec', 0):.2f} pages/second")
    
    print(f"\n--- Batch Size Distribution ---")
    print(f"Average batch size: {analysis['avg_batch_size']:.1f} pages")
    print(f"Smallest batch: {analysis['min_batch_size']} pages")
    print(f"Largest batch: {analysis['max_batch_size']} pages")
    print(f"Median batch size: {analysis['median_batch_size']:.1f} pages")
    
    print(f"\n--- Performance Metrics ---")
    print(f"Average request duration: {analysis['avg_duration']:.2f}s")
    print(f"Fastest request: {analysis['min_duration']:.2f}s")
    print(f"Slowest request: {analysis['max_duration']:.2f}s")
    print(f"Average individual throughput: {analysis['avg_individual_throughput']:.2f} pages/s")
    
    print(f"\n--- Concurrency Analysis ---")
    print(f"Max concurrent requests: {analysis['max_concurrent_requests']}")
    print(f"Peak concurrent pages: {analysis['peak_concurrent_pages']}")
    
    if analysis['failure_details']:
        print(f"\n--- Failures ---")
        for failure in analysis['failure_details']:
            print(f"Request {failure['request_id']} ({failure['batch_size']} pages): {failure['error'][:100]}")
    
    # Performance breakdown by batch size
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"\n--- Performance by Batch Size ---")
        print(f"{'Batch Size':<12} {'Duration(s)':<12} {'Pages/s':<10} {'Efficiency'}")
        print("-" * 50)
        
        # Group by batch size ranges
        size_ranges = [
            (1, 10, "Small (1-10)"),
            (11, 50, "Medium (11-50)"), 
            (51, 200, "Large (51-200)"),
            (201, 1000, "XLarge (201+)")
        ]
        
        for min_size, max_size, label in size_ranges:
            range_results = [r for r in successful_results if min_size <= r.batch_size <= max_size]
            if range_results:
                avg_duration = statistics.mean(r.duration for r in range_results)
                avg_throughput = statistics.mean(r.pages_per_second for r in range_results)
                efficiency = avg_throughput / (analysis['max_individual_throughput'] / 100) if analysis['max_individual_throughput'] > 0 else 0
                print(f"{label:<12} {avg_duration:<12.2f} {avg_throughput:<10.2f} {efficiency:<10.1f}%")

def main():
    """Test variable concurrent load processing"""
    
    print("VARIABLE LOAD CONCURRENT TESTING")
    print("=" * 60)
    
    # Load diverse images
    available_images = load_diverse_images(1000)
    print(f"Loaded {len(available_images)} diverse images for testing")
    
    # Test different concurrent load scenarios
    test_scenarios = [
        {"concurrent_requests": 5, "max_pages_per_request": 100},
        {"concurrent_requests": 10, "max_pages_per_request": 200}, 
        {"concurrent_requests": 15, "max_pages_per_request": 300},
        {"concurrent_requests": 20, "max_pages_per_request": 500},
        {"concurrent_requests": 25, "max_pages_per_request": 750},
        {"concurrent_requests": 30, "max_pages_per_request": 1000},
    ]
    
    all_test_results = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['concurrent_requests']} concurrent requests, max {scenario['max_pages_per_request']} pages each")
        print(f"{'='*80}")
        
        try:
            results = test_variable_concurrent_load(
                available_images, 
                scenario['concurrent_requests'],
                scenario['max_pages_per_request']
            )
            
            analysis = analyze_variable_load_results(results)
            print_variable_load_summary(results, analysis)
            
            all_test_results.append({
                "scenario": scenario,
                "results": results,
                "analysis": analysis
            })
            
        except Exception as e:
            print(f"ERROR in scenario: {e}")
            continue
        
        # Brief pause between scenarios
        print(f"\nWaiting 10 seconds before next scenario...")
        time.sleep(10)
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL SCENARIO COMPARISON")
    print(f"{'='*80}")
    print(f"{'Scenario':<25} {'Success Rate':<12} {'Total Pages':<12} {'Throughput':<15} {'Max Concurrent'}")
    print("-" * 80)
    
    for test_result in all_test_results:
        scenario = test_result['scenario']
        analysis = test_result['analysis']
        
        scenario_name = f"{scenario['concurrent_requests']}req×{scenario['max_pages_per_request']}p"
        success_rate = f"{analysis.get('success_rate', 0):.1f}%"
        total_pages = analysis.get('total_pages_processed', 0)
        throughput = f"{analysis.get('overall_throughput_pages_per_sec', 0):.1f} p/s"
        max_concurrent = analysis.get('max_concurrent_requests', 0)
        
        print(f"{scenario_name:<25} {success_rate:<12} {total_pages:<12} {throughput:<15} {max_concurrent}")
    
    # Find optimal configuration
    if all_test_results:
        successful_tests = [t for t in all_test_results if t['analysis'].get('success_rate', 0) > 90]
        if successful_tests:
            best_test = max(successful_tests, key=lambda x: x['analysis'].get('overall_throughput_pages_per_sec', 0))
            best_scenario = best_test['scenario']
            best_analysis = best_test['analysis']
            
            print(f"\n🏆 OPTIMAL CONFIGURATION FOUND:")
            print(f"   {best_scenario['concurrent_requests']} concurrent requests")
            print(f"   Max {best_scenario['max_pages_per_request']} pages per request")
            print(f"   Achieved {best_analysis['overall_throughput_pages_per_sec']:.1f} pages/second")
            print(f"   {best_analysis['success_rate']:.1f}% success rate")
            print(f"   Peak {best_analysis['max_concurrent_requests']} concurrent requests")

if __name__ == "__main__":
    main()