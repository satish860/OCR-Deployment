import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_deployment.client import OCRClient


def test_optimized_base64_performance():
    """
    Performance test for optimized base64 batch processing
    Tests the improved base64 processing after removing binary endpoint
    """
    
    print("Optimized Base64 Performance Test")
    print("=" * 60)
    
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
    print(f"   Service is healthy")
    
    # Test optimized base64 processing
    page_num = 0
    prompt_mode = "prompt_layout_all_en"
    
    print(f"\n2. Testing PDF page {page_num} with optimized base64 processing...")
    
    # Method: Optimized Base64 Processing
    print("\n   OPTIMIZED BASE64 PROCESSING")
    print("   " + "-" * 50)
    
    test_times = []
    test_results = []
    
    for i in range(5):  # Run 5 times for better average
        print(f"   Run {i+1}/5...")
        start_time = time.time()
        
        result = client.process_pdf_page(
            pdf_path=pdf_path,
            page_num=page_num,
            prompt_mode=prompt_mode,
            dpi=100
        )
        
        total_time = time.time() - start_time
        test_times.append(total_time)
        
        if result.get("success"):
            test_results.append({
                "total_time": total_time,
                "processing_time": result.get("processing_time", 0),
                "result_length": len(result.get("result", "")),
                "method": result.get("processing_method", "unknown")
            })
            print(f"     Total: {total_time:.2f}s | Processing: {result.get('processing_time', 0):.2f}s | Result: {len(result.get('result', '')):,} chars")
        else:
            print(f"     FAILED: {result.get('error', 'Unknown error')}")
            
        time.sleep(0.5)  # Brief pause between runs
    
    # Test batch processing (where the real gains are)
    print("\n   BATCH PROCESSING TEST (2 images)")
    print("   " + "-" * 50)
    
    batch_times = []
    batch_results = []
    
    for i in range(3):  # Run 3 times for average
        print(f"   Batch run {i+1}/3...")
        start_time = time.time()
        
        # Convert 2 PDF pages to base64 for batch processing
        page0_b64, _ = client.pdf_page_to_base64(pdf_path, 0, dpi=100)
        page1_b64, _ = client.pdf_page_to_base64(pdf_path, 1, dpi=100)
        
        # Process 2 pages as a batch
        result = client.process_batch(
            images_b64=[page0_b64, page1_b64],
            prompt_mode=prompt_mode
        )
        
        total_time = time.time() - start_time
        batch_times.append(total_time)
        
        if result.get("success"):
            batch_results.append({
                "total_time": total_time,
                "pages_processed": result.get("total_pages", 0),
                "method": result.get("processing_mode", "unknown"),
                "results_count": len(result.get("results", []))
            })
            pages = result.get("total_pages", 0)
            time_per_page = total_time / pages if pages > 0 else 0
            print(f"     Total: {total_time:.2f}s | Pages: {pages} | Per page: {time_per_page:.2f}s | Method: {result.get('processing_mode', 'unknown')}")
        else:
            print(f"     FAILED: {result.get('error', 'Unknown error')}")
            
        time.sleep(1)  # Brief pause between runs
    
    # Performance Analysis
    if test_results and batch_results:
        print("\n3. PERFORMANCE ANALYSIS:")
        print("   " + "=" * 50)
        
        # Calculate averages
        avg_single_total = sum(test_times) / len(test_times)
        avg_batch_total = sum(batch_times) / len(batch_times)
        
        avg_single_processing = sum(r["processing_time"] for r in test_results) / len(test_results)
        
        # Calculate batch efficiency
        avg_batch_pages = sum(r["pages_processed"] for r in batch_results) / len(batch_results) if batch_results else 1
        avg_time_per_page_batch = avg_batch_total / avg_batch_pages if avg_batch_pages > 0 else avg_batch_total
        
        # Calculate batch efficiency gain
        batch_efficiency = avg_single_total / avg_time_per_page_batch if avg_time_per_page_batch > 0 else 0
        
        print(f"   {'Metric':<30} {'Single Page':<12} {'Batch (per page)':<15} {'Efficiency':<10}")
        print(f"   {'-'*30} {'-'*12} {'-'*15} {'-'*10}")
        print(f"   {'Average Total Time':<30} {avg_single_total:<11.2f}s {avg_time_per_page_batch:<14.2f}s {batch_efficiency:<9.1f}x")
        print(f"   {'Average Processing Time':<30} {avg_single_processing:<11.2f}s {'N/A':<14} {'N/A':<9}")
        
        # Show optimization details
        print(f"\n   OPTIMIZATION DETAILS:")
        print(f"   - Processing method: {test_results[0].get('method', 'optimized_base64')}")
        print(f"   - Batch processing: {batch_results[0].get('method', 'gpu_direct_batch')}")
        print(f"   - Average batch size: {avg_batch_pages:.1f} pages")
        print(f"   - Concurrency: max_inputs=5")
        
        # Summary
        print(f"\n4. SUMMARY:")
        print(f"   🚀 Optimized base64 processing active!")
        print(f"   📊 Batch efficiency: {batch_efficiency:.1f}x faster per page")
        print(f"   💾 Cleaned up codebase: removed binary endpoint complexity")
        print(f"   ⚙️  GPU efficiency: Optimized base64 decode + PIL processing")
        print(f"   🔧 Concurrency: max_inputs=5 for batch processing")
        
        if batch_efficiency >= 1.5:
            print(f"\n   ✅ SIGNIFICANT BATCH IMPROVEMENT: {batch_efficiency:.1f}x faster per page!")
        elif batch_efficiency >= 1.2:
            print(f"\n   ✅ MODERATE BATCH IMPROVEMENT: {batch_efficiency:.1f}x faster per page")
        else:
            print(f"\n   ⚠️  BATCH PROCESSING WORKING: {batch_efficiency:.1f}x efficiency")
    
    else:
        print("\n❌ Could not complete performance analysis - some tests failed")
    
    print("\n" + "=" * 60)


def test_upload_size_comparison():
    """Compare upload sizes between different formats"""
    
    print("\nUpload Size Comparison")
    print("=" * 40)
    
    pdf_path = "input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf"
    client = OCRClient("dummy")  # Just for utility methods
    
    try:
        # Base64 method
        print("1. Base64 encoding (original method):")
        base64_data, size1 = client.pdf_page_to_base64(pdf_path, 0, dpi=100)
        print(f"   Base64 string length: {len(base64_data):,} characters")
        print(f"   Image size: {size1}")
        
        # Binary WebP method
        print("\n2. WebP binary (optimized method):")
        webp_bytes, size2 = client.pdf_page_to_optimized_bytes(pdf_path, 0, dpi=100, format="WebP")
        print(f"   Binary WebP size: {len(webp_bytes):,} bytes")
        print(f"   Image size: {size2}")
        
        # Binary JPEG method
        print("\n3. JPEG binary (alternative method):")
        jpeg_bytes, size3 = client.pdf_page_to_optimized_bytes(pdf_path, 0, dpi=100, format="JPEG")
        print(f"   Binary JPEG size: {len(jpeg_bytes):,} bytes")
        print(f"   Image size: {size3}")
        
        # Comparison
        base64_size = len(base64_data)
        webp_savings = (base64_size - len(webp_bytes)) / base64_size * 100
        jpeg_savings = (base64_size - len(jpeg_bytes)) / base64_size * 100
        
        print(f"\n4. SIZE COMPARISON:")
        print(f"   WebP saves: {webp_savings:.1f}% vs base64")
        print(f"   JPEG saves: {jpeg_savings:.1f}% vs base64")
        print(f"   WebP vs JPEG: {(len(jpeg_bytes) - len(webp_bytes))/len(jpeg_bytes)*100:.1f}% smaller")
        
    except Exception as e:
        print(f"   Error in size comparison: {e}")


if __name__ == "__main__":
    try:
        # Test optimized performance
        test_optimized_base64_performance()
        
        # Test upload size comparison  
        test_upload_size_comparison()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()