import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_deployment.client import OCRClient


def test_ocr_client():
    """Simple test of the clean OCR client"""
    
    # Configuration
    process_url = "https://marker--process.modal.run"
    health_url = "https://marker--health.modal.run"
    pdf_path = "input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf"
    
    print("Testing OCR Client")
    print("=" * 50)
    
    # Initialize client
    client = OCRClient(process_url, health_url)
    
    # 1. Check health
    print("1. Checking service health...")
    health = client.check_health()
    if health["healthy"]:
        print(f"   Service is healthy: {health['details']['status']}")
        print(f"   Model loaded: {health['details'].get('model_loaded', 'unknown')}")
    else:
        print(f"   Service unhealthy: {health['error']}")
        return
    
    # 2. Process single PDF page
    print("\n2. Processing single PDF page...")
    start_time = time.time()
    result = client.process_pdf_page(
        pdf_path=pdf_path,
        page_num=0,
        prompt_mode="prompt_layout_all_en",
        dpi=100
    )
    
    if result.get("success"):
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   Result length: {len(result.get('result', '')):,} characters")
        print(f"   Image size: {result.get('image_size', 'unknown')}")
        print(f"   Processing method: {result.get('processing_method', 'unknown')}")
        
        # Save result
        output_file = "page_0_result_client.md"
        if client.save_result(result, output_file):
            print(f"   Result saved to: {output_file}")
        
        # Show preview
        content = result.get("result", "")
        print(f"\n   Preview (first 300 chars):")
        print("   " + "-" * 40)
        print("   " + content[:300] + ("..." if len(content) > 300 else ""))
        print("   " + "-" * 40)
    else:
        print(f"   Processing failed: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    print(f"\n3. Total test time: {total_time:.2f}s")
    print(f"   Available prompt modes: {', '.join(client.get_available_prompts())}")


def test_batch_vs_sequential():
    """Test batch processing vs sequential processing with 10 pages"""
    
    print("\n\nTesting Batch vs Sequential Processing (10 pages)")
    print("=" * 60)
    
    # Configuration
    process_url = "https://marker--process.modal.run"
    pdf_path = "input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf"
    num_pages = 10
    
    client = OCRClient(process_url)
    
    # Convert multiple pages to base64
    print("1. Converting PDF pages to base64...")
    images = []
    for page_num in range(num_pages):
        try:
            # Cycle through available pages (PDF might have fewer than 10 pages)
            actual_page = page_num % 3  # Assuming PDF has at least 3 pages
            image_b64, size = client.pdf_page_to_base64(pdf_path, actual_page, dpi=100)
            images.append(image_b64)
            print(f"   Page {page_num} (using PDF page {actual_page}): {len(image_b64):,} chars, size {size}")
        except Exception as e:
            print(f"   Failed to convert page {page_num}: {e}")
            # Try to continue with fewer pages
            break
    
    if not images:
        print("   No images converted successfully")
        return
        
    print(f"   Successfully converted {len(images)} images")
    
    # Test 1: Batch Processing
    print(f"\n2. BATCH PROCESSING: Processing {len(images)} pages in single batch...")
    start_batch = time.time()
    batch_result = client.process_batch(
        images_b64=images,
        prompt_mode="prompt_layout_all_en"
    )
    batch_time = time.time() - start_batch
    
    batch_success_count = 0
    if batch_result.get("success"):
        print(f"   Batch processing time: {batch_result.get('processing_time', 0):.2f}s")
        print(f"   Total pages: {batch_result.get('total_pages', 0)}")
        print(f"   Processing mode: {batch_result.get('processing_mode', 'unknown')}")
        
        # Count successful results
        for i, page_result in enumerate(batch_result.get("results", [])):
            if page_result.get("success"):
                batch_success_count += 1
                content_len = len(page_result.get("result", ""))
                print(f"   Page {i}: {content_len:,} characters")
            else:
                print(f"   Page {i}: FAILED - {page_result.get('error', 'Unknown error')}")
    else:
        print(f"   Batch processing failed: {batch_result.get('error', 'Unknown error')}")
    
    # Test 2: Sequential Processing
    print(f"\n3. SEQUENTIAL PROCESSING: Processing {len(images)} pages one by one...")
    sequential_results = []
    start_sequential = time.time()
    
    for i, image_b64 in enumerate(images):
        print(f"   Processing page {i}...")
        page_start = time.time()
        result = client.process_image(
            image_b64=image_b64,
            prompt_mode="prompt_layout_all_en"
        )
        page_time = time.time() - page_start
        
        if result.get("success"):
            content_len = len(result.get("result", ""))
            print(f"     Page {i}: {content_len:,} characters in {page_time:.2f}s")
            sequential_results.append(result)
        else:
            print(f"     Page {i}: FAILED - {result.get('error', 'Unknown error')}")
    
    sequential_time = time.time() - start_sequential
    sequential_success_count = len(sequential_results)
    
    # Performance Comparison
    print(f"\n4. PERFORMANCE COMPARISON:")
    print(f"   {'Method':<20} {'Time':<12} {'Success':<10} {'Avg/Page':<12} {'Speedup':<10}")
    print(f"   {'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    print(f"   {'Batch':<20} {batch_time:<11.2f}s {batch_success_count:<9} {batch_time/max(1,batch_success_count):<11.2f}s {'1.0x':<10}")
    
    if sequential_time > 0:
        speedup = sequential_time / batch_time
        print(f"   {'Sequential':<20} {sequential_time:<11.2f}s {sequential_success_count:<9} {sequential_time/max(1,sequential_success_count):<11.2f}s {speedup:<9.1f}x")
        
        if speedup > 1:
            print(f"\n   🚀 Batch processing is {speedup:.1f}x FASTER than sequential!")
        else:
            print(f"\n   ⚠️  Sequential processing is {1/speedup:.1f}x faster (unexpected)")
    
    print(f"\n5. SUMMARY:")
    print(f"   - Batch processing: {batch_time:.2f}s total ({batch_time/max(1,len(images)):.2f}s per page)")
    print(f"   - Sequential processing: {sequential_time:.2f}s total ({sequential_time/max(1,len(images)):.2f}s per page)")
    print(f"   - Container warm-up effect: {'Minimal' if sequential_time/max(1,len(images)) < 3 else 'Significant'}")
    print(f"   - True batching benefit: {'Yes' if batch_time < sequential_time * 0.8 else 'Marginal'}")


if __name__ == "__main__":
    try:
        # Test single processing
        test_ocr_client()
        
        # Test batch vs sequential processing with 10 pages
        test_batch_vs_sequential()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()