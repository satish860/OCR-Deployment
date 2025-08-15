import requests
import time
import json
from pathlib import Path
import base64

# Modal endpoints
GENERATE_URL = "https://marker--dotsocr-v2.modal.run"

def load_diverse_test_images(max_images=1000):
    """Load diverse test images from multiple PDFs and pages"""
    import fitz  # PyMuPDF
    
    input_dir = Path("input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found, using minimal test PNG")
        minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        return [minimal_png]
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    images = []
    total_pages = 0
    
    # Extract pages from all PDFs
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing {pdf_file.name}...")
            doc = fitz.open(pdf_file)
            pages_in_pdf = doc.page_count
            total_pages += pages_in_pdf
            print(f"  Pages in PDF: {pages_in_pdf}")
            
            # Extract all pages from this PDF (or up to remaining limit)
            pages_to_extract = min(pages_in_pdf, max_images - len(images))
            
            for page_num in range(pages_to_extract):
                page = doc[page_num]
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image_b64 = base64.b64encode(img_data).decode('utf-8')
                images.append(image_b64)
                
                if len(images) >= max_images:
                    break
            
            doc.close()
            print(f"  Extracted {pages_to_extract} pages")
            
            if len(images) >= max_images:
                break
                
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
    
    print(f"\nTotal images extracted: {len(images)}")
    print(f"Total pages available: {total_pages}")
    
    if len(images) == 0:
        # Fallback
        minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        images = [minimal_png]
    
    # Show size distribution
    sizes = [len(img) for img in images]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    print(f"Image size stats: avg={avg_size/1024:.0f}KB, min={min_size/1024:.0f}KB, max={max_size/1024:.0f}KB")
    
    return images

def test_batch_size_limit(available_images, batch_size):
    """Test a specific batch size to see if it succeeds or fails"""
    print(f"\n=== Testing Batch Size: {batch_size} images ===")
    
    # Use diverse images, cycling through available images if needed
    if batch_size <= len(available_images):
        # Use the first batch_size images
        images = available_images[:batch_size]
        print(f"Using {batch_size} diverse images")
    else:
        # Cycle through available images to reach batch_size
        images = []
        for i in range(batch_size):
            images.append(available_images[i % len(available_images)])
        
        unique_images = len(set(images))  # Count unique images
        print(f"Using {batch_size} images ({unique_images} unique, recycled to reach batch size)")
    
    payload = {
        "images": images,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    # Calculate approximate memory usage
    total_size_bytes = sum(len(img) for img in images)
    avg_image_size_kb = (total_size_bytes / len(images)) / 1024 if images else 0
    image_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Approximate input size: {image_size_mb:.1f} MB ({batch_size} images, avg {avg_image_size_kb:.0f}KB per image)")
    
    try:
        start_time = time.time()
        response = requests.post(
            GENERATE_URL, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=900  # 15 minutes for large batch processing
        )
        total_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Time per Image: {total_time/batch_size:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                results = result.get('results', [])
                successful_results = [r for r in results if r.get('success')]
                
                print(f"SUCCESS: {len(successful_results)}/{len(results)} images processed")
                
                if successful_results:
                    avg_result_length = sum(len(r.get('result', '')) for r in successful_results) / len(successful_results)
                    print(f"Average result length: {avg_result_length:.0f} chars")
                
                return True, total_time, len(successful_results), None
            else:
                error = result.get('error', 'Unknown error')
                print(f"FAILED: {error}")
                return False, total_time, 0, error
        else:
            error_text = response.text
            print(f"HTTP ERROR {response.status_code}: {error_text}")
            return False, total_time, 0, f"HTTP {response.status_code}: {error_text}"
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, 0, 0, str(e)

def find_batch_size_limit(available_images):
    """Find the maximum batch size before hitting memory limits"""
    print("FINDING MAXIMUM BATCH SIZE LIMITS")
    print("=" * 60)
    
    # Test progressively larger batch sizes
    test_sizes = [
        10,    # Baseline (we know this works)
        20,    # 2x
        30,    # 3x  
        50,    # 5x
        75,    # 7.5x
        100,   # 10x
        150,   # 15x
        200,   # 20x
        300,   # 30x
        500,   # 50x
        750,   # 75x
        1000,  # 100x
    ]
    
    results = []
    max_successful_batch = 0
    
    for batch_size in test_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING BATCH SIZE: {batch_size}")
        print(f"{'='*60}")
        
        success, total_time, successful_count, error = test_batch_size_limit(available_images, batch_size)
        
        results.append({
            "batch_size": batch_size,
            "success": success,
            "total_time": total_time,
            "successful_count": successful_count,
            "time_per_image": total_time / batch_size if batch_size > 0 and total_time > 0 else 0,
            "error": error
        })
        
        if success:
            max_successful_batch = batch_size
            print(f"BATCH SIZE {batch_size}: SUCCESS")
        else:
            print(f"BATCH SIZE {batch_size}: FAILED - {error}")
            
            # Check if this looks like an OOM error
            if error and any(keyword in error.lower() for keyword in ['memory', 'oom', 'cuda', 'out of memory', 'allocation']):
                print(f"MEMORY LIMIT REACHED at batch size {batch_size}")
                break
            elif error and 'timeout' in error.lower():
                print(f"TIMEOUT at batch size {batch_size} - may need longer timeout")
                break
        
        # Small delay between tests to let GPU memory clear
        time.sleep(10)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH SIZE LIMIT ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Batch Size':<12} {'Status':<10} {'Time(s)':<10} {'Per Image(s)':<12} {'Error':<30}")
    print("-" * 80)
    
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        error_short = (result['error'][:25] + "...") if result['error'] and len(result['error']) > 25 else (result['error'] or "")
        
        print(f"{result['batch_size']:<12} {status:<10} {result['total_time']:<10.1f} "
              f"{result['time_per_image']:<12.2f} {error_short:<30}")
    
    print("-" * 80)
    print(f"MAXIMUM SUCCESSFUL BATCH SIZE: {max_successful_batch}")
    
    if max_successful_batch > 0:
        # Find the best performing batch size
        successful_results = [r for r in results if r['success'] and r['time_per_image'] > 0]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['time_per_image'])
            print(f"OPTIMAL BATCH SIZE: {best_result['batch_size']} ({best_result['time_per_image']:.2f}s per image)")
            
            # Calculate theoretical throughput
            images_per_minute = 60 / best_result['time_per_image']
            print(f"THEORETICAL THROUGHPUT: {images_per_minute:.0f} images/minute")
            
            # GPU memory efficiency estimate
            gpu_memory_gb = 40  # A100 40GB
            print(f"GPU MEMORY UTILIZATION: {max_successful_batch} images fit in {gpu_memory_gb}GB")
            print(f"MEMORY PER IMAGE: ~{gpu_memory_gb*1024/max_successful_batch:.1f}MB per image")
    
    return max_successful_batch, results

def main():
    """Find the maximum batch size we can handle"""
    
    # Load diverse test images from all PDFs
    available_images = load_diverse_test_images(max_images=1000)
    
    if len(available_images) == 1:
        print(f"Only 1 image available - will recycle for larger batches")
    
    print(f"Available for testing: {len(available_images)} diverse images")
    
    # Find batch size limits
    max_batch, results = find_batch_size_limit(available_images)
    
    print(f"\nFINAL RESULTS:")
    print(f"Maximum batch size: {max_batch} images")
    print(f"This means you can process up to {max_batch} pages in a single API call!")
    
    if max_batch >= 100:
        print(f"EXCELLENT: Can handle 100+ image batches!")
    elif max_batch >= 50:
        print(f"GOOD: Can handle 50+ image batches") 
    elif max_batch >= 20:
        print(f"DECENT: Can handle 20+ image batches")
    else:
        print(f"LIMITED: Max batch size is quite small")

if __name__ == "__main__":
    main()