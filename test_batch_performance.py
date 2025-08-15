import requests
import time
import json
from pathlib import Path
import base64

# Modal endpoints
GENERATE_URL = "https://marker--dotsocr-v2.modal.run"

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

def test_batch_processing(image_b64, num_images=5):
    """Test true batch processing performance"""
    print(f"\n=== Testing TRUE Batch Processing ({num_images} images) ===")
    
    # Create array of the same image (simulating multi-page document)
    images = [image_b64] * num_images
    
    payload = {
        "images": images,
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
            timeout=600  # 10 minutes for batch processing
        )
        total_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Batch Time: {total_time:.2f}s")
        print(f"Average Time per Image: {total_time/num_images:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', 'N/A')}")
            print(f"Processing Mode: {result.get('processing_mode', 'N/A')}")
            print(f"Total Pages Processed: {result.get('total_pages', 'N/A')}")
            
            if result.get('success') and result.get('results'):
                results = result.get('results', [])
                successful_results = [r for r in results if r.get('success')]
                print(f"Successful Results: {len(successful_results)}/{len(results)}")
                
                if successful_results:
                    avg_result_length = sum(len(r.get('result', '')) for r in successful_results) / len(successful_results)
                    print(f"Average Result Length: {avg_result_length:.0f} chars")
                
                print("BATCH PROCESSING PASSED")
                return True, total_time, total_time/num_images
            else:
                print(f"Batch processing failed: {result.get('error', 'Unknown error')}")
                return False, total_time, 0
        else:
            print(f"HTTP Error: {response.text}")
            return False, total_time, 0
            
    except Exception as e:
        print(f"Batch Processing ERROR: {e}")
        return False, 0, 0

def test_sequential_processing_simulation(image_b64, num_images=5):
    """Simulate sequential processing by calling single image endpoint multiple times"""
    print(f"\n=== Testing Sequential Processing Simulation ({num_images} images) ===")
    
    results = []
    total_time = 0
    
    for i in range(num_images):
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
                timeout=300
            )
            request_time = time.time() - start_time
            total_time += request_time
            
            print(f"Image {i+1}: {request_time:.2f}s", end=" | ")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    results.append(result)
                    print("SUCCESS")
                else:
                    print(f"FAILED: {result.get('error', 'Unknown')}")
            else:
                print(f"HTTP ERROR: {response.status_code}")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    avg_time_per_image = total_time / num_images if num_images > 0 else 0
    successful_count = len(results)
    
    print(f"\nSequential Processing Summary:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Image: {avg_time_per_image:.2f}s")
    print(f"Successful Results: {successful_count}/{num_images}")
    
    return successful_count == num_images, total_time, avg_time_per_image

def main():
    """Compare batch vs sequential processing performance"""
    print("BATCH VS SEQUENTIAL OCR PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Load test image
    image_b64 = load_test_image()
    print(f"Image base64 length: {len(image_b64)}")
    
    # Test different batch sizes
    test_sizes = [2, 5, 10]
    
    results_summary = []
    
    for num_images in test_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {num_images} IMAGES")
        print(f"{'='*60}")
        
        # Test batch processing
        batch_success, batch_total_time, batch_avg_time = test_batch_processing(image_b64, num_images)
        
        time.sleep(5)  # Brief pause between tests
        
        # Test sequential processing
        seq_success, seq_total_time, seq_avg_time = test_sequential_processing_simulation(image_b64, num_images)
        
        # Calculate speedup
        if batch_success and seq_success and batch_total_time > 0:
            speedup = seq_total_time / batch_total_time
            efficiency = (seq_avg_time - batch_avg_time) / seq_avg_time * 100
            
            results_summary.append({
                "num_images": num_images,
                "batch_time": batch_total_time,
                "sequential_time": seq_total_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "batch_avg": batch_avg_time,
                "seq_avg": seq_avg_time
            })
            
            print(f"\nPERFORMANCE COMPARISON ({num_images} images):")
            print(f"Batch Processing:     {batch_total_time:>8.2f}s ({batch_avg_time:.2f}s per image)")
            print(f"Sequential Processing: {seq_total_time:>8.2f}s ({seq_avg_time:.2f}s per image)")
            print(f"Speedup:              {speedup:>8.1f}x faster")
            print(f"Efficiency Gain:      {efficiency:>8.1f}%")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Images':<8} {'Batch(s)':<10} {'Sequential(s)':<14} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 60)
    
    for result in results_summary:
        print(f"{result['num_images']:<8} {result['batch_time']:<10.1f} {result['sequential_time']:<14.1f} "
              f"{result['speedup']:<10.1f}x {result['efficiency']:<11.1f}%")
    
    if results_summary:
        avg_speedup = sum(r['speedup'] for r in results_summary) / len(results_summary)
        avg_efficiency = sum(r['efficiency'] for r in results_summary) / len(results_summary)
        print("-" * 60)
        print(f"{'AVERAGE':<8} {'':<10} {'':<14} {avg_speedup:<10.1f}x {avg_efficiency:<11.1f}%")

if __name__ == "__main__":
    main()