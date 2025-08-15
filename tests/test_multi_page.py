import time
import requests
import base64
import json
from PIL import Image
import io
import fitz  # PyMuPDF

def pdf_to_base64_pages(pdf_path, dpi=150, max_pages=None):
    """Convert all PDF pages to base64 encoded images"""
    doc = fitz.open(pdf_path)
    pages = []
    
    total_pages = len(doc)
    if max_pages:
        total_pages = min(total_pages, max_pages)
    
    print(f"Converting {total_pages} pages from PDF...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image and then to base64
        pil_image = Image.open(io.BytesIO(img_data))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        pages.append({
            "page_number": page_num,
            "base64": img_base64,
            "size": pil_image.size
        })
        print(f"  Page {page_num}: {pil_image.size}")
    
    doc.close()
    return pages

def test_multi_page_batch():
    # Configuration
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    modal_endpoint = "https://marker--dotsocr-batch-v2.modal.run"
    max_pages = 3  # Limit for testing
    
    print(f"Testing multi-page batch OCR with {pdf_path}")
    print("=" * 60)
    
    # Convert all pages to base64
    print("Step 1: Converting PDF pages to images...")
    start_convert = time.time()
    pages = pdf_to_base64_pages(pdf_path, max_pages=max_pages)
    convert_time = time.time() - start_convert
    print(f"[OK] {len(pages)} pages converted in {convert_time:.2f}s")
    
    # Prepare batch request
    images_b64 = [page["base64"] for page in pages]
    total_size = sum(len(img) for img in images_b64)
    
    request_data = {
        "images": images_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    print(f"\nStep 2: Sending batch request...")
    print(f"Total images: {len(images_b64)}")
    print(f"Total data size: {total_size:,} characters")
    
    # Send batch request
    start_request = time.time()
    try:
        response = requests.post(modal_endpoint, json=request_data, timeout=600)  # 10 minute timeout
        end_request = time.time()
        
        request_time = end_request - start_request
        total_time = convert_time + request_time
        avg_time_per_page = request_time / len(pages)
        
        print(f"[OK] Batch request completed in {request_time:.2f}s")
        print(f"[INFO] Average time per page: {avg_time_per_page:.2f}s")
        print(f"[DONE] Total time: {total_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                batch_results = result.get("results", [])
                total_pages = result.get("total_pages", 0)
                
                print(f"\n[SUCCESS] Processed {total_pages} pages")
                
                # Save results
                output_data = {
                    "pdf_path": pdf_path,
                    "total_pages": total_pages,
                    "processing_time": request_time,
                    "avg_time_per_page": avg_time_per_page,
                    "results": batch_results
                }
                
                with open("multi_page_results.json", "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"[SAVED] Results saved to multi_page_results.json")
                
                # Print summary for each page
                print(f"\n[SUMMARY] Per-page results:")
                print("-" * 50)
                for i, page_result in enumerate(batch_results):
                    if page_result.get("success"):
                        ocr_text = page_result.get("result", "")
                        try:
                            # Try to parse as JSON to count elements
                            json_data = json.loads(ocr_text)
                            element_count = len(json_data) if isinstance(json_data, list) else 1
                        except:
                            element_count = "N/A"
                        
                        print(f"Page {i}: {len(ocr_text):,} chars, {element_count} elements")
                        
                        # Show first element for verification
                        if ocr_text and len(ocr_text) > 50:
                            preview = ocr_text[:100].replace('\n', ' ')
                            print(f"  Preview: {preview}...")
                    else:
                        error = page_result.get("error", "Unknown error")
                        print(f"Page {i}: ERROR - {error}")
                print("-" * 50)
                
            else:
                print(f"[ERROR] Batch OCR failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"[ERROR] HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out after 10 minutes")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def test_parallel_vs_sequential_comparison():
    """Compare parallel vs sequential batch processing"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Sequential vs Parallel Batch")
    print("=" * 60)
    
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    sequential_endpoint = "https://marker--dotsocr-batch-v2.modal.run"
    parallel_endpoint = "https://marker--dotsocr-batch-parallel-v2.modal.run"
    test_pages = 3  # Test with 3 pages for comparison
    
    # Convert pages
    pages = pdf_to_base64_pages(pdf_path, max_pages=test_pages)
    images_b64 = [page["base64"] for page in pages]
    
    print(f"Testing with {test_pages} pages...")
    
    # Test sequential batch endpoint
    print(f"\n1. Testing Sequential Batch Endpoint...")
    start_time = time.time()
    sequential_request = {
        "images": images_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(sequential_endpoint, json=sequential_request, timeout=600)
        sequential_time = time.time() - start_time
        print(f"Sequential batch: {sequential_time:.2f}s ({sequential_time/test_pages:.2f}s per page)")
        sequential_success = response.status_code == 200
    except Exception as e:
        print(f"Sequential batch failed: {e}")
        sequential_time = None
        sequential_success = False
    
    # Test parallel batch endpoint
    print(f"\n2. Testing Parallel Batch Endpoint...")
    start_time = time.time()
    parallel_request = {
        "images": images_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(parallel_endpoint, json=parallel_request, timeout=600)
        parallel_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            timing_info = result.get("timing", {})
            launch_time = timing_info.get("launch_time", 0)
            collection_time = timing_info.get("collection_time", 0)
            
            print(f"Parallel batch: {parallel_time:.2f}s ({parallel_time/test_pages:.2f}s per page)")
            print(f"  Launch time: {launch_time:.2f}s")
            print(f"  Collection time: {collection_time:.2f}s")
            parallel_success = True
        else:
            print(f"Parallel batch failed with status {response.status_code}")
            parallel_success = False
    except Exception as e:
        print(f"Parallel batch failed: {e}")
        parallel_time = None
        parallel_success = False
    
    # Compare results
    if sequential_time and parallel_time and sequential_success and parallel_success:
        improvement = sequential_time - parallel_time
        percentage = (improvement / sequential_time) * 100
        speedup = sequential_time / parallel_time
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"Sequential: {sequential_time:.2f}s")
        print(f"Parallel:   {parallel_time:.2f}s")
        print(f"Improvement: {improvement:.2f}s ({percentage:.1f}% faster)")
        print(f"Speedup: {speedup:.1f}x")
        
        if percentage > 50:
            print("🚀 Excellent parallel speedup!")
        elif percentage > 20:
            print("✅ Good parallel performance")
        else:
            print("⚠️ Limited parallel benefit")


def test_single_vs_batch_comparison():
    """Compare single page processing vs batch processing"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Single vs Batch")
    print("=" * 60)
    
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    single_endpoint = "https://marker--dotsocr-v2.modal.run"
    batch_endpoint = "https://marker--dotsocr-batch-v2.modal.run"
    
    # Convert first page only
    pages = pdf_to_base64_pages(pdf_path, max_pages=1)
    image_b64 = pages[0]["base64"]
    
    # Test single endpoint
    print("\nTesting single endpoint...")
    start_time = time.time()
    single_request = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(single_endpoint, json=single_request, timeout=300)
        single_time = time.time() - start_time
        print(f"Single endpoint: {single_time:.2f}s")
    except Exception as e:
        print(f"Single endpoint failed: {e}")
        single_time = None
    
    # Test batch endpoint
    print("Testing batch endpoint...")
    start_time = time.time()
    batch_request = {
        "images": [image_b64],
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(batch_endpoint, json=batch_request, timeout=300)
        batch_time = time.time() - start_time
        print(f"Batch endpoint: {batch_time:.2f}s")
    except Exception as e:
        print(f"Batch endpoint failed: {e}")
        batch_time = None
    
    if single_time and batch_time:
        overhead = batch_time - single_time
        print(f"\nBatch overhead: {overhead:.2f}s ({overhead/single_time*100:.1f}%)")


def test_large_batch():
    """Test with larger number of pages to demonstrate parallel scaling"""
    print("\n" + "=" * 60)
    print("LARGE BATCH TEST: 9 Pages")
    print("=" * 60)
    
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    parallel_endpoint = "https://marker--dotsocr-batch-parallel-v2.modal.run"
    
    # Convert more pages for scaling test
    pages = pdf_to_base64_pages(pdf_path, max_pages=9)
    images_b64 = [page["base64"] for page in pages]
    
    print(f"Testing parallel processing with {len(pages)} pages...")
    print(f"Expected sequential time: ~{len(pages) * 15:.0f} seconds")
    print(f"Expected parallel time: ~15-20 seconds")
    
    start_time = time.time()
    request_data = {
        "images": images_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(parallel_endpoint, json=request_data, timeout=1200)  # 20 minute timeout
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            timing_info = result.get("timing", {})
            results = result.get("results", [])
            
            successful_pages = sum(1 for r in results if r.get("success"))
            
            print(f"\n🎉 SUCCESS! Processed {successful_pages}/{len(pages)} pages")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average per page: {total_time/len(pages):.2f}s")
            print(f"Launch time: {timing_info.get('launch_time', 0):.2f}s")
            print(f"Collection time: {timing_info.get('collection_time', 0):.2f}s")
            
            # Calculate theoretical improvement
            sequential_estimate = len(pages) * 15
            improvement = ((sequential_estimate - total_time) / sequential_estimate) * 100
            print(f"Estimated improvement vs sequential: {improvement:.1f}%")
            
            # Save results
            with open("large_batch_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Results saved to large_batch_results.json")
            
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Large batch test failed: {e}")

if __name__ == "__main__":
    print("Testing multi-page OCR batch processing")
    test_multi_page_batch()
    test_single_vs_batch_comparison()
    test_parallel_vs_sequential_comparison()
    test_large_batch()